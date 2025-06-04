#!/usr/bin/env python3
# detect_repo_gen_github_fast.py

import os
import sys
import json
import time
import random
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration & thresholds (all hyperparameters at top)
# ──────────────────────────────────────────────────────────────────────────────

# Map file extensions to language-specialized surrogate checkpoints
SURROGATE_BY_EXT = {
    ".py":   "microsoft/CodeGPT-small-py",
    ".java": "Salesforce/codegen-350M-mono-java",
    ".cpp":  "NinedayWang/PolyCoder-160M",
    ".c":    "NinedayWang/PolyCoder-160M",
    ".js":   "Salesforce/codegen-350M-mono-javascript",
    ".rs":   "Salesforce/codegen-350M-mono-rust",
    ".go":   "Salesforce/codegen-350M-mono-go",
    # If you find R-specific or other-language checkpoints, add here.
}

# Fallback generic multi-language surrogate
FALLBACK_SURROGATE = "Salesforce/codet5-small"

# FIM model (used to generate perturbations)—we use the same for all languages
FIM_MODEL = "facebook/incoder-1B"

# Device (CPU only, since CUDA is not available)
DEVICE = "cpu"

# Number of perturbations (paper suggests 20–32 for best accuracy)
N_PERTURB = 8

# Number of contiguous lines to mask in each perturbation
MASK_LINES = 8

# Truncation ratio γ for suffix scoring (paper uses ~0.99)
GAMMA = 0.99

# File extensions to consider for analysis
SOURCE_EXTS = (".py", ".js", ".java", ".cpp", ".c", ".rs", ".go", ".R", ".r")

# Only analyze the top K files by “last commit”
MAX_FILES = 5

# Skip any file longer than this many lines (avoid huge contexts)
MAX_LINES_PER_FILE = 1000

# Per-file timeout threshold (in seconds). If detect_one_snippet()
# takes longer than this, we skip that file immediately.
TIMEOUT_PER_FILE = 60

# How many threads to use for fetching timestamps in parallel
TIMESTAMP_WORKERS = 20

# How many worker processes to use for detection
DETECT_WORKERS = 1

# GitHub API base
GITHUB_API = "https://api.github.com"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Hardcoded GitHub token (you can replace this with an env var)
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN = "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"
if not GITHUB_TOKEN:
    print("WARNING: No GITHUB_TOKEN found. You will be rate-limited (60 req/hr).", file=sys.stderr)

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
}

# ──────────────────────────────────────────────────────────────────────────────
# 3) Globals loaded once (in each worker) via init_worker()
# ──────────────────────────────────────────────────────────────────────────────

# Surrogate caches: maps checkpoint name → tokenizer/model/max_length
surr_tokenizer: dict[str, AutoTokenizer] = {}
surr_model:     dict[str, AutoModelForCausalLM] = {}
SURR_MAX_LEN:   dict[str, int] = {}

# FIM model/tokenizer/global max length
fim_tokenizer = None
fim_model     = None
FIM_MAX_LEN   = None

def init_worker():
    """
    This initializer runs exactly once per worker process when the Pool is created.
    It loads the FIM model into globals. Surrogates are loaded lazily per-file.
    """
    global fim_tokenizer, fim_model, FIM_MAX_LEN

    if fim_model is None:
        print(f"Loading FIM model ({FIM_MODEL}) in worker…")
        fim_tokenizer = AutoTokenizer.from_pretrained(FIM_MODEL)
        fim_model     = AutoModelForCausalLM.from_pretrained(FIM_MODEL).to(DEVICE).eval()

        cfg = fim_model.config
        if hasattr(cfg, "n_positions"):
            FIM_MAX_LEN = cfg.n_positions
        elif hasattr(cfg, "n_ctx"):
            FIM_MAX_LEN = cfg.n_ctx
        elif hasattr(cfg, "max_position_embeddings"):
            FIM_MAX_LEN = cfg.max_position_embeddings
        else:
            raise ValueError("Cannot determine FIM_MAX_LEN from model config")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Helper: lazy-load surrogate by file extension
# ──────────────────────────────────────────────────────────────────────────────

def get_surrogate_for_extension(ext: str):
    """
    Given a file extension (e.g. ".py", ".java"), return (tokenizer, model, max_len).
    Loads and caches the surrogate the first time it’s needed in this process.
    """
    global surr_tokenizer, surr_model, SURR_MAX_LEN

    # Choose checkpoint name
    name = SURROGATE_BY_EXT.get(ext, FALLBACK_SURROGATE)

    # Load if not already in cache
    if name not in surr_model:
        print(f"  → Loading surrogate model {name} for *{ext}* files…")
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()

        cfg = mod.config
        if hasattr(cfg, "n_positions"):
            max_len = cfg.n_positions
        elif hasattr(cfg, "n_ctx"):
            max_len = cfg.n_ctx
        elif hasattr(cfg, "max_position_embeddings"):
            max_len = cfg.max_position_embeddings
        else:
            raise ValueError("Cannot determine max_length from surrogate config")

        surr_tokenizer[name] = tok
        surr_model[name]     = mod
        SURR_MAX_LEN[name]   = max_len

    return surr_tokenizer[name], surr_model[name], SURR_MAX_LEN[name]

# ──────────────────────────────────────────────────────────────────────────────
# 5) Core functions: compute_suffix_logprob & make_fim_perturbation
# ──────────────────────────────────────────────────────────────────────────────

def compute_suffix_logprob(
    code: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_len: int,
    gamma: float,
) -> float:
    """
    Compute log-probability of only the suffix (γ-truncation).
    """
    enc = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False
    ).to(DEVICE)

    input_ids = enc["input_ids"][0]  # [L], L ≤ max_len
    L = input_ids.size(0)
    if L < 2:
        k = 0
    else:
        k = max(int(gamma * L), 1)

    suffix_len = L - k
    if suffix_len <= 0:
        return float("-inf")

    full_ids = input_ids.unsqueeze(0)  # [1, L]
    labels   = full_ids.clone()        # [1, L]
    labels[0, :k] = -100               # ignore prefix

    with torch.no_grad():
        out = model(full_ids, labels=labels)
        mean_nll = out.loss.item()

    return - mean_nll * suffix_len

def make_fim_perturbation(
    code: str,
    tok: AutoTokenizer,
    mod: AutoModelForCausalLM,
    max_len: int,
    mask_lines: int,
) -> str:
    """
    One FIM perturbation on code:
    - If snippet ≤ mask_lines lines, return code unchanged.
    - Otherwise, remove a random block of mask_lines lines, build:
        <fim_prefix>
        {prefix}
        <fim_suffix>
        {suffix}
      and let Incoder fill in the “masked block.”  
    - Ensure prompt + generated ≤ max_len by computing safe max_new_tokens.
    """
    lines = code.splitlines()
    L_lines = len(lines)
    if L_lines <= mask_lines:
        return code

    start = random.randrange(0, L_lines - mask_lines + 1)
    end = start + mask_lines
    prefix_lines = lines[:start]
    suffix_lines = lines[end:]

    prefix = "\n".join(prefix_lines).rstrip()
    suffix = "\n".join(suffix_lines).lstrip()
    prompt = f"<fim_prefix>\n{prefix}\n<fim_suffix>\n{suffix}"

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False
    ).to(DEVICE)
    inputs.pop("token_type_ids", None)

    prompt_len = inputs["input_ids"].size(1)
    allowed_gen = max_len - prompt_len - 1
    if allowed_gen <= 0:
        return code

    estimated_new = mask_lines * 25
    max_new_tokens = min(estimated_new, allowed_gen)

    gen = mod.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    decoded = tok.decode(gen[0], skip_special_tokens=True).strip()
    if "<fim_suffix>" not in decoded:
        return decoded

    parts = decoded.split("<fim_suffix>", 1)
    after_prefix = parts[1].lstrip("\n\r ")
    final_code = prefix + "\n" + after_prefix
    return final_code

def detect_one_snippet_with_model(
    code_str: str,
    surr_tok: AutoTokenizer,
    surr_mod: AutoModelForCausalLM,
    surr_len: int
) -> float:
    """
    Compute flatness_score = logp_suffix(original)
                       - avg_{i=1..N_PERTURB}(logp_suffix(perturbation_i)).  
    Uses the supplied surrogate tokenizer/model.
    """
    lp_orig = compute_suffix_logprob(code_str, surr_tok, surr_mod, surr_len, GAMMA)

    perturb_lps = []
    for _ in range(N_PERTURB):
        pert_code = make_fim_perturbation(code_str, fim_tokenizer, fim_model, FIM_MAX_LEN, MASK_LINES)
        lp_pert = compute_suffix_logprob(pert_code, surr_tok, surr_mod, surr_len, GAMMA)
        perturb_lps.append(lp_pert)

    avg_pert = sum(perturb_lps) / len(perturb_lps)
    return lp_orig - avg_pert

# ──────────────────────────────────────────────────────────────────────────────
# 6) GitHub helper routines
# ──────────────────────────────────────────────────────────────────────────────

def parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    Parse a GitHub URL and return (owner, repo). Strips any "/tree/..." or "/blob/..." suffix.
    """
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    prefix_https = "https://github.com/"
    prefix_http  = "http://github.com/"
    if repo_url.startswith(prefix_https):
        path = repo_url[len(prefix_https):]
    elif repo_url.startswith(prefix_http):
        path = repo_url[len(prefix_http):]
    else:
        raise ValueError("Unsupported GitHub URL. Expect https://github.com/owner/repo")

    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError("URL must be of the form github.com/owner/repo")
    owner = parts[0]
    repo  = parts[1]
    return owner, repo

def get_default_branch(owner: str, repo: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch repo info: {resp.status_code} {resp.text}")
    return resp.json().get("default_branch", "main")

def list_branches(owner: str, repo: str) -> list[str]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/branches"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 403:
        reset_ts = resp.headers.get("X-RateLimit-Reset")
        if reset_ts:
            reset_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(reset_ts)))
            raise RuntimeError(f"GitHub API rate limit exceeded. Limit resets at {reset_time} UTC.")
        else:
            raise RuntimeError("GitHub API rate limit exceeded (no reset time).")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to list branches: {resp.status_code} {resp.text}")
    return [b["name"] for b in resp.json()]

def list_all_files(owner: str, repo: str, branch: str) -> list[str]:
    """
    Uses the Git Trees API: returns all blob paths in that branch.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tree: {resp.status_code} {resp.text}")
    tree = resp.json().get("tree", [])
    return [entry["path"] for entry in tree if entry["type"] == "blob"]

def get_file_last_commit_timestamp(owner: str, repo: str, path: str, branch: str) -> int:
    """
    Returns the UNIX timestamp of the latest commit that touched path.
    If it fails, returns 0.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {"path": path, "sha": branch, "per_page": 1}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return 0
    commits = resp.json()
    if not commits:
        return 0
    date_str = commits[0]["commit"]["committer"]["date"]  # "YYYY-MM-DDTHH:MM:SSZ"
    t_struct = time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(time.mktime(t_struct))

def fetch_raw_file_content(owner: str, repo: str, path: str, branch: str) -> str:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch raw file: {raw_url} → {resp.status_code}")
    return resp.text

# ──────────────────────────────────────────────────────────────────────────────
# 7) Worker task: runs in each Pool worker
# ──────────────────────────────────────────────────────────────────────────────

def worker_task(owner: str, repo: str, branch: str, rel_path: str, ts: int):
    """
    This function runs in the worker process:
    1) Fetch raw file content. If skip (too many lines or fetch fails), return a skip code.
    2) Select the correct surrogate by extension, loading it if necessary.
    3) Run detect_one_snippet_with_model(...) and return (rel_path, ts, flatness_score, elapsed).
    """
    start = time.time()

    # 1) Fetch content
    try:
        content = fetch_raw_file_content(owner, repo, rel_path, branch)
    except Exception:
        return ("SKIP_FETCH_ERROR", rel_path)

    # 2) Skip if file is too large
    num_lines = len(content.splitlines())
    if num_lines > MAX_LINES_PER_FILE:
        return ("SKIP_TOO_LONG", rel_path, num_lines)

    # 3) Determine surrogate by file extension
    _, ext = os.path.splitext(rel_path.lower())
    surr_tok, surr_mod, surr_len = get_surrogate_for_extension(ext)

    # 4) Run AI-detection
    try:
        flatness_score = detect_one_snippet_with_model(content, surr_tok, surr_mod, surr_len)
    except Exception:
        return ("SKIP_DETECT_ERROR", rel_path)

    elapsed = time.time() - start
    return (rel_path, ts, flatness_score, elapsed)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Main logic: GitHub integration + timestamp fetch + Pool-based detection
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 8.1) URL (modify as needed or accept via CLI)
    repo_url = "https://github.com/SiamakFarshidi/AI_Models"
    TARGET_BRANCH = None

    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"Error parsing URL: {e}")
        sys.exit(1)

    print(f"Owner: {owner}    Repo: {repo}")

    # Fetch branch list (handles rate-limit 403 explicitly)
    try:
        branches = list_branches(owner, repo)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not branches:
        print("No branches found. Exiting.")
        sys.exit(1)

    default_branch = get_default_branch(owner, repo)
    # If URL contained "/tree/<branch>", use that as TARGET_BRANCH, else None
    if "/tree/" in repo_url:
        TARGET_BRANCH = repo_url.rstrip("/").split("/tree/")[-1]
    branch = TARGET_BRANCH if (TARGET_BRANCH in branches) else default_branch
    print(f"Using branch: {branch}")

    # 8.2) List all files & filter by extension
    print("Listing every file in the tree…")
    all_paths = list_all_files(owner, repo, branch)
    candidates = [p for p in all_paths if p.lower().endswith(SOURCE_EXTS)]
    if not candidates:
        print("No source files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(candidates)} total source files. Fetching commit dates in parallel…")

    # 8.3) Fetch each file's last-commit timestamp in parallel
    file_dates = []
    with ThreadPoolExecutor(max_workers=TIMESTAMP_WORKERS) as pool:
        futures = {
            pool.submit(get_file_last_commit_timestamp, owner, repo, path, branch): path
            for path in candidates
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                ts = future.result()
            except Exception:
                ts = 0
            if ts > 0:
                file_dates.append((path, ts))

    if not file_dates:
        print("No valid commit timestamps found. Exiting.")
        sys.exit(1)

    # 8.4) Sort by timestamp descending, keep only top MAX_FILES
    file_dates.sort(key=lambda x: x[1], reverse=True)
    top_files = file_dates[:MAX_FILES]
    print(f"Analyzing up to {len(top_files)} most-recent files (each has a {TIMEOUT_PER_FILE}s timeout)…\n")

    ai_count = 0
    human_count = 0

    # 8.5) Create a Pool of worker processes (each loads FIM + caches surrogates lazily)
    with multiprocessing.Pool(
        processes=DETECT_WORKERS,
        initializer=init_worker
    ) as pool:
        for rel_path, ts in top_files:
            ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))

            async_res = pool.apply_async(worker_task, (owner, repo, branch, rel_path, ts))
            try:
                # Wait up to TIMEOUT_PER_FILE seconds
                result = async_res.get(timeout=TIMEOUT_PER_FILE)
            except multiprocessing.TimeoutError:
                # Timed out: skip this file
                print(f"  • {rel_path}    (last commit: {ts_human} UTC)  →  exceeded {TIMEOUT_PER_FILE}s, SKIPPED")
                continue

            # Unpack result
            if not isinstance(result, tuple):
                print(f"  • {rel_path}    (last commit: {ts_human} UTC)  →  unexpected result, SKIPPED")
                continue

            tag = result[0]
            if tag == "SKIP_FETCH_ERROR":
                print(f"  • {rel_path}    (last commit: {ts_human} UTC)  →  SKIPPED (fetch error)")
                continue
            elif tag == "SKIP_TOO_LONG":
                _, bad_path, num_lines = result
                print(f"  • Skipping {bad_path} (too many lines: {num_lines} > {MAX_LINES_PER_FILE})")
                continue
            elif tag == "SKIP_DETECT_ERROR":
                print(f"  • {rel_path}    (last commit: {ts_human} UTC)  →  SKIPPED (detection error)")
                continue

            # Otherwise: result = (rel_path, ts, flatness_score, elapsed)
            _, _, flatness_score, elapsed = result
            label = "[AI]" if flatness_score > 0 else "[HUMAN]"
            print(f"  {label}    {rel_path}    (last commit: {ts_human} UTC)  →  took {elapsed:.1f}s")

            if flatness_score > 0:
                ai_count += 1
            else:
                human_count += 1

    # 8.6) Summarize results into JSON
    total = ai_count + human_count
    ai_ratio = (ai_count / total) if total > 0 else 0.0
    if ai_ratio > 0.66:
        estimation = "High"
    elif ai_ratio > 0.33:
        estimation = "Medium"
    else:
        estimation = "Low"

    result = {
        "#Files":     total,
        "#Human":     human_count,
        "#AI":        ai_count,
        "ai_ratio":   round(ai_ratio, 3),
        "estimation": estimation
    }

    out_path = Path.cwd() / "repo_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path.resolve()}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
