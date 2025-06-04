#!/usr/bin/env python3
# ai-detection.py

import os
import sys
import json
import time
import random
import torch
import requests

from pathlib import Path
from typing import List, Tuple, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Detection mode: "zeroshot", "supervised", or "both"
METHOD = "both"

# ─── Zero-Shot (DetectGPT4Code) ────────────────────────────────────────────────

SURROGATE_MODEL  = "Salesforce/codegen-350M-mono"
SURROGATE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INFILL_MODEL     = "facebook/incoder-1B"
INFILL_DEVICE    = SURROGATE_DEVICE

ZS_N_PERTURB = 20       # Number of perturbations per file
ZS_MASK_FRAC = 0.1      # Fraction of lines to mask
ZS_GAMMA     = 0.9      # Fraction of trailing tokens to score

# ─── Supervised MageCode ───────────────────────────────────────────────────────

SEMANTIC_MODEL = "Salesforce/codet5-base"
SEMANTIC_DEVICE = SURROGATE_DEVICE

MAGECODE_CLASSIFIER_PATH = "magecode_classifier.pt"
# If this file is missing, supervised mode will be skipped.

# ─── GitHub / File Selection ───────────────────────────────────────────────────

GITHUB_API   = "https://api.github.com"
GITHUB_TOKEN = "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"
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
}

# File extensions to consider (for both search and fallback)
EXTENSIONS = ["py", "js", "java", "cpp", "c", "rs", "go"]
MAX_FILES  = 20

# ──────────────────────────────────────────────────────────────────────────────
# 2) INITIALIZE TOKENIZERS & MODELS
# ──────────────────────────────────────────────────────────────────────────────

# 2.1) SURROGATE CodeGen for log-probs + metric features
surrogate_tokenizer = AutoTokenizer.from_pretrained(SURROGATE_MODEL)
surrogate_model     = AutoModelForCausalLM.from_pretrained(SURROGATE_MODEL).eval().to(SURROGATE_DEVICE)

if hasattr(surrogate_model.config, "n_positions"):
    SURROGATE_MAX_LEN = surrogate_model.config.n_positions
elif hasattr(surrogate_model.config, "n_ctx"):
    SURROGATE_MAX_LEN = surrogate_model.config.n_ctx
else:
    SURROGATE_MAX_LEN = 1024

# 2.2) INFILLING (Incoder as causal LM; we manually insert "<mask>")
infill_tokenizer = AutoTokenizer.from_pretrained(INFILL_MODEL)
infill_model     = AutoModelForCausalLM.from_pretrained(INFILL_MODEL).eval().to(INFILL_DEVICE)

# 2.3) SEMANTIC Encoder (CodeT5)
semantic_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL)
semantic_model     = AutoModel.from_pretrained(SEMANTIC_MODEL).eval().to(SEMANTIC_DEVICE)

# 2.4) SUPERVISED CLASSIFIER (MageCode)
classifier = None
if METHOD in ("supervised", "both"):
    if os.path.isfile(MAGECODE_CLASSIFIER_PATH):
        classifier = torch.load(MAGECODE_CLASSIFIER_PATH, map_location=SEMANTIC_DEVICE)
        classifier.eval()
    else:
        print(f"[WARN] Could not find `{MAGECODE_CLASSIFIER_PATH}`; skipping supervised method.")

# ──────────────────────────────────────────────────────────────────────────────
# 3) ZERO-SHOT CURVATURE HELPERS (DetectGPT4Code-style)
# ──────────────────────────────────────────────────────────────────────────────

def compute_trailing_logprob(code_str: str, gamma: float = ZS_GAMMA) -> float:
    """
    Tokenize `code_str`, run SURROGATE_MODEL to get per-token logits,
    then sum the log-prob of only the *last* ceil((1−gamma)·N) tokens.
    """
    toks     = surrogate_tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=SURROGATE_MAX_LEN
    ).to(SURROGATE_DEVICE)
    input_ids = toks["input_ids"]  # (1, seq_len)
    seq_len   = input_ids.size(1)
    if seq_len < 2:
        return 0.0

    with torch.no_grad():
        outputs = surrogate_model(input_ids)
        logits  = outputs.logits  # (1, seq_len, V)

    log_probs = []
    for i in range(1, seq_len):
        logp_i   = torch.log_softmax(logits[0, i - 1], dim=-1)  # (V,)
        token_id = input_ids[0, i].item()
        log_probs.append(logp_i[token_id].item())

    N = len(log_probs)
    start_idx = int(gamma * N)
    if start_idx < 0:
        start_idx = 0
    return sum(log_probs[start_idx:])


def generate_perturbation_infill(code_str: str, mask_frac: float = ZS_MASK_FRAC) -> str:
    """
    Mask out a random contiguous span ≈ mask_frac·total_lines with "<mask>",
    then use Incoder (causal) to infill, returning the perturbed code.
    """
    lines       = code_str.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return code_str

    mask_count = max(1, int(total_lines * mask_frac))
    start_idx  = random.randint(0, total_lines - mask_count)
    end_idx    = start_idx + mask_count

    prefix = "\n".join(lines[:start_idx])
    suffix = "\n".join(lines[end_idx:])

    masked_code = ""
    if prefix:
        masked_code += prefix + "\n"
    masked_code += "<mask>"
    if suffix:
        masked_code += "\n" + suffix

    toks = infill_tokenizer(
        masked_code,
        return_tensors="pt",
        truncation=True,
        max_length=SURROGATE_MAX_LEN
    ).to(INFILL_DEVICE)

    unmasked_text = prefix + "\n" + suffix
    unmasked_toks = infill_tokenizer(
        unmasked_text,
        return_tensors="pt",
        truncation=True,
        max_length=SURROGATE_MAX_LEN
    )["input_ids"]
    avg_per_line  = max(1, unmasked_toks.size(1) // max(1, total_lines - mask_count))
    max_new       = mask_count * avg_per_line + 10

    with torch.no_grad():
        generated = infill_model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            bos_token_id=infill_tokenizer.bos_token_id,
            eos_token_id=infill_tokenizer.eos_token_id,
            pad_token_id=infill_tokenizer.pad_token_id,
        )

    gen_ids = generated[0].tolist()
    decoded = infill_tokenizer.decode(gen_ids, skip_special_tokens=False)

    parts = decoded.split("<mask>")
    if len(parts) >= 2:
        infilled  = parts[1]
        pert_code = ""
        if prefix:
            pert_code += prefix + "\n"
        pert_code += infilled.strip()
        if suffix:
            pert_code += "\n" + suffix
        return pert_code

    return code_str


def is_ai_zero_shot(
    code_str: str,
    n_perturb: int = ZS_N_PERTURB,
    mask_frac: float = ZS_MASK_FRAC,
    gamma: float = ZS_GAMMA
) -> Tuple[bool, float]:
    """
    Return (is_ai, curvature_gap) per DetectGPT4Code:
      gap = trailing_logprob(orig) − average(trailing_logprob(perturbed_i))
    If gap > 0 → “AI”.
    """
    baseline_lp = compute_trailing_logprob(code_str, gamma=gamma)
    pert_lps     = []
    for _ in range(n_perturb):
        pert = generate_perturbation_infill(code_str, mask_frac=mask_frac)
        pert_lps.append(compute_trailing_logprob(pert, gamma=gamma))
    avg_pert_lp = sum(pert_lps) / len(pert_lps)
    gap = baseline_lp - avg_pert_lp
    return (gap > 0.0, gap)


# ──────────────────────────────────────────────────────────────────────────────
# 4) SUPERVISED MageCode HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_semantic_embedding(code_str: str) -> torch.Tensor:
    """
    Use CodeT5 to get a [CLS] embedding (shape: [hidden_size]).
    """
    toks = semantic_tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(SEMANTIC_DEVICE)

    with torch.no_grad():
        outputs = semantic_model(**toks, return_dict=True)
        cls_emb  = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    return cls_emb.squeeze(0)  # (hidden_size,)


def compute_metric_features(code_str: str) -> List[float]:
    """
    Compute six “metric” features using SURROGATE_MODEL:
      1) avg_token_logprob
      2) avg_rank
      3) avg_entropy
      4) prop_top1
      5) prop_top10
      6) LRR (Language-Model Repetition Rate)
    """
    toks        = surrogate_tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=SURROGATE_MAX_LEN
    ).to(SURROGATE_DEVICE)
    input_ids   = toks["input_ids"]      # (1, L)
    attention_m = toks["attention_mask"]
    L           = input_ids.size(1)
    if L < 2:
        return [0.0] * 6

    with torch.no_grad():
        outputs = surrogate_model(input_ids, attention_mask=attention_m)
        logits  = outputs.logits  # (1, L, V)

    total_lp     = 0.0
    total_rank   = 0.0
    total_ent    = 0.0
    top1_count   = 0
    top10_count  = 0
    prev_tokens  = []
    repeated_cnt = 0
    window_size  = 50

    for i in range(1, L):
        logp_dist = torch.log_softmax(logits[0, i - 1], dim=-1)  # (V,)
        probs     = torch.exp(logp_dist)                         # (V,)
        tid       = input_ids[0, i].item()

        total_lp   += logp_dist[tid].item()
        total_rank += (logp_dist > logp_dist[tid]).sum().item() + 1
        total_ent  += -torch.sum(probs * logp_dist).item()

        topk = torch.topk(probs, k=10).indices.tolist()
        if tid == topk[0]:
            top1_count += 1
        if tid in topk:
            top10_count += 1

        if tid in prev_tokens[-window_size:]:
            repeated_cnt += 1
        prev_tokens.append(tid)

    n = L - 1
    return [
        total_lp / n,
        total_rank / n,
        total_ent / n,
        top1_count / n,
        top10_count / n,
        repeated_cnt / n
    ]


def is_ai_supervised(code_str: str) -> Optional[Tuple[bool, float]]:
    """
    Run MageCode:
      1) Get CodeT5 [CLS] embedding
      2) Compute six metric features
      3) Concatenate → MLP classifier
      4) Output p(AI); threshold 0.5
    """
    if classifier is None:
        return None

    sem_emb = get_semantic_embedding(code_str)  # (hidden_size,)
    metr    = compute_metric_features(code_str) # list of 6 floats

    metr_tensor = torch.tensor(metr, dtype=torch.float32, device=SEMANTIC_DEVICE)
    feature_vec = torch.cat([sem_emb, metr_tensor], dim=0).unsqueeze(0)  # (1, hidden_size+6)

    with torch.no_grad():
        logits = classifier(feature_vec)
        if logits.size(-1) == 1:
            p_ai = torch.sigmoid(logits).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            p_ai = probs[0, 1].item()
    return (p_ai > 0.5, p_ai)


# ──────────────────────────────────────────────────────────────────────────────
# 5) GITHUB FILE-SELECTION “HYBRID”
# ──────────────────────────────────────────────────────────────────────────────

def get_top_n_file_paths(owner: str, repo: str, n: int) -> List[str]:
    """
    Use GitHub’s /search/code to get up to n most-recently updated files
    in {owner}/{repo} matching EXTENSIONS. Returns a list of “path” strings.
    """
    ext_q = "+".join(f"extension:{e}" for e in EXTENSIONS)
    url   = f"{GITHUB_API}/search/code"
    params = {
        "q":    f"repo:{owner}/{repo}+{ext_q}",
        "sort": "updated",
        "order":"desc",
        "per_page": n
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Code Search failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return [item["path"] for item in data.get("items", [])]


def list_all_files(owner: str, repo: str, branch: str) -> List[str]:
    """
    Fallback: call /git/trees/{branch}?recursive=1 to list every file path.
    Returns a list of “path” strings (blobs only).
    """
    url  = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tree: {resp.status_code} {resp.text}")
    tree = resp.json().get("tree", [])
    return [entry["path"] for entry in tree if entry["type"] == "blob"]


def get_file_last_commit_timestamp(
    owner: str, repo: str, path: str, branch: str
) -> int:
    """
    Fallback: call /repos/{owner}/{repo}/commits?path={path}&sha={branch}&per_page=1
    Returns UNIX timestamp of the most recent commit touching `path`, or 0 on error.
    """
    url    = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {"path": path, "sha": branch, "per_page": 1}
    resp   = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return 0
    commits = resp.json()
    if not commits:
        return 0
    date_str = commits[0]["commit"]["committer"]["date"]  # e.g. "2025-05-28T14:23:00Z"
    t_struct = time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(time.mktime(t_struct))


def fetch_raw_file_content(
    owner: str, repo: str, path: str, branch: str
) -> str:
    """
    Download the file text via raw.githubusercontent.com.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch raw file: {raw_url} → {resp.status_code}")
    return resp.text


def get_default_branch(owner: str, repo: str) -> str:
    """
    Returns the default branch (e.g. “main” or “master”).
    """
    url  = f"{GITHUB_API}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch repo info: {resp.status_code} {resp.text}")
    return resp.json().get("default_branch", "main")


# ──────────────────────────────────────────────────────────────────────────────
# 6) MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    repo_url = "https://github.com/keras-team/keras"

    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    # Parse “owner/repo” from various GitHub URL formats
    if repo_url.startswith("git@github.com:"):
        _, path = repo_url.split(":", 1)
    elif repo_url.startswith("https://github.com/"):
        path = repo_url[len("https://github.com/"):]
    elif repo_url.startswith("http://github.com/"):
        path = repo_url[len("http://github.com/"):]
    else:
        print("[ERROR] Unsupported GitHub URL format.")
        sys.exit(1)

    path = path.rstrip("/")
    parts = path.split("/")
    if len(parts) != 2:
        print("[ERROR] URL must be of the form github.com/owner/repo")
        sys.exit(1)
    owner, repo = parts

    print(f"\n→ Scanning GitHub repo: {owner}/{repo}\n")

    # 6.1) Find default branch
    default_branch = get_default_branch(owner, repo)
    print(f"• Default branch: '{default_branch}'\n")

    # 6.2) Attempt Code Search for top N files
    top_paths: List[str] = []
    try:
        print(f"• Trying GitHub Code Search for top {MAX_FILES} files…")
        top_paths = get_top_n_file_paths(owner, repo, MAX_FILES)
    except Exception as e:
        print(f"[WARN] Code Search failed: {e}")

    # 6.3) If Code Search yields none, fall back to listing & commit-dating all files
    if not top_paths:
        print("• Code Search returned no results. Falling back to tree+commits…")
        all_paths = list_all_files(owner, repo, default_branch)
        # Filter by extension
        candidates = [
            p for p in all_paths
            if any(p.lower().endswith(f".{ext}") for ext in EXTENSIONS)
        ]
        print(f"  → Found {len(candidates)} total source-file paths.")

        # Get each file’s last commit timestamp
        dated: List[Tuple[str,int]] = []
        for p in candidates:
            ts = get_file_last_commit_timestamp(owner, repo, p, default_branch)
            if ts > 0:
                dated.append((p, ts))
        if not dated:
            print("  → No valid commit timestamps. Exiting.")
            sys.exit(0)

        # Sort by timestamp descending, pick top MAX_FILES
        dated.sort(key=lambda x: x[1], reverse=True)
        top_paths = [p for (p, _) in dated[:MAX_FILES]]
        print(f"  → Selected {len(top_paths)} most-recent files via fallback.\n")
    else:
        print(f"• Code Search found {len(top_paths)} files. Fetching their contents…\n")

    # 6.4) For each selected file, fetch content & run detection
    results = []
    for path in top_paths:
        try:
            code = fetch_raw_file_content(owner, repo, path, default_branch)
        except Exception as e:
            print(f"[WARN] Skipping {path} (fetch error): {e}")
            continue

        entry = {"path": path}

        # 6.4.1) Zero-Shot Curvature
        if METHOD in ("zeroshot", "both"):
            try:
                ai_zs, gap = is_ai_zero_shot(code)
                entry["zero_shot"] = {"is_ai": ai_zs, "curvature_gap": gap}
                print(f"[ZS] {'[AI]' if ai_zs else '[H]'}  {path}  (gap={gap:.2f})")
            except Exception as e:
                entry["zero_shot"] = {"error": str(e)}
                print(f"[ZS][ERR] {path} → {e}")

        # 6.4.2) Supervised MageCode
        if METHOD in ("supervised", "both") and (classifier is not None):
            try:
                sup = is_ai_supervised(code)
                if sup is None:
                    entry["supervised"] = {"skipped": True}
                    print(f"[SP] Skipped supervised for {path}")
                else:
                    ai_sp, p_ai = sup
                    entry["supervised"] = {"is_ai": ai_sp, "prob_ai": p_ai}
                    print(f"[SP] {'[AI]' if ai_sp else '[H]'}  {path}  (p_ai={p_ai:.3f})")
            except Exception as e:
                entry["supervised"] = {"error": str(e)}
                print(f"[SP][ERR] {path} → {e}")

        results.append(entry)

    # 6.5) Write JSON summary
    summary = {
        "repo": f"{owner}/{repo}",
        "method": METHOD,
        "files_analyzed": len(results),
        "timestamp_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results
    }
    out_path = Path.cwd() / f"{repo.replace('/', '_')}_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n→ Results written to {out_path}\n")


if __name__ == "__main__":
    main()
