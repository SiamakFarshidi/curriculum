from django.shortcuts import render
import csv
import os
import json
from difflib import SequenceMatcher
NULL=-10000
#---------------------------------------------------------------------------------------------------------------------

def similar(a, b, threshold=0.75):
    """
    Checks if two strings are similar based on a similarity threshold.
    """
    matcher = SequenceMatcher(None, a, b)
    return matcher.ratio() >= threshold
#---------------------------------------------------------------------------------------------------------------------
def generateSearchSpace(csv_files,folder_path):
    searchSpace={}
    id=0
    for csv_file in csv_files:
        csv_data = open_csv_file(folder_path+csv_file)
        for row in csv_data:
            if 'courseid' not in row:
                searchSpace[id]={'csv_file':csv_file,'row':row}
                id=id+1
    return searchSpace
#---------------------------------------------------------------------------------------------------------------------
def detectSimilarNodes(searchSpace):
    similarNodes={}
    id=0
    for travers1 in searchSpace:
        for travers2 in searchSpace:
            if ((searchSpace[travers1]['csv_file'] != searchSpace[travers2]['csv_file'])) and ("Obj" not in searchSpace[travers1]['row'][6]) and ("Obj" not in searchSpace[travers2]['row'][6]):
                if similar(searchSpace[travers1]['row'][6],searchSpace[travers2]['row'][6]):
                    similarNodes[id]={'node1':searchSpace[travers1],'node2':searchSpace[travers2]}
                    id=id+1
    return similarNodes
#---------------------------------------------------------------------------------------------------------------------

def buildSimilarityGraph(csv_files,folder_path):
    dataset_nodes=[]
    dataset_edges=[]
    searchSpace=generateSearchSpace(csv_files,folder_path)
    similarNodes=detectSimilarNodes(searchSpace)
    #print(similarNodes)
    dataset_nodes,dataset_edges = createSimilarityGraph(similarNodes)

    return dataset_nodes,dataset_edges
#---------------------------------------------------------------------------------------------------------------------

def graph(request):

    try:
        courseTitle = request.GET['courseTitle']
    except:
        courseTitle = ''

    dataset_nodes={}
    dataset_edges={}
    courseTitles=[]
    folder_path=os.getcwd()+"\\graphvisualiztion\\CSV\\"
    csv_files = get_csv_files_in_folder(folder_path)

    for csv_file in csv_files:
        courseTitles.append(csv_file[:len(csv_file)-4])

    id=0

    if courseTitle=="Similarity graph":
        dataset_nodes,dataset_edges= buildSimilarityGraph(csv_files,folder_path)
    else:        
        if courseTitle:
            csv_file=courseTitle+".csv"
        else:
            csv_file = csv_files[0]
            
        if not csv_files:
            print("No CSV files found!")
        else:
            csv_data = open_csv_file(folder_path+csv_file)
            dataset_nodes,dataset_edges,id = createGraph(csv_file,csv_data,id)

               
    return render(request,'graph.html',{
        "dataset_nodes":json.dumps(dataset_nodes),
        "dataset_edges":json.dumps(dataset_edges),
        "courseTitles":courseTitles
    })
#---------------------------------------------------------------------------------------------------------------------
def mergeList(first_list, second_list):
    resulting_list = list(first_list)
    resulting_list.extend(x for x in second_list if x not in resulting_list)
    return resulting_list
#---------------------------------------------------------------------------------------------------------------------

def contentmgmt(request):
       return render(request,'contentmgmt.html') 


#---------------------------------------------------------------------------------------------------------------------
def open_csv_file(file_path):
    """
    Opens a CSV file and returns its contents as a list of lists.
    Each inner list represents a row in the CSV file.
    """
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        csv_data = list(csv_reader)
    return csv_data
#---------------------------------------------------------------------------------------------------------------------
def access_csv_cell(csv_data, row_index, col_index):
    """
    Accesses a cell in the CSV data based on row and column indices.
    Returns the value of the cell.
    """
    if row_index < 0 or row_index >= len(csv_data):
        return None
    if col_index < 0 or col_index >= len(csv_data[row_index]):
        return None
    return csv_data[row_index][col_index]
#---------------------------------------------------------------------------------------------------------------------
def get_csv_files_in_folder(folder_path):
    """
    Retrieves a list of CSV files in the specified folder.
    """
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files
#---------------------------------------------------------------------------------------------------------------------
def createNode(id, cap, url, tooltip, img ,shape='image', size=12,color='white'):

    if len(cap)>30:
        tooltip=cap+"\n"+tooltip

    newNode={
        'id': id,
        'url': url.replace('json',''),
        'label': cap[:30] + (cap[30:] and '...'),
        'title': tooltip[:100]  + (tooltip[100:] and '...'),
        'color':color,
        'shape': shape,
        'image': img,
        'size':size,
    }
    return newNode
#---------------------------------------------------------------------------------------------------------------------
def createEdge(from_node_id, to_node_id):
    Newedge={'from': from_node_id, 'to': to_node_id, 'title': '' }    
    return Newedge
#---------------------------------------------------------------------------------------------------------------------
def createCourseNode(id,label,tooltip):
    Course={
        'id': id,
        'url':'N/A',
        'widthConstraint': { 'maximum': 150,'minimum': 100  },
        'heightConstraint': { 'minimum': 70, 'maximum': 100 },
        'title': tooltip[:100]  + (tooltip[100:] and '...'),
        'label': label,
        'x': -150,
        'y': -150,
        'shape': "circle",
    }
    return Course
#---------------------------------------------------------------------------------------------------------------------
def createGraph(courseTitle,csv_data,id):
    colors=['BlueViolet','DeepPink','Lime','orange','Indigo','DarkSlateBlue','DarkMagenta',"brown","lightgray"]
    nodes=[]
    edges=[]
    nodes.append(createCourseNode(id,courseTitle[:len(courseTitle)-4],csv_data[1][6]))
    CourseID=id
    id=id+1
    
    Prev_courseid	= NULL
    Prev_chapterid	= NULL
    Prev_sectionid	= NULL
    Prev_unitid	    = NULL
    Prev_subunitid   = NULL
    Prev_topic       = NULL
    Prev_chapter     = NULL
    Prev_CognitiveLevel = NULL
    Prev_description = NULL

    cnt=0
    
    for row in csv_data:
        
        if cnt<2: #ignore the first two rows (title)
            cnt=cnt+1
            continue    
                
        Cur_courseid	= row[0]
        Cur_chapterid	= row[1]
        Cur_sectionid	= row[2]
        Cur_unitid	    = row[3]
        Cur_subunitid   = row[4]
        Cur_topic       = row[6]
        Cur_chapter     = row[8]
        Cur_CognitiveLevel = row[10]
        Cur_description = row[12]
        
        if not(Prev_courseid):
            parentID=id
            nodes.append(createNode(id,Cur_topic,"",Cur_description,"","box",10,colors[3]))
            edges.append(createEdge(CourseID,id))
        else:                       
            if (Cur_chapterid != Prev_chapterid):
                parentID=id
                nodes.append(createNode(id,Cur_topic,"",Cur_description,"","box",10,colors[3]))
                edges.append(createEdge(CourseID,id))
            else:
                nodes.append(createNode(id,Cur_topic,"",Cur_description,"","image",10,colors[0]))
                edges.append(createEdge(parentID,id))
            
        Prev_courseid	= Cur_courseid
        Prev_chapterid	= Cur_chapterid
        Prev_sectionid	= Cur_sectionid
        Prev_unitid	    = Cur_unitid
        Prev_subunitid   = Cur_subunitid
        Prev_topic       = Cur_topic
        Prev_chapter     = Cur_chapter
        Prev_CognitiveLevel = Cur_CognitiveLevel
        Prev_description = Cur_description
        id=id+1


        #print(row)
        #parentID=id 
        #for i in range(6,14):
        #    if  row[i] =="-" or row[i] =="": continue
        #    nodes.append(createNode(id,row[i],"","","",10,colors[i-6]))
        #    if parentID==id:
        #        edges.append(createEdge(CourseID,id))
        #    else:
        #        edges.append(createEdge(parentID,id))
        #    id=id+1
    return nodes,edges,id
#---------------------------------------------------------------------------------------------------------------------
def createSimilarityGraph(similarNodes):
    colors=['BlueViolet','DeepPink','Lime','orange','Indigo','DarkSlateBlue','DarkMagenta',"brown"]
    nodes=[]
    edges=[]
    courses={}
    id=0

    index=[]

    for similarPair in similarNodes:
        #-----------------------------------------------------------------
        node1= similarNodes[similarPair]['node1']
        if node1['csv_file'] not in courses:
            id=id+1          
            nodes.append(createCourseNode(id,node1['csv_file'][:len(node1['csv_file'])-4],""))
            courses[node1['csv_file']]=id

        row1 = node1['row']
        parentID1= courses[node1['csv_file']]

        #-----------------------------------------------------------------
        node2= similarNodes[similarPair]['node2']        
        if node2['csv_file'] not in courses:
            id=id+1
            nodes.append(createCourseNode(id,node2['csv_file'][:len(node2['csv_file'])-4],""))
            courses[node2['csv_file']]=id

        row2 = node2['row']
        parentID2= courses[node2['csv_file']]

        #-----------------------------------------------------------------
        id=id+1
        nodes.append(createNode(id,row1[6]+","+ row2[6],"","","",10,colors[4]))

        edges.append(createEdge(parentID2,id))
        edges.append(createEdge(parentID1,id))

    return nodes,edges
#---------------------------------------------------------------------------------------------------------------------