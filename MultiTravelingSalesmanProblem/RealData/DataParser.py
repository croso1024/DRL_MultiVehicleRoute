""" 
    Use to load the txt file for the Data2Graph.py 
"""

import matplotlib.pyplot as plt 


def tspParse(file_name): 
    with open("./model/MultiTravelingSalesmanProblem/RealData/"+file_name+".tsp" , "r") as f : 
        lines = f.readlines() 
    
    nodes = [] 
    for i in range(len(lines)): 
        if lines[i].startswith("NODE_COORD_SECTION"): 
            for j in range(i+1 , len(lines)):
                if lines[j].startswith("EOF"):break 
                
                node_info = lines[j].strip().split()
                node_id = int(node_info[0])
                x = float(node_info[1])
                y = float(node_info[2])
                nodes.append( (node_id , x , y) )        
            break 
    instance_info = {"file_name":file_name } 
    return nodes , instance_info

def boundParse(file_name , vehicle_num): 
    with open("./model/MultiTravelingSalesmanProblem/RealData/bound_file.txt" , "r") as f : 
        for line in f : 
            values = line.split() 
            if values[0] == file_name and values[1] == str(vehicle_num) : 
                return round(float(values[3]) , 1)

if __name__ == "__main__": 
    
    nodes = tspParse("bier127")
    for node in nodes : 
        print(node)
                    
