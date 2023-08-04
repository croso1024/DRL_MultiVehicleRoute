import numpy as np  , json 
import matplotlib.pyplot as plt 




def performance_diagram(file_path , filter_word):  
    """
        Plot the performance diagram of "specific vehicle num" 
            x-axis : Problem scale 
            y-axis : objective value     
        every data line represent a one "method" (OR-Tools-GD , N50V5 with Greedy , N100V10 with POMO ... etc.)

    """
    with open(file_path , "r" ) as file : 
        logger = json.load(file) 
        logger:dict  
    
    vehicle_num = logger.pop("Vehicle_num")
    method_score = {}
    
    for method , data in logger.items() :
        
        # Only draw specific vehicle_nums model : 
        if any([ (word in method) for word in filter_word])  : continue   
        
        nodes_data = sorted(data.keys() , key= lambda x : int(x) ) 
        score_list = [  data[node_num]["obj"] for node_num in  nodes_data  ]
        # print(f"Method : {method}")
        # print(f"Nodes-data : {nodes_data}\n")
        # print(f"Score list : {score_list}")
        method_score.update({method:score_list})
        line_style = "dotted" if "V10" in method else "solid"
        plt.plot([ int(i) for i in nodes_data] , score_list ,label=method, marker="o" , linestyle=line_style )

    plt.xlabel("Node-num")
    plt.ylabel("Objective Value")
    plt.title(f"Generlization ability test : {vehicle_num} vehicle") 
    plt.legend()
    plt.grid(True) 
    
    plt.xticks([int(i) for i in nodes_data])
    plt.show()
 

def efficiency_diagram(*file_path , filter_word): 
    """ 
        Plot the computation time , use a range of node-nums , 
        only use OR-Tools , Our-model with greedy , Our-model with POMO , 
        and use V5 , V10 , two dataset  , so we totally have 6 line ! ( fetch from two .json log )
    """
    
    
    for path in file_path : 
        with open(path , "r") as file :
            logger = json.load(file) 
            logger:dict

        vehicle_num = logger.pop("Vehicle_num")
        method_time = {} 
        
        for method , data in logger.items() : 
            
            if any([ (word in method) for word in filter_word])  : continue   
            
            nodes_data = sorted(data.keys() , key=lambda x : int(x))
            time_list = [data[node_num]["time"] for node_num in nodes_data ]
            
            method_time .update({method:time_list})
            line_style = "dotted" if "V10" in method else "solid"  
            
            if not "OR"  in method : 
                inference_method = "Greedy" if "-G" in method else "POMO"
                legend_name =   f"Our Model with {inference_method} (V{vehicle_num})"
            else : 
                legend_name = f"OR-Tools Greedy descent (V{vehicle_num})"

            
            plt.plot([int(i) for i in nodes_data] , time_list , label=legend_name , marker='o' , linestyle=line_style)
            
        plt.xlabel("Node-num") 
        plt.ylabel("Computation time(seconds)")
        plt.title(f"Time complexity v.s. Problem scale")
        plt.legend() 
        plt.grid(True) 
    
    
    plt.xticks([int(i) for i in nodes_data])

    plt.yscale('log')

    plt.show()
    
    
# performance_diagram("./model/MultiTravelingSalesmanProblem/ComputationLogger.json" , filter_word=["V10"]) 
efficiency_diagram("./model/MultiTravelingSalesmanProblem/ComputationLogger.json",
                   filter_word=["N100","V10"])