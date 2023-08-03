from RealData.DSExperimentsAlgorithms import * 
from RealData.OSM_Plotting import OSMRoute_plotter
import osmnx as ox 
import time 

def Testing(algo , test_times = 50) : 
    average_obj , average_fulfill = 0 , 0 
    algorithms = Algorithms[algo]
    for i in tqdm(range(test_times)): 
        obj , fulfill = algorithms(Agent = Agent , config= config , BaseGraph=BaseGraph , node_coordinate_table=node_coordinate_table) 
        average_obj += obj 
        average_fulfill += fulfill 
    average_obj /= test_times 
    average_fulfill /= test_times
    print(f"{algo} Average objective : {average_obj}  Fulfill {average_fulfill} in {test_times} tests")
    print(f"\n-----------\n")


def PlottingExperiment(algo): 
    Plotter = OSMRoute_plotter(
        config=config , BaseGraph=BaseGraph , map=config["map"]
    )
    obj , fulfill , logger = Algorithms[algo](Agent = Agent , config= config , BaseGraph=BaseGraph ,
                                     node_coordinate_table=node_coordinate_table,
                                     return_looger=True) 
    #print(f"Objective : {obj} , fulfill : {fulfill}")
    Plotter.plotting(logger) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = {
    "vehicle_num" : 5,
    "init_Request_num" : 40 , 
    "init_capacity": 1 , 
    "maximum_Request_num": 60 , 
    "step_duration" : 0.15 , 
    "release_num":(4,5) , 
    "total_update_times": 4 , 
    "device": device , 
    "DE_transform_type" : 8 , 
    "ORTools_algo":"GREEDY_DESCENT",
    "map":"TAIPEI", 
    # "ORTools_algo":"GUIDED_LOCAL_SEARCH",
    "deterministic" : False, 
}

Algorithms = {
    "ORTools":ORTools , 
    "Greedy":Greedy , 
    "PMPO":PMPO , 
    "DE":DE,
    "Mix":mix , 
}

from net.ver16.PolicyNetwork import PolicyNetwork
Agent = PolicyNetwork(
        node_feature_dim=5,fleet_feature_dim=6,vehicle_feature_dim=5,
        hidden_dim=192, heads = 6 , num_layers=4 , 
        skip_connection=True , clip_coe=10,temp=1.5
    ).to(device)
Agent.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v16_n50_SC2.pth"))
Agent.eval() 


# 載入地圖 ,並且先建立一個simu用來拿出node_coordinate_table
BaseGraph = ox.load_graphml("./model/DynamicStochasticVehicleRouting/Taipei7500D.graphml")
import networkx as nx 
node_coordinate_table = simulator_maker(config=config , BaseGraph=BaseGraph).node_coordinate_table
print(f"--- Base Graph Loading complete ! ")
#PlottingExperiment("PMPO")

    
Testing("ORTools" , test_times=30)
Testing("Greedy" , test_times=30)
Testing("PMPO" , test_times=30)
Testing("DE" , test_times=30)
Testing("Mix" , test_times=30)
        
