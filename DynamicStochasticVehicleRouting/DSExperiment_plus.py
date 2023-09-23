from RealData.DSExperimentsAlgorithms_plus import * 
from RealData.OSM_Plotting import OSMRoute_plotter
import osmnx as ox 
import time 
import copy 

def Testing(algo , test_times = 50) : 
    average_obj , average_fulfill = 0 , 0 
    algorithms = Algorithms[algo]
    test_config = copy.deepcopy(config)
    test_config["ORTools_algo"] = "GUIDED_LOCAL_SEARCH" if algo == "ORTools-GL" else "GREEDY_DESCENT"
    for i in tqdm(range(test_times)): 
    # for i in range(test_times): 
        obj , fulfill = algorithms(Agent_depot = Agent_depot , Agent_random = Agent_random  , config= test_config , BaseGraph=BaseGraph , node_coordinate_table=node_coordinate_table) 
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
    obj , fulfill , logger = Algorithms[algo](Agent_depot = Agent_depot , Agent_random = Agent_random   , config= config , BaseGraph=BaseGraph ,
                                     node_coordinate_table=node_coordinate_table,
                                     return_looger=True) 
    #print(f"Objective : {obj} , fulfill : {fulfill}")
    Plotter.plotting(logger) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# low


config = {
    "vehicle_num" : 5,
    "init_Request_num" : 30 , 
    "init_capacity": 0.8 , 
    "maximum_Request_num": 70 , 
    "step_duration" : 0.3 , 
    "release_num":(6,6) , 
    "total_update_times": 4 , 
    "device": device , 
    "DE_transform_type" : 8 , 
    "ORTools_algo":"GREEDY_DESCENT",
    "map":"TAIPEI", 
    # "ORTools_algo":"GUIDED_LOCAL_SEARCH",
    "deterministic" : False, 
}
print(f"Config : {config}\n\n")
Algorithms = {
    "ORTools-GD":ORTools , 
    "ORTools-GL":ORTools,
    "Greedy":Greedy , 
    "PMPO":PMPO , 
    "DE":DE,
    "Mix":mix , 
}

from net.ver20.PolicyNetwork import PolicyNetwork
Agent_depot = PolicyNetwork(
        node_feature_dim=5,fleet_feature_dim=6,vehicle_feature_dim=5,
        hidden_dim=192, heads = 6 , num_layers=3 , 
        skip_connection=True , clip_coe=10,temp=1.5
    ).to(device)
Agent_random = PolicyNetwork(
        node_feature_dim=5,fleet_feature_dim=6,vehicle_feature_dim=5,
        hidden_dim=192, heads = 6 , num_layers=3 , 
        skip_connection=True , clip_coe=10,temp=1.5
    ).to(device)
# Agent_depot.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v20_n50_SC_DEPOT_0703.pth"))
Agent_depot.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v20_n50_SC_DEPOT.pth"))
Agent_depot.eval() 
# Agent_random.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v20_n50_SC_RANDOM_0703.pth"))
Agent_random.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v20_n50_SC_RANDOM.pth"))
Agent_random.eval()

# 載入地圖 ,並且先建立一個simu用來拿出node_coordinate_table
BaseGraph = ox.load_graphml("./model/DynamicStochasticVehicleRouting/Taipei7500D.graphml")
import networkx as nx 
node_coordinate_table = simulator_maker(config=config , BaseGraph=BaseGraph).node_coordinate_table
print(f"--- Base Graph Loading complete ! ")

# Use to plot the Dynamic stochastic routing
# PlottingExperiment("PMPO")
# exit()

# Use to testing the performance of the model 
testtimes = 64
Testing("ORTools-GD" , test_times=testtimes)
# Testing("ORTools-GL" , test_times=testtimes)
Testing("Greedy" , test_times=testtimes)
Testing("PMPO" , test_times=testtimes)
Testing("DE" , test_times=testtimes)
# Testing("Mix" , test_times=testtimes)
        
config = {
    "vehicle_num" : 5,
    "init_Request_num" : 30 , 
    "init_capacity": 0.8 , 
    "maximum_Request_num": 70 , 
    "step_duration" : 0.3 , 
    "release_num":(4,4) , 
    "total_update_times": 6 , 
    "device": device , 
    "DE_transform_type" : 8 , 
    "ORTools_algo":"GREEDY_DESCENT",
    "map":"TAIPEI", 
    # "ORTools_algo":"GUIDED_LOCAL_SEARCH",
    "deterministic" : False, 
}
print(f"Config : {config}\n\n")
Testing("ORTools-GD" , test_times=testtimes)
# Testing("ORTools-GL" , test_times=testtimes)
Testing("Greedy" , test_times=testtimes)
Testing("PMPO" , test_times=testtimes)
Testing("DE" , test_times=testtimes)