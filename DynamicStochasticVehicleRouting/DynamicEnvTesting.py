""" 
    主要的測試Code , 將Routeing Simulator實際應用到模型以及OR-tools上 , 同時在此定義了整個
    Reinforcement learning environemnt & Routing simulator environment的互動架構 
"""
from Env.Environment_SC import DSVRP_Environment
from RealData.RoutingSimulator import RoutingSimulator
from torch_geometric.data import Batch 
from tqdm import tqdm 
import argparse ,torch,time  , copy 
from torch.distributions import Categorical 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "vehicle_num" : 3,
    "init_Request_num" : 20 , 
    "init_capacity": 0.7 , 
    "maximum_Request_num": 30 , 
    "step_duration" : 0.4 , 
    "total_update_times":  2 , 
}
# Step1. 建立Simulator 
simulator = RoutingSimulator(
    init_Request_num = config["init_Request_num"] , 
    vehicle_num = config["vehicle_num"], 
    init_capacity = config["init_capacity"] , 
    maximum_Request_num = config["maximum_Request_num"], 
    step_duration = config["step_duration"] , 
    total_update_times = config["total_update_times"] ,
    PMPO = False ,
    BaseGraph=None ,
)
from net.ver18.PolicyNetwork import PolicyNetwork
Agent = PolicyNetwork(
        node_feature_dim=5,fleet_feature_dim=6,vehicle_feature_dim=5,
        hidden_dim=192, heads = 6 , num_layers=4 , 
        skip_connection=True , clip_coe=10,temp=1.5
    ).to(device)
Agent.load_state_dict(torch.load("./model/DynamicStochasticVehicleRouting/checkpoint/N50_v18_2.pth"))
Agent.eval() 

def env_maker(batch_size,batch_data,node_num , vehicle_pos_set ,vehicle_capacity , StateEQ): 
    return DSVRP_Environment(
        batch_size = batch_size , 
        node_nums = node_num , 
        vehicle_num = config["vehicle_num"], 
        batch_data=batch_data , 
        vehicle_capacity=vehicle_capacity , 
        vehicle_pos_mode=vehicle_pos_set , 
        StateEQ=StateEQ , 
        graph_transform=None , 
        training=False , 
        device=device , 
    )

RoadNetwork ,vehicle_pos , vehicle_capacity , final_change = simulator.reset()  
node_num = simulator.get_node_num()
batch_size = 1 
terminate = False 
while not terminate : 
    # Terminate代表整個環境變化是否結束 , 會需要simulator做出最後一輪改變後才為True,使模型輸出最後一輪預測結果
    if final_change : terminate = True 
    
    with torch.no_grad(): 
        env = env_maker(batch_size = batch_size , batch_data = RoadNetwork.to(device) , node_num=simulator.get_node_num(),
                      vehicle_pos_set=vehicle_pos , vehicle_capacity=vehicle_capacity,StateEQ=None) 
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state ,mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        objective , fulfill_rate  = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (objective + (node_num/(2*config['vehicle_num'])))
        value , index = torch.min(score ,dim=0) 
        
        Model_Predict = env.get_ModelRoute(index)
        print(f"Route : {Model_Predict} index : {index}")
        if not terminate  : 
            RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.step(Model_Predict)
        else : 
            objective , fulfill = simulator.step(Model_Predict)
            print(f"Objective : {objective} , Fulfill : {fulfill}")
print(f"----------\n\n")
print(simulator.current_request_Recorder)
print(simulator.complete_request_Recorder)
print(simulator.vehicle_pos_Recorder)
print(simulator.vehicle_routes_Recorder)