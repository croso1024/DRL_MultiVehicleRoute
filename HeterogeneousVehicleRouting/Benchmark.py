""" 
    Used to testing the model performance in the benchmark dataset 
"""
from Env.Environment import HCVRP_Environment 
from torch.distributions import Categorical 
from utils.HCVRPD_solver import ORtools_HCVRPD 
from utils.RouteVisulizer import RouteVisulizer
from tqdm import tqdm 
import argparse , torch , time ,  copy , random ,numpy as np  , matplotlib.pyplot as plt 
from torch_geometric.data import Batch 
from math import exp 
from RealData.DataParser import cParse , BrandaoParse
from RealData.Data2Graph import GraphDataParse , get_Instance_from_coordinate , get_InstancePMPO_from_coordinate
from RealData.DataStatistic import get_txt_files 
#############  SEED  ###############
seed = 33
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.utils.backcompat.broadcast_warning.enabled = True 
########## Parameters ############## 

Color_map = ["red","orange","yellow","green","blue","indigo","purple","grey","peru",
                 "dodgerblue","crimson","sandybrown","khaki","lime","royalblue","navy","darkslateblue"
                 ,"violet","orchid","teal","olive"] 

def env_maker_validation(batch_size,batch_data,node_num,vehicle_num,vehicle_charateristic_set,StateEQ): 
    return HCVRP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_capacity=[ v[0] for v in vehicle_charateristic_set ], 
        vehicle_velocity=[ v[1] for v in vehicle_charateristic_set ], 
        vehicle_pos_mode="Depot", 
        DE_transform_type=DE_transform_type , 
        StateEQ=StateEQ , 
        training=False , 
        graph_transform={"transform":"knn" , "value":12}, 
        device = device ,
    )

def Greedy(Agent , instance , vehicle_charateristic_set , instance_info ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list([instance])
    batch_size = 1
    with torch.no_grad() : 
        start = time.time() 
        batch = batch.to(device) 
        print(vehicle_charateristic_set)
        env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num,vehicle_num=vehicle_num,
                                   vehicle_charateristic_set=vehicle_charateristic_set ,StateEQ=None )
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        # 取出在所有batch中最好的解 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        value , index  = torch.min(score,dim=0) 
        #print(f"best : {value} at {index}")
        #objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=True , store=True , show=False , 
            method="GREEDY" , 
            store_path="./model/HeterogeneousVehicleRouting/plot/"
            )
            routes = env.get_ModelRoute(index = best_index)
            visulizer.Visualize(fig_id = file_name , instance=instance , routes=routes , vehicle_attribute=vehicle_charateristic_set, objective_value=objective)
        
    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap =  ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"Greedy : {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (objective , fulfill_rate[best_index] , end-start ,Gap)

def PMPO(Agent, instance , vehicle_charateristic_set , instance_info ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list(instance)  # PMPO_instance is list 
    batch_size = 8 
    with torch.no_grad(): 
        start = time.time() 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num,vehicle_num=vehicle_num,
                                   vehicle_charateristic_set=vehicle_charateristic_set ,StateEQ="PMPO"  )
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        value , index = torch.min(score,dim=0) 
        # objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective  , best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=True , store=True , show=False , 
            method="POMO" , 
            store_path="./model/HeterogeneousVehicleRouting/plot/"
            )
            routes = env.get_ModelRoute(index = best_index)
            visulizer.Visualize(fig_id = file_name , instance=instance[0] , routes=routes , vehicle_attribute=vehicle_charateristic_set, objective_value=objective)
        
    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap =  ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"POMO: {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")

    return (objective , fulfill_rate[best_index] , end-start , Gap )

def PMPO_sample(Agent, instance , vehicle_charateristic_set , instance_info ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list(instance*160)  # PMPO_instance is list 
    batch_size = 8 * 160
    with torch.no_grad(): 
        start = time.time() 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num,vehicle_num=vehicle_num,
                                   vehicle_charateristic_set=vehicle_charateristic_set ,StateEQ="PMPO" )
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Categorical(Agent(batch,fleet_state,vehicle_state , mask) )
            action = action_dist.sample()
            #action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        value , index = torch.min(score,dim=0) 
        # objective , best_index  = env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective  , best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=True , store=True , show=False , 
            method="POMO_sampled" , 
            store_path="./model/HeterogeneousVehicleRouting/plot/"
            )
            routes = env.get_ModelRoute(index = best_index)
            visulizer.Visualize(fig_id = file_name , instance=instance[0] , routes=routes , vehicle_attribute=vehicle_charateristic_set, objective_value=objective)
            
    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap =  ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"POMO: {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (objective , fulfill_rate[best_index] , end-start ,Gap )

def DE(Agent, instance , vehicle_charateristic_set , instance_info ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list([instance])
    DE_batch_size = 1 * DE_transform_type 
    with torch.no_grad() : 
        start = time.time() 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=DE_batch_size , batch_data=batch , node_num=node_num , vehicle_num=vehicle_num,
            vehicle_charateristic_set=vehicle_charateristic_set , StateEQ="DE")
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        value , index = torch.min(score,dim=0) 
        # objective  , best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective  , best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=True , store=True , show=False , 
            method="DE" , 
            store_path="./model/HeterogeneousVehicleRouting/plot/"
            )
            routes = env.get_ModelRoute(index = best_index)
            visulizer.Visualize(fig_id = file_name , instance=instance , routes=routes ,vehicle_attribute=vehicle_charateristic_set, objective_value=objective)
            
    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap = ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"DE: {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    
    
    return (objective , fulfill_rate[best_index] , end-start  ,Gap)
    
def mix(Agent, instance , vehicle_charateristic_set , instance_info): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list(instance) 
    DE_batch_size = 8 * DE_transform_type
    with torch.no_grad() : 
        start = time.time() 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=DE_batch_size ,batch_data=batch , node_num=node_num , vehicle_num=vehicle_num,
            vehicle_charateristic_set=vehicle_charateristic_set , StateEQ="mix" )
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        # score = minSum_tour
        value , index = torch.min(score,dim=0) 
        # objective , best_index = env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective , best_index = env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=True , store=True , show=False , 
            method="MIX" , 
            store_path="./model/HeterogeneousVehicleRouting/plot/"
            )
            routes = env.get_ModelRoute(index = best_index)
            visulizer.Visualize(fig_id = file_name , instance=instance[0] , routes=routes , vehicle_attribute=vehicle_charateristic_set, objective_value=objective)


    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap = ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"Mix: {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")  

    
    return (objective , fulfill_rate[best_index] , end-start ,Gap)

def mix_sample(Agent, instance , vehicle_charateristic_set , instance_info ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list(instance * 4) 
    DE_batch_size = 8 * 4 * DE_transform_type
    with torch.no_grad() : 
        start = time.time() 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=DE_batch_size ,batch_data=batch , node_num=node_num , vehicle_num=vehicle_num,
            vehicle_charateristic_set=vehicle_charateristic_set , StateEQ="mix" )
        batch = env.get_initilize_Batch() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Categorical(Agent(batch,fleet_state,vehicle_state , mask) ) 
            action = action_dist.sample()
            # action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
        end = time.time() 
        minSum_tour , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (minSum_tour + (node_num/(2*vehicle_num)))
        # score = minSum_tour
        value , index = torch.min(score,dim=0) 
        objective , best_index = env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
    benchmark_value = exp( 5*(1-fulfill_rate[best_index].item() ) ) * objective
    Gap =  ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"Mix: {objective:.3f} objective with {fulfill_rate[best_index].item():.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (objective , fulfill_rate[best_index] , end-start ,Gap)

def ORTools(instance  ,vehicle_charateristic_set , instance_info  , time_limit=1 ,algo="GD" ):
    algo_table = {
        "GD":"GREEDY_DESCENT" ,
        "GL":"GUIDED_LOCAL_SEARCH",
        "SA":"SIMULATED_ANNEALING",
        "TS":"TABU_SEARCH" ,
        "GTS":"GENERIC_TABU_SERACH" ,
    }
    average_tour_length = 0 
    average_fulfill_rate = 0 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]

    start = time.time() 
    demand_vector = (instance.x[:,2]*10000).tolist() 
    demand_vector[0] = 0 
    # dist_matrix = torch.round(instance.node_route_cost*1000).int().tolist()
    dist_matrix = torch.round(torch.tensor(instance_info["Distance_matrix"]) * 10000)

    tour_length , fulfill_rate = ORtools_HCVRPD(
        dist_matrix=dist_matrix.int().tolist(), 
        # dist_matrix = instance_info["Distance_matrix"] , 
        demand_vector=demand_vector , 
        vehicle_num=vehicle_num , 
        depot_index=0, # Avoid error
        capacity_vector=[ 10000*v[0] for v in vehicle_charateristic_set ], 
        velocity_vector=[ v[1] for v in vehicle_charateristic_set ], 
        # time_limit=time_limit ,
        time_limit=1,
        algo=algo_table[algo] , 
    ).solve() 
    average_tour_length += tour_length 
    average_fulfill_rate += fulfill_rate
    end = time.time() 
    average_tour_length = average_tour_length / 10000
    average_fulfill_rate = average_fulfill_rate 
    benchmark_value = exp( 5*(1-average_fulfill_rate ) ) * average_tour_length
    Gap = ((benchmark_value - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"ORTools-{algo}({time_limit}s): {average_tour_length:.3f} objective with {average_fulfill_rate:.3f} fulfill Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (average_tour_length , average_fulfill_rate , end-start ,torch.tensor(Gap))





def Instance_testing(instance_type , instance_name): 
    
    # -- Get the normalized coordinate(node list) , demand list and vehicle-set , instance-info
    coordinates , demand_list ,vehicle_charateristic_set ,instance_info = GraphDataParse(instance_type=instance_type , instance_name=instance_name)
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    opt = instance_info["bound"]
    print(f"\n\n-- Instance : {file_name} haves {node_num} nodes {vehicle_num} vehicles) -- \n\n")
    instance = get_Instance_from_coordinate(coordinates=coordinates , demand_list=demand_list)
    PMPO_instance = get_InstancePMPO_from_coordinate(coordinate=coordinates , demand_list=demand_list) 
    # Recorder = {
    #     "Greedy":[] , "PMPO":[] , "DE":[] , "mix":[] , "ORTools(10s)":[] 
    # }
    Recorder = dict()
    Recorder["Greedy"] = (
    Greedy(Agent=Agent , instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , 
           instance_info=instance_info )
                           )
    Recorder["POMO"] = (
    PMPO( Agent=Agent , instance=PMPO_instance, vehicle_charateristic_set=vehicle_charateristic_set , 
           instance_info=instance_info)
    )
    Recorder["DE"] = (
    DE( Agent=Agent , instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , 
           instance_info=instance_info )
    )
    Recorder["mix"] = (
    mix( Agent=Agent , instance=PMPO_instance, vehicle_charateristic_set=vehicle_charateristic_set , 
        instance_info=instance_info )
    )

    Recorder["ORTools-GD"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="GD"
            , instance_info=instance_info , time_limit=10)
    ) 
    Recorder["ORTools-GL(2s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="GL"
            , instance_info=instance_info , time_limit=2)
    ) 
    Recorder["ORTools-GL(5s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="GL"
            , instance_info=instance_info , time_limit=5)
    ) 
    Recorder["ORTools-GL(10s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="GL"
            , instance_info=instance_info , time_limit=10)
    ) 
    
    Recorder["ORTools-TS(2s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="TS"
            , instance_info=instance_info , time_limit=2)
    ) 
    Recorder["ORTools-TS(5s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="TS"
            , instance_info=instance_info , time_limit=5)
    ) 
    Recorder["ORTools-TS(10s)"] = (
    ORTools(instance=instance, vehicle_charateristic_set=vehicle_charateristic_set , algo="TS"
            , instance_info=instance_info , time_limit=10)
    ) 
    
    return Recorder 
    
    






# Step1. Load benchmark dataset data 
if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser() 
    argParser.add_argument("-benchmark","--benchmark_mode", action="store_true", default=False)
    argParser.add_argument("-vis","--visual_mode", action="store_true", default=False)
    argParser.add_argument("-map" ,"--map_name" , type=str , default="None")
    argParser.add_argument("-t" , "--type" , type=str ,default="brandao") 
    argParser.add_argument("-trans" , "--DE_transform_type" , type=int,default=8) 
    argParser.add_argument("-samples" , "--sample_num" , type=int,default=1280) 
    argParser.add_argument("-m" , "--model" , type=str , default="RL_agent")
    arg = argParser.parse_args() 

    Benchmark_mode = arg.benchmark_mode
    visualize = arg.visual_mode
    file_name = arg.map_name 
    model_path = "./model/HeterogeneousVehicleRouting/checkpoint/" + arg.model + ".pth"
    instance_type = arg.type 
    DE_transform_type = arg.DE_transform_type
    Samples_num = arg.sample_num
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert instance_type in ["brandao" , "c"] , "instance type error"
    from net.ver20.PolicyNetwork import PolicyNetwork 
    Agent = PolicyNetwork(
        node_feature_dim=5,fleet_feature_dim=7,vehicle_feature_dim=6 , 
        hidden_dim=192, heads = 6 , num_layers=3 , 
        skip_connection=True , clip_coe=10,temp=1.5
    ).to(device) 
    Agent.eval() 
    model_path = "./model/HeterogeneousVehicleRouting/checkpoint/N55H_v20k12_0604.pth"
    Agent.load_state_dict(torch.load(model_path))
    
    if Benchmark_mode: 
        dataset_list = get_txt_files("./model/HeterogeneousVehicleRouting/RealData2/")
        Performance_Matrix = {
            "Greedy":[0,0],
            "POMO":[0,0],
            "DE":[0,0],
            "mix":[0,0],
            "ORTools-GD":[0,0],
            "ORTools-GL(2s)":[0,0],
            "ORTools-GL(5s)":[0,0],
            "ORTools-GL(10s)":[0,0],
            "ORTools-TS(2s)":[0,0],
            "ORTools-TS(5s)":[0,0],
            "ORTools-TS(10s)":[0,0]
            }
        for map in dataset_list: 
            map = map.split(".")[0]
            instance_type = "c" if map[0]=="c" else "brandao"
            # map[0] -> type of dataset 
            Recorder = Instance_testing(instance_type=instance_type , instance_name=map )
            for key,item in Recorder.items() : 
                Performance_Matrix[key][0] += item[3] # Performance Gap 
                Performance_Matrix[key][1] += item[2] # Time per instance 
        
        for key in Performance_Matrix.keys() : 
            Performance_Matrix[key][0] /= len(dataset_list) 
            Performance_Matrix[key][1] /= len(dataset_list) 
            print(f"{key} avaerage Gap:{Performance_Matrix[key][0]} , time : {Performance_Matrix[key][1]}")
            Performance_Matrix[key][0] = Performance_Matrix[key][0].cpu()
        # Performance （　time , gap , method ) 
        Performance = [(Performance_Matrix[key][1],Performance_Matrix[key][0],key) for key in Performance_Matrix.keys()]
        # 使用時間作為pareto的排序 , 把速度最快的算法加入pareto前沿 ( 速度最快或性能最好的必定在pareto frontier內 )
        sorted_performance = sorted(Performance)
        pareto_frontier = [sorted_performance[0]] 
        
        for method in sorted_performance[1:] :  # Note: 這個sorted_performance是按照時間排序
            if method[1] <= pareto_frontier[-1][1]: #如果他的性能好過目前pareto frontier內速度最慢的那個算法 
                pareto_frontier.append(method) 
        
        pareto_frontier_x = [x for x,y,_ in pareto_frontier]
        pareto_frontier_y = [y for x,y,_ in pareto_frontier]
        
        
        ORTools_Performance = [] 
        Model_Performance = [] 
        for consumption_time,gap,method in Performance:
            if method in ["Greedy","POMO","DE","mix"] : 
                Model_Performance.append((consumption_time,gap,method))
            else : 
                ORTools_Performance.append((consumption_time,gap,method))
        
        colors = range(len(Performance))
        
        scatter= plt.scatter([x for x, _, _ in Performance], [y for _, y, _ in Performance],c=colors, cmap="Spectral",s=150 ,marker=".")
        for i in range(len(pareto_frontier) - 1):
            plt.plot([pareto_frontier[i][0], pareto_frontier[i+1][0]], [pareto_frontier[i][1], pareto_frontier[i+1][1]], 'r--')
        
        plt.title('Performance vs Time')
        plt.grid()
        plt.xlabel('Avg. Computing times(s)')
        plt.ylabel('Avg. Performance - Gap(%)')

        # 加入legend1 
        # (method name on the scatter point)
        for x, y, key in Performance:
            plt.text(x, y, key)
        
        # 加入legend2
        # (method name & corresponding colors on the right hand side)
        legend_labels = [key for _,_,key in Performance]
        plt.legend(scatter.legend_elements()[0],legend_labels)

        
        plt.show()
        
    else: 
        Instance_testing(instance_type=instance_type , instance_name=file_name)
        
        
        





