""" 
    Used to testing the model performance in the benchmark dataset 
"""
from Env.Environment import MTSP_Environment
from torch.distributions import Categorical 
from utils.VRP_solver import ORtools_VRP
from utils.RouteVisulizer import RouteVisulizer
from tqdm import tqdm 
import argparse , torch , time ,  copy , random ,numpy as np  , matplotlib.pyplot as plt 
from torch_geometric.data import Batch 
from math import exp 
from RealData.DataParser import tspParse , boundParse
from RealData.DataPlotting import get_tsp_files
from RealData.Data2Graph import GraphDataParse , get_Instance_from_coordinate , get_InstancePMPO_from_coordinate
#############  SEED  ###############
seed = 74
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

def env_maker_validation(batch_size,batch_data,node_num,vehicle_num,StateEQ,knn): 
    return MTSP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_pos_mode="Depot", 
        StateEQ=StateEQ , 
        graph_transform={"transform":"knn" , "value":12} ,
        device = device ,
    )

def Greedy(Agent , instance , instance_info,  knn=False  ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list([instance])
    batch_size = 1
    with torch.no_grad() : 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num
                                   ,vehicle_num=vehicle_num,StateEQ=None , knn=knn )
        batch = env.get_initilize_Batch() 
        start = time.time() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        end = time.time() 
        # 取出在所有batch中最好的解 
        value , index  = torch.min(reward,dim=0) 
        objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=False , store=True , show=False , 
            method="GREEDY" , 
            store_path="./model/MultiTravelingSalesmanProblem/plot/"
            )
            routes = env.get_SplitedRoute()[best_index]
            visulizer.Visualize(fig_id = file_name , instance=instance , routes=routes , objective_value=objective)

    Gap =  ((objective - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"Greedy : {objective:.3f} objective  Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (objective  , end-start ,Gap)

def PMPO(Agent, instance  , instance_info , knn = False): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch = Batch.from_data_list(instance)  # PMPO_instance is list 
    batch_size = 8 
    with torch.no_grad(): 
        batch = batch.to(device) 
        env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num
                                   ,vehicle_num=vehicle_num ,StateEQ="PMPO" , knn=knn )
        batch = env.get_initilize_Batch() 
        start = time.time() 
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        while not done : 
            action_dist = Agent(batch,fleet_state,vehicle_state , mask) 
            # action = action_dist.sample()
            action = torch.argmax(action_dist , dim=1 , keepdim=False) 
            batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
        end = time.time() 
        value , index = torch.min(reward,dim=0) 
        # objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=index)
        objective ,best_index= env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)

        if visualize:
            visulizer = RouteVisulizer(
            node_num = node_num , vehicle_num=vehicle_num , capacity=False , store=True , show=False , 
            method="POMO" , 
            store_path="./model/MultiTravelingSalesmanProblem/plot/"
            )
            routes = env.get_SplitedRoute()[best_index]
            visulizer.Visualize(fig_id = file_name , instance=instance[0] , routes=routes , objective_value=objective)

    Gap =  ((objective - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"POMO: {objective:.3f} objective  Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (objective  , end-start , Gap )

def PMPO_sample(Agent, instance , instance_info , knn = False ): 
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]
    batch_ = Batch.from_data_list(instance*64)  # PMPO_instance is list 
    batch_size = 8 * 64
    best_objective = float("inf")
    with torch.no_grad(): 
        for i in range( Samples_num//batch_size ):
            batch = copy.deepcopy(batch_).to(device)       
            env = env_maker_validation(batch_size=batch_size , batch_data = batch ,node_num=node_num
                        ,vehicle_num=vehicle_num ,StateEQ="PMPO" , knn=knn )
            batch = env.get_initilize_Batch() 
            start = time.time() 
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Categorical(Agent(batch,fleet_state,vehicle_state , mask) )
                action = action_dist.sample()
                #action = torch.argmax(action_dist , dim=1 , keepdim=False) 
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
            end = time.time()
            value , index = torch.min(reward , dim=0) 
            objective , best_index  = env.Calculate_Objective_byDist(Dist_matrix=instance_info["Distance_matrix"],best_index=None)
            best_objective = min(objective, best_objective)
        Gap =  ((best_objective - instance_info["bound"]) / instance_info["bound"] )*100
        print(f"POMO sample: {best_objective:.3f} objective  Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
        print(f"           --               ")
        return (best_objective , end-start ,Gap )
                


def ORTools(instance   , instance_info  , time_limit=1 ,algo="GD" ):
    algo_table = {
        "GD":"GREEDY_DESCENT" ,
        "GL":"GUIDED_LOCAL_SEARCH",
        "SA":"SIMULATED_ANNEALING",
        "TS":"TABU_SEARCH" ,
        "GTS":"GENERIC_TABU_SERACH" ,
    }
    file_name,node_num ,vehicle_num = instance_info["file_name"] , instance_info["node_num"] , instance_info["vehicle_num"]

    start = time.time() 
    dist_matrix = torch.round(torch.tensor(instance_info["Distance_matrix"]) * 10000)

    tour_length = ORtools_VRP(
        dist_matrix=dist_matrix.int().tolist(), 
        vehicle_num=vehicle_num , 
        time_limit=time_limit ,
        # time_limit=1,
        algo=algo_table[algo]
    ).solve() 
    average_tour_length = tour_length 
    end = time.time() 
    average_tour_length = average_tour_length / 10000
    Gap = ((average_tour_length - instance_info["bound"]) / instance_info["bound"] )*100
    print(f"ORTools_{algo}({time_limit}s): {average_tour_length:.3f} objective  Gap:{Gap:.3f}%  in {end-start:.3f} seconds")
    print(f"           --               ")
    return (average_tour_length  , end-start ,torch.tensor(Gap))


def Instance_testing(instance_name , vehicle_num ): 
    
    # -- Get the normalized coordinate(node list) , demand list and vehicle-set , instance-info
    coordinates ,  instance_info = GraphDataParse(instance_name=instance_name)
    file_name,node_num = instance_info["file_name"] , instance_info["node_num"] 
    # Manual add the vehicle num into the instance info
    instance_info["vehicle_num"] = vehicle_num
    instance_info["bound"] = boundParse(file_name = instance_name , vehicle_num=vehicle_num)
    print(f"\n\n-- Instance : {file_name} haves {node_num} nodes {vehicle_num} vehicles) bound:{instance_info['bound']} -- \n\n")
    instance = get_Instance_from_coordinate(coordinates=coordinates )
    PMPO_instance = get_InstancePMPO_from_coordinate(coordinates=coordinates ) 
    # Recorder = {
    #     "Greedy":[] , "PMPO":[] , "DE":[] , "mix":[] , "ORTools(10s)":[] 
    # }
    Recorder = dict() 
    Recorder["Greedy"] = (
    Greedy(Agent=Agent , instance=instance,instance_info=instance_info , knn=12)
                           )
    Recorder["POMO"] = (
    PMPO( Agent=Agent , instance=PMPO_instance,instance_info=instance_info , knn=12)
    )
    Recorder["POMO_sample"] = (
        PMPO_sample( Agent=Agent , instance=PMPO_instance,instance_info=instance_info , knn=12)
    )
    Recorder["ORTools_GD"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=15,algo="GD")
    ) 
    # ----------- GL ----------------
    Recorder["ORTools_GL(2s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=2 , algo="GL")
    ) 
    
    Recorder["ORTools_GL(5s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=5 , algo="GL")
    ) 
    
    Recorder["ORTools_GL(10s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=10 , algo="GL")
    ) 
    # ----------- TS ----------------
    Recorder["ORTools_TS(2s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=2 , algo="TS")
    ) 
    
    Recorder["ORTools_TS(5s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=5 , algo="TS")
    ) 
    
    Recorder["ORTools_TS(10s)"] = (
    ORTools(instance=instance, instance_info=instance_info , time_limit=10 , algo="TS")
    ) 
    
    return Recorder 
    

# Step1. Load benchmark dataset data 
if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser() 
    argParser.add_argument("-benchmark","--benchmark_mode", action="store_true", default=False)
    argParser.add_argument("-vis","--visual_mode", action="store_true", default=False)
    argParser.add_argument("-map" ,"--map_name" , type=str , default="None")
    argParser.add_argument("-v" , "--vehicle_num" , type=int , default=5)
    argParser.add_argument("-samples" , "--sample_num" , type=int,default=512) 
    argParser.add_argument("-m" , "--model" , type=str , default="RL_agent")
    arg = argParser.parse_args() 

    Benchmark_mode = arg.benchmark_mode
    visualize = arg.visual_mode 
    file_name = arg.map_name 
    vehicle_num = arg.vehicle_num 
    model_path = "./model/MultiTravelingSalesmanProblem/checkpoint" + arg.model + ".pt"
    Samples_num = arg.sample_num
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from net.ver20.PolicyNetwork import PolicyNetwork 
    Agent = PolicyNetwork(
        node_feature_dim=4,fleet_feature_dim=5,vehicle_feature_dim=4, 
        hidden_dim=192 , heads = 6 , num_layers=3 , 
        skip_connection=True , clip_coe=10,temp = 1.2
    ).to(device) 
    model_path = "./model/MultiTravelingSalesmanProblem/checkpoint/N50_v20k12_0605.pth"
    Agent.load_state_dict(torch.load(model_path))
    
    
    
    if Benchmark_mode: 
        vehicle_num_set = [5,7,9]
        dataset_list = get_tsp_files("./model/MultiTravelingSalesmanProblem/RealData/")
        Performance_Matrix = {
            "Greedy":[0,0],
            "POMO":[0,0],
            "POMO_sample":[0,0],
            "ORTools_GD":[0,0],
            "ORTools_GL(2s)":[0,0],
            "ORTools_GL(5s)":[0,0],
            "ORTools_GL(10s)":[0,0],
            "ORTools_TS(2s)":[0,0],
            "ORTools_TS(5s)":[0,0],
            "ORTools_TS(10s)":[0,0],
            }
        for map in dataset_list: 
            for vehicle_num in vehicle_num_set: 
                map = map.split(".")[0]
                Recorder = Instance_testing( instance_name=map , vehicle_num=vehicle_num)
                for key,item in Recorder.items() : 
                    Performance_Matrix[key][0] += item[2] # Performance Gap 
                    Performance_Matrix[key][1] += item[1] # Time per instance 
            
        for key in Performance_Matrix.keys() : 
            Performance_Matrix[key][0] /= (len(dataset_list) * len(vehicle_num_set)) 
            Performance_Matrix[key][1] /= (len(dataset_list) * len(vehicle_num_set ))
            print(f"{key} avaerage Gap:{Performance_Matrix[key][0]} , time : {Performance_Matrix[key][1]}")
            Performance_Matrix[key][0] = Performance_Matrix[key][0]
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
        
        for i in range(len(pareto_frontier) - 1):
            plt.plot([pareto_frontier[i][0], pareto_frontier[i+1][0]],
                     [pareto_frontier[i][1], pareto_frontier[i+1][1]], 'r--')

        colors = range(len(Performance))
        # colors = ["red","orange","yellow","green","blue","indigo","purple","grey","cyan",'brown']
        
        
        scatter = plt.scatter([x for x, _, _ in Performance], [y for _, y, _ in Performance] ,c=colors ,cmap="Spectral",s=150 , marker=".")
        plt.title('Performance vs Time')
        plt.xlabel('Avg. Computing time(s)')
        plt.grid()
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
        Instance_testing(instance_name=file_name , vehicle_num=vehicle_num)
        
        
        





