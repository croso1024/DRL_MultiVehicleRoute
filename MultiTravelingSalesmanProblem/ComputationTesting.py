""" 
    2023-07-27 - 用來提供隨著問題規模的改變計算時間的成長折線圖 ,
    總共包含 ORtools-GD , Our-Greedy , Our-POMO 三種算法 , V5 與 V10的兩個Set (共6條折線)
"""

from Env.Environment import MTSP_Environment 
from torch.distributions import Categorical 
from utils.MTSPGenerator import MTSP_DataGenerator 
from utils.VRP_solver import ORtools_VRP
import argparse ,torch,time  , copy 
from net.ver20.PolicyNetwork import PolicyNetwork 
import numpy as np 
from tqdm import tqdm 
import os ,json 
from utils.ValidationDataset import * 

def env_maker_validate(batch_size , batch_data  , StateEQ    ): 
    return MTSP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_pos_mode="Depot" , 
        StateEQ=StateEQ , 
        graph_transform={"transform":"knn" , "value":12} , 
        device = device ,
    )


def Greedy(Agent,validation_set , batch_size , vehicle_nums ): 

    recorder = [] 
    consumption_time = 0 
    average_tour_length = 0 
    with torch.no_grad(): 
        for i , batch in enumerate(tqdm(validation_set)): 
            start = time.time()
            batch = batch.to(device) 
            env = env_maker_validate(batch_size=batch_size , batch_data=batch , StateEQ=None  ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask ,done = env.reset() 
            while not done : 
                action_dist = Agent(batch ,fleet_state, vehicle_state , mask) 
                action = torch.argmax(action_dist, dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask , done = env.step(action.view(batch_size,-1))
            objective_value = reward 
            consumption_time += time.time()-start
            recorder.extend( objective_value.detach().cpu()  )
            average_tour_length += objective_value.mean() 
    print(f"Greedy Complete in {consumption_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"-----------------------------------------")
    return {"obj":(average_tour_length/len(validation_set)).item() , "time":consumption_time/arg.dataset_size }
    
def PMPO(Agent,validation_set , batch_size , vehicle_nums  , maximum_batch_size=None ): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    average_tour_length = 0 
    recorder = [] 
    consumption_time = 0 
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    PMPO_dataset = validation_set
    with torch.no_grad() : 
        
        for i , batch in enumerate(tqdm(PMPO_dataset)): 
            start = time.time() 
            batch.to(device)
            env = env_maker_validate(batch_size=batch_size , batch_data=batch , StateEQ="PMPO"   )
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
            objective_value = reward 
            consumption_time += time.time() - start
            PMPO_objective_value = env.StateEQ_reward_function(objective_value , mode="min")
            average_tour_length += PMPO_objective_value.mean() 
            for i in range(maximum_batch_size//8):
                recorder.append(PMPO_objective_value[i*8].cpu())
                
    print(f"PMPO Complete in {consumption_time:.3f} seconds")
    print(f"Average objective : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")
    return {"obj":(average_tour_length/len(PMPO_dataset)).item() , "time":consumption_time/arg.dataset_size}

def ORTools_MTSP(validation_set , batch_size ,vehicle_num , time_limit=1 ,algo="GD"):
    algo_table = {
        "GD":"GREEDY_DESCENT" ,
        "GL":"GUIDED_LOCAL_SEARCH",
        "SA":"SIMULATED_ANNEALING",
        "TS":"TABU_SEARCH" ,
        "GTS":"GENERIC_TABU_SERACH" ,
    }
    
    recorder = [] 
    consumption_time = 0 
    average_tour_length = 0 
    start = time.time() 
    for i ,batch_data in enumerate(tqdm(validation_set)): 
        for instance in tqdm(batch_data.to_data_list()): 
            start = time.time() 
            dist_matrix = torch.round(instance.node_route_cost*10000).int().tolist()
            tour_length  = ORtools_VRP(
                dist_matrix=dist_matrix , 
                vehicle_num=vehicle_num , 
                depot_index=0, # Avoid error
                time_limit=time_limit ,
                algo = algo_table[algo] ,
            ).solve() 
            consumption_time += time.time() - start 
            average_tour_length += tour_length 
            recorder.append(tour_length/10000)
    average_tour_length = average_tour_length / (10000*len(validation_set)*batch_size)
    OR_algo_set = "_OR"+algo+str(time_limit)
    
    print(f"ORTools-{algo}({time_limit}s) Complete in {consumption_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length:.3f}")
    print(f"-----------------------------------------")
    return {"obj":average_tour_length , "time":consumption_time/arg.dataset_size}

if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-ln" , "--lowerbound",type=int , default=50)
    argParser.add_argument("-hn" , "--highbound",type=int , default=100)
    argParser.add_argument("-i","--interval",type=int , default=5) 
    argParser.add_argument("-d","--dataset_size", type=int , default=32)
    argParser.add_argument("-b","--batch_size", type=int , default=8) 
    argParser.add_argument("-v","--vehicle_num", type=int , default=5) 
    argParser.add_argument("-m","--model", type=str , default="N50V5") 
    argParser.add_argument("-p","--data_path",type=str,default="Dataset/")
    
    arg = argParser.parse_args()
    
    dataset_size = arg.dataset_size
    batch_size = arg.batch_size
    vehicle_num = arg.vehicle_num
    lowerbound_nodes = arg.lowerbound
    highbound_nodes = arg.highbound
    interval = arg.interval 
    assert  (highbound_nodes > lowerbound_nodes) and not (highbound_nodes-lowerbound_nodes) % interval 
    
    model = {
                "N50V5":"N50V5_v20k12_0605.pth" , "N50V10":"N50V10_v20k12_0802.pth",
                "N100V5":"N100V5_v20k12_0726.pth" , "N100V10":"N100V10_v20k12_0729.pth" ,
            }
    
    model_path = "./model/MultiTravelingSalesmanProblem/checkpoint/" + model[arg.model]
    data_path = "./model/MultiTravelingSalesmanProblem/" + arg.data_path 
    ##################### Prepare Model   ######################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Agent = PolicyNetwork(
        node_feature_dim=4,
        fleet_feature_dim=5,
        vehicle_feature_dim=4 , 
        hidden_dim=192 , 
        heads = 6 , 
        num_layers=3 , 
        skip_connection=True , 
        clip_coe=10,
        temp=1
    ).to(device) 
    Agent.load_state_dict(torch.load(model_path))
    total = sum([parameters.nelement() for parameters in Agent.parameters()]) 
    print(f"Parameters of the Model : {total}")
    print("--------------\n")
    maximum_batch_size = 32
    
    #################### Create log #######################
    Method = ["ORTools-GD" , "Our-Greedy","Our-POMO"]
    Logger = { method:{  str(nodes):None for nodes in range(lowerbound_nodes , highbound_nodes+interval , interval ) }   for method in Method  }
    print(f"Initilize logger : {Logger}") 
    
    
    ################### Start testing ######################
   
    for node_num in tqdm(range(lowerbound_nodes , highbound_nodes+interval , interval)): 
        CreateDataset(dataset_size=dataset_size , batch_size=batch_size,node_num =node_num , vehicle_num=vehicle_num)
        DataName = f"Data-D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}.pt"
        print(f"Create data : Data-D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}.pt")
        dataset = LoadDataset(
            dataset_size=dataset_size,batch_size=batch_size,node_num=node_num,vehicle_num=vehicle_num , 
            maximum_batch_size=maximum_batch_size  , PMPO=False
        )
        PMPO_dataset = LoadDataset(
            dataset_size=dataset_size,batch_size=batch_size,node_num=node_num,vehicle_num=vehicle_num , 
            maximum_batch_size=maximum_batch_size  , PMPO=True
        )
        print("Start")
        
        Logger["Our-Greedy"][str(node_num)] = Greedy(Agent=Agent , validation_set=copy.deepcopy(dataset) , batch_size=batch_size,vehicle_nums=vehicle_num)
        
        Logger["Our-POMO"][str(node_num)]   = PMPO(Agent=Agent  ,validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num , 
                                                 maximum_batch_size=maximum_batch_size)

        Logger["ORTools-GD"][str(node_num)]  = ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num,time_limit=10000 ,algo='GD')

    Logger.update({"vehicle_num":vehicle_num , "Model":arg.model})
    print(Logger) 
    with open("./model/MultiTravelingSalesmanProblem/ComputationLogger.json" , "w") as jsonfile : 
        json.dump(Logger , jsonfile)
        print("Logger Complete !")
    
    if os.path.isfile(data_path+DataName) : 
        print(f"Remove data : {data_path + DataName}")   
        os.remove(data_path+DataName)
    else : 
        print("not find the data to remove")
        