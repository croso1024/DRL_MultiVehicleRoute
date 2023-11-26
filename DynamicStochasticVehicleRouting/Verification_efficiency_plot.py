""" 
    DSVRP 驗證集測試
    
    2023-10-01 : 
        配合Journal版本 , 在環境中加入KNN , 同時修復了ORTools的DSVRP_solver在計算滿足率的bug , 即他們的1~(v+1)是有demand會扣車輛的
        因此這邊先把demand_vector[1~(v+1)]也變為 0 , 同時計算這一塊被消除的demand和 作為fix_demand傳入Solver
        讓OR-Tools計算完後的完成與總需求 , 分子分母都要加上這一塊fix_demand
        
    2023-11-11 : 
        加上vis的引數 , 在Greedy做了一個plot 
        
"""
# 對於Journal版本來說 , Environment用MaskInit或Journal基本一樣 , 因為journal主要改了在訓練時的部份
from Env.Environment_MaskInit import DSVRP_Environment 
from torch.distributions import Categorical 
from utils.DSVRPGenerator import DSVRP_DataGenerator 
from utils.DSVRP_solver_Journal import ORtools_DSVRP
from tqdm import tqdm 
import argparse ,torch,time  , copy 
from utils.ValidationDataset import LoadDataset
from utils.RouteVisulizer import RouteVisulizer
import random , numpy as np 

#############  SEED  ###############
seed = 105
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


def env_maker_validate(batch_size , batch_data  ,capacity_vector, StateEQ    ): 
    return DSVRP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_capacity=capacity_vector , 
        vehicle_pos_mode="Random" , 
        StateEQ=StateEQ , 
        DE_transform_type=DE_transform_type , 
        graph_transform={"transform":"knn" , "value":12} , 
        # graph_transform=None , 
        training=False , 
        device = device ,
    )


def Greedy(Agent,validation_set , batch_size , vehicle_nums): 

    
    average_tour_length = 0 
    average_fulfill_rate = 0  
    computation_time = 0
    with torch.no_grad(): 
        for i , (batch,capacity_vector) in enumerate(tqdm(validation_set)): 
            start = time.time()
            batch = batch.to(device) 
            env = env_maker_validate(batch_size=batch_size , batch_data=batch , capacity_vector=capacity_vector.tolist() ,StateEQ=None ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask ,done = env.reset() 
            while not done : 
                action_dist = Agent(batch ,fleet_state, vehicle_state , mask) 
                action = torch.argmax(action_dist, dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask , done = env.step(action.view(batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time += time.time() - start 
            average_tour_length += minSum_tour.mean() 
            average_fulfill_rate += batchwise_fulfill_rate.mean() 
            
        
            if visulize : 
                visulizer = RouteVisulizer(
                    node_num = node_num , vehicle_num=vehicle_num , method="GREEDY" , 
                    capacity=True , show=False , store=True , 
                    store_path="./model/DynamicStochasticVehicleRouting/plot/" 
                )
                route = env.get_SplitedRoute()[0]
                visulizer.Visualize(
                    fig_id=i , instance=batch , routes=route , objective_value=0
                )
                
            
            
            
    print(f"Greedy Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(validation_set):.3f}")
    # effective_route = env.get_effective_route()
    # type_dict , decision_sequence =env.get_DecisionSequence() 
    # print(f"type-dict : {type_dict} ")
    # print(f"Decision sequence : {decision_sequence}")
    # print(f"Effective route : {effective_route}")
    
    print(f"-----------------------------------------")
    
def PMPO(Agent,validation_set , batch_size , vehicle_nums , maximum_batch_size=None ): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    average_tour_length = 0 
    average_fulfill_rate = 0  
    computation_time = 0
    #PMPO_dataset = instance_generator.dataset_to_PMPOdataset_SingleProcess(validation_set)
    charateristic_set_number = 8 if not maximum_batch_size  else (8* batch_size) // maximum_batch_size  
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    PMPO_dataset = validation_set
    with torch.no_grad() : 
        
        for i , (batch,capacity_vector) in enumerate(tqdm(PMPO_dataset)): 
            # print(f"Debug batch : {batch}\n")
            start = time.time() 
            batch.to(device)
            env = env_maker_validate(batch_size=batch_size , batch_data=batch ,  capacity_vector=capacity_vector.tolist(), StateEQ="PMPO")
            batch = env.get_initilize_Batch()
            # print(f"Debug get batch : {batch}\n")
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time += time.time() - start 
            PMPO_minSum_tour , PMPO_fulill_rate = env.calculate_PMPO_inference(minSum_tour)
            average_tour_length += PMPO_minSum_tour.mean() 
            average_fulfill_rate += PMPO_fulill_rate.mean() 
    print(f"PMPO Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")
    
def DE(Agent,validation_set , batch_size , vehicle_nums   ): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    DE_batch_size = batch_size * DE_transform_type
    average_tour_length = 0 
    average_fulfill_rate = 0
    computation_time = 0   
    with torch.no_grad() : 
        
        for i, (batch,capacity_vector) in enumerate(tqdm(validation_set)): 
            start = time.time() 
            batch.to(device)
            # env = env_maker_validate(batch)
            env = env_maker_validate(batch_size=DE_batch_size  , batch_data=batch ,  capacity_vector=capacity_vector.tolist(),  StateEQ="DE"  ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time += time.time() - start 
            DE_minSum_tour , DE_fulill_rate = env.calculate_DE_inference(minSum_tour)
            average_tour_length += DE_minSum_tour.mean() 
            average_fulfill_rate += DE_fulill_rate.mean() 
        
    print(f"DE Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(validation_set):.3f}")
    
    # effective_route = env.get_effective_route()
    # type_dict , decision_sequence =env.get_DecisionSequence() 
    # assert len(decision_sequence) == len(effective_route) 
    # for i in range(len(effective_route)): 
    #     print(f"SE:{decision_sequence[i]} --> {effective_route[i]} \n")

    print(f"-----------------------------------------")
    
def mix(Agent,validation_set , batch_size ,vehicle_nums  ,maximum_batch_size=None): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    charateristic_set_number = 8 if not maximum_batch_size else  (8*batch_size) // maximum_batch_size
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    DE_batch_size = batch_size * DE_transform_type  
    average_tour_length = 0 
    computation_time = 0
    average_fulfill_rate = 0  
    # PMPO_dataset = instance_generator.dataset_to_PMPOdataset_SingleProcess(validation_set)
    PMPO_dataset = validation_set
    with torch.no_grad() : 
        for i, (batch,capacity_vector) in enumerate(tqdm(PMPO_dataset)): 
            start = time.time() 
            batch.to(device)
            env = env_maker_validate(batch_size=DE_batch_size  , batch_data=batch , capacity_vector=capacity_vector.tolist() , StateEQ="mix"  ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time += time.time() -start
            mix_minSum_tour , mix_fulill_rate = env.calculate_mix_inference(minSum_tour)
            average_tour_length += mix_minSum_tour.mean() 
            average_fulfill_rate += mix_fulill_rate.mean() 
    print(f"Mix Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")
def PMPO_sample(Agent ,validation_set ,batch_size,vehicle_nums  , maximum_batch_size = None,sample_size=3): 
    """ 
        更改模式 , 變為一樣sample PMPO-batch , 只是多跑幾次去紀錄最好的成果
    """
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    average_tour_length = 0 
    average_fulfill_rate = 0 
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    PMPO_dataset = validation_set
    with torch.no_grad(): 
        start = time.time() 
        for i ,  (batch,capacity_vector) in tqdm(enumerate( copy.deepcopy(PMPO_dataset) )): 
            # recorder for this batch 
            batch = copy.deepcopy(batch)
            best_objective = torch.ones(size=(batch_size,1) , dtype=torch.float32,device=device) * 100 

            for sample_times in range(sample_size) : 
                batch.to(device) 
                env = env_maker_validate(batch_size=batch_size , batch_data=batch ,capacity_vector=capacity_vector, StateEQ="PMPO" )
                batch = env.get_initilize_Batch() 
                batch.state , fleet_state , vehicle_state ,mask , done = env.reset() 
                while not done : 
                    action_dist = Categorical(Agent(batch,fleet_state , vehicle_state,mask) )
                    action = action_dist.sample() 
                    batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
                objective_value , fulfill_rate = reward 
                
                
                PMPO_objective_value = env.StateEQ_reward_function(objective_value , mode="min")
                index = PMPO_objective_value < best_objective
                # update the best solution
                best_objective[index] = PMPO_objective_value[index]
                
            average_tour_length += best_objective.mean()
            average_fulfill_rate += fulfill_rate.mean()
        end = time.time() 
    print(f"PMPO-Sample Complete in {(end-start):.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"Average fulfill : {average_fulfill_rate:.3f}")
    print(f"-----------------------------------------")

    
def ORTools_DSVRP(validation_set , batch_size ,vehicle_num ,algo="GD" ,time_limit=1 ):

    algo_table = {
        "GD":"GREEDY_DESCENT" ,
        "GL":"GUIDED_LOCAL_SEARCH",
        "SA":"SIMULATED_ANNEALING",
        "TS":"TABU_SEARCH" ,
        "GTS":"GENERIC_TABU_SERACH" ,
    }
    
    average_tour_length = 0 
    average_fulfill_rate = 0
    computation_time = 0 
    for i ,(batch_data,capacity_vector) in enumerate(tqdm(validation_set)): 
        
        for instance in tqdm(batch_data.to_data_list()): 
            start = time.time() 
            demand_vector = (instance.x[:,2]*10000).tolist()
            # 2023-09-27 : 修正 , demand vector應該是0-(v+1)也都0 , 並且計算fix_demand ,傳入ORTools_DSVRP內補償滿足率計算
            # demand_vector[0] = 0 
            fix_demand = sum(demand_vector[1:vehicle_num+1])
            for i in range(vehicle_num+1) : demand_vector[i] = 0
            dist_matrix = torch.round(instance.node_route_cost*10000).int().tolist()
            # dist_matrix = torch.round(instance.node_route_cost_std*10000).int().tolist()
            dist_matrix_std = torch.round(instance.node_route_cost_std*10000).int().tolist() 
            tour_length ,fulfill_rate = ORtools_DSVRP(
                dist_matrix=dist_matrix, 
                dist_matrix_std =dist_matrix_std ,  
                demand_vector= demand_vector , 
                vehicle_num=vehicle_num,
                start_pos =  list(range(1 , vehicle_num+1)) , 
                capacity_vector= [c*10000 for c in capacity_vector.tolist()], 
                algo=algo_table[algo],
                depot_index=0, # Avoid error
                fix_demand = fix_demand ,
                time_limit=time_limit 
            ).solve() 
            average_tour_length += tour_length 
            average_fulfill_rate += fulfill_rate
            computation_time += time.time() -start 
    average_tour_length = average_tour_length / (10000*len(validation_set)*batch_size)
    average_fulfill_rate = average_fulfill_rate / (len(validation_set)*batch_size) 
    print(f"ORTools-{algo}({time_limit}s) Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length:.3f}")
    print(f"Average fulfill : {average_fulfill_rate:.3f}")
    print(f"-----------------------------------------")


if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d","--dataset_size", type=int, default=128)
    argParser.add_argument("-b","--batch_size", type=int, default=32)
    argParser.add_argument("-n","--node_num", type=int, default=20)
    argParser.add_argument("-v","--vehicle_num", type=int, default=1)
    argParser.add_argument("-vis","--visulize", action="store_true", default=False)
    argParser.add_argument("-t","--ortools_times", type=int, default=1)
    argParser.add_argument("-trans" , "--transform_type" , type=int , default=8)
    argParser.add_argument("-m","--model", type=str, default="RL_agent") 
    arg = argParser.parse_args() 
    
    dataset_size = arg.dataset_size 
    batch_size = arg.batch_size
    node_num = arg.node_num 
    vehicle_num = arg.vehicle_num 
    visulize =  arg.visulize 
    ortools_timelimit = arg.ortools_times 
    DE_transform_type = arg.transform_type 
    model_path = "./model/DynamicStochasticVehicleRouting/checkpoint/" + arg.model + ".pth" 
    
    ##################### Prepare Model   ######################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from net.ver20.PolicyNetwork import PolicyNetwork 
    Agent = PolicyNetwork(
        node_feature_dim=5,
        fleet_feature_dim=6,
        vehicle_feature_dim=5 , 
        hidden_dim=192 , 
        heads = 6 , 
        num_layers=3 , 
        skip_connection=True , 
        clip_coe=10,
        temp=1
    ).to(device) 
    model_path = "./model/DynamicStochasticVehicleRouting/checkpoint/N50V5_Journal.pth"
    # model_path = "./model/DynamicStochasticVehicleRouting/checkpoint/N55_v20_0606.pth"
    Agent.load_state_dict(torch.load(model_path))
    total = sum([parameters.nelement() for parameters in Agent.parameters()]) 
    print(f"Parameters of the Model : {total}")
    print("--------------\n")
    ################ Testing Parameters   #################
    DE_transform_type = 8 
    maximum_batch_size = 32

    ##################### Prepare dataset   ######################
    print(f"Verification dataset : {dataset_size} instance \n {node_num} nodes per instance with {vehicle_num} vehicles" )
    instance_generator = DSVRP_DataGenerator(workers=4 , batch_size=batch_size , node_num=node_num)
    dataset = LoadDataset(
        dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
         maximum_batch_size=maximum_batch_size , PMPO=False )
    # PMPO_dataset  = LoadDataset(
    #     dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
    #      maximum_batch_size=maximum_batch_size , PMPO=True )
    

    Greedy(Agent=Agent, validation_set=copy.deepcopy(dataset) , batch_size=batch_size , vehicle_nums=vehicle_num  )

    # PMPO(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num , 
    #      maximum_batch_size=maximum_batch_size)

    # DE(Agent=Agent, validation_set=copy.deepcopy(dataset), batch_size=batch_size , vehicle_nums=vehicle_num )

    # mix(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num ,maximum_batch_size=maximum_batch_size)

    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=10 ,algo="GD")
    
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=1 ,algo="GL")
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=2 ,algo="GL")
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=10 ,algo="GL")
    
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=1 ,algo="TS")
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=2 ,algo="TS")
    # ORTools_DSVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=10 ,algo="TS")