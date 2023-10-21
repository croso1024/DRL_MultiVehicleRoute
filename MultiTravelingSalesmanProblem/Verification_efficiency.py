""" 
    2023-04-05 HCVRP版本 , 測試模型性能使用OR-tools 
    2023-04-10 實現DE, MIX inference method
    2023-04-11 實現每一個Batch重新sample一次capacity , velocity 
    2023-04-13 透過增大batch來增加效率 : 
        1. 將PMPO_dataset單獨拿出來 , 免去重複處理PMPO 
        2. 增加PMPO_dataset的大小 , 設定maximum_batch_size , 即允許PMPO自己擴大batch_size , 直到滿足maximum
        3. 修正DE遇到非8 type無法運作的bug , 同樣允許DE,mix做擴大batchsize 
        4. 修正Sample-based method的bug 
        5. 加入DE-sample-based  method 
    2023-04-14 加入新建的Dataset loader , 改用固定的Dataset做測試, 主要只有修改產生dataset的部份 ,改以預先製作的dataset,capacity,velocity setting
"""
from Env.Environment_DSE import MTSP_Environment 
from torch.distributions import Categorical 
from utils.MTSPGenerator import MTSP_DataGenerator 
from utils.VRP_solver import ORtools_VRP
from tqdm import tqdm 
import argparse ,torch,time  , copy 
from utils.ValidationDataset import LoadDataset
import numpy as np 


def env_maker_validate(batch_size , batch_data  , StateEQ    ): 
    return MTSP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_pos_mode="Depot" , 
        DE_transform_type=DE_transform_type , 
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
    np.save("./model/MultiTravelingSalesmanProblem/experiment_data/"+recorder_file_name+"_Greedy.npy" , np.array([v.item() for v in recorder]))
    print(f"Greedy Complete in {consumption_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"-----------------------------------------")
    
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
                
    np.save("./model/MultiTravelingSalesmanProblem/experiment_data/"+recorder_file_name+"_POMO.npy" , np.array([v.item() for v in recorder]))
    print(f"PMPO Complete in {consumption_time:.3f} seconds")
    print(f"Average objective : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")
    
def PMPO_sample(Agent ,validation_set ,batch_size,vehicle_nums  , maximum_batch_size = None,sample_size=3): 
    """ 
        更改模式 , 變為一樣sample PMPO-batch , 只是多跑幾次去紀錄最好的成果
    """
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    average_tour_length = 0 
    consumption_time = 0 
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    PMPO_dataset = validation_set
    with torch.no_grad(): 
        for i ,  batch in tqdm(enumerate( copy.deepcopy(PMPO_dataset) )): 
            start = time.time() 
            # recorder for this batch 
            batch = copy.deepcopy(batch)
            best_objective = torch.ones(size=(batch_size,1) , dtype=torch.float32,device=device) * 100 
            for sample_times in range(sample_size) : 
                batch.to(device) 
                env = env_maker_validate(batch_size=batch_size , batch_data=batch , StateEQ="PMPO"  )
                batch = env.get_initilize_Batch() 
                batch.state , fleet_state , vehicle_state ,mask , done = env.reset() 
                while not done : 
                    action_dist = Categorical(Agent(batch,fleet_state , vehicle_state,mask) )
                    action = action_dist.sample() 
                    batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
                objective_value = reward 
                PMPO_objective_value = env.StateEQ_reward_function(objective_value , mode="min")
                index = PMPO_objective_value < best_objective
                # update the best solution
                best_objective[index] = PMPO_objective_value[index]
            consumption_time += time.time() - start 
            average_tour_length += best_objective.mean()
            
    print(f"PMPO-Sample Complete in {consumption_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")





def DE(Agent , validation_set , batch_size , vehicle_num , maximum_batch_size = None): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    DE_batch_size = batch_size * DE_transform_type
    average_tour_length = 0 
    computation_time = 0 
    with torch.no_grad() : 
         
        for i, batch in enumerate(tqdm(validation_set)): 
            start = time.time()
            batch.to(device)
            # env = env_maker_validate(batch)
            # env = env_maker_validate(batch_size=DE_batch_size  , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
            #                         , vehicle_velocity=vehicle_charateristic[1].tolist() , StateEQ="DE"  ) 
            env = env_maker_validate(batch_size=DE_batch_size , batch_data=batch , StateEQ="DE")
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
            minSum_tour = reward 
            computation_time += time.time() - start 
            # DE_minSum_tour , DE_fulill_rate = env.calculate_DE_inference(minSum_tour)
            DE_minSum_tour = env.StateEQ_reward_function(minSum_tour , mode="min")
            average_tour_length += DE_minSum_tour.mean() 
    print(f"DE Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"-----------------------------------------")
    


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
    np.save("./model/MultiTravelingSalesmanProblem/experiment_data/"+recorder_file_name+OR_algo_set+".npy" , np.array(recorder) )
    
    print(f"ORTools-{algo}({time_limit}s) Complete in {consumption_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length:.3f}")
    print(f"-----------------------------------------")


if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d","--dataset_size", type=int, default=128)
    argParser.add_argument("-b","--batch_size", type=int, default=32)
    argParser.add_argument("-n","--node_num", type=int, default=20)
    argParser.add_argument("-v","--vehicle_num", type=int, default=1)
    argParser.add_argument("-t","--ortools_times", type=int, default=1)
    argParser.add_argument("-trans" , "--transform_type" , type=int , default = 8)
    argParser.add_argument("-m","--model", type=str, default="RL_agent") 
    arg = argParser.parse_args() 
    
    dataset_size = arg.dataset_size 
    batch_size = arg.batch_size
    node_num = arg.node_num 
    vehicle_num = arg.vehicle_num 
    ortools_timelimit = arg.ortools_times 
    DE_transform_type = arg.transform_type 
    model_path = "./model/MultiTravelingSalesmanProblem/checkpoint/" + arg.model + ".pth" 
    
    ##################### Prepare Model   ######################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from net.ver20.PolicyNetwork import PolicyNetwork 
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
    model_path = "./model/MultiTravelingSalesmanProblem/checkpoint/Journal/N100V5_v20k12_DSE_0829.pth"
    Agent.load_state_dict(torch.load(model_path))
    total = sum([parameters.nelement() for parameters in Agent.parameters()]) 
    print(f"Parameters of the Model : {total}")
    print("--------------\n")
    ################ Testing Parameters   #################
    start_pos = "Depot"
    maximum_batch_size = 32

    ##################### Prepare dataset   ######################
    print(f"Verification dataset : {dataset_size} instance \n {node_num} nodes per instance with {vehicle_num} vehicles" )
    recorder_file_name = f"N{node_num}V{vehicle_num}"
    # instance_generator = MTSP_DataGenerator(workers=4 , batch_size=batch_size , node_num=node_num)
    dataset = LoadDataset(
        dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
         maximum_batch_size=maximum_batch_size , PMPO=False )
    PMPO_dataset  = LoadDataset(
        dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
         maximum_batch_size=maximum_batch_size , PMPO=True )
    



    Greedy(Agent=Agent, validation_set=copy.deepcopy(dataset) , batch_size=batch_size  , vehicle_nums=vehicle_num , 
           )

    PMPO(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num , 
       maximum_batch_size=maximum_batch_size)

    PMPO_sample(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num , 
            maximum_batch_size=maximum_batch_size , sample_size=10 )

    DE(Agent=Agent , validation_set=copy.deepcopy(dataset),batch_size=batch_size ,vehicle_num = vehicle_num ,
        maximum_batch_size=maximum_batch_size )

    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*300 ,algo='GD')
    
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*2 ,algo="GL")
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*5 ,algo="GL")
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*300 ,algo="GL")
    
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*2 ,algo="TS")
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*5 ,algo="TS")
    # ORTools_MTSP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num  ,  time_limit=ortools_timelimit*300 ,algo="TS")