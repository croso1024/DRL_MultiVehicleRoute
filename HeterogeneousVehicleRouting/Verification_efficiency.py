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
from Env.Environment import HCVRP_Environment 
from torch.distributions import Categorical 
from utils.CVRPGenerator import CVRP_DataGenerator 
from utils.HCVRPD_solver import ORtools_HCVRPD
from tqdm import tqdm 
import argparse ,torch,time  , copy 
from utils.ValidationDataset import LoadDataset



def env_maker_validate(batch_size , batch_data, vehicle_capacity , vehicle_velocity  , StateEQ   ): 
    return HCVRP_Environment(
        batch_size=batch_size , 
        node_nums = node_num , 
        vehicle_num=vehicle_num , 
        batch_data = batch_data , 
        vehicle_capacity=vehicle_capacity, 
        vehicle_velocity=vehicle_velocity , 
        vehicle_pos_mode="Depot", 
        DE_transform_type=DE_transform_type , 
        StateEQ=StateEQ , 
        training=False , 
        graph_transform={"transform":"knn" , "value":12}, 
        device = device ,
    )


def Greedy(Agent,validation_set , batch_size , vehicle_nums): 

    
    average_tour_length = 0 
    average_fulfill_rate = 0  
    computation_time = 0 
    with torch.no_grad(): 
        for i , (batch,vehicle_charateristic) in enumerate(tqdm(validation_set)): 
            start = time.time()
            batch = batch.to(device) 
            env = env_maker_validate(batch_size=batch_size , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
                                     , vehicle_velocity=vehicle_charateristic[1].tolist() , StateEQ=None ) 
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
    print(f"Greedy Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(validation_set):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(validation_set):.3f}")
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
        for i , (batch,vehicle_charateristic) in enumerate(tqdm(PMPO_dataset)): 
            start = time.time() 
            # print(f"Debug batch : {batch}\n")
            batch.to(device)
            env = env_maker_validate(batch_size=batch_size , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
                                     , vehicle_velocity=vehicle_charateristic[1].tolist()  , StateEQ="PMPO"  )
            batch = env.get_initilize_Batch()
            # print(f"Debug get batch : {batch}\n")
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time+= time.time() - start 
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
         
        for i, (batch,vehicle_charateristic) in enumerate(tqdm(validation_set)): 
            start = time.time()
            batch.to(device)
            # env = env_maker_validate(batch)
            env = env_maker_validate(batch_size=DE_batch_size  , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
                                     , vehicle_velocity=vehicle_charateristic[1].tolist() , StateEQ="DE"  ) 
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
    print(f"-----------------------------------------")
    
def mix(Agent,validation_set , batch_size ,vehicle_nums  ,maximum_batch_size=None): 
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    charateristic_set_number = 8 if not maximum_batch_size else  (8*batch_size) // maximum_batch_size
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    DE_batch_size = batch_size * DE_transform_type  
    average_tour_length = 0 
    average_fulfill_rate = 0  
    computation_time = 0
    # PMPO_dataset = instance_generator.dataset_to_PMPOdataset_SingleProcess(validation_set)
    PMPO_dataset = validation_set
    with torch.no_grad() : 
        
        for i, (batch,vehicle_charateristic) in enumerate(tqdm(PMPO_dataset)): 
            start = time.time() 
            batch.to(device)
            env = env_maker_validate(batch_size=DE_batch_size  , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
                                , vehicle_velocity=vehicle_charateristic[1].tolist() , StateEQ="mix"  ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch,fleet_state , vehicle_state ,mask )
                action = torch.argmax(action_dist,dim=1,keepdim=False)
                batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(DE_batch_size,-1))
            minSum_tour , batchwise_fulfill_rate = reward 
            computation_time += time.time() - start 
            mix_minSum_tour , mix_fulill_rate = env.calculate_mix_inference(minSum_tour)
            average_tour_length += mix_minSum_tour.mean() 
            average_fulfill_rate += mix_fulill_rate.mean() 
    print(f"Mix Complete in {computation_time:.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")


def PMPO_sample(Agent ,validation_set ,batch_size,vehicle_nums 
                , maximum_batch_size = None,sample_size=3): 
    """ 
        更改模式 , 變為一樣sample PMPO-batch , 只是多跑幾次去紀錄最好的成果
    """
    assert batch_size % 8 ==0 , "Batch size not match PMPO"
    average_tour_length = 0 
    average_fulfill_rate = 0  
    charateristic_set_number = 8 if not maximum_batch_size else (8*batch_size) // maximum_batch_size
    batch_size = batch_size if not maximum_batch_size else maximum_batch_size
    PMPO_dataset = validation_set
    with torch.no_grad(): 
        start = time.time() 
        for i ,  (batch,vehicle_charateristic) in tqdm(enumerate( copy.deepcopy(PMPO_dataset) )): 
            # recorder for this batch 
            batch = copy.deepcopy(batch)
            best_score = torch.ones(size=(batch_size,1) , dtype=torch.float32 , device=device) * 100 
            best_tour_length =torch.zeros(size=(batch_size,1) , dtype=torch.float32 , device=device)
            best_tour_fulfill =torch.zeros(size=(batch_size,1) , dtype=torch.float32 , device=device)

            for sample_times in range(sample_size) : 
                batch.to(device) 
                env = env_maker_validate(batch_size=batch_size , batch_data=batch , vehicle_capacity=vehicle_charateristic[0].tolist()
                                     , vehicle_velocity=vehicle_charateristic[1].tolist() , StateEQ="PMPO"  )
                batch = env.get_initilize_Batch() 
                batch.state , fleet_state , vehicle_state ,mask , done = env.reset() 
                while not done : 
                    action_dist = Categorical(Agent(batch,fleet_state , vehicle_state,mask) )
                    action = action_dist.sample() 
                    batch.state , fleet_state , vehicle_state , reward , mask ,done = env.step(action.view(batch_size,-1))
                minSum_tour , batchwise_fulfill_rate = reward 
                PMPO_minSum_tour , PMPO_fulfill_rate = env.calculate_PMPO_inference(minSum_tour)
                score = torch.exp( 5*(1-PMPO_fulfill_rate)  ) * ( PMPO_minSum_tour + (node_num/(2*vehicle_num)))
                index = score <= best_score 
                # print(f"Debug before update : {best_score}\n {best_tour_length}\n {best_tour_fulfill}\n")
                best_score[index] = score[index] 
                best_tour_length[index] = PMPO_minSum_tour[index]
                best_tour_fulfill[index] = PMPO_fulfill_rate[index]
                
            average_tour_length += best_tour_length.mean() 
            average_fulfill_rate += best_tour_fulfill.mean() 
            
        end = time.time() 
    print(f"PMPO-Sample Complete in {(end-start):.3f} seconds")
    print(f"Average minSum : {average_tour_length/len(PMPO_dataset):.3f}")
    print(f"Average fulfill : {average_fulfill_rate/len(PMPO_dataset):.3f}")
    print(f"-----------------------------------------")

    
def ORTools_HCVRP(validation_set , batch_size ,vehicle_num  , time_limit=1 ):

    average_tour_length = 0 
    average_fulfill_rate = 0 
    start = time.time() 
    for i ,(batch_data,vehicle_charateristic) in enumerate(tqdm(validation_set)): 
        for instance in tqdm(batch_data.to_data_list()): 
            demand_vector = (instance.x[:,2]*10000).tolist() 
            demand_vector[0] = 0 
            dist_matrix = torch.round(instance.node_route_cost*10000).int().tolist()
            tour_length , fulfill_rate = ORtools_HCVRPD(
                dist_matrix=dist_matrix , 
                demand_vector=demand_vector , 
                vehicle_num=vehicle_num , 
                depot_index=0, # Avoid error
                capacity_vector=[ c*10000 for c in  vehicle_charateristic[0].tolist()] , 
                velocity_vector=vehicle_charateristic[1].tolist(), 
                time_limit=time_limit 
            ).solve() 
            average_tour_length += tour_length 
            average_fulfill_rate += fulfill_rate
    end = time.time() 
    average_tour_length = average_tour_length / (10000*len(validation_set)*batch_size)
    average_fulfill_rate = average_fulfill_rate / (len(validation_set)*batch_size)
    print(f"ORTools({time_limit}s) Complete in {(end-start):.3f} seconds")
    print(f"Average minSum : {average_tour_length:.3f}")
    print(f"Average fulfill : {average_fulfill_rate:.3f}")
    print(f"-----------------------------------------")


            

if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d","--dataset_size", type=int, default=128)
    argParser.add_argument("-b","--batch_size", type=int, default=32)
    argParser.add_argument("-n","--node_num", type=int, default=20)
    argParser.add_argument("-v","--vehicle_num", type=int, default=1)
    argParser.add_argument("-t","--ortools_times", type=int, default=1)
    argParser.add_argument("-rp","--random_pos", action="store_true", default=False)
    argParser.add_argument("-he","--heterogeneous", action="store_true", default=False)
    argParser.add_argument("-m","--model", type=str, default="RL_agent") 
    arg = argParser.parse_args() 
    
    dataset_size = arg.dataset_size 
    batch_size = arg.batch_size
    node_num = arg.node_num 
    vehicle_num = arg.vehicle_num 
    random_pos = arg.random_pos 
    heterogeneous = arg.heterogeneous
    ortools_timelimit = arg.ortools_times 
    model_path = "./model/HeterogeneousVehicleRouting/checkpoint/" + arg.model + ".pth" 
    
    ##################### Prepare Model   ######################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from net.ver20.PolicyNetwork import PolicyNetwork as PolicyNetwork
    Agent = PolicyNetwork(
        node_feature_dim=5,
        fleet_feature_dim=7,
        vehicle_feature_dim=6 , 
        # GCN_prelayer=False , 
        hidden_dim=192 , 
        heads = 6 , 
        num_layers=3 , 
        skip_connection=True , 
        clip_coe=10,
        temp=1.5
    ).to(device) 
    model_path = "./model/HeterogeneousVehicleRouting/checkpoint/N55H_v20k12_0604.pth"
    Agent.load_state_dict(torch.load(model_path))
    total = sum([parameters.nelement() for parameters in Agent.parameters()]) 
    print(f"Parameters of the Model : {total}")
    print("--------------\n")
    ################ Testing Parameters   #################
    DE_transform_type = 16
    maximum_batch_size = 64

    ##################### Prepare dataset   ######################
    print(f"Verification dataset : {dataset_size} instance \n {node_num} nodes per instance with {vehicle_num} vehicles" )
    instance_generator = CVRP_DataGenerator(workers=4 , batch_size=batch_size , node_num=node_num)
    dataset = LoadDataset(
        dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
        heterogeneous= heterogeneous , maximum_batch_size=maximum_batch_size , PMPO=False )
    PMPO_dataset  = LoadDataset(
        dataset_size=dataset_size , batch_size=batch_size , node_num= node_num , vehicle_num= vehicle_num , 
        heterogeneous= heterogeneous , maximum_batch_size=maximum_batch_size , PMPO=True )
    

    Greedy(Agent=Agent, validation_set=copy.deepcopy(dataset) , batch_size=batch_size , vehicle_nums=vehicle_num )

    PMPO(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num ,maximum_batch_size=maximum_batch_size)

    if heterogeneous: 
        DE(Agent=Agent, validation_set=copy.deepcopy(dataset), batch_size=batch_size , vehicle_nums=vehicle_num )

        mix(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num ,maximum_batch_size=maximum_batch_size)
    
    # PMPO_sample(Agent=Agent, validation_set=copy.deepcopy(PMPO_dataset) , batch_size=batch_size , vehicle_nums=vehicle_num  ,maximum_batch_size=maximum_batch_size , sample_size=5 )
    #ORTools_HCVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num , 
    #           time_limit= ortools_timelimit)
    # ORTools_HCVRP(validation_set=dataset , batch_size=batch_size , vehicle_num=vehicle_num , 
    #         time_limit= 2)

    
