""" 
    This code is use for testing different algorithms include OR-tools on the Study Case scenario
    
    
    
    
    
    
"""

from Env.Environment_SC_inference import DSVRP_Environment
from RealData.RoutingSimulator import RoutingSimulator
from RealData.simulator_utils import ORTools_input_transform
from torch_geometric.data import Batch 
from tqdm import tqdm 
from utils.DSVRP_solver import ORtools_DSVRP
import argparse ,torch,time  , copy , time 
from torch.distributions import Categorical 



def simulator_maker( config ,PMPO=False, BaseGraph=None ,node_coordinate_table=None ): 
    return  RoutingSimulator(
    init_Request_num = config["init_Request_num"] , 
    vehicle_num = config["vehicle_num"], 
    init_capacity = config["init_capacity"] , 
    maximum_Request_num = config["maximum_Request_num"], 
    step_duration = config["step_duration"] , 
    total_update_times = config["total_update_times"] ,
    deterministic=config["deterministic"] , 
    release_num=config["release_num"],
    map=config["map"],
    PMPO = PMPO,
    BaseGraph=BaseGraph, 
    node_coordinate_table=node_coordinate_table , 
)
    
def env_maker(batch_size , batch_data , node_num,vehicle_num , vehicle_pos_set , vehicle_capacity , StateEQ, config): 
    return DSVRP_Environment(
        batch_size= batch_size , 
        node_nums= node_num , 
        vehicle_num= vehicle_num,
        batch_data= batch_data ,
        vehicle_capacity= vehicle_capacity , 
        vehicle_pos_mode= vehicle_pos_set , 
        training=False , 
        graph_transform= None , 
        DE_transform_type=config["DE_transform_type"] , 
        device=config["device"] , 
        StateEQ= StateEQ ,
    )
    
def Greedy(Agent_depot , Agent_random ,config,BaseGraph = None , node_coordinate_table=None , return_looger=False  ): 
    simulator = simulator_maker(config ,PMPO=False, BaseGraph=BaseGraph,node_coordinate_table=node_coordinate_table )
    RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.reset()
    batch_size = 1  
    Agent = Agent_depot  # specify the depot model for first decision 
    terminate = False 
    while not terminate : 
        if final_change : terminate = True 
        with torch.no_grad(): 
            env= env_maker(
                batch_size=batch_size , batch_data=RoadNetwork.to(config["device"]) , 
                node_num= simulator.get_node_num() , vehicle_num=config['vehicle_num'] , 
                vehicle_capacity=vehicle_capacity , vehicle_pos_set=vehicle_pos ,StateEQ = None,config=config
            )
            batch = env.get_initilize_Batch()
            batch.state ,fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state, mask) 
                action = torch.argmax(action_dist , dim=1 , keepdim=False)
                batch.state ,fleet_state , vehicle_state , reward , mask  ,done = env.step(action.view(batch_size,-1))
        objective , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (objective + (simulator.get_node_num()/(1*config['vehicle_num'])))
        value , index = torch.min(score ,dim=0) 
        Model_Predict = env.get_ModelRoute(index)
        
        if not terminate: 
            RoadNetwork , vehicle_pos , vehicle_capacity , final_change , infopackage = simulator.step(Model_Predict)
            if infopackage["early_stop"] and return_looger :
                return infopackage["obj"] , infopackage["fulfill"] , simulator.get_Logger() 
            elif infopackage["early_stop"] and not return_looger : 
                return  infopackage["obj"] , infopackage["fulfill"]
            else : pass    
            s = time.time() 
            Agent = Agent_random            
            # print(f"Re assign agent time:{time.time() - s}")
        else : 
            objective , fulfill = simulator.step(Model_Predict) 
            #print(f"Greedy -- Objective : {objective} , fulfill : {fulfill}") 
    
    if return_looger : 
        return objective , fulfill , simulator.get_Logger()
    else :
        return objective , fulfill
    
def PMPO(Agent_depot ,Agent_random, config ,  BaseGraph = None ,node_coordinate_table=None ,  return_looger=False   ):
    simulator = simulator_maker(config ,PMPO=True, BaseGraph=BaseGraph, node_coordinate_table=node_coordinate_table)
    RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.reset()
    batch_size = 8 
    Agent = Agent_depot  # specify the depot model for first decision 
    terminate = False 
    while not terminate : 
        if final_change : terminate = True 
        with torch.no_grad(): 
            env= env_maker(
                batch_size=batch_size , batch_data=RoadNetwork.to(config["device"]) , 
                node_num= simulator.get_node_num() , vehicle_num=config['vehicle_num'] , 
                vehicle_capacity=vehicle_capacity , vehicle_pos_set=vehicle_pos ,StateEQ = "PMPO",config=config
            )
            batch = env.get_initilize_Batch()
            batch.state ,fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state, mask) 
                action = torch.argmax(action_dist , dim=1 , keepdim=False)
                batch.state ,fleet_state , vehicle_state , reward , mask  ,done = env.step(action.view(batch_size,-1))
        objective , fulfill_rate = reward 
        score = torch.exp(5*(1-fulfill_rate)) * (objective + (simulator.get_node_num()/(1*config['vehicle_num'])))
        value , index = torch.min(score ,dim=0) 
        Model_Predict = env.get_ModelRoute(index)
        if not terminate: 
            RoadNetwork , vehicle_pos , vehicle_capacity , final_change , infopackage = simulator.step(Model_Predict)
            if infopackage["early_stop"] and return_looger :
                return infopackage["obj"] , infopackage["fulfill"] , simulator.get_Logger() 
            elif infopackage["early_stop"] and not return_looger : 
                return  infopackage["obj"] , infopackage["fulfill"]
            else : pass       
            Agent = Agent_random                  
        else : 
            objective , fulfill = simulator.step(Model_Predict) 
            #print(f"PMPO -- Objective : {objective} , fulfill : {fulfill}") 
    if return_looger : 
        return objective , fulfill , simulator.get_Logger()
    else :
        return objective , fulfill

def DE(Agent_depot,Agent_random,config , BaseGraph=None , node_coordinate_table=None ,  return_looger=False  ): 
    simulator = simulator_maker(config ,PMPO=False, BaseGraph=BaseGraph,node_coordinate_table=node_coordinate_table)
    RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.reset()
    DE_batch_size = config["DE_transform_type"]
    Agent = Agent_depot  # specify the depot model for first decision 
    terminate = False 
    while not terminate : 
        if final_change : terminate = True 
        with torch.no_grad() : 
            env= env_maker(
                batch_size=DE_batch_size , batch_data=RoadNetwork.to(config["device"]) , 
                node_num= simulator.get_node_num() , vehicle_num=config['vehicle_num'] , 
                vehicle_capacity=vehicle_capacity , vehicle_pos_set=vehicle_pos ,StateEQ = "DE" ,config=config
            )
            batch = env.get_initilize_Batch()
            batch.state ,fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state, mask) 
                action = torch.argmax(action_dist , dim=1 , keepdim=False)      
                batch.state ,fleet_state , vehicle_state ,reward ,mask ,done = env.step(action.view(DE_batch_size,-1))      
            objective , fulfill_rate = reward 
            score = torch.exp(5*(1-fulfill_rate)) * (objective + (simulator.get_node_num()/(1*config['vehicle_num'])))
            value , index = torch.min(score ,dim=0) 
            Model_Predict = env.get_ModelRoute(index)  
            if not terminate: 
                
                RoadNetwork , vehicle_pos , vehicle_capacity , final_change , infopackage = simulator.step(Model_Predict)
                if infopackage["early_stop"] and return_looger :
                    return infopackage["obj"] , infopackage["fulfill"] , simulator.get_Logger() 
                elif infopackage["early_stop"] and not return_looger : 
                    return  infopackage["obj"] , infopackage["fulfill"]
                else : pass                
                Agent = Agent_random
            else : 
                objective , fulfill = simulator.step(Model_Predict) 
                #print(f"DE -- Objective : {objective} , fulfill : {fulfill}") 
    if return_looger : 
        return objective , fulfill , simulator.get_Logger()
    else :
        return objective , fulfill

def mix(Agent_depot,Agent_random,config , BaseGraph=None , node_coordinate_table=None ,  return_looger=False  ): 
    simulator = simulator_maker(config ,PMPO=True, BaseGraph=BaseGraph,node_coordinate_table=node_coordinate_table)
    RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.reset()
    DE_batch_size = 8 *  config["DE_transform_type"]
    Agent = Agent_depot
    terminate = False 
    while not terminate : 
        if final_change : terminate = True 
        with torch.no_grad() : 
            env= env_maker(
                batch_size=DE_batch_size , batch_data=RoadNetwork.to(config["device"]) , 
                node_num= simulator.get_node_num() , vehicle_num=config['vehicle_num'] , 
                vehicle_capacity=vehicle_capacity , vehicle_pos_set=vehicle_pos ,StateEQ = "mix" ,config=config
            )
            batch = env.get_initilize_Batch()
            batch.state ,fleet_state , vehicle_state , mask , done = env.reset() 
            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state, mask) 
                action = torch.argmax(action_dist , dim=1 , keepdim=False)      
                batch.state ,fleet_state , vehicle_state ,reward ,mask ,done = env.step(action.view(DE_batch_size,-1))      
            objective , fulfill_rate = reward 
            score = torch.exp(5*(1-fulfill_rate)) * (objective + (simulator.get_node_num()/(2*config['vehicle_num'])))
            value , index = torch.min(score ,dim=0) 
            Model_Predict = env.get_ModelRoute(index)  
            if not terminate: 
                RoadNetwork , vehicle_pos , vehicle_capacity , final_change ,infopackage= simulator.step(Model_Predict)
                if infopackage["early_stop"] and return_looger :
                    return infopackage["obj"] , infopackage["fulfill"] , simulator.get_Logger() 
                elif infopackage["early_stop"] and not return_looger : 
                    return  infopackage["obj"] , infopackage["fulfill"]
                else : pass   
                Agent = Agent_random
            else : 
                objective , fulfill = simulator.step(Model_Predict) 
                #print(f"mix -- Objective : {objective} , fulfill : {fulfill}") 
    if return_looger : 
        return objective , fulfill , simulator.get_Logger()
    else :
        return objective , fulfill

def ORTools(Agent_depot,Agent_random , config , BaseGraph=None,node_coordinate_table = None ,  return_looger=False  ) : 
    simulator = simulator_maker(config ,PMPO=False, BaseGraph=BaseGraph,node_coordinate_table=node_coordinate_table)
    terminate = False 
    RoadNetwork , vehicle_pos , vehicle_capacity , final_change = simulator.reset()
    Distance_matrix , demand_list , vehicle_pos, vehicle_capacity = ORTools_input_transform(RoadNetwork , vehicle_pos ,vehicle_capacity,simulator.done_vehicle)
    while not terminate : 
        if final_change: terminate = True 
        Route = ORtools_DSVRP(
            dist_matrix= Distance_matrix , 
            # 這邊傳進dist_matrix_std只是給不報錯 , 最終的Cost計算是交給RoutingSimulator , 故不影響
            dist_matrix_std= Distance_matrix ,  
            demand_vector= demand_list , 
            #vehicle_num = config["vehicle_num"] , 
            vehicle_num = len(vehicle_pos) , 
            start_pos = vehicle_pos ,  
            capacity_vector=vehicle_capacity , 
            algo=config["ORTools_algo"],
            time_limit=1 , 
        ).solve(route_only=True)    
        # OR-tools輸出的路線數量等同於活著的車 , 因此需要補正,把OR-tools的路徑放在正確的車上
        #print(f"Before compensation : {Route}")
        Model_Predict , i = [] ,0
        for done in simulator.done_vehicle: 
            if done : 
                Model_Predict.append([0]) 
            else : 
                Model_Predict.append(Route[i])
                i+=1 
        assert i == len(vehicle_pos) , "available vehicles number not consistent"
        #print(f"Compensation Route : {Model_Predict}")          
        
        
        if not terminate: 
            RoadNetwork , vehicle_pos , vehicle_capacity , final_change , infopackage  = simulator.step(Model_Predict)
            # 除了正常的final_change-> terminate , 如果車輛提前全部用完,則infopackage會給出early_stop和obj , fulfill     
            if infopackage["early_stop"] and return_looger :
                return infopackage["obj"] , infopackage["fulfill"] , simulator.get_Logger() 
            elif infopackage["early_stop"] and not return_looger : 
                return  infopackage["obj"] , infopackage["fulfill"]
            else : pass                 
            
            Distance_matrix , demand_list , vehicle_pos, vehicle_capacity  = ORTools_input_transform(RoadNetwork , vehicle_pos ,vehicle_capacity,simulator.done_vehicle)
 
        else : 
            objective , fulfill = simulator.step(Model_Predict) 
            #print(f"OR-tools -- Objective : {objective} , fulfill : {fulfill}") 
            
    if return_looger : 
        return objective , fulfill , simulator.get_Logger()
    else :
        return objective , fulfill