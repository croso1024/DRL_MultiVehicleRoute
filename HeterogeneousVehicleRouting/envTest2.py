from utils.CVRPGenerator import CVRP_DataGenerator 
# from utils.TSPGenerator import TSP_DataGenerator
from torch_geometric.data import Batch 
from Env.Environment_2 import HCVRP_Environment
import torch , numpy as np  , random 
import sys , os 

class hiddenprint : 
    def __enter__(self): 
        self._original_stdout = sys.stdout 
        sys.stdout = open(os.devnull , 'w')
    
    def __exit__(self,exc_type ,exc_val , exc_tb): 
        sys.stdout.close() 
        sys.stdout = self._original_stdout

StateEQ  = "DE"
batch_size = 4
DE_type = 1
node_nums = 6

generator_batch_size = batch_size // DE_type if StateEQ in ["DE","mix"]  else batch_size

ig = CVRP_DataGenerator(workers=1,batch_size=generator_batch_size,node_num=node_nums) 
batch = ig.getInstance_Batch() if StateEQ=="DE" else ig.getInstance_BatchPMPO_SingleProcess(dataset_size=1)[0]
# batch =  ig.getInstance_BatchPMPO_SingleProcess(dataset_size=1)[0] if StateEQ in  else ig.getInstance_Batch()


env = HCVRP_Environment(
    batch_size=batch_size,
    node_nums=node_nums,
    batch_data=batch , 
    vehicle_num=8,
    vehicle_capacity=[0.2,1] , 
    vehicle_velocity=[0.4,1] ,
    StateEQ=StateEQ , 
    DE_transform_type=DE_type  ,
    vehicle_pos_mode="Depot",
    training=True , 
    device='cpu' 
)
# with hiddenprint():
observation , fleet_state ,vehicle_state , mask , done = env.reset()
print(f"observation : \n{observation}")
print(f"Fleet state : \n{fleet_state}" )
print(f"vehicle_state : \n{vehicle_state}")
print(f"mask : \n{mask}")
print(f"done : \n{done }")
print(f"path log : {env.path_log}")
print(f"step counter : {env.step_counter}")
print("\n\n ################## \n\n")

while not done : 
    action_list = list() 
    for i in range(batch_size):
        action = int(input("input action :  ")) 
        action_list.append(action) 
    action = torch.tensor(action_list).view(batch_size ,-1).int()
    print(f" your action : {action} , {action.shape}")
    #with hiddenprint():
    observation , vehicle_state , vehicle_state ,reward , mask , done = env.step(action)    
    # print(f"observation : \n{observation}")
    print(f"Fleet state : \n{fleet_state}" )
    print(f"vehicle_state : \n{vehicle_state}")
    print(f"mask : \n{mask}")
    print(f"path log : \n{env.path_log}")
    print(f"-------- terminal state : {env.get_terminal_state()}")
    
    print(f"reward : \n{reward}")
print(f"effective route : {env.get_effective_route()}")    
print(f"terminal log probs : {env.get_terminal_mask_log_probs()}")