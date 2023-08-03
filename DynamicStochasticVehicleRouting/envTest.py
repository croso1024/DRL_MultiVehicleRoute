from utils.DSVRPGenerator import DSVRP_DataGenerator
# from utils.TSPGenerator import TSP_DataGenerator
from torch_geometric.data import Batch 
from Env.Environment_SC import DSVRP_Environment
import torch 
import sys , os 
from RealData.simulator_utils import PMPO_vehicle_pos

class hiddenprint : 
    def __enter__(self): 
        self._original_stdout = sys.stdout 
        sys.stdout = open(os.devnull , 'w')
    
    def __exit__(self,exc_type ,exc_val , exc_tb): 
        sys.stdout.close() 
        sys.stdout = self._original_stdout

# StateEQ  = "mix"
StateEQ = "PMPO"
DE_type = 2
batch_size=8
generate_batch_size = batch_size // DE_type if StateEQ in ["DE","mix"] else batch_size
node_nums = 5
vehicle_num = 3
ig = DSVRP_DataGenerator(workers=1,batch_size=generate_batch_size,node_num=node_nums) 
# batch = ig.getInstance_Batch()
batch = ig.getInstance_Batch() if StateEQ == "DE" or StateEQ == None else ig.getInstance_BatchPMPO_SingleProcess(dataset_size=1)[0]

coordinate_pos = [(0.12,0.32),(0.66,0.88),(0.99,0.25)]
if StateEQ == "DE" or StateEQ == None : pass 
else : 
    coordinate_pos = PMPO_vehicle_pos(coordinate_pos) 
    
print("----------------------")
env = DSVRP_Environment( 
    batch_size=batch_size,
    node_nums=node_nums,
    batch_data=batch , 
    vehicle_num=vehicle_num,
    # vehicle_capacity=[1,2,3], # only use in inference mode ! 
    # vehicle_pos_mode="Random" , 
    vehicle_pos_mode=coordinate_pos,
    StateEQ=StateEQ , 
    DE_transform_type=DE_type,
    graph_transform={"transform":"knn", "value":12},
    training=True,
    # training=False , 
    device='cpu' 
)
# with hiddenprint():
observation , fleet_state ,vehicle_state , mask , done = env.reset()
print(f"observation : \n{observation}")
print(f"Fleet state : \n{fleet_state}" )
print(f"vehicle_state : \n{vehicle_state}")
print(f"mask : \n{mask}")
#print(f"done : \n{done }")
print(f"path log : {env.path_log}")
print(f"step counter : {env.step_counter}")
print("\n\n ################## \n\n")

print(f"Init pos :\n{env.init_pos}")
print(f"permutation tensor : \n{env.premutation_tensor}")

while not done : 
    action_list = list() 
    for i in range(batch_size):
        action = int(input("input action :  ")) 
        action_list.append(action) 
    action = torch.tensor(action_list).view(batch_size ,-1).int()
    print(f" your action : {action} , {action.shape}")
    #with hiddenprint():
    observation , vehicle_state , vehicle_state ,reward , mask , done = env.step(action)    
    #print(f"observation : \n{observation}")
    print(f"Fleet state : \n{fleet_state}" )
    print(f"vehicle_state : \n{vehicle_state}")
    print(f"mask : \n{mask}")
    print(f"path log : \n{env.path_log}")
    print(f"reward : \n{reward}")
    
print("-------\n\n")
effective_route = env.get_effective_route()
print(f"Effective route : {effective_route}")
print(batch.node_route_cost_std)