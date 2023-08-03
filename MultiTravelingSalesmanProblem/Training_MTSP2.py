from torch.cuda.amp import autocast , GradScaler
import sys ,os,copy , torch, random , numpy as np 
from utils.MTSPGenerator import MTSP_DataGenerator 
from tqdm import tqdm 
from time import time 
from torch.optim import AdamW 
from torch.distributions import Categorical 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR 
from torch.nn.functional import normalize 
# from Env.Environment_HCVRP import HCVRP_Environment 
from Env.Environment import MTSP_Environment 
from math import exp
from random import shuffle 
from utils.ValidationDataset import LoadDataset
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
num_epoch , per_dataset_size , batch_size = 300, 400 , 32
validation_size = 6 # 1 validation_size x batch 
lr , decay_rate  , grad_cummulate = 1e-4, 1e-6   , 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
writter = SummaryWriter("./model/MultiTravelingSalesmanProblem/training_log")
############ Model ################## 
node_nums , vehicle_nums = 100,10
node_features_dim , fleet_features_dim , vehicle_features_dim = 4, 5 , 4
hidden_dim =192 
from net.ver20.PolicyNetwork import PolicyNetwork
Agent = PolicyNetwork(
    node_feature_dim=node_features_dim , 
    fleet_feature_dim= fleet_features_dim , 
    vehicle_feature_dim= vehicle_features_dim , 
    hidden_dim=hidden_dim , 
    edge_dim=1,
    heads = 6 , 
    num_layers = 3, 
    skip_connection=True , 
    clip_coe=10 ,
).to(device)
# Agent.load_state_dict(torch.load("./model/MultiTravelingSalesmanProblem/checkpoint/N50_v20k12.pth"))

optimizer = AdamW( Agent.parameters() , lr = lr , weight_decay=decay_rate ) 
scheduler = StepLR(optimizer=optimizer , step_size=per_dataset_size, gamma = 0.99)
scaler = GradScaler()
total = sum([param.nelement() for param in Agent.parameters()])
print("Number of model parameter: %.2fM" % (total/1e6))

############################################
Training_Generator = lambda node_num : MTSP_DataGenerator(
    workers = 4 , batch_size = batch_size , node_num=node_num 
)
# NV_table = [(40,4),(50,5),(60,6)]
NV_table = [(100,10)]
validation_setting = {
    "D":1280 , "B":64 , "N":100 , "V": 10
}


def env_maker(batch_size , batch_data , node_num , vehicle_num , StateEQ):

    return MTSP_Environment(
        batch_size = batch_size , 
        node_nums = node_num , 
        batch_data = batch_data , 
        vehicle_num = vehicle_num , 
        vehicle_pos_mode="Depot",
        StateEQ=StateEQ,
        graph_transform={"transform":"knn" , "value":12}, 
        device =device 
    )

def training_one_epoch(dataset): 
    
    Agent.train()
    optimizer.zero_grad() 
    avg_reward = 0  
    model_time , env_time = 0 , 0 
    
    for episode , (n , v , batch)  in tqdm(enumerate(dataset)) : 
        batch = batch.to(device) 
        env = env_maker(batch_size , batch , node_num=n , vehicle_num=v , StateEQ="PMPO") 
        batch = env.get_initilize_Batch()
        batch.state , fleet_state , vehicle_state , mask , done = env.reset() 
        log_probs = list() 
        
        with autocast(): 
            while not done : 
                ts = time() 
                action_dist = Categorical(probs= Agent(batch , fleet_state , vehicle_state ,mask) )
                model_time += time() - ts
                action = action_dist.sample() 
                log_probs.append( action_dist.log_prob(action) )
                ts = time()
                batch.state , fleet_state , vehicle_state , reward , mask , done = env.step(action.view(batch_size, -1 ))
                env_time += time() -ts 
            avg_reward += reward.mean()  
            PMPO_reward = env.StateEQ_reward_function(reward ,mode="mean")
            Adavanced = normalize( ( reward - PMPO_reward ) , dim=0 )
            log_probs = torch.stack(log_probs , dim = 1) 
            loss = (log_probs * Adavanced).mean() 
            
        scaler.scale(loss).backward() 
        if episode %  grad_cummulate == 0 : 
            scaler.step(optimizer=optimizer) 
            scaler.update()
            # optimizer.step() 
            scheduler.step() 
            optimizer.zero_grad()
    print(f"total time consumption in this epoch : model : {model_time:.3f} , env : {env_time:.3f}")
    return avg_reward/len(dataset) 

def validation(dataset): 
    
    Agent.eval() 
    average_tour_length = 0 
    with torch.no_grad() : 
        for batch in dataset: 
            batch = batch.to(device) 
            env = env_maker(batch_size = validation_setting["B"]  , 
                            batch_data = batch,
                            node_num=validation_setting["N"], 
                            vehicle_num=validation_setting["V"] , 
                            StateEQ=None) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask ,done = env.reset()

            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state ,mask) 
                action = torch.argmax(action_dist,dim=1 ,keepdim=False)
                batch.state , fleet_state ,vehicle_state ,reward , mask ,done = env.step(action.view(validation_setting["B"]  , -1))
            average_tour_length += reward.mean()
    average_validation_reward =   average_tour_length / len(dataset)           
    print(f"Validation reward average  : { average_validation_reward }")
    return average_validation_reward 


def training(): 
    
    vadlidation_set = LoadDataset(
        dataset_size=validation_setting["D"],
        batch_size=validation_setting["B"] , 
        node_num=validation_setting["N"] , 
        vehicle_num=validation_setting["V"] , 
    )
    best = float("inf")
    for epoch in tqdm(range(num_epoch)): 
        s = time() 
        dataset = MixBatchGenerator()
        print(f"In {epoch} generate {per_dataset_size}*{batch_size} of {node_nums} Graph data in {time()-s} seconds ")
        avg_reward = training_one_epoch(dataset=dataset)
        print(f"-- Complete {epoch}-th epoch , reward : {avg_reward:.3f}  ")
        
        validation_reward   = validation(dataset=vadlidation_set) 
        
        if validation_reward < best : 
            torch.save(Agent.state_dict() , "./model/MultiTravelingSalesmanProblem/checkpoint/N100V10_v20k12_0729.pth")
            print(f"\n\n -- Save the model parameters \n\n")
            best = validation_reward
        
        writter.add_scalars(
            "Episode loss" , 
            {"Training ":avg_reward , "Validation":validation_reward} , epoch
        )

    writter.close() 


def MixBatchGenerator(): 
    temp = list()
    for setting in NV_table: 
        Batches = Training_Generator(setting[0]).getInstance_BatchPMPO_SingleProcess(
            dataset_size=per_dataset_size//len(NV_table)) 
        Batches = [ (setting[0] , setting[1] , batch ) for batch in Batches ]
        temp.append(Batches)
    Dataset = [temp[i][j] for j in range(per_dataset_size//len(NV_table)) for i in range(len(temp))]
    return Dataset 

    
training()

