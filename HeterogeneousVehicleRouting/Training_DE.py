""" 
    HCVRP的訓練框架 , 這個版本加入每一個batch重新sample一次車隊大小,車隊容量的設定
    -- 0407 
        加入Decision Equivalent ,先去除PMPO做測試
    -- 0420 
        Hybrid training architecture 
"""
from torch.cuda.amp import autocast , GradScaler
import sys ,os,copy , torch, random , numpy as np 
from utils.CVRPGenerator import CVRP_DataGenerator 
from utils.ValidationDataset import LoadDataset
from tqdm import tqdm 
from time import time 
from torch.optim import AdamW ,SGD , Adam
from torch.distributions import Categorical 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR 
from torch.nn.functional import normalize 
from Env.Environment import HCVRP_Environment 
from net.ver20.PolicyNetwork import PolicyNetwork
from utils.KNN_transform import KNNGraph_transformer
from random import uniform ,shuffle

from math import exp
#############  SEED  ###############
seed = 102
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
num_epoch , per_dataset_size , batch_size = 5, 3  , 16
lr , decay_rate  , grad_cummulate = 1e-4, 1e-6   , 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
writter = SummaryWriter("./model/HeterogeneousVehicleRouting/training_log")
############ Model ################## 
node_nums , vehicle_nums = 55 , 8 

node_features_dim , fleet_features_dim , vehicle_features_dim = 5, 7 , 6
hidden_dim = 192
Agent = PolicyNetwork(node_feature_dim=node_features_dim , fleet_feature_dim= fleet_features_dim , vehicle_feature_dim= vehicle_features_dim ,                                                  
    hidden_dim=hidden_dim , edge_dim=1,heads = 6 , num_layers = 3, skip_connection=True , clip_coe=10 ,
).to(device)
#Agent.load_state_dict(torch.load("./model/HeterogeneousVehicleRouting/checkpoint/N55H_v15.pth"))

optimizer = AdamW( Agent.parameters() , lr = lr , weight_decay=decay_rate ) 

scheduler = StepLR(optimizer=optimizer , step_size=per_dataset_size//grad_cummulate, gamma = 0.989)
scaler = GradScaler()
total = sum([param.nelement() for param in Agent.parameters()])
print("Number of model parameter: %.2fM" % (total/1e6))

############### Environemnt & DataGenerator ###############

DE_transform_type = 8
Training_Generator = lambda node_num : CVRP_DataGenerator(
    workers= 4 , batch_size= batch_size // DE_transform_type , 
    node_num= node_num 
)
#NV_table = [(50,8),(57,9),(62,10)]
NV_table = [(55,8)]
validation_setting = {
    "D":128 , "B":8 , "N":55 , "V" : 8 , "HE":True
}


def env_maker(batch_size , batch_data , node_num , vehicle_num ,vehicle_charateristic=None,StateEQ=None , training=False): 
    return HCVRP_Environment(
        batch_size = batch_size , 
        node_nums = node_num , 
        batch_data = batch_data , 
        vehicle_num = vehicle_num , 
        vehicle_capacity=True if training else vehicle_charateristic[0].tolist(), 
        vehicle_velocity=True if training else vehicle_charateristic[1].tolist(), 
        StateEQ=StateEQ,
        DE_transform_type = DE_transform_type , 
        training = training , 
        graph_transform={"transform":"knn" , "value":12},
        vehicle_pos_mode="Depot",
        device =device 
    )



def training_one_epoch(dataset): 
    
    Agent.train()
    optimizer.zero_grad() 
    avg_reward = 0  
    model_time , env_time = 0 , 0 
    
    for episode , (n,v , batch) in tqdm(enumerate(dataset)) : 
        batch = batch.to(device) 
        #print(f"N:{n} , V:{v}")
        env = env_maker(batch_size , batch ,node_num=n , vehicle_num=v   ,StateEQ="DE",training=True) 
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
            DE_reward = env.StateEQ_reward_function(reward, mode='mean')
            # Adavanced = (reward - DE_reward) / DE_reward
            Adavanced = normalize( ( reward - DE_reward ) , dim=0 )
            log_probs = torch.stack(log_probs , dim = 1) 
            log_probs_mask = env.get_terminal_mask_log_probs()
            log_probs = torch.mul(log_probs, log_probs_mask)
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
    average_fulfill_rate = 0 
    with torch.no_grad() : 

        for (batch,vehicle_charateristic) in dataset: 
            batch = batch.to(device) 
            env = env_maker(
                            batch_size=validation_setting["B"],
                            batch_data=batch , 
                            node_num=validation_setting["N"],
                            vehicle_num=validation_setting["V"],
                            vehicle_charateristic=vehicle_charateristic , 
                            StateEQ=None,
                            training=False
                            ) 
            batch = env.get_initilize_Batch()
            batch.state , fleet_state , vehicle_state , mask ,done = env.reset()

            while not done : 
                action_dist = Agent(batch , fleet_state , vehicle_state ,mask) 
                action = torch.argmax(action_dist,dim=1 ,keepdim=False)
                batch.state , fleet_state ,vehicle_state ,reward , mask ,done = env.step(action.view(validation_setting["B"]  , -1))
            minSum_tour , fulfill_rate = reward 
            average_tour_length += minSum_tour.mean()     
            average_fulfill_rate  += fulfill_rate.mean()
    average_validation_reward =   average_tour_length / len(dataset)           
    average_fulfill_rate = average_fulfill_rate / len(dataset) 
    print(f"Validation reward average  : { average_validation_reward } , fulfill : {average_fulfill_rate}")
    average_validation_score= exp( 5*(1-average_fulfill_rate ) ) * ( average_validation_reward + (node_nums/(2*vehicle_nums)))
    print(f"Validation score : {average_validation_score}")
    return average_validation_reward , average_validation_score


def training(): 

    score_logger = [] 
    
    validation_set = LoadDataset(
        dataset_size=validation_setting["D"], 
        batch_size=validation_setting["B"] ,
        node_num=validation_setting["N"] ,
        vehicle_num=validation_setting["V"] ,
        heterogeneous=validation_setting["HE"] 
    )
    best = float("inf")
    for epoch in tqdm(range(num_epoch)): 
        s = time() 
        dataset = MixBatchGenerator()
        print(f"In {epoch} generate {per_dataset_size}*{batch_size} of {node_nums} Graph data in {time()-s} seconds ")
        avg_reward = training_one_epoch(dataset=dataset)
        print(f"-- Complete {epoch}-th epoch , reward : {avg_reward:.3f}  ")
        
        validation_reward , score = validation(dataset=validation_set) 
        score_logger.append(score.cpu())
        if score< best : 
            #torch.save(Agent.state_dict() , "./model/HeterogeneousVehicleRouting/checkpoint/N55H_v20k12.pth")
            print(f"\n\n -- Save the model parameters \n\n")
            best = score
        
        writter.add_scalars(
            "Episode loss" , 
            {"Training ":avg_reward , "Validation":validation_reward} , epoch
        )
    score_logger_filename = f"HVRP_DE_n{node_nums}_v{vehicle_nums}.npy"
    np.save("./model/HeterogeneousVehicleRouting/"+score_logger_filename , np.array(score_logger))
    writter.close() 



def MixBatchGenerator():
    temp = list() 
    for setting in NV_table:  
        Batches = Training_Generator(setting[0]).getInstance_Batch_SingleProcess(dataset_size=per_dataset_size//len(NV_table))
        # Add a label to represent the vehicle num in the batch
        Batches = [ (setting[0] , setting[1] , batch)  for batch in Batches  ]
        temp.append(Batches) 
    Dataset = [temp[i][j] for j in range(per_dataset_size//len(NV_table)) for i in range(len(temp))]
    return Dataset 


training()