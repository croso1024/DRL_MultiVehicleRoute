""" 
    03-19 PolicyNetwork 
    
        -- init -- 架構延伸自MVRP版本P7 , 
        測試一下Sparse tensor的作法對內存佔用有沒有比較友善 ,
        此版本直接支援不同車隊大小
        
"""

import torch.nn as nn 
import torch 
from torch_geometric.nn import Sequential as GraphSequential
from torch_geometric.utils import to_dense_batch 
from math import sqrt 
from net.ver2.Embedding import Node_Embedding , Fleet_Embedding , Vehicle_Embedding
from net.ver2.GraphNeuralNetwork import GraphNeuralNetwork
from net.ver2.Attention import Attention_Block

class PolicyNetwork(nn.Module): 
    
    def __init__(self , node_feature_dim , fleet_feature_dim , vehicle_feature_dim , hidden_dim ,GCN_prelayer=False ,edge_dim=1, 
                 heads=8 ,num_layers=3 , skip_connection=False  , temp=1,clip_coe=10): 
        super().__init__()
        
        self.node_feature_dim = node_feature_dim 
        self.fleet_feature_dim = fleet_feature_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.hidden_dim = hidden_dim 
        self.heads = heads 
        self.num_layers = num_layers 
        self.skip_connection = skip_connection
        self.temp = temp 
        self.clip_coe = clip_coe
        
        self.Node_embedding = Node_Embedding(
            node_feature_dim=node_feature_dim , hidden_dim=hidden_dim 
        )
        self.Fleet_embedding = Fleet_Embedding(
            fleet_feature_dim=fleet_feature_dim , hidden_dim=hidden_dim ,heads=heads 
        )
        self.Vehicle_embedding = Vehicle_Embedding(
            vehicle_feature_dim=vehicle_feature_dim , hidden_dim=hidden_dim 
        )

        self.GraphNeuralNetwork = GraphNeuralNetwork(
            node_hidden_dim=hidden_dim,edge_dim=edge_dim , skip_connection=skip_connection , num_layers=num_layers , GCN_prelayer=GCN_prelayer
        )
        self.Pointer = PointerNetwork(hidden_dim=hidden_dim,heads=heads , skip_connection=True , clip_coe=clip_coe)
    
    def forward(self,batch ,Fleet_state , Vehicle_state , mask):  
                
        Graph_state , edge_index , edge_attr , batch_ptr = batch.state , batch.edge_index , batch.edge_attr ,batch.batch
        # Graph feature extract 
        Graph_state = self.Node_embedding(Graph_state ,batch_ptr)
        # Graph_state_init = Graph_state.clone()
        Graph_state = self.GraphNeuralNetwork(Graph_state ,edge_index , edge_attr , batch_ptr )
        # Fleet embedding 
        Fleet_state = self.Fleet_embedding( Fleet_state  )
        # Vehicle embedding 
        Vehicle_state = self.Vehicle_embedding(Vehicle_state)
        
        return self.Pointer(Graph_state , Fleet_state, Vehicle_state , batch_ptr , mask)        
        
    

class PointerNetwork(nn.Module):
    
    def __init__(self,hidden_dim , heads , temp=1,skip_connection=False, clip_coe=5 ): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.heads = heads 
        self.clip_coe = clip_coe 
        self.temp = temp 
        
        # Fleet ( Query ) & Graph ( Key ) Attention 
        self.Fleet_Graph_Attention = Attention_Block(hidden_dim=hidden_dim,heads=heads)

        ## Vehicle ( Query ) & Fleet ( Key ) Attention
        self.Vehicle_Fleet_Attention1 = Attention_Block(hidden_dim=hidden_dim , heads=heads )
        self.Vehicle_Fleet_Attention2 = Attention_Block(hidden_dim=hidden_dim , heads=heads )
    
        
        # Vehicle ( Query ) & Graph ( Key ) Attention 
        self.Vehicle_Graph_Attention1= Attention_Block(hidden_dim=hidden_dim , heads=heads) 
        self.Vehicle_Graph_Attention2= Attention_Block(hidden_dim=hidden_dim , heads=heads)
        self.Vehicle_Graph_Attention3= Attention_Block(hidden_dim=hidden_dim , heads=heads)
        #self.Vehicle_Graph_Attention4= Attention_Block(hidden_dim=hidden_dim , heads=heads)
        # Graph ( Query ) & Fleet ( Key ) Attention 
        self.Graph_Fleet_Attention = Attention_Block(hidden_dim=hidden_dim , heads=heads)
        
        self.point = nn.Sequential(
            nn.Linear(3*hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU() , 
        )     
        
        
        self.tanh = nn.Tanh()
        self.flatten =nn.Flatten()
        self.softmax = nn.Softmax( dim= 1 )
        
        #self.log_softmax=nn.LogSoftmax(dim=1)
        

    
    def forward(self,Graph_state , Fleet_state , Vehicle_state , batch_ptr , mask): 
        """ 
            Graph-state : batch x node_num x hidden 
            Fleet-state : batch x vehicle-num x hidden 
            Vehicle-state : batch x 1 x hidden 
        """
        Graph_state , ret = to_dense_batch(Graph_state,batch_ptr)
        # Aggregate the vehicle & Fleet 
        Vehicle_state = self.Vehicle_Fleet_Attention1(Vehicle_state,Fleet_state,Fleet_state)
        Vehicle_state = self.Vehicle_Fleet_Attention2(Vehicle_state,Fleet_state,Fleet_state)
     
        # Fleet readout also need to pass through the graph-state 
        Fleet_readout = torch.mean(Fleet_state,dim=1,keepdim=True) 
        Fleet_readout = self.Fleet_Graph_Attention(Fleet_readout,Graph_state,Graph_state)
     
        # Use Fleet state to enhance the graph-readout , use old graph-state as readout
        Graph_readout = torch.mean(Graph_state,dim=1,keepdim=True)
        Graph_state = self.Graph_Fleet_Attention(Graph_state,Fleet_state,Fleet_state)
        
        # Query : batch x 3 x hidden_dim 
        Query = torch.concat(
            (Graph_readout,Fleet_readout,Vehicle_state) , dim=1 
        )
        # Query : batch x 3 x hidden_dim --> batch x 1 x hidden_dim 
        Query = self.point(self.flatten(Query)).unsqueeze(1)
        
        Query = self.Vehicle_Graph_Attention1(Query,Graph_state,Graph_state) 
        Query = self.Vehicle_Graph_Attention2(Query,Graph_state,Graph_state) 
        Query = self.Vehicle_Graph_Attention3(Query,Graph_state,Graph_state) 
        
        # probablity_dist = batch x node_num x 1
        probablity_dist = torch.bmm(Graph_state,Query.permute(0,2,1)) / sqrt(self.hidden_dim) 
        probablity_dist = torch.clip(self.clip_coe*self.tanh(probablity_dist) , min=-self.clip_coe , max = self.clip_coe)
        
        if mask is not None:
            probablity_dist = probablity_dist.float().masked_fill(mask , float('-inf')) 
        
        return self.softmax(probablity_dist).squeeze(2)
        