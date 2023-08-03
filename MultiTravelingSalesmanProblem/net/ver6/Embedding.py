"""
    2023-03-28
""" 
import torch 
import torch.nn as nn 
from torch_geometric.nn import Linear , LayerNorm , BatchNorm , GraphNorm
from net.ver6.Attention import Attention_Block
from net.ver6.PermuteLayer import Permute
class Node_Embedding(nn.Module): 
    """
        Linear embedding without batch norm
    """
    def __init__(self,node_feature_dim , hidden_dim): 
        super().__init__() 
        self.linear = Linear(node_feature_dim , hidden_dim , weight_initializer="kaiming_uniform") 
        # self.Norm= BatchNorm(hidden_dim) 
        self.Norm = LayerNorm(hidden_dim,mode="node") 
    
    def forward(self,node,batch_ptr): 
        return self.Norm(  self.linear(node) , batch_ptr )
        # return self.Norm(self.linear(node)) 
    
    
class Fleet_Embedding(nn.Module): 
    """ 
        Use follow process 
        1. linear layer 
        2. 2  x  self-attention block 
    """
    def __init__(self,fleet_feature_dim , hidden_dim  , heads = 4 ) :  
        super().__init__() 
        self.Embedding = nn.Sequential( 
            Linear(fleet_feature_dim , hidden_dim  , weight_initializer="kaiming_uniform") , 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU() ,
        )        
        
        self.Self_attention1 = Attention_Block(hidden_dim=hidden_dim , heads=heads)
        self.Self_attention2 = Attention_Block(hidden_dim=hidden_dim , heads=heads)
    
    def forward(self,fleet_vector): 
        # batch x vehicle_num x fleet_feature_dim  -> batch x vehicle_num x hidden_dim 
        fleet_embedding = self.Embedding(fleet_vector.clone()) 
        # batch x vehicle_num x hidden_dim
        out = self.Self_attention1(fleet_embedding , fleet_embedding,fleet_embedding)
        out = self.Self_attention2(fleet_embedding , fleet_embedding,fleet_embedding)
        return out     
    
class Vehicle_Embedding(nn.Module): 
    
    def __init__(self,vehicle_feature_dim , hidden_dim) : 
        super().__init__() 
        self.linear = Linear(vehicle_feature_dim , hidden_dim , weight_initializer="kaiming_uniform")
        self.Norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vehicle_vector): 
        # return self.Norm(self.linear(vehicle_vector.squeeze(1))).unsqueeze(1) 
        return self.Norm(self.linear(vehicle_vector)) 