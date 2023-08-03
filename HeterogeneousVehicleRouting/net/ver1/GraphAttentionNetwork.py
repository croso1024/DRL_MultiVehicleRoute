""" 
    2023-03-28 Graph Attention Network that consider edge infomation 
"""

import torch 
import torch.nn as nn 
from torch_geometric.nn import GraphNorm ,  LGConv, GATv2Conv

class GraphAttentionNetwork(nn.Module): 
    
    def __init__(self,hidden_dim , heads=4,skip_connection=False,num_layers=3): 
        super().__init__() 
        self.skip_connection = skip_connection 
        self.Network = nn.ModuleList(
            [GAT_layer(hidden_dim=hidden_dim,heads=heads,skip_connection=skip_connection) for i in range(num_layers)]
        )
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        
        for layer in self.Network: 
            node = layer(node,edge_index,edge_attr,batch_ptr)
        return node 

class GAT_layer(nn.Module):  

    def __init__(self,hidden_dim , heads=4,  skip_connection=False): 
        super().__init__() 
        self.Conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads = 4 , 
            edge_dim=1 ,
            concat=True , 
        )
        self.Norm = GraphNorm(hidden_dim*heads) 
        self.skip_connection = skip_connection
        self.Relu = nn.ReLU()  
        self.Linear = nn.Linear(heads*hidden_dim , hidden_dim)
        self.LinearNorm = GraphNorm(hidden_dim)
        
        
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        if self.skip_connection: 
            out = self.Relu( self.Norm( self.Conv(x=node , edge_index=edge_index, edge_attr=edge_attr , return_attention_weights=None ) , batch_ptr  )  )
            print(f"Debug out : {out.shape}")
        else: 
            out = self.Relu(self.Norm(self.Conv(x=node,edge_index=edge_index,edge_attr=edge_attr , return_attention_weights=None),batch_ptr))
            
        return self.Relu(self.LinearNorm( self.Linear(out) ,batch_ptr ))
        