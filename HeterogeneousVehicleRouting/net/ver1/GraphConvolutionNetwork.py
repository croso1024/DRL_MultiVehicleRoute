""" 
    2023-03-28 Graph Convolution Network that consider edge infomation 
"""

import torch 
import torch.nn as nn 
from torch_geometric.nn import GraphConv,GraphNorm , InstanceNorm,GCNConv , LGConv , EGConv,LayerNorm

class GraphConvolutionNetwork(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False,num_layers=3): 
        super().__init__() 
        self.skip_connection = skip_connection 
        self.Network = nn.ModuleList(
            [GCN_layer(hidden_dim=hidden_dim,skip_connection=skip_connection) for i in range(num_layers)]
        )
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        
        for layer in self.Network: 
            node = layer(node,edge_index,edge_attr,batch_ptr)
        return node 

class LGConv_layer(nn.Module):  

    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.Conv = LGConv(normalize=True) 
        self.Norm = GraphNorm(hidden_dim) 
        self.skip_connection = skip_connection
        self.Relu = nn.ReLU() 
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        if self.skip_connection: 
            return self.Relu( self.Norm( node + self.Conv(x=node , edge_index=edge_index, edge_weight=edge_attr ) , batch_ptr  )  )
        else: 
            return self.Relu(self.Norm(self.Conv(x=node,edge_index=edge_index,edge_weight=edge_attr),batch_ptr))

class EGConv_layer(nn.Module):  

    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.Conv = EGConv(
            in_channels=hidden_dim,out_channels=hidden_dim , aggregators=["symnorm" , "sum" , "max"] , 
        )
        self.Norm = GraphNorm(hidden_dim) 
        self.skip_connection = skip_connection
        self.Relu = nn.ReLU() 
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        if self.skip_connection: 
            return self.Relu( self.Norm( node + self.Conv(x=node , edge_index=edge_index) , batch_ptr  )  )
        else: 
            return self.Relu(self.Norm(self.Conv(x=node,edge_index=edge_index),batch_ptr))

class GCN_layer(nn.Module): 
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.Conv = GCNConv(
            in_channels=hidden_dim,out_channels=hidden_dim, normalize=True 
        )
        self.Norm = GraphNorm(hidden_dim) 
        self.skip_connection = skip_connection
        self.Relu = nn.ReLU() 
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        if self.skip_connection: 
            return self.Relu( self.Norm( node + self.Conv(x=node , edge_index=edge_index , edge_weight=edge_attr) , batch_ptr  )  )
        else: 
            return self.Relu(self.Norm(self.Conv(x=node,edge_index=edge_index, edge_weight=edge_attr),batch_ptr))