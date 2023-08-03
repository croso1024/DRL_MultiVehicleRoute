""" 
    2023-04-04 
    HCVRP version model : 
    try to use the edge-based GNN to extract the graph-feature associate edge-feature 
    
    The architecture is :  prelayer GAT + GIN , use hierarchical representation


"""
import torch 
import torch.nn as nn 
from torch_geometric.utils import to_dense_batch 
from torch_geometric.nn import GATv2Conv as GATConv , GCNConv ,GINConv , EGConv , MLP 
from torch_geometric.nn import GraphNorm 
from net.ver1.Attention import Attention_Block 


class GraphNeuralNetwork(nn.Module): 
    
    def __init__(self,node_hidden_dim , edge_dim , heads=4,skip_connection=False,num_layers=4,GCN_prelayer=False): 
        super().__init__()
        self.node_hidden_dim  = node_hidden_dim 
        self.edge_dim = edge_dim 
        self.skip_connection = skip_connection 
        self.GCN_prelayer = GCN_prelayer
        
        self.GCN_prelayer = GCNConv_layer(hidden_dim=node_hidden_dim, skip_connection=skip_connection)
        
        self.GraphConvolution = nn.ModuleList(
            [GCNConv_layer(hidden_dim=node_hidden_dim,skip_connection=skip_connection) for i in range(num_layers)]
        )
    
    def forward(self,node,edge_index , edge_attr , batch_ptr ): 
        
        # Pass the prelayer to extract the Edge infomation 
        if self.GCN_prelayer: 
            node  = self.GCN_prelayer(node , edge_index , edge_attr , batch_ptr) 
        #print(f"Debug attention score : {attention_score} , edge attr : {edge_attr}")
        for Convolution_layer in self.GraphConvolution:
            node = Convolution_layer(node , edge_index , edge_attr , batch_ptr)
        return node


class GCNConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim ,skip_connection=False): 
        super().__init__() 
        self.network = GCNConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            add_self_loops=True , 
            normalize=True , 
        )
        self.Norm = GraphNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection 
    
    def forward(self,node, edge_index , edge_attr , batch_ptr): 
        if self.skip_connection : 
            
            return  self.ReLU( 
                    self.Norm(node+ self.network(x = node , edge_index =edge_index,edge_weight=edge_attr ) ,
                    batch_ptr  )) 
        else: 
            return  self.ReLU( 
                    self.Norm( self.network(x = node , edge_index =edge_index,edge_weight=edge_attr  ) ,
                    batch_ptr  )) 


class EGConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__()
        self.network = EGConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            # aggregators=["sum","symnorm","std","max"], 
            aggregators=["sum","max"], 
            num_bases=4,
            num_heads=8,
        ) 
        self.Norm = GraphNorm(hidden_dim) 
        self.ReLU = nn.ReLU()
        self.skip_connection=skip_connection
    
    def forward(self,node,edge_index,edge_attr,batch_ptr): 
        if self.skip_connection: 
            return self.ReLU(self.Norm(node+ self.network(x=node,edge_index=edge_index),batch_ptr))
        else:
            return self.ReLU(self.Norm( self.network(x=node,edge_index=edge_index),batch_ptr))