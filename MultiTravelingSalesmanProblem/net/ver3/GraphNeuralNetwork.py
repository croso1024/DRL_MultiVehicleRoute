""" 
    2023-04-04 
    HCVRP version model : 
    try to use the edge-based GNN to extract the graph-feature associate edge-feature 
    
    The architecture is :  prelayer GAT + GIN , use hierarchical representation


"""
import torch 
import torch.nn as nn 
from torch_geometric.utils import to_dense_batch 
from torch_geometric.nn import GATv2Conv as  GCNConv ,GINConv , MLP , GraphConv 
from torch_geometric.nn import GraphNorm , LayerNorm , BatchNorm
from net.ver3.Attention import Attention_Block 


class GraphNeuralNetwork(nn.Module): 
    
    def __init__(self,node_hidden_dim , edge_dim , heads=4,skip_connection=False,num_layers=4 ): 
        super().__init__()
        self.node_hidden_dim  = node_hidden_dim 
        self.edge_dim = edge_dim 
        self.skip_connection = skip_connection 
        self.GraphConvolution = nn.ModuleList(
            [GINConv_layer(hidden_dim=node_hidden_dim,skip_connection=skip_connection) for i in range(num_layers)]
        )
    
    def forward(self,node,edge_index , edge_attr , batch_ptr ): 

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
        # self.Norm = GraphNorm(hidden_dim) 
        self.Norm = LayerNorm(hidden_dim, mode="node")
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection 
    
    def forward(self,node, edge_index , edge_attr , batch_ptr): 
        return  self.ReLU( 
                self.Norm(node+ self.network(x = node , edge_index =edge_index,edge_weight=edge_attr ) ,
                batch_ptr  )) 


class GraphConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim ,skip_connection=False): 
        super().__init__() 
        self.network = GraphConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim, 
            aggr="add"
        )
        # self.Norm = GraphNorm(hidden_dim) 
        self.Norm = LayerNorm(hidden_dim, mode="node")
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection 
    
    def forward(self,node, edge_index , edge_attr , batch_ptr): 
        return  self.ReLU( 
                self.Norm(node+ self.network(x = node , edge_index =edge_index,edge_weight=edge_attr ) ,
                batch_ptr  )) 


class GINConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim ,skip_connection=False): 
        super().__init__()
        self.network = GINConv(
            nn = MLP(
                in_channels=hidden_dim , 
                hidden_channels = hidden_dim , 
                out_channels= hidden_dim , 
                norm = LayerNorm(hidden_dim), 
                num_layers = 3
            ) , train_eps=True 
        )
        self.Norm = BatchNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection
    
    def forward(self,node, edge_index,edge_attr ,batch_ptr): 
        if self.skip_connection : 
            return self.ReLU(self.Norm(node+ self.network(x=node,edge_index=edge_index) ) )
        else: 
            return self.ReLU(self.Norm( self.network(x=node,edge_index=edge_index) ) )











