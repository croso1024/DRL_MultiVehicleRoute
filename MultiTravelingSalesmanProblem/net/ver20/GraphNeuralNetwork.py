""" 
    2023-04-04 
    HCVRP version model : 
    try to use the edge-based GNN to extract the graph-feature associate edge-feature 
    
    The architecture is :  prelayer GAT + GIN , use hierarchical representation


"""
import torch 
import torch.nn as nn 
from torch_geometric.utils import to_dense_batch 
from torch_geometric.nn import  GCNConv ,GINConv , MLP , GraphConv ,Sequential as GraphSequential
from torch_geometric.nn import GraphNorm , LayerNorm , BatchNorm , Linear
from torch_geometric.nn import GCN , JumpingKnowledge
from net.ver20.Attention import Attention_Block 


class GraphNeuralNetwork(nn.Module): 
    
    def __init__(self,node_hidden_dim , edge_dim , heads=4,skip_connection=False,num_layers=3 ): 
        super().__init__()
        self.node_hidden_dim  = node_hidden_dim 
        self.edge_dim = edge_dim 
        self.skip_connection = skip_connection 
        self.num_layers = num_layers
        self.GraphConvolution = nn.ModuleList(
            # Let skip_connection == False 
            # [GCNConv_layer(hidden_dim=node_hidden_dim ) for i in range(num_layers)]
            [GraphConv_layer(hidden_dim=node_hidden_dim , skip_connection=skip_connection ) for i in range(num_layers)]
        )

        self.JumpingKnowledge = JumpingKnowledge(mode="cat")
        #self.JumpingKnowledge = JumpingKnowledge(mode="cat",channels=node_hidden_dim,num_layers=2)
        self.JumpingKnowledge_Linear = nn.Linear( num_layers*node_hidden_dim , node_hidden_dim) 
              
        
    def forward(self,node,edge_index , edge_attr , batch_ptr ): 
        out = [None] * self.num_layers 
        for  i ,Convolution_layer in enumerate(self.GraphConvolution ): 
            node = Convolution_layer(node , edge_index , edge_attr , batch_ptr) 
            out[i] = node 
            
        
        return self.JumpingKnowledge_Linear(self.JumpingKnowledge(out))
        # return node

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
                self.Norm( node+ self.network(x = node , edge_index =edge_index,edge_weight=edge_attr ) ,
                batch_ptr  )) 


# class GraphConv_layer(nn.Module): 
    
#     def __init__(self,hidden_dim ,skip_connection=False): 
#         super().__init__() 
#         self.network = GraphConv(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim, 
#             aggr="mean"
#         )
#         # self.Norm = GraphNorm(hidden_dim) 
#         self.Norm = LayerNorm(hidden_dim, mode="node")
#         self.ReLU = nn.ReLU() 
#         self.skip_connection = skip_connection 
    
#     def forward(self,node, edge_index , edge_attr , batch_ptr): 
#         return  self.ReLU( 
#                 self.Norm(node+ self.network(x = node , edge_index =edge_index,edge_weight=edge_attr ) ,
#                 batch_ptr  )) 


class GraphConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.GraphConvolution =GraphConv(
            in_channels=hidden_dim , 
            out_channels= hidden_dim , 
            aggr = "mean" ,
        )
        self.Norm1 = LayerNorm(hidden_dim,mode="node")
        self.Norm2 = LayerNorm(hidden_dim,mode="node")
        #self.Norm3 = LayerNorm(hidden_dim,mode="node")
        self.Linear1 = Linear(hidden_dim,hidden_dim ,weight_initializer="kaiming_uniform")
        self.Linear2 = Linear(hidden_dim,hidden_dim, weight_initializer="kaiming_uniform")
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection 
    def forward(self, node , edge_index,edge_attr , batch_ptr): 
        if self.skip_connection: 
            node = node + self.GraphConvolution(x=node, edge_index =edge_index,edge_weight=edge_attr)
        else : 
            node =  self.GraphConvolution(x=node, edge_index =edge_index,edge_weight=edge_attr)
        node = self.ReLU(self.Norm1( self.Linear1(node ) , batch_ptr)) 
        node = self.ReLU(self.Norm2( self.Linear2(node ) , batch_ptr))
        #node = self.ReLU(self.Norm3( self.Linear2(node ) , batch_ptr  ))
        return node 



        


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
        self.Norm = LayerNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection
    
    def forward(self,node, edge_index,edge_attr ,batch_ptr): 
        if self.skip_connection : 
            return self.ReLU(self.Norm(node+ self.network(x=node,edge_index=edge_index) ) )
        else: 
            return self.ReLU(self.Norm( self.network(x=node,edge_index=edge_index) ) )











