""" 
    2023-04-04 
    HCVRP version model : 
    try to use the edge-based GNN to extract the graph-feature associate edge-feature 
    
    The architecture is :  prelayer GAT + GIN , use hierarchical representation


"""
import torch 
import torch.nn as nn 
from torch_geometric.utils import to_dense_batch 
from torch_geometric.nn import  GCNConv ,GINConv , MLP , GraphConv , TAGConv,FAConv ,Sequential as GraphSequential
from torch_geometric.nn import GraphNorm , LayerNorm , BatchNorm , Linear
from torch_geometric.nn import GCN , JumpingKnowledge
from net.ver17.Attention import Attention_Block 


class GraphNeuralNetwork(nn.Module): 
    
    def __init__(self,node_hidden_dim , edge_dim , heads=4,skip_connection=False,num_layers=3 ): 
        super().__init__()
        self.node_hidden_dim  = node_hidden_dim 
        self.edge_dim = edge_dim 
        self.skip_connection = skip_connection 
        self.num_layers = num_layers
        self.GraphConvolution = nn.ModuleList(
            # Let skip_connection == False 
            # [FAConv_layer(hidden_dim=node_hidden_dim , skip_connection=skip_connection) for i in range(num_layers)]
            [TAGConv_layer(hidden_dim=node_hidden_dim , skip_connection=skip_connection) for i in range(num_layers)]
        )

        # self.MLP = MLP(in_channels=node_hidden_dim , hidden_channels=node_hidden_dim , 
        #                out_channels=node_hidden_dim , num_layers=num_layers , 
        #                norm=nn.LayerNorm(node_hidden_dim))
        
    
    def forward(self,node,edge_index , edge_attr , batch_ptr ): 
        # # initial-node-feature 
        # node_0 = node 
        for  i ,Convolution_layer in enumerate(self.GraphConvolution ): 
            # node = Convolution_layer(node ,node_0, edge_index , edge_attr , batch_ptr) 
            node = Convolution_layer(node , edge_index , edge_attr , batch_ptr) 
        return node
        # # return node

        #return self.MLP(node)
        return self.ReLU(self.Norm(self.network(x=node , edge_index=edge_index , edge_weight=edge_attr) , batch_ptr))
class FAConv_layer(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False) : 
        super().__init__() 
        self.network = FAConv(
            channels= hidden_dim , 
            eps = 0.1   ,
            normalize=False , 
        )
        self.Norm = LayerNorm(hidden_dim,mode="node") 
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection 
        
    def forward(self, node , node_0 , edge_index , edge_attr ,batch_ptr): 
        
        return self.ReLU(
                self.Norm( 
                    self.network(x=node , x_0 = node_0 , edge_index=edge_index, edge_weight=edge_attr) 
                    , batch_ptr )   
            )
        


class TAGConv_layer(nn.Module): 
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__()
        self.network = TAGConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim , 
            K = 3  , 
        )
        self.Norm = LayerNorm(hidden_dim,mode="node")
        self.ReLU = nn.ReLU()
    
    def forward( self, node , edge_index , edge_attr , batch_ptr): 
        return self.ReLU(
            self.Norm( node + self.network(x=node,edge_index=edge_index,edge_weight=edge_attr),batch_ptr    )
            )
        

