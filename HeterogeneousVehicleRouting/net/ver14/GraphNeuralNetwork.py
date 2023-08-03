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
from net.ver14.Attention import Attention_Block 


class GraphNeuralNetwork(nn.Module): 
    
    def __init__(self,node_hidden_dim , edge_dim , heads=4,skip_connection=False,num_layers=3 ): 
        super().__init__()
        self.node_hidden_dim  = node_hidden_dim 
        self.edge_dim = edge_dim 
        self.skip_connection = skip_connection 
        self.num_layers = num_layers
        self.GraphConvolution = nn.ModuleList(
            # Let skip_connection == False 
            [GCNConv_layer(hidden_dim=node_hidden_dim ) for i in range(num_layers)]
        )
        self.JumpingKnowledge = JumpingKnowledge(mode="cat")
        self.JumpingKnowledge_Linear = nn.Linear( num_layers*node_hidden_dim , node_hidden_dim) 
        #self.JumpingKnowledge_Norm = LayerNorm(node_hidden_dim , mode="node") 
        #self.ReLU = nn.ReLU()         
        
    def forward(self,node,edge_index , edge_attr , batch_ptr ): 
        out = [None] * self.num_layers 
        for  i ,Convolution_layer in enumerate(self.GraphConvolution ): 
            node = Convolution_layer(node , edge_index , edge_attr , batch_ptr) 
            out[i] = node 
            
        #return self.ReLU( self.JumpingKnowledge_Norm( self.JumpingKnowledge_Linear( self.JumpingKnowledge(out)) , batch_ptr ) ) 
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
        self.Norm = LayerNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection = skip_connection
    
    def forward(self,node, edge_index,edge_attr ,batch_ptr): 
        if self.skip_connection : 
            return self.ReLU(self.Norm(node+ self.network(x=node,edge_index=edge_index) ) )
        else: 
            return self.ReLU(self.Norm( self.network(x=node,edge_index=edge_index) ) )











