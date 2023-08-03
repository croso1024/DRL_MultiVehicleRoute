
import torch 
import torch.nn as nn 
from torch_geometric.nn import GraphConv , LayerNorm,GINConv , GATConv , MLP , GINEConv
from torch_sparse import SparseTensor



class GIN_Network(nn.Module): 
    
    def __init__(self , hidden_dim , skip_connection=False ,num_layers=3 ,sparse_conv=False): 
        super().__init__() 
        
        if sparse_conv: 
            self.Network = nn.ModuleList(
                [GIN_Layer_Sparse(hidden_dim=hidden_dim,skip_connection=skip_connection)
                for i in range(num_layers) ]
            )
        else: 
            self.Network = nn.ModuleList(
                [GIN_Layer(hidden_dim=hidden_dim , skip_connection=skip_connection) 
                for i in range(num_layers) ]
            )
        self.sparse_conv = sparse_conv

    def forward(self, node, edge_index , edge_attr , batch_ptr) :
        if self.sparse_conv : 
            adj = SparseTensor( 
                    row = edge_index[0] , col=edge_index[1] , value = edge_attr 
                )
            
            for layer in self.Network : 
                node = layer(node, adj , batch_ptr)
            
        else :
            
            for layer in self.Network : 
                node = layer(node, edge_index , edge_attr , batch_ptr) 
            
        return node 


class GIN_Layer(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.Conv = GINConv(
            nn = MLP(
                in_channels=hidden_dim , 
                hidden_channels= hidden_dim, 
                out_channels= hidden_dim , 
                norm = nn.LayerNorm(hidden_dim) , 
                num_layers=3 
            ) , train_eps=True 
        ) 
        self.Norm = LayerNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection =skip_connection  
    
    def forward(self , node , edge_index , edge_attr ,batch_ptr ): 
        if self.skip_connection : 
            return self.ReLU( self.Norm( node + self.Conv( x = node , edge_index=edge_index   ),batch_ptr))
        else : 
            return self.ReLU(self.Norm( self.Conv( x = node , edge_index=edge_index  ),batch_ptr ))



class GIN_Layer_Sparse(nn.Module): 
    
    def __init__(self,hidden_dim , skip_connection=False): 
        super().__init__() 
        self.Conv = GINConv(
            nn = MLP(
                in_channels=hidden_dim , 
                hidden_channels= hidden_dim, 
                out_channels= hidden_dim , 
                norm = nn.LayerNorm(hidden_dim) , 
                num_layers=3 
            ) , train_eps=True 
        ) 
        self.Norm = LayerNorm(hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.skip_connection =skip_connection  
    
    def forward(self , node , adj , batch_ptr): 
        if self.skip_connection : 
            return self.ReLU( self.Norm( node + self.Conv( node , adj.t() ) , batch_ptr))
        else : 
            return self.ReLU(self.Norm(self.Conv( node, adj.t() ) , batch_ptr ))