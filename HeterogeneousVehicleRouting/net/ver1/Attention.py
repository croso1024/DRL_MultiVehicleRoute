""" 
 0328 -version3 

"""


import torch 
import torch.nn as nn 

class Attention_Block(nn.Module): 
    
    def __init__(self, hidden_dim ,heads=4 ): 
        super().__init__()
        self.hidden_dim = hidden_dim 

        self.Attention_layer = nn.MultiheadAttention(hidden_dim , num_heads=heads , batch_first=True)
        self.AttnNorm = nn.LayerNorm(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim,hidden_dim) 
        self.ReLU = nn.ReLU() 
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.LinearNorm = nn.LayerNorm(hidden_dim) 
        
    def forward(self,Q,K,V): 
        
        out =  self.AttnNorm(Q + self.Attention_layer(Q,K,V)[0])
        out =  self.LinearNorm( out + self.linear2( self.ReLU( self.linear1( out ) )   )  )
        return out 


