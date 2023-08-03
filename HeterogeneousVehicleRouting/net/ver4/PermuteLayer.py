import torch.nn as nn 
import torch 


class Permute(nn.Module): 
    """ 
        use to permutation the tensor then pass through BN , 
        ==> tensor.permute(0,2,1)
    """
    def __init__(self): 
        super().__init__() 
    def forward(self,data:torch.Tensor): 
        return data.permute(0,2,1)