""" 
    2023-03-28 Add Knn - transformation 
"""

from torch_geometric.transforms import KNNGraph, Distance , Compose, RadiusGraph
from torch_geometric.data import Batch 
import torch_geometric.utils as utils 
import torch_geometric.nn as nn 
import torch 
def KNN_Transform(batch_data:Batch): 
    # add the "pos" attribute to use
    batch_data.pos = batch_data.x[:,:2]
    
"""    

    在DynamicStochastic Routing裡面 , 使用randomness factor代替原始的edge_attr 
"""    
class KNNGraph_transformer(object):
    
    def __init__(self, k=4 , self_loop=False,num_workers=1) : 
        self.transform = Compose(
            [
                KNNGraph(k=k , loop=self_loop  ,  num_workers=num_workers),
                Distance(norm=False , cat = False )
            ]
            )
        
    def batch_transform(self,batch): 
        # add the "pos" attribute to the batch 
        batch.pos = batch.x[:,:2]
        out = self.transform(batch) 
        del out.pos 
        out.edge_attr = out.edge_attr.squeeze()
        return out


class KNNGraph_transformer2(object):
    
    def __init__(self, k=4 , self_loop=False,num_workers=1) : 
        self.transform = Compose(
            [
                KNNGraph(k=k , loop=self_loop ,num_workers=num_workers),
                Distance(norm=False , cat = False )
            ]
            )
        
    def batch_transform(self,batch): 
        # add the "pos" attribute to the batch 
        origin_edge_attr = batch.edge_attr
        origin_edge_index = batch.edge_index  
        batch.pos = batch.x[:,:2]
        print(f"Debug before transform --- \n")
        print(f"batch - {batch}\n")
        print(f"edge index - {batch.edge_index}\n")
        print(f"edge attr -{batch.edge_attr}\n")
        """
        Method.1  
        for i , (u,v) in enumerate(origin_edge_index.t()) : 
            #print(f"{i}-th edge: {u}->{v}")
            mask = ((out.edge_index[0] == u) & (out.edge_index[1] == v )) 
            #print(mask)
            out.edge_attr[mask] = origin_edge_attr[i]
        """
        
        """
        Method.2
        """
        out = self.transform(batch)
        index = torch.zeros(out.edge_index.shape[1],dtype=torch.long) 
        for i , e  in enumerate( origin_edge_index.t() ): 
            index[(out.edge_index.t() == e).all(dim=1)] = i 
        out.edge_attr = origin_edge_attr[index]
        
        
        del out.pos 
        out.edge_attr = out.edge_attr.squeeze()
        print(f"Debug after transform --- \n")
        print(f"batch - {out}\n")
        print(f"edge index - {out.edge_index}\n")
        print(f"edge attr -{out.edge_attr}\n")

        
        return out

class  RadiusGraph_transformer(object): 
    
    def __init__(self, r =0.2 , self_loop = False, num_workers=1): 
        self.transform = Compose(
            [
                RadiusGraph(r = r , loop=self_loop , num_workers=num_workers) ,
                Distance(norm=False , cat=False) 
            ]
        )
    
    def batch_transform(self,batch): 
        batch.pos = batch.x[:,:2]
        out = self.transform(batch)
        del out.pos 
        out.edge_attr = out.edge_attr.squeeze()
        return out 
    