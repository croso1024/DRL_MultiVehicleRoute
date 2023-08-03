""" 
    2023-03-28 Add Knn - transformation 
"""

from torch_geometric.transforms import KNNGraph, Distance , Compose , RadiusGraph
from torch_geometric.data import Batch 

def KNN_Transform(batch_data:Batch): 
    # add the "pos" attribute to use
    #batch_data.pos = batch_data.x[:,:2]
    batch_data.pos = batch_data.x # In MTSP version
    
    
class KNNGraph_transformer(object):
    
    def __init__(self, k=4 , self_loop=False,num_workers=1) : 
        self.transform = Compose(
            [
                KNNGraph(k=k , loop=self_loop , num_workers=num_workers),
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
    