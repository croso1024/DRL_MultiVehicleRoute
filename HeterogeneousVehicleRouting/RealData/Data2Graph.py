""" 
    DataParser that covert the  txt file to pytorch-geometric graph data 
    1. load the raw data from the txt file 
    2. Covert the node coordinate to normalize distance_matrix    
    3. Normalize demand , vehicle charteristic 
"""

from torch_geometric.data import Data 
import numpy as np 
from math import sqrt 
from sklearn.manifold import MDS , Isomap
from scipy.linalg import eigh
import torch 
from RealData.DataParser import cParse , BrandaoParse
# from DataParser import cParse , BrandaoParse
class Instance(Data): 
    def __cat_dim__(self, key: str, value, *args, **kwargs) :
        if key in ['y' , 'node_route_cost'] : return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 

class Demand_Node(object): 
    
    def __init__(self,x,y,demand): 
        self.x = x 
        self.y = y 
        self.demand = demand 
    def node_attribute(self)->tuple: 
        return (self.x , self.y , self.demand) 

def PMPO_node_argument(init_x , init_y , demand):  
    
    node_1 = Demand_Node(init_x , init_y , demand)
    node_2 = Demand_Node(init_y , init_x , demand) 

    node_3 = Demand_Node(1-init_x , init_y, demand)
    node_4 = Demand_Node(init_x , 1-init_y, demand) 

    node_5 = Demand_Node(init_y , 1-init_x, demand)
    node_6 = Demand_Node(1-init_y , init_x, demand)
    
    node_7 = Demand_Node(1-init_x , 1-init_y, demand)
    node_8 = Demand_Node(1-init_y , 1-init_x, demand)
    
    return [node_1 , node_2 , node_3 , node_4 ,node_5 , node_6 , node_7 , node_8 ]


def GraphDataParse(instance_type:str , instance_name:str): 
    """ 
        Transform the txt file to the "normalize" node coordinates list & demand
        產生 Graph-data , vehicle-data , vehicle-info , 後兩者要到實際送進inference才用得到
    """
    # Step1. Load data from the txt file 
    if instance_type == "brandao" : 
        parse_func = BrandaoParse 
    elif instance_type == "c": 
        parse_func = cParse 
    else: raise RuntimeError("Unknown data type")
    node_list , vehicle_list ,instance_info = parse_func(instance_name) 
    demand_list = [node[3] for node in node_list] 
    # Step2. Covert the node list to adajency matrix 
    Unnormalized_Distance_matrix = [] 
    for src in node_list : 
        temp = [] 
        for dst in node_list : 
            temp.append(  sqrt( 
                               pow( (src[1]-dst[1]) , 2)  + pow( (src[2]-dst[2]) , 2)
            ) ) 
        Unnormalized_Distance_matrix.append(temp) 
    Distance_matrix = np.array(Unnormalized_Distance_matrix , dtype=np.float64) 


    #normalized_coordinate = MultiDimensionScaling(Distance_matrix=Distance_matrix)
    normalized_coordinate = ProportionScaling(node_list=node_list)
    #normalized_coordinate = ISOMAP(Distance_matrix=Distance_matrix)
    
    assert len(normalized_coordinate) == len(demand_list) 
    # np.random.shuffle(vehicle_list)
    instance_info.update({"Distance_matrix":Unnormalized_Distance_matrix})
    
    return normalized_coordinate , demand_list , vehicle_list , instance_info 
    

def get_Instance_from_coordinate(coordinates, demand_list): 
    """ 
        使用TSPGenerator 的getInstance function  
    """
    node_num = len(coordinates) 
    nodes = [ Demand_Node( x=coordinates[i][0],y=coordinates[i][1],demand=demand_list[i] )  for i in range(node_num)]
    node_features = np.zeros( (node_num ,3 ) , dtype=np.float32) 
    for ith , node in enumerate(nodes):  node_features[ith] = node.node_attribute() 
    
    Distance_matrix = np.zeros( (node_num , node_num) , dtype=np.float32) 
    edge_index , edge_attr = list() , list() 
        
    for u ,src_node in enumerate(nodes): 
        for v , dst_node in enumerate(nodes): 
            distance = sqrt(pow( ( src_node.x - dst_node.x ),2) + pow( ( src_node.y - dst_node.y ),2) )
            Distance_matrix[u][v] = distance
            edge_index.append((u,v)) 
            edge_attr.append(distance) 
            
    return Instance(
        x = torch.tensor( node_features , dtype=torch.float32) , 
        node_route_cost = torch.tensor(Distance_matrix , dtype=torch.float32) ,
        edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
        edge_attr = torch.tensor(edge_attr , dtype=torch.float32)  ,
    )
    
def get_InstancePMPO_from_coordinate(coordinate , demand_list): 
    node_num = len(coordinate) 
    PMPO_node_list = [ PMPO_node_argument(
            init_x= coordinate[i][0] , init_y=coordinate[i][1] , demand=demand_list[i]
        ) for i in range(node_num)]
    assert len(PMPO_node_list[0]) == 8 , "PMPO dimension error"
    PMPO_data = list() 
    PMPO_node_list = list(zip(*PMPO_node_list)) 
    
    for argument_th , nodes in enumerate(PMPO_node_list): 
        node_features = np.zeros((node_num,3) ,dtype=np.float32)
        for ith , node in enumerate(nodes): 
            node_features[ith] = node.node_attribute() 
        Distance_matrix = np.zeros( (node_num , node_num) , dtype=np.float32) 
        edge_index , edge_attr = list() , list() 
        for u ,src_node in enumerate(nodes): 
            for v , dst_node in enumerate(nodes): 
                distance = sqrt(pow( ( src_node.x - dst_node.x ),2) + pow( ( src_node.y - dst_node.y ),2) )
                Distance_matrix[u][v] = distance
                edge_index.append((u,v)) 
                edge_attr.append(distance) 
        PMPO_data.append(
            Instance(
                x = torch.tensor( node_features , dtype=torch.float32) , 
                node_route_cost = torch.tensor(Distance_matrix , dtype=torch.float32) ,
                edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
                edge_attr = torch.tensor(edge_attr , dtype=torch.float32)  ,
            )
        )
    # 回傳的是一組包含8個等效data的list
    return PMPO_data 

def MultiDimensionScaling(Distance_matrix:np.array): 
    mds = MDS(n_components=2 , dissimilarity="precomputed")
    coordinate = mds.fit_transform(Distance_matrix) 
    min_val , max_val = np.min(coordinate) , np.max(coordinate)
    normalized_coordinate = (coordinate - min_val ) / (max_val - min_val) 
    return normalized_coordinate

def ISOMAP(Distance_matrix:np.array):  
    isomap = Isomap(n_neighbors=6, n_components= 2 , metric="precomputed" )
    coordinate = isomap.fit_transform(Distance_matrix) 
    min_val , max_val = np.min(coordinate) , np.max(coordinate)
    normalized_coordinate = (coordinate - min_val ) / (max_val - min_val) 
    return normalized_coordinate

def ProportionScaling(node_list): 
    coor = np.array([(x,y) for id,x,y,demand in node_list]) 
    normalized_coordinate = np.zeros( (len(node_list) , 2 ) , dtype=np.float32)
    x_max , y_max = np.max(coor[:,0]) , np.max(coor[:,1])
    maxima = x_max if x_max > y_max else y_max
    normalized_coordinate = coor / maxima
    return normalized_coordinate
    