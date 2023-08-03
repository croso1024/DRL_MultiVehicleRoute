from torch_geometric.data import Data 
import numpy as np 
from math import sqrt 
from sklearn.manifold import MDS , Isomap
from scipy.linalg import eigh
import torch 
from RealData.DataParser import tspParse
# from DataParser import tspParse

class Instance(Data): 
    def __cat_dim__(self, key: str, value, *args, **kwargs) :
        if key in ['y' , 'node_route_cost'] : return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 
    
class Node(object): 
    
    def __init__(self,x,y): 
        self.x = x 
        self.y = y 
    def node_attribute(self)->tuple: 
        return (self.x , self.y) 
    
def PMPO_node_argument(init_x , init_y ):  
    
    node_1 = Node(init_x , init_y )
    node_2 = Node(init_y , init_x ) 

    node_3 = Node(1-init_x , init_y)
    node_4 = Node(init_x , 1-init_y) 

    node_5 = Node(init_y , 1-init_x)
    node_6 = Node(1-init_y , init_x)
    
    node_7 = Node(1-init_x , 1-init_y)
    node_8 = Node(1-init_y , 1-init_x)
    
    return [node_1 , node_2 , node_3 , node_4 ,node_5 , node_6 , node_7 , node_8 ]


def GraphDataParse(instance_name:str): 
    
    node_list , instance_info = tspParse(instance_name) 
    Unnormalized_Distance_matrix = [] 
    for src in node_list : # node_list : ( node_id , x , y )
        temp = [] 
        for dst in node_list : 
            temp.append(
                sqrt(
                    pow( (src[1]-dst[1]) ,2) + pow((src[2]-dst[2]) , 2 ) 
                )
            )
        Unnormalized_Distance_matrix.append(temp) 
    Distance_matrix = np.array(Unnormalized_Distance_matrix, dtype=np.float64) 
    #print(node_list)
    #normalized_coordinate = ProportionScaling(node_list = node_list)
    #normalized_coordinate =  ISOMAP(Distance_matrix=Distance_matrix) 
    normalized_coordinate =  MultiDimensionScaling(Distance_matrix=Distance_matrix) 
    
    instance_info.update(
        {"Distance_matrix":Unnormalized_Distance_matrix,
                          "node_num":len(node_list)}
                         )
    
    return normalized_coordinate , instance_info



def get_Instance_from_coordinate(coordinates): 
    node_num = len(coordinates) 
    nodes = [Node(x=coordinates[i][0] ,y=coordinates[i][1] ) for i in range(node_num)]
    node_features = np.zeros( (node_num , 2 ) , dtype=np.float32) 
    for ith , node in enumerate(nodes) : node_features[ith] = node.node_attribute() 
    Distance_matrix = np.zeros( ( node_num , node_num ) , dtype= np.float32) 
    edge_index ,edge_attr = list() , list() 
    
    for u , src_node in enumerate(nodes): 
        for v , dst_node in enumerate(nodes): 
            distance = sqrt(pow( ( src_node.x - dst_node.x)  ,2 ) +   pow( (src_node.y - dst_node.y) , 2 )) 
            Distance_matrix[u][v] = distance                
            edge_index.append((u,v)) 
            edge_attr.append(distance) 
            
    return Instance(
        x = torch.tensor(node_features , dtype=torch.float32) , 
        node_route_cost = torch.tensor(Distance_matrix , dtype=torch.float32) , 
        edge_index = torch.tensor( edge_index ,dtype=torch.long).t().contiguous() , 
        edge_attr = torch.tensor( edge_attr , dtype=torch.float32) , 
    )
    

def get_InstancePMPO_from_coordinate(coordinates) : 
    node_num = len(coordinates) 
    PMPO_node_list = [PMPO_node_argument(init_x=coordinates[i][0] , init_y=coordinates[i][1]) for i in range(node_num)]
    
    assert len(PMPO_node_list[0]) == 8 ,"PMPO dimension error !"
    PMPO_data = list() 
    PMPO_node_list = list(zip(*PMPO_node_list)) 
    
    for argumenth_th , nodes in enumerate(PMPO_node_list): 
        node_features = np.zeros((node_num,2) ,dtype=np.float32)
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
    
def ISOMAP(Distance_matrix:np.array):  
    isomap = Isomap(n_neighbors=6, n_components= 2 , metric="precomputed" )
    coordinate = isomap.fit_transform(Distance_matrix) 
    min_val , max_val = np.min(coordinate) , np.max(coordinate) 
    normalized_coordinate = (coordinate-min_val) / (max_val-min_val)
    return normalized_coordinate


def MultiDimensionScaling(Distance_matrix:np.array): 
    mds = MDS(n_components=2 , dissimilarity="precomputed")
    coordinate = mds.fit_transform(Distance_matrix) 
    min_val , max_val = np.min(coordinate) , np.max(coordinate) 
    normalized_coordinate = (coordinate-min_val) / (max_val-min_val)
    return normalized_coordinate

def PrincipleCoordinateAnalysis(Distance_matrix:np.array): 
    B = -0.5 * (Distance_matrix**2 )
    N = B.shape[0]
    I = np.eye(N) 
    J = np.ones((N,N)) 
    B = I @ B @ I 
    B -= (1/N) * (B@J + J@B - 2 * B @ J @ B)
    eigen_value , eigen_vector = eigh(B)
    coordinate = eigen_vector[:,:2]
    min_val , max_val = np.min(coordinate) , np.max(coordinate) 
    normalized_coordinate = (coordinate-min_val) / (max_val-min_val)
    return normalized_coordinate


def ProportionScaling(node_list): 
    # coor = np.array(node_list)  
    coor =  np.array( [(x,y) for id, x, y in node_list]  )
    normalized_coordinate = np.zeros( (len(node_list) , 2 ) , dtype=np.float32)
    x_max , y_max = np.max(coor[:,0]) , np.max(coor[:,1])
    maxima = x_max if x_max > y_max else y_max
    normalized_coordinate = coor / maxima
    return normalized_coordinate
    