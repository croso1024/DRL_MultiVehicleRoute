from torch_geometric.data import Data 
import numpy as np ,os 
from sklearn.manifold import MDS ,Isomap
import osmnx as ox , torch 
from math import sqrt

class Instance(Data): 
    def __cat_dim__(self, key: str, value, *args, **kwargs) :
        if key in ['y' , 'node_route_cost',"node_route_cost_std"] : return None 
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

def Distance_matrix_to_Coordinate(distance_matrix , method="mds"): 
    """ 
        Only need for model ,Transform the Distance matrix , and divided into two part
        represent the request and vehicle pos respectively
    """
    if method == "mds": 
        mds = MDS(n_components=2 , dissimilarity="precomputed")
        coordinate = mds.fit_transform(distance_matrix) 
    elif method == "isomap" : 
        isomap = Isomap(n_neighbors=6, n_components= 2 , metric="precomputed" )
        coordinate = isomap.fit_transform(distance_matrix) 
    else : raise RuntimeError("Unknown transform method")
    
    min_val , max_val = np.min(coordinate) , np.max(coordinate)
    normalized_coordinate = (coordinate - min_val ) / (max_val - min_val) 
    normalized_coordinate = [(normalized_coordinate[i][0],normalized_coordinate[i][1]) for i in range(normalized_coordinate.shape[0])]
    return normalized_coordinate





def node_distance(x1,y1,x2,y2): 
    return sqrt(pow((x1-x2),2) + pow((y1-y2),2) )

def interpolation( x1,y1 , x2,y2 ,duration): 
    """ 
        專用在duration已經無法滿足車輛從當前位置移動到下一個節點上
    """
    distance = sqrt(pow((x1-x2),2) + pow((y1-y2),2) )
    ratio = duration / distance 
    new_x = x1 + (x2-x1) * ratio
    new_y = y1 + (y2-y1) * ratio
    return (new_x , new_y) 


def PMPO_vehicle_pos(vehicle_pos) : 
    """ 
        將vehicle-pos做PMPO轉換 
        由 :[(v1x,v1y),(v2x,v2y),(v3x,v3y)] --> [ [(v1x1,v1y1) ,(v1x2,v1y2)...]  , [(v2x1,v2y1) ,(v2x2,v2y2)...] ...]
        到 :[ [(v1x1,v1y1),(v2x1,v2y1),(v3x1,v3y1)] , [(v1x2,v1y2),(v2x2,v2y2),(v3x2,v3y2)] , ....]  
        
    """
    PMPO_vehicle_data = list() 
    for ith_vehicle, (x,y) in enumerate(vehicle_pos) : 
        PMPO_vehicle_data.append( [(x,y),(y,x),(1-x,y),(x,1-y),(y,1-x),(1-y,x),(1-x,1-y),(1-y,1-x)] ) 
    PMPO_vehicle_data = list(zip(*PMPO_vehicle_data)) 
    
    return PMPO_vehicle_data


def ORTools_input_transform(RoadNetworkData , vehicle_pos , vehicle_capacity , done_vehicle): 
    """ 
        要把將RouteSimulator的資料與OR-tools可用的格式串起來 , 即將RoadNetwork與vehicle_pos結合 , 產生新的節點代表車輛,同時更新deamnd-list
        另一部份要幫助OR-tools ,把死掉的車去除 , 同時要在OR-tools輸出路徑時 , 用[0]把死掉的車位置補起來
    """
    nodes = RoadNetworkData.x[:,:2]
    # print(f"Debug origin  nodes : {nodes}")
    # print(f"Debug origin  vehicle_pos : {vehicle_pos}") 
    available_vehicle_pos , available_vehicle_capacity = [] , [] 
    # 只保留還能用的車子
    for ith_vehicle ,done in enumerate(done_vehicle): 
        if done : pass 
        else : 
            available_vehicle_pos.append(  vehicle_pos[ith_vehicle] )
            available_vehicle_capacity.append(vehicle_capacity[ith_vehicle])
    
    
    new_nodes = nodes.tolist()  + available_vehicle_pos
    
    Distance_matrix = np.zeros( (len(new_nodes),len(new_nodes)) , dtype=np.float32 )
    for u,src in enumerate(new_nodes) : 
        for v , dst  in  enumerate(new_nodes):  
            Distance_matrix[u][v] =  node_distance( src[0] , src[1] , dst[0] , dst[1])
    Distance_matrix = torch.tensor(Distance_matrix , dtype=torch.float32)
    # print(f"RoadNetwork distance matrix : \n{RoadNetworkData.node_route_cost}\n")
    # print(f"New Distance matrix :\n{Distance_matrix}\n")
    # print(f"New Distance matrix2 :\n{(Distance_matrix*10000).int()}\n")
    # print(f"New Distance matrix3 :\n{(Distance_matrix*10000).int().tolist()}\n")
    demand_list = torch.round(RoadNetworkData.x[:,2] * 10000).tolist() 
    demand_list = demand_list + [0] * len(vehicle_pos) 
    # print(f"Demand list : \n{demand_list}")
    # print(f"vehicle_capacity: \n{vehicle_capacity}")
    vehicle_pos = [i for i in range( nodes.shape[0]  , len(new_nodes))]
    # print(f"vehicle pos : {vehicle_pos}")
    vehicle_capacity = [c*10000 for c in available_vehicle_capacity]
    
    
    return torch.round(Distance_matrix*10000).int().tolist() ,demand_list , vehicle_pos , vehicle_capacity  
    
    
    
    
    
    