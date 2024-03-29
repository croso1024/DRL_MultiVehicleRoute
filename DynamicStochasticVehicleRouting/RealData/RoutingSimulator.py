""" 
    For the Taipai routing case study , simulate the road map transistion and 
    dynamic reqeust generation 

"""
import osmnx as ox ,networkx as nx , torch ,random , numpy as np
import matplotlib.pyplot as plt 
from random import betavariate , normalvariate , randint
from sklearn.manifold import MDS , Isomap
from RealData.location_set import loc_dict_encode
from torch_geometric.data import Batch , Data 
from math import sqrt 
from osmnx.utils_graph import get_route_edge_attributes
import copy 

import time
from RealData.simulator_utils import * 

""" 
            作圖功能會需要的紀錄 , 所有的紀錄都是分時刻 ,總共應該有(
                初始狀態 --> predict1 --> 更新1 --> predict2 --> 更新2 -->  predict3 --> 最終狀態 --> predict4
                ( cur1 )  (Predict1)   ( cur2 )  (Predict2) ( cur3 )     (Predict3)   ( cur4 )   (Predict4)
                ( com1 )   (done1)     ( com2 )   (done2)   ( com3 )      (done3)     ( com4 )    (done4)
                ( SDR1 )               ( SDR2 )             ( SDR3 )                  ( SDR4 )
                ( route1 )             ( rout2 )            ( route3 )                ( route4 )
            )   
                2 + total_update_times 組資料 , 注意各自紀錄的時間點!            
            ------------------------------------------------------------ 
            1. Current Request(用以紀錄當前時刻的Request)
            2. Complete Request(用以在地圖上畫出已經完成的那些節點)
            Complete-Request + Current Request在地圖上會用兩個不同顏色一起出現 
            3. Vehicle_SrcDstRatio (改為紀錄車輛前一節點與下一節點 , 移動比例 , 這樣才能在OSM map上真實呈現)
            4. Model_Predict (用以紀錄當前時刻算法的規劃結果) (在初始狀態是沒有的)
            5. Vehicle-Routes (紀錄截自此刻每一車輛實際的路線)
"""

class RoutingSimulator : 
    

    
    def __init__(
                self ,
                 init_Request_num ,
                 maximum_Request_num , 
                 vehicle_num ,
                 init_capacity , 
                 step_duration, 
                 total_update_times , 
                 release_num ,
                 PMPO=False, 
                 deterministic = False , 
                 BaseGraph = None , 
                 node_coordinate_table = None , # 每次都計算這一部份時間相當久 !!! 
                 map = "NTU"
                 ): 
        # Base Graph of the Open Street Map 
        if BaseGraph :
            self.BaseGraph = BaseGraph
        else :
            raise RuntimeError("Base graph loading error !")
            #self.BaseGraph = ox.graph_from_point((25.0416,121.5438) , dist=7000 , simplify=True, network_type="walk")        
        self.node_set = loc_dict_encode[map]
            
            
        # Parameters of the simulator environment 
        self.init_Vehicle_capacity = init_capacity 
        self.init_Request_num = init_Request_num 
        self.maximum_Request_num = maximum_Request_num 
        self.vehicle_num = vehicle_num                                
        self.step_duration = step_duration 
        self.total_update_times = total_update_times 
        self.update_times = 0 # already update time for dynamic scenario
        self.RoadNetwork_Generator = self.get_CurrentRoadNetwork if not PMPO else self.get_CurrentRoadNetwork_PMPO
        self.PMPO = PMPO
        self.release_num = release_num
        self.deterministic = deterministic
        assert maximum_Request_num >=  init_Request_num + ( total_update_times * self.release_num[1] ) , "Improper parameter setting on Request"
        
        # Setup the set of all ,unknown , current and complete node 
        self.node_set = { i :self.node_set[str(i)] for i in range(self.maximum_Request_num) }
        self.unknown_request = set(self.node_set.keys()) 
        self.current_request = set()
        self.complete_request = set() 
       
        # 使用所有節點建立Distance matrix , 並將其轉換成為節點表再後續重複使用 !
        if node_coordinate_table : 
            self.node_coordinate_table = node_coordinate_table
        else : 
            self.node_coordinate_table = self.get_initial_Coordinates()
        # 所有車輛的位置,預設是建立在node-set"0"
        # Random generate demand for all request : 
        #self.demand_set = {node_id : max(0.01 ,(betavariate(alpha =2,beta=5) /4) ) for node_id in self.node_set.keys()}
        self.demand_set = {node_id : max(0.01 ,normalvariate(mu=0.07,sigma=0.02)) for node_id in self.node_set.keys()}
        self.demand_set[0] = 0 
        # Setup the parameters for all vehicle : 
        self.capacity_set = [float(self.init_Vehicle_capacity) for i in range(self.vehicle_num)]
        self.vehicle_cost = [0 for i in range(self.vehicle_num)]
        self.vehicle_last_node = [0 for i in range(self.vehicle_num)] 
        self.vehicle_pos_set = [self.node_coordinate_table[0] for i in range(self.vehicle_num)]
        self.vehicle_SrcDstRatio =  [  (0,0,0)  for i in range(self.vehicle_num) ]
        self.vehicle_route_cumulative = [ [0] for i in range(self.vehicle_num) ]
        self.done_vehicle = [False for i in range(self.vehicle_num)]
        # Setup the randomness matrix ( for all available request node )

        if self.deterministic :
            self.randomness_factor_matrix = np.zeros((self.maximum_Request_num,self.maximum_Request_num) ,dtype=np.float32)
        else : 
            self.randomness_factor_matrix = np.random.uniform(low=0.05,high=0.25,
                                        size=(self.maximum_Request_num,self.maximum_Request_num))

        self.randomness_factor_matrix = (self.randomness_factor_matrix + self.randomness_factor_matrix.T ) /2
        for i in range(self.maximum_Request_num): self.randomness_factor_matrix[i][i] =0 
        # Recorder 
        self.current_request_Recorder = list() 
        self.complete_request_Recorder = list() 
        self.vehicle_SDR_Recorder = list() 
        self.model_predict_Recorder = list()
        self.vehicle_routes_Recorder = list() 
        self.vehicle_done_Recorder = list()
    
        # print(f"Debug :-------- \n")
        # print(f"node_coordinate table : {self.node_coordinate_table}\n")
        # print(f"demand_set : {self.demand_set}\n")
        # print(f"vehicle -pos : {self.vehicle_pos_set}\n")
        # print(f"vehicle-capacity : {self.capacity_set}\n")
        # print(f"Randomness factor matrix : \n{self.randomness_factor_matrix}\n")
        
    def reset(self): 
        """ 
            1.Sample在初始狀態使用的節點 , 從Coordinate-set ,demand-set , randomness-matrix中取出對應部份
            2.使用取出的部份建立Graph-data , vehicle-data
            3.更新狀態變數 , Recorder 
        """
        # Sampled the request exclude the depot , until the init_Request num
        if self.deterministic : 
            Sampled_node = sorted(list(range(self.init_Request_num)))
        else : 
            Sampled_node = sorted(
                [0] + random.sample( self.unknown_request-{0} , self.init_Request_num-1 ) 
            )
        self.current_request = self.current_request.union(Sampled_node)
        self.unknown_request = self.unknown_request.difference(Sampled_node)
        self.node_id_table = {i : node  for i ,node in enumerate(self.current_request) }
        # print(f"Debug update node-id-table :\n{self.node_id_table}")
        # print(f"Debug update current request : \n{self.current_request}")
        # print(f"Debug update unknown request : \n{self.unknown_request}")
        nodes = [self.node_coordinate_table[node] for node in self.current_request]
        demands = [self.demand_set[node] for node in self.current_request]
        randomness_matrix = [  [  self.randomness_factor_matrix[src][dst] for dst in self.current_request] for src in self.current_request]
        # print(f"Debug nodes : \n{nodes}")
        # print(f"Debug demands : \n{demands}")
        # print(f"Debug randomness matrix: \n{randomness_matrix}")
        # Recorde the initial State
        self.current_request_Recorder.append(copy.deepcopy(self.current_request))
        self.complete_request_Recorder.append(copy.deepcopy(self.complete_request) )
        self.vehicle_SDR_Recorder.append(copy.deepcopy(self.vehicle_SrcDstRatio)) 
        self.vehicle_routes_Recorder.append(copy.deepcopy(self.vehicle_route_cumulative))

        
        
        self.terminate = False 
        if self.PMPO : 
            RoadNetwork =self.get_CurrentRoadNetwork_PMPO(nodes=nodes , demands=demands , randomness_matrix=randomness_matrix)
            Vehicle_Pos = PMPO_vehicle_pos(self.vehicle_pos_set)
            return Batch.from_data_list(RoadNetwork) , Vehicle_Pos , self.capacity_set ,self.terminate
        else : 
            RoadNetwork = self.get_CurrentRoadNetwork(nodes=nodes , demands=demands , randomness_matrix=randomness_matrix)
            Vehicle_Pos = self.vehicle_pos_set
            return Batch.from_data_list([RoadNetwork]) , Vehicle_Pos , self.capacity_set ,self.terminate 
        
        
        
    def step(self,vehicle_routes): 
        """ 
            此step是主動使用模型已經output的結果來觸發 , 根據我們設定的切斷時刻T, 去計算每一台車輛在該時刻的位置 , 剩餘容量 ,
            同時更新已經完成的Request, 並且加入新的Request ,
            Note : 假設Vehicle-routes的順序與原始車輛順序相同
        """

        
        self.model_predict_Recorder.append( self.cover_predictID_to_nodeID(copy.deepcopy(vehicle_routes))   )
        self.vehicle_done_Recorder.append(  copy.deepcopy(self.done_vehicle) )

        #更新車輛位置,容量 , current_request , complete_request
        if not self.terminate :
            info_package = self.update_VehiclePosition(vehicle_routes=vehicle_routes)
            self.update_NewRequest()
            self.update_Randomness() 
            # check the complete route == complete request 
            self.checkComplete() 
            
            nodes = [self.node_coordinate_table[node] for node in self.current_request]
            demands = [self.demand_set[node] for node in self.current_request]
            randomness_matrix = [  [  self.randomness_factor_matrix[src][dst] for dst in self.current_request] for src in self.current_request]
            # Recorder 
            self.current_request_Recorder.append(copy.deepcopy(self.current_request))
            self.complete_request_Recorder.append(copy.deepcopy(self.complete_request) )
            self.vehicle_SDR_Recorder.append(copy.deepcopy(self.vehicle_SrcDstRatio) )
            
            self.vehicle_routes_Recorder.append(copy.deepcopy(self.vehicle_route_cumulative))
            
            if self.PMPO : 
                RoadNetwork =self.get_CurrentRoadNetwork_PMPO(nodes=nodes , demands=demands , randomness_matrix=randomness_matrix)
                Vehicle_Pos = PMPO_vehicle_pos(self.vehicle_pos_set)
                return Batch.from_data_list(RoadNetwork) ,Vehicle_Pos, self.capacity_set ,self.terminate , info_package
            else : 
                RoadNetwork = self.get_CurrentRoadNetwork(nodes=nodes , demands=demands , randomness_matrix=randomness_matrix)
                Vehicle_Pos = self.vehicle_pos_set
                return Batch.from_data_list([RoadNetwork]) ,Vehicle_Pos, self.capacity_set ,self.terminate , info_package
        # 在terminate狀態下拿到vehicle_routes , 即要結算Route Cost ,fulfill-rate了
        else: 
            # print(f"Debug Predict \n{self.model_predict_Recorder}")
            # print(f"Debug cumulative route : \n{self.vehicle_routes_Recorder}")
            # print(f"Debug vehicle done Recorder : {self.vehicle_done_Recorder}")
            #print(f"\n--------------\n")
            objective , fulfill_rate = self.Total_Objective(vehicle_routes)
            return objective , fulfill_rate 
        
   
    def update_VehiclePosition(self,vehicle_routes): 
        """ 
            在經過的時間T下更新 車輛位置,容量, 已經完成的節點 , 另外要注意模型在route內只會輸出 0~current_request-1, 
            因此需要有一個table把模型輸出的node編號換成在所有節點下的node編號. 
        """
        
        # print(f"Vehicle Route : {vehicle_routes}")
        for ith_vehicle , path in enumerate(vehicle_routes): 
            
            # 依據此一台車是否完成 , 開始更新參數            
            if self.done_vehicle[ith_vehicle] :  continue
            # 初始化與該車更新相關的參數
            last_node = self.vehicle_last_node[ith_vehicle] # last_node為global的node
            vehicle_x , vehicle_y = self.vehicle_pos_set[ith_vehicle]
            duration = self.step_duration
            capacity = self.capacity_set[ith_vehicle]
            complete_list = list() 
            
            for node in path : 
                # 把模型輸出的編號轉換成為Global node id 
                node = self.node_id_table[node]
                # 取出該車的上一個節點 , 同樣也是global id 
                last_node = self.vehicle_last_node[ith_vehicle] 
                # 使用節點的座標計算空間距離
                node_x , node_y = self.node_coordinate_table[node][0] , self.node_coordinate_table[node][1]
                distance = node_distance( vehicle_x , vehicle_y ,node_x , node_y) 
                # 加入隨機性
                distance = distance * (1 + self.randomness_factor_matrix[last_node][node])
                
                # 目前剩餘的時間足夠走到下一個節點: ( 如果是走到depot了,那就是要結束了 )
                if duration > distance:  
                    
                    duration -= distance 
                    self.vehicle_cost[ith_vehicle] += distance 
                    # self.vehicle_cost[ith_vehicle] += ( distance * (1+)
                    capacity -= self.demand_set[node] 
                    assert capacity >=-0.0005 , f"negative capacity error , rest capacity:{capacity} "          
                    complete_list.append(node) 
                    vehicle_x , vehicle_y = node_x , node_y # 走上下一個節點
                    self.vehicle_last_node[ith_vehicle] = node # 更新該車最後一個節點
                    self.vehicle_route_cumulative[ith_vehicle].append(node) # 在該車的實際路徑中添加該節點 

                    
                    if not node == 0 : continue
                    # 走回depot, 該車已經結束,後續不再參與更新 , 紀錄SDR
                    # else : 
                    #     self.done_vehicle[ith_vehicle] = True 
                    #     self.vehicle_SrcDstRatio[ith_vehicle] = (0,0,0) 
                    #     break 
                    # 0622 test modify the rule of done_vehicle 
                    elif self.vehicle_last_node[ith_vehicle] == 0: 
                        self.vehicle_SrcDstRatio[ith_vehicle] = (0,0,0)
                        continue
                    else : 
                        self.done_vehicle[ith_vehicle] = True 
                        self.vehicle_SrcDstRatio[ith_vehicle] = (0,0,0)
                        break 


                # 目前剩餘的時間已經不足到下一個節點 : --> Interpolation + break 
                elif duration < distance : 
                    self.vehicle_cost[ith_vehicle] += duration 
                    vehicle_x,vehicle_y = interpolation(vehicle_x,vehicle_y , node_x , node_y , duration)
                    # 更新該車輛的 Src , Dst , Ratio , 用來繪製車輛位置
                    self.vehicle_SrcDstRatio[ith_vehicle] = (last_node , node , duration/distance)
                    # 如果duration足夠走超過一半 , 那我們更新該車的最後一個節點為他的目標  
                    
                    """ 
                     0510為了測試關掉的
                    if duration > 0.5 * distance : 
                        self.vehicle_last_node[ith_vehicle] = node 
                    """   
                    # Open when plot the route 
                    # if duration > 0.5 * distance : 
                    #     self.vehicle_last_node[ith_vehicle] = node 
                    # # 到此這一台車的更新完畢
                    break 

                # 剩餘時間剛好走上下一個節點(理論上應該不太可能)
                else : raise RuntimeError("Duration match distance error")
                
            # 此處已經更新完車輛ith的complete-list , capacity , position 
            self.vehicle_pos_set[ith_vehicle] = ( vehicle_x , vehicle_y )
            self.capacity_set[ith_vehicle] = capacity 
            self.complete_request = self.complete_request.union( set(complete_list) )
            self.current_request  = self.current_request.difference( set(complete_list) )
        #print(f"Vehicle Done : \n{self.done_vehicle}\n")
        # print(f"vehicle -pos : {self.vehicle_pos_set}\n")
        #print(f"vehicle-capacity : {self.capacity_set}\n")
        
        
        """ 
            為了解決ortools沒有車的可以用的bug , 如果所有車都掛了 , 則提前計算obj , fulfill  ,透過infopackage讓環境提前結束 
        """
        if all( self.done_vehicle  ) : 
            print(f"\n USE Early Stop \n")
            self.terminate = True 
            total_objective ,total_demand , total_fulfill = 0 , 0 , 0  
            for cost_of_vehicle_i in self.vehicle_cost : total_objective += cost_of_vehicle_i
            # Step2. Fulfill rate 
            for demand in self.demand_set.values() : total_demand += demand
            for node in self.complete_request :  total_fulfill += self.demand_set[node]
            return {"early_stop":True, "obj":total_objective, "fulfill":total_fulfill/total_demand}
        else : 
            return {"early_stop":False}
        
    
    def update_NewRequest(self): 
        """ 
            從Unkown-set中產生Sampled節點產生新的request , 每一次隨機加入2-5個節點 
            總共更新total_update_times次 , 每次最多5個節點, 故Request num必須滿足 
            maximum_Request_num >=  init_Request_num + ( total_update_times * 5 )
            當更新達到total_update_times次後, 下一次一口氣將所有unknown節點放出
        """
        # 最後一次的更新, 取出所有unknown request加入 
        if self.update_times == self.total_update_times : 
            Sampled_node = sorted( [0] + list(self.unknown_request) )
            #print(f"Debug final sampled : {Sampled_node}")
            self.terminate = True 
        else : 
            if self.deterministic : 
                Sampled_node = [0] + sorted(list(self.unknown_request)[:3])
            else : 
                Sampled_node = sorted( [0]  + random.sample( self.unknown_request , randint(self.release_num[0],self.release_num[1]) ) )
        
        self.current_request = set(sorted(self.current_request.union(Sampled_node)))
        self.unknown_request = self.unknown_request.difference(Sampled_node)
        self.node_id_table = {i : node  for i ,node in enumerate(self.current_request) }
        # print(f"Debug update node-id-table :\n{self.node_id_table}")
        # print(f"Debug update current request : \n{self.current_request}")
        # print(f"Debug update unknown request : \n{self.unknown_request}")
        # print(f"Debug update complete request : \n{self.complete_request}")
        
        self.update_times += 1 
        return None 

    def update_Randomness(self): 
        """ 
            在每一次更新的時候都會連帶更新隨機矩陣,更新完成後才經由get-RoadNetwork產生Instance
        """
        if self.deterministic : return 
        else : 
            for u in range(self.maximum_Request_num): 
                for v in range(self.maximum_Request_num): 
                    self.randomness_factor_matrix[u][v] = normalvariate( self.randomness_factor_matrix[u][v] ,sigma=0.02 )

    def checkComplete(self): 
        """ 
            用來檢查累積路徑的結果和車輛已經完成的node實際上是一樣的
        """
        complete_route = list() 
        for ith_vehicle in range(self.vehicle_num):  
            #print(f"Debug vehicle cumulative route :{self.vehicle_route_cumulative[ith_vehicle]}")
            complete_route.extend(   self.vehicle_route_cumulative[ith_vehicle]    )
        #print(f"Debug complete request {self.complete_request}")
        #print(f"Debug complete route {complete_route}")
        assert self.complete_request-{0} == set(complete_route)-{0} , "Complete request not equal to the complete route error"
        return 
    
    
            
    def get_initial_Coordinates(self): 
        loc_dict_nearest = {
            node : ox.nearest_nodes(self.BaseGraph,self.node_set[node][1] , self.node_set[node][0])
            for node in self.node_set.keys() 
        }
        distance_matrix_dict = {
            src_name : {
                #dst_name :  nx.shortest_path_length(self.BaseGraph , src_node , dst_node) 
                dst_name : self.calculate_route_length(ox.shortest_path(self.BaseGraph , src_node , dst_node))
                for dst_name , dst_node in loc_dict_nearest.items()
            } for src_name , src_node in loc_dict_nearest.items()
        }
        

        distance_matrix = np.array([list(node.values()) for node in distance_matrix_dict.values() ] , dtype=np.float32) 
        # 確保對稱
        n = distance_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = (distance_matrix[i, j] + distance_matrix[j, i])/2
        normalized_coordinate = Distance_matrix_to_Coordinate(distance_matrix)
        return normalized_coordinate 
    
    def get_node_num(self): 
        return len(self.current_request)
    
    
    def calculate_route_length(self,route_points): 
        route = get_route_edge_attributes(self.BaseGraph , route_points) 
        total_distance = sum(edge["length"] for edge in route )
        return total_distance
    
    
    
    
    def Total_Objective(self,vehicle_routes): 
        """ 
            最終的Cost就不使用Batch和tensor操作了
        """
        # Step1. Route cost objective 
        for ith_vehicle , path in enumerate(vehicle_routes): 
            if self.done_vehicle[ith_vehicle]:continue  
            vehicle_x , vehicle_y = self.vehicle_pos_set[ith_vehicle]
            capacity = self.capacity_set[ith_vehicle]
            complete_list = list() 
            for node in path : 
                node = self.node_id_table[node]
                last_node = self.vehicle_last_node[ith_vehicle]
                node_x , node_y = self.node_coordinate_table[node][0] , self.node_coordinate_table[node][1] 

                distance = node_distance(vehicle_x,vehicle_y , node_x,node_y )
                distance = distance * (1 + self.randomness_factor_matrix[last_node][node] )
                self.vehicle_cost[ith_vehicle] += distance 
                capacity -= self.demand_set[node] 
                assert capacity >=-0.0005 , f"negative capacity error , rest capacity:{capacity} "                          
                complete_list.append(node) 
                vehicle_x , vehicle_y = node_x , node_y 
                if node == 0 : 
                    self.done_vehicle[ith_vehicle] = True 
                    break 
                else : 
                    self.vehicle_last_node[ith_vehicle] = node 
                    
                    
            self.vehicle_pos_set[ith_vehicle] = (vehicle_x,vehicle_y)
            self.complete_request = self.complete_request.union(set(complete_list)) 
            self.current_request = self.current_request.difference(set(complete_list))
        
        for i in self.done_vehicle : assert i == True , "Not all vehicles back to the depot error"

        total_objective ,total_demand , total_fulfill = 0 , 0 , 0  
        
        for cost_of_vehicle_i in self.vehicle_cost : total_objective += cost_of_vehicle_i
        # Step2. Fulfill rate 
        for demand in self.demand_set.values() : total_demand += demand
        for node in self.complete_request :  total_fulfill += self.demand_set[node]
        #print(f"Final rest request : {self.current_request}")
        return total_objective ,  total_fulfill/total_demand
        

    def get_CurrentRoadNetwork(self , nodes,demands,randomness_matrix ): 
        """ 
            Generate the graph-data from distance-matrix , demand-set , vehicle-pos , vehicle_capacity
            注意此處是直接吃node,demands,randomness, 即randomness再初始化的時候已經設定過了 , 這邊會再加入微小的擾動
        """
        assert len(nodes) == len(demands) == len(randomness_matrix[0]) , "Road Info dimension error"
        node_num = len(nodes)
        nodes = [ Demand_Node(x = nodes[i][0] , y=nodes[i][1] , demand=demands[i]) for i in range(node_num) ]
        node_feature = np.zeros((node_num , 3) , dtype=np.float32) 
        for ith , node in enumerate(nodes): node_feature[ith] = node.node_attribute()
        Distance_matrix = np.zeros( (node_num , node_num ) , dtype=np.float32 )
        Stochastic_Distance_matrix =  np.zeros( (node_num , node_num ) , dtype=np.float32 )
        
        edge_index , edge_attr = list() , list() 
        
        for u , src_node in enumerate(nodes): 
            for v , dst_node in enumerate(nodes): 

                distance = sqrt(pow( ( src_node.x - dst_node.x ),2) + pow( ( src_node.y - dst_node.y ),2) )
                #randomness_factor = normalvariate( mu = randomness_matrix[u][v] , sigma=0.02 )
                randomness_factor = randomness_matrix[u][v]
                # stochastic_distance = distance *  ( 1 + randomness_factor   )
                stochastic_distance = max(
                    normalvariate(mu=distance , sigma = randomness_factor*distance) ,
                    0.75*distance
                )
                
                Distance_matrix[u][v] = distance 
                Stochastic_Distance_matrix[u][v] = stochastic_distance
                edge_index.append((u,v)) 
                edge_attr.append(randomness_factor)
        
        
        return  Instance(
            x = torch.tensor(node_feature , dtype=torch.float32) , 
            node_route_cost = torch.tensor(Distance_matrix , dtype=torch.float32) , 
            node_route_cost_std = torch.tensor(Stochastic_Distance_matrix , dtype=torch.float32) , 
            edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
            edge_attr = torch.tensor(edge_attr , dtype=torch.float32) , 
        )
    
    def get_CurrentRoadNetwork_PMPO(self,nodes, demands , randomness_matrix): 
        assert len(nodes) == len(demands) == len(randomness_matrix[0]) , "Road Info dimension error"
        node_num = len(nodes)
        
        PMPO_node_list = [
            PMPO_node_argument(init_x=nodes[i][0] ,init_y = nodes[i][1] , demand=demands[i] )
            for i in range(node_num) 
            ]
        assert len(PMPO_node_list[0])==8 , "PMPO dimension error"
        PMPO_data = list() 
        PMPO_node_list = list(zip(*PMPO_node_list))
        # Create the common part of PMPO graph data
        Distance_matrix = np.zeros( (node_num , node_num ) , dtype=np.float32 )
        Stochastic_Distance_matrix =  np.zeros( (node_num , node_num ) , dtype=np.float32 )
        edge_index , edge_attr = list() , list() 
        for u ,src_node in enumerate(PMPO_node_list[0]): 
            for v , dst_node in enumerate(PMPO_node_list[0]): 
                distance = sqrt(pow( ( src_node.x - dst_node.x ),2) + pow( ( src_node.y - dst_node.y ),2) )
                # randomness_factor = normalvariate( mu = randomness_matrix[u][v] , sigma=0.02 )
                randomness_factor = randomness_matrix[u][v]
                # stochastic_distance = distance * (1+randomness_factor) 
                stochastic_distance = max(
                    normalvariate(mu=distance , sigma = randomness_factor*distance) ,
                    0.75*distance
                )
                Distance_matrix[u][v] = distance 
                Stochastic_Distance_matrix[u][v] = stochastic_distance
                edge_index.append((u,v)) 
                edge_attr.append(randomness_factor)
        
        
        for argumenth_th , nodes in enumerate(PMPO_node_list): 
            node_feature = np.zeros((node_num , 3) , dtype=np.float32) 
            for ith , node in enumerate(nodes): node_feature[ith] = node.node_attribute()
            PMPO_data.append(
                Instance(
                    x = torch.tensor(node_feature , dtype=torch.float32) , 
                    node_route_cost = torch.tensor(Distance_matrix , dtype=torch.float32) , 
                    node_route_cost_std = torch.tensor(Stochastic_Distance_matrix , dtype=torch.float32) , 
                    edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
                    edge_attr = torch.tensor(edge_attr , dtype=torch.float32) , 
                )
            )    
    

        return PMPO_data 

    def cover_predictID_to_nodeID(self,vehicle_routes): 
        coverted_routes = list() 
        
        for ith_vehicle , path in enumerate(vehicle_routes): 
            temp = [ self.node_id_table[node] for node in path ] 
            coverted_routes.append(temp) 
        return coverted_routes                
            

    def get_Logger(self): 
        
        Logger = {
            "Current_Request" : self.current_request_Recorder , 
            "Complete_Request" : self.complete_request_Recorder , 
            "Vehicle_SDR" : self.vehicle_SDR_Recorder , 
            "Model_Predict": self.model_predict_Recorder , 
            "Vehicle_Routes" : self.vehicle_routes_Recorder , 
            "Done_Vehicle": self.vehicle_done_Recorder , 
        }
        return Logger 