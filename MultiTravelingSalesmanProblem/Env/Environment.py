""" 
2023-04-10 
    HCVRP environment 統整版 , 整合Decision-equivalent + PMPO同時使用這個env  
    1. 加入累積capacity/total capacity 作為fleet-state的positional encoding , note 這一步是加在fleet permute之後 , 才不會被影響

    2.混合PMPO以及DE模式

    3.增加self.usePMPO屬性 , 支援同一個batch內的random capacity ,velocity ,
      初始化的capa,velo都變為range , 在reset的時候進行初始化

    4. 修正DE transform時疊加batch的順序 , capacity , velocity以及轉換DE graph-data都改以repeat_inferleave形式 
       即DETransform疊graph-data 由 1,2 -> 1,2,1,2 變為 1,2 -> 1,1,2,2 , capacity , velocity也都滿足這個特性 , 
       才能確保計算reward時取一個範圍內的data這件事情正確
    
    Note : 只要是使用 PMPO or mix , 就記得要用getInstance -- Batch系列
2023-04-11 
    修改inference , 從單純取min改為取criterion最好的 
2023-04-16
    移植回MTSP的版本 , 用於MTSP , 只有用在MTSP沒有SE版 , 使用PMPO做計算
    移除不必要的車隊 , 決策車輛狀態 , 移除Terminal-State等的計算 
    
    
"""
import torch 
from torch_geometric.utils import to_dense_batch , mask_to_index 
from torch_geometric.data import Batch 
from utils.KNN_transform import KNNGraph_transformer  , RadiusGraph_transformer
from random import randint 
from math import sqrt ,exp


class MTSP_Environment : 
    def __init__(   self , 
                    batch_size , 
                    node_nums , 
                    batch_data , 
                    vehicle_num = 1 , 
                    vehicle_pos_mode  = "Depot" ,  
                    StateEQ = "PMPO" ,
                    graph_transform: dict = None ,
                    device = 'cpu' , 
                ): 
        self.batch_size = batch_size     
        self.num_nodes = node_nums 
        self.vehicle_num = vehicle_num 
        self.vehicle_pos_mode = vehicle_pos_mode
        self.path_length = node_nums + vehicle_num - 1 
        self.stateEQ_mode = StateEQ
        self.reward_func = self.reward_training
        self.device = device
        # Graph-state = static state + dynamic state 
        # [ pos_x , pos_y , depot , have_request ]
        if self.stateEQ_mode == "PMPO": 
            self.batch = batch_data 
            self.usePMPO = True 
        else: 
            self.batch = batch_data 
            self.usePMPO = False


        if graph_transform :
            if graph_transform["transform"] == "knn" : 
                knn_graph = KNNGraph_transformer(k = graph_transform["value"] , self_loop=False , num_workers=4) 
                self.batch = knn_graph.batch_transform(self.batch)
            elif graph_transform["transform"] == "radius": 
                radius_graph = RadiusGraph_transformer(r=graph_transform["value"] , self_loop=False , num_workers=4) 
                self.batch = radius_graph.batch_transform(self.batch)
            else : raise RuntimeError("Graph Transformer error")
        

        # setup the stateEQ reward function 
        self.StateEQ_reward_function = self.calculate_PMPO
        
        self.static_state , ret = to_dense_batch(self.batch.x , self.batch.batch) 
        self.dynamic_state = torch.zeros(size=(self.batch_size , self.num_nodes ,2) , dtype=torch.float32 ,device=self.device)
        # Fleet [pos_x , pos_y ,vehicle-number( i vehicle / total vehicle num ), on-decision , complete-decision ]
        # Vehicle [pos_x , pos_y , vehicle-number ,cumulative-visited ]
        self.Fleet_state = torch.zeros(size=(self.batch_size , self.vehicle_num, 5) , dtype=torch.float32, device=self.device)
        self.init_pos = torch.zeros(size=(self.batch_size,self.vehicle_num,2) , dtype=torch.float32 ,device=self.device) 
        # Support tensor 
        self.path_log = torch.zeros(size=(self.batch_size,self.path_length) , dtype=torch.int64,device=self.device)
        self.mask = self.mask = torch.zeros((self.batch_size, self.num_nodes , 1) ,dtype=torch.bool,device=self.device)
        self.mask_help = torch.arange(self.num_nodes,device=self.device).unsqueeze(0).unsqueeze(2).expand_as(self.mask)
        self.sup_batchwise_indexing = torch.arange(self.batch_size,device=self.device).unsqueeze(1).long()
        self.depot_token_counter = torch.zeros(size=(self.batch_size,1) , dtype=torch.int8 , device=self.device)
        self.sup_temp_dynamicGraph_state = torch.zeros(size=(self.batch_size, self.num_nodes,1),dtype=torch.float32 , device=self.device)
        self.sup_temp_dynamicGraph_state_src = torch.ones(size=(self.batch_size,1,1) , dtype=torch.float32 , device=self.device) 
        self.step_counter = 0 
        # Record  the effective iteraction log-probs 
        self.terminal_list = [] 
        self.sup_terminal_shape =  torch.zeros(self.batch_size,device=self.device ).bool() 
        self.sup_reuse_mask = torch.zeros(size=(self.num_nodes,1),device=self.device).bool() 
        self.total_visted_reqeust = torch.ones(size=(self.batch_size, 1 , 1 ) , dtype=torch.int16 , device=self.device) *  self.num_nodes
        self.cumulative_visted = torch.zeros(size=(self.batch_size,1,1),dtype=torch.float32 , device=self.device)
        self.penality_coeifficient = self.num_nodes / (2*self.vehicle_num)
      
    def reset(self): 
        """ 
            Reset: 
            step1. 選擇depot起點位置, 更新Graph-data的內容轉為Graph-state , 紀錄Total demand size 
            step2. 初始化車隊向量, 賦予車輛位置( Depot , random , specific ) ,保存車輛初始位置
            step3. 更新 mask , decision_vehicle
            step4. 取出現在正要決策的車輛 (pos和capacity即可)
            step5. 輸出Graph-state , fleet-state , decision-vehicle-state (此版本不做reset時的mask)
        """
        self.depot_index = 0 
        self.dynamic_state[:,1:,1] = 1  #除了depot index都把have request開為1
        self.dynamic_state[:,self.depot_index,0] = 1  # depot屬性設置
        self.total_demand = self.num_nodes # total demand直接替換成總共的節點數 
        if self.vehicle_pos_mode == "Depot": 
            self.init_pos = torch.gather(
                self.static_state , dim=1 , index = (
                    torch.zeros(size=(self.batch_size,self.vehicle_num,2) , dtype=torch.long ,device=self.device) 
                )
            )
        else: raise RuntimeError("Pos mode only Depot available")
        # 紀錄每一個batch目前的Decision vehicle
        self.decision_vehicle = torch.zeros(size=(self.batch_size,1),dtype=torch.long,device=self.device)
        self.Fleet_state[:,:,:2] = self.init_pos.clone()   
        # add vehicle_number for every vehicle in the fleet 
        self.Fleet_state[:,:,2] =  torch.arange(self.vehicle_num).unsqueeze(0).expand(size=(self.batch_size,self.vehicle_num))  / self.vehicle_num
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 3 ] = 1  # 把decision-vehicle的正在決策打開
        
        mask = self.update_mask( (torch.ones(size=(self.batch_size ,1 )
                                        ,device=self.device) * self.depot_index) , reset=True )

        # -- reuse-mask , 當某個batch互動結束後 , 使用reuse_mask(只有depot被擋住)給該batch繼續互動 , 在MTSP移除
        #self.sup_reuse_mask[self.depot_index] = 1 
        #mask = self.update_terminal(mask)
        
        ## 取出 pos-x , pos-y , vehicle-number
        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim=1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,3)
        )
        ## concat cumulative visited
        self.vehicle_vector = torch.concat(
            (self.vehicle_vector , self.cumulative_visted/self.total_visted_reqeust) , dim=2    
        )
        
        self.observation = torch.concat(
            (self.static_state , self.dynamic_state) , dim= 2 
        ).view(self.batch_size*self.num_nodes , -1)
        
        
        self.done = False 
        return self.observation , self.Fleet_state , self.vehicle_vector , mask , self.done 
    
    def step(self,action:torch.Tensor): 
        """ 
            需要更新項目 :
                step1. 更新Graph-state (have request , demand不動) , Fleet-state(目前車輛pos , capacity) , path_log 
                step2. 更新決策車輛 , 並順便判斷這次action有哪些車輛回到depot(先更新決策車輛 , 因為mask的capacity與其有關)
                step3. 以更新過的決策車輛 , 以及depot token來更新mask ( depot / normal node / capacity not enough node)
                step4. 更新terminal mask ,確認哪些batch提早結束 
                step5. 檢查是否done , 計算reward以及更新狀態
        """
        self.update_Graph_state(action)
        self.update_Fleet_state(action) 
        
        self.path_log[self.sup_batchwise_indexing , self.step_counter] = action.long()
        
        depot_token = self.update_decision_vehicle(action) 
        
        mask = self.update_mask(action , depot_token = depot_token)
        
        # mask = self.update_terminal(mask) 
        self.step_counter += 1  
        if self.step_counter == self.path_length - 1 : 
            reward = self.reward_func() 
            self.done = True 
        else : 
            reward =  None 
            
        return self.observation , self.Fleet_state , self.vehicle_vector , reward , mask ,self.done 
    
    def update_Graph_state(self, action:torch.Tensor): 
        temp_dynamic_Graph_state = torch.scatter(
            self.sup_temp_dynamicGraph_state , 
            dim = 1 , 
            index = action.unsqueeze(1).long() , 
            src = self.sup_temp_dynamicGraph_state_src
        )
        self.dynamic_state[:,:,1] -= temp_dynamic_Graph_state.squeeze(2)
        self.dynamic_state[:,:,1] = torch.clip(self.dynamic_state[:,:,1],min=0 , max=1)
        self.observation = torch.concat(
            (self.static_state , self.dynamic_state) ,dim=2 
        ).view(self.batch_size*self.num_nodes , -1)
    
    def update_Fleet_state(self,action:torch.Tensor) : 
        temp_state_graph_state = torch.gather(
            self.static_state , dim = 1 , 
            index = action.unsqueeze(2).expand(size=(self.batch_size, 1, 2 )).long()
        )
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle,:2] = temp_state_graph_state
        # 把有拜訪新節點的 增加cumulative_visted  
        visited_new_node = (action != self.depot_index)  
        self.cumulative_visted[visited_new_node] += 1 
      
        
    def update_decision_vehicle(self, action: torch.Tensor): 
        depot_token = (action == self.depot_index).int() 
        depot_token_index = mask_to_index(depot_token.squeeze(1)) 
        self.Fleet_state[depot_token_index.squeeze(),self.decision_vehicle[depot_token_index].squeeze(),4] = 1 
        self.Fleet_state[:,:,3] = 0 
        self.decision_vehicle += depot_token 
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 3 ] = 1 
        
        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim = 1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,3).long()
        )
        self.vehicle_vector = torch.concat(
            (self.vehicle_vector,self.cumulative_visted/self.total_visted_reqeust) , dim=2 
        )
        return depot_token 
    
    def update_mask(self,action:torch.Tensor , reset=False, depot_token=None): 
        # 去掉了CVRP中需要看當下車輛容量的限制
        if reset : 
            depot_token = (action == self.depot_index).int() 
            self.depot_token_counter += depot_token 
            depot_need_mask = self.depot_token_counter == self.vehicle_num 
            mask_visited_node = torch.eq( self.mask_help , action.unsqueeze(1).expand_as(self.mask))
            self.mask = self.mask + mask_visited_node 
            self.mask[:,self.depot_index , :] = depot_need_mask 
            return self.mask 
        else : 
            self.depot_token_counter += depot_token 
            mask_depot = self.depot_token_counter == self.vehicle_num
            mask_visited_node = torch.eq( self.mask_help , action.unsqueeze(1).expand_as(self.mask))
            self.mask = self.mask + mask_visited_node 
            self.mask[:,self.depot_index,:] = mask_depot 
            return self.mask 
        
    """
        -- MTSP 版本中使用不到    
    
    def update_terminal(self, mask:torch.Tensor):
        terminal = mask.all(dim=1).squeeze() + self.sup_terminal_shape
        self.terminal_list.append(terminal) 
        mask[terminal] = self.sup_reuse_mask #已經結束互動的那些batch , mask改成reuse_mask 
        return mask 

    def get_terminal_mask_route(self): 
        terminal_mask = torch.stack(self.terminal_list , dim=1) 
        terminal_mask[: , -1] = False  
        return (1-terminal_mask.int()).bool() 
    def get_terminal_mask_log_probs(self): 
        terminal_mask = torch.stack(self.terminal_list , dim=1) 
        terminal_mask = terminal_mask[: , :-1] 
        return (1-terminal_mask.int()).bool() 
    
    def get_effective_route(self): 
        path_log_mask = self.get_terminal_mask_route() 
        effective_route = list() 
        for batch_idx in range(self.batch_size): 
            effective_route.append(self.path_log[batch_idx][path_log_mask[batch_idx]])
        return effective_route

    def get_fulfill_rate(self): 
        return self.batchwise_fulfill_rate.mean() 
    
    
    def get_terminal_state(self):
        return self.terminal_list 
    """
    def get_initilize_Batch(self): 
        return self.batch 

    def get_SplitedRoute(self): 
        batchwise_SplitedRoute = list() 
        for batch_idx in range(self.batch_size): batchwise_SplitedRoute.append(self.path_log_split(self.path_log[batch_idx]))
        return batchwise_SplitedRoute
    
    def path_log_split(self,path): 
        splited_route , temp = [] , [] 
        for node in path : 
            if node == self.depot_index: 
                temp.append(node) 
                splited_route.append(temp) 
                temp=[] 
            else : temp.append(node) 
        assert len(splited_route) == self.vehicle_num 
        return splited_route 

    def calculate_splited_tour_length(self,batch_idx , splited_route): 
        path_distance_cost = torch.zeros(self.vehicle_num)
        assert len(splited_route) == self.vehicle_num 
        with torch.no_grad(): 
            for vehicle , path in enumerate(splited_route): 
                init_x , init_y = self.init_pos[batch_idx][vehicle] 
                first_x , first_y  = self.static_state[batch_idx][path[0]]
                distance = self.batch.node_route_cost[batch_idx][path[:-1],path[1:]] 
                path_distance_cost[vehicle] = ( torch.sum(distance) + sqrt(  pow((init_x-first_x),2) + pow((init_y-first_y),2)  )) 
        return path_distance_cost    
    
    
    """ 
        MTSP版本中 training 和 inference所使用的環境reward function都一樣 , 故都只保留Training mode (PMPO與普通)
    """
    def reward_training(self ) : 
        batchwise_objective_value = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device)
        for batch_idx in range(self.batch_size): 
            splited_route = self.path_log_split(self.path_log[batch_idx])
            distance_of_All_vehicle = self.calculate_splited_tour_length(batch_idx=batch_idx , splited_route=splited_route)
            objective_value = torch.max(distance_of_All_vehicle) 
            batchwise_objective_value[batch_idx] = objective_value
        return batchwise_objective_value 
    
    """
    
    def reward_inference(self) :   
        batchwise_objective_value = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device)
        for batch_idx in range(self.batch_size): 
            splited_route = self.path_log_split(self.path_log[batch_idx])
            distance_of_All_vehicle = self.calculate_splited_tour_length(batch_idx=batch_idx , splited_route=splited_route)
            minSum_tour = torch.sum(distance_of_All_vehicle) 
            batchwise_objective_value[batch_idx] = minSum_tour
        return batchwise_objective_value 
    """ 
    
    def calculate_PMPO(self , batchwise_objective_value , mode="mean"):
        assert self.batch_size % 8 == 0 , "PMPO need batch size can divided by 8 "
        mean = True if mode == "mean" else False 
        PMPO_reward = torch.zeros(size=(self.batch_size ,1 ) , dtype=torch.float32 , device=self.device)
        if mean : 
            for i in range(self.batch_size //8 ): 
                PMPO_reward[ i*8 : (i+1)*8   ] = torch.mean(batchwise_objective_value[i*8 : (i+1)*8]) 
        else :
            for i in range(self.batch_size //8 ): 
                PMPO_reward[ i*8 : (i+1)*8   ] = torch.min(batchwise_objective_value[i*8 : (i+1)*8]) 
        return PMPO_reward
    
    """
    def calculate_PMPO_inference(self,batchwise_objective_value):   
        assert self.batch_size % 8 == 0 , "PMPO need batch size can divided by 8 "         
        PMPO_reward = torch.zeros(size=(self.batch_size ,1 ) , dtype=torch.float32 , device=self.device)
        PMPO_fulfill = torch.zeros(size=(self.batch_size ,1 ) , dtype=torch.float32 , device=self.device)
        score = torch.exp( 5*(1-self.batchwise_fulfill_rate)  ) * ( batchwise_objective_value + self.penality_coeifficient)
        for i in range(self.batch_size //8 ): 
            ret , index = torch.min(score[i*8 : (i+1)*8] , dim=0)  # fine the index of best criterion score !
            PMPO_reward[i*8 : (i+1)*8 ] = batchwise_objective_value[ (i*8)+index ]
            PMPO_fulfill[i*8 : (i+1)*8 ] = self.batchwise_fulfill_rate[(i*8)+index]
        return PMPO_reward, PMPO_fulfill
    """


    def Calculate_Objective_byDist(self,Dist_matrix , best_index=None):

        batchwise_splite_route = self.get_SplitedRoute()
        best_minMax = float("inf")

        if isinstance(best_index , torch.Tensor) : 
            best_index = best_index.item() 
            splited_route = batchwise_splite_route[best_index]
            assert len(splited_route) == self.vehicle_num 
            max_vehicle = float("-inf")
            for vehicle , path in enumerate(splited_route): 
                vehicle_distance = Dist_matrix[0][path[0]]
                for step in range(len(path)-1): 
                    vehicle_distance += Dist_matrix[path[step]][path[step+1]]
                max_vehicle = max(max_vehicle , vehicle_distance) 

            best_minMax = max_vehicle

        else: 
            
            for batch_idx in range(self.batch_size): 
                splited_route = batchwise_splite_route[batch_idx]
                assert len(splited_route) == self.vehicle_num 
                max_vehicle = float("-inf")
                for vehicle , path in enumerate(splited_route): 
                    vehicle_distance = Dist_matrix[0][path[0]]
                    for step in range(len(path)-1): 
                        vehicle_distance += Dist_matrix[path[step]][path[step+1]]
                    max_vehicle = max(max_vehicle , vehicle_distance) 
                
                if max_vehicle < best_minMax : 
                    best_minMax = max_vehicle 
                    best_index = batch_idx
                
        return best_minMax , best_index 