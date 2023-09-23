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
2023-04-15  
    發現會有depot-token-index異常造成車輛同時有on-decision和complete decision , 加入1行確保正在決策的車輛不會同時有complete-decision的簡單修復
2023-04-18 
    調整reset的順序 , 把mask更新移動到vehicle-vector後 , 在mask reset也加入capacity limit , 
    用來處理HVRP instance裡面有時候剛開始第一台車就無法服務某些節點的狀況
    
"""
import torch 
from torch_geometric.utils import to_dense_batch , mask_to_index 
from torch_geometric.data import Batch 
from utils.KNN_transform import KNNGraph_transformer , RadiusGraph_transformer
from random import randint  , random  
from math import sqrt ,exp


class HCVRP_Environment : 
    def __init__(   self , 
                    batch_size , 
                    node_nums , 
                    batch_data , 
                    vehicle_num = 1 , 
                    vehicle_capacity : list = [1,1],
                    vehicle_velocity : list = [1,1],
                    vehicle_pos_mode : str or list = "Depot" ,  
                    StateEQ = "PMPO" ,
                    DE_transform_type = 8, 
                    training = False , 
                    graph_transform : dict = None ,
                    device = 'cpu' , 
                ): 
        self.batch_size = batch_size     
        self.num_nodes = node_nums 
        self.vehicle_num = vehicle_num 
        self.vehicle_pos_mode = vehicle_pos_mode
        self.path_length = node_nums + vehicle_num - 1 
        self.stateEQ_mode = StateEQ
        self.DE_transform_type = DE_transform_type
        # setup capacity vector 
        if type(vehicle_capacity) == list and len(vehicle_capacity) == self.vehicle_num : 
            if training : raise RuntimeError("list charateristic only support inference mode")
            self.vehicle_capacity = vehicle_capacity
        else: 
            if not training : raise RuntimeError("Need specify the vehicle charateristic except the inference")
   
        # setyp the vehicle velocity 
        if type(vehicle_velocity) == list and len(vehicle_velocity) == self.vehicle_num: 
            if training : raise RuntimeError("list charateristic only support inference mode")
            self.vehicle_velocity = vehicle_velocity
        else : 
            if not training : raise RuntimeError("Need specify the vehicle charateristic except the inference")

        # setup the criterion 
        if training : 
            self.training = True 
            self.reward_func = self.reward_training  
        else : 
            self.training = False 
            self.reward_func = self.reward_inference  
        self.device = device
        
        # Graph-state = static state + dynamic state 
        # [ pos_x , pos_y , demand , depot , have_request ]
        if self.stateEQ_mode == "DE" : 
            self.batch = self.DecisionEquivalentTransform(batch_data)
            self.usePMPO = False
        elif self.stateEQ_mode == "mix": 
            assert self.batch_size % (DE_transform_type*8) == 0 ,"Batch size ERROR with mix mode"
            self.batch = self.DecisionEquivalentTransform(batch_data)
            self.usePMPO = True 
        elif self.stateEQ_mode == "PMPO": 
            self.batch = batch_data 
            self.usePMPO = True 
        else: 
            self.batch = batch_data 
            self.usePMPO = False
            
        if graph_transform : 
            if graph_transform["transform"]   == "knn": 
                knn_graph = KNNGraph_transformer(k=graph_transform["value"] , self_loop=False , num_workers=4)
                self.batch = knn_graph.batch_transform(self.batch) 
            elif graph_transform["transform"] == "radius" : 
                radius_graph = RadiusGraph_transformer(r=graph_transform["value"] , self_loop=False, num_workers=4)
                self.batch = radius_graph.batch_transform(self.batch) 
            else : raise RuntimeError("Graph Transformer error")
        
        # setup the stateEQ reward function 
        if self.stateEQ_mode == "DE" : 
            self.StateEQ_reward_function = self.calculate_DE_training
        elif self.stateEQ_mode == "PMPO": 
            self.StateEQ_reward_function = self.calculate_PMPO_training
        elif self.stateEQ_mode == "mix": 
            self.StateEQ_reward_function = self.calculate_mix_training

        self.static_state , ret = to_dense_batch(self.batch.x , self.batch.batch) 
        self.dynamic_state = torch.zeros(size=(self.batch_size , self.num_nodes ,2) , dtype=torch.float32 ,device=self.device)
        # Fleet [pos_x , pos_y , remaining ,vehicle_velocity ,vehicle-number(by cumulative capacity), on-decision , complete-decision ]
        # Vehicle [pos_x , pos_y , remaining ,vehicle_velocity , vehicle-number(by cumulative capacity) ,cumulative-percentage ]
        self.Fleet_state = torch.zeros(size=(self.batch_size , self.vehicle_num, 7) , dtype=torch.float32, device=self.device)
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
        # FulFill rate 
        self.total_demand : torch.Tensor
        # self.total_capacity = torch.ones(size=(self.batch_size,1,1),dtype=torch.float32 ,device=self.device) * sum(self.vehicle_capacity)
        self.total_capacity = torch.Tensor
        self.cumulative_capacity = torch.zeros(size=(self.batch_size,1,1),dtype=torch.float32 , device=self.device)
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
        # let the demand of depot = 0 
        self.static_state[:,self.depot_index,2] = 0 
        self.dynamic_state[:,1:,1] = 1  #除了depot index都把have request開為1
        # self.dynamic_state[:,self.depot_index,1] = 0 
        self.dynamic_state[:,self.depot_index,0] = 1 
        self.total_demand = torch.sum(self.static_state[:,:,2] , dim=1 , keepdim=True)     
        if self.vehicle_pos_mode == "Depot": 
            self.init_pos = torch.gather(
                self.static_state , dim=1 , index = (
                    torch.zeros(size=(self.batch_size,self.vehicle_num,2) , dtype=torch.long ,device=self.device) 
                )
            )
        else : raise RuntimeError("Only Depot mode in HVRP")
            
        self.decision_vehicle = torch.zeros(size=(self.batch_size,1),dtype=torch.long,device=self.device)
        self.Fleet_state[:,:,:2] = self.init_pos.clone()   
       
        if self.stateEQ_mode == "mix" : random_num = 8 * self.DE_transform_type  # only batch//(8*DEtype) independant data 
        elif self.usePMPO :random_num = 8                                                        # only batch//8 independant data 
        elif self.stateEQ_mode=="DE" : random_num = self.DE_transform_type                       # only batch // DEtype independant data 
        else : random_num = 1                                                                    # every datas are independant

        
        if self.training : 
            
            random_capacity , random_velocity  = self.RandomCharateristic( sample_size=  self.batch_size // random_num )
            
            
            
            # PMPO first , then DE 
            if self.stateEQ_mode == "mix": 
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([random_num for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
                random_velocity = torch.repeat_interleave(
                    input=random_velocity , repeats=torch.tensor([random_num for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device), dim=0 
                )
                
            elif self.stateEQ_mode == "DE": 
                
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([self.DE_transform_type for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
                random_velocity = torch.repeat_interleave(
                    input=random_velocity , repeats=torch.tensor([self.DE_transform_type for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device), dim=0 
                )
            elif self.stateEQ_mode == "PMPO": 
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([8 for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
                random_velocity = torch.repeat_interleave(
                    input=random_velocity , repeats=torch.tensor([8 for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device), dim=0 
                )
                
            self.total_capacity = torch.sum(random_capacity , dim=2 , keepdim=True)
            self.Fleet_state[self.sup_batchwise_indexing,:,2] = random_capacity
            self.Fleet_state[self.sup_batchwise_indexing,:,3] = random_velocity

        else : 
            # 設定上在validation的時候使用的vehicle-charateristic set在同一個batch都是固定的,所以才用普通的sum
            self.total_capacity = torch.ones(size=(self.batch_size,1,1),dtype=torch.float32 ,device=self.device) * sum(self.vehicle_capacity)
            self.Fleet_state[:,:,2] = torch.tensor(self.vehicle_capacity ,dtype=torch.float32,device=self.device)
            self.Fleet_state[:,:,3] = torch.tensor(self.vehicle_velocity ,dtype=torch.float32,device=self.device)


        if self.stateEQ_mode in ["DE","mix"] :
            self.Fleet_state = self.Fleet_state[self.sup_batchwise_indexing,self.permutation_tensor,:]

            
            
        # print(f"Debug after update the Fleet-state : \n{self.Fleet_state}")
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 5 ] = 1 
        cumulative_capacity =  torch.cumsum( self.Fleet_state[:,:,2] , dim=1  )
        self.Fleet_state[:,:,4] = cumulative_capacity / self.total_capacity.squeeze(1)
        
        
        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim=1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,5)
        )
      
        self.vehicle_vector = torch.concat(
            (self.vehicle_vector , self.cumulative_capacity/self.total_capacity) , dim=2    
        )
        self.observation = torch.concat(
            (self.static_state , self.dynamic_state) , dim= 2 
        ).view(self.batch_size*self.num_nodes , -1)
        
        # 把mask移到最下 , 處理某些車輛在一開始就有無法運送的節點( capacity limit )
        mask = self.update_mask( (torch.ones(size=(self.batch_size ,1 ),device=self.device) * self.depot_index) , reset=True )
        self.sup_reuse_mask[self.depot_index] = 1 
        mask = self.update_terminal(mask)
        
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
        
        mask = self.update_terminal(mask) 
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
            index = action.unsqueeze(2).expand(size=(self.batch_size, 1, 3 )).long()
        )
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle,:2] = temp_state_graph_state[:,:,:2]
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle,2] -= temp_state_graph_state[:,:,2]
        self.cumulative_capacity  +=  temp_state_graph_state[:,:,2].unsqueeze(2)
        
    def update_decision_vehicle(self, action: torch.Tensor): 
        depot_token = (action == self.depot_index).int() 
        depot_token_index = mask_to_index(depot_token.squeeze(1)) 
        self.Fleet_state[depot_token_index.squeeze(),self.decision_vehicle[depot_token_index].squeeze(),6] = 1 
        self.Fleet_state[:,:,5] = 0 
        self.decision_vehicle += depot_token 
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 5 ] = 1 
        
        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim = 1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,5).long()
        )
        self.vehicle_vector = torch.concat(
            (self.vehicle_vector,self.cumulative_capacity/self.total_capacity) , dim=2 
        )
        return depot_token 
    
    def update_mask(self,action:torch.Tensor , reset=False, depot_token=None): 
        if reset : 
            depot_token = (action == self.depot_index).int() 
            self.depot_token_counter += depot_token 
            depot_need_mask = self.depot_token_counter == self.vehicle_num 
            mask_visited_node = torch.eq( self.mask_help , action.unsqueeze(1).expand_as(self.mask))

            vehicle_capacity = self.vehicle_vector[:,:,2].unsqueeze(1) 
            meet_constrain_node = self.static_state[self.sup_batchwise_indexing,:,2] > vehicle_capacity[self.sup_batchwise_indexing,:,0]
            
            self.mask = self.mask + mask_visited_node 
            self.mask[:,self.depot_index , :] = depot_need_mask 
            return self.mask  + meet_constrain_node.permute(0,2,1)
        else : 
            self.depot_token_counter += depot_token 
            mask_depot = self.depot_token_counter == self.vehicle_num
            mask_visited_node = torch.eq( self.mask_help , action.unsqueeze(1).expand_as(self.mask))
            
            vehicle_capacity = self.vehicle_vector[:,:,2].unsqueeze(1) 
            meet_constrain_node = self.static_state[self.sup_batchwise_indexing,:,2] > vehicle_capacity[self.sup_batchwise_indexing,:,0]

            self.mask = self.mask + mask_visited_node 
            self.mask[:,self.depot_index,:] = mask_depot 
            return self.mask + meet_constrain_node.permute(0,2,1)
    
    def update_terminal(self, mask:torch.Tensor):
        terminal = mask.all(dim=1).squeeze() + self.sup_terminal_shape
        self.terminal_list.append(terminal) 
        mask[terminal] = self.sup_reuse_mask 
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
        # TODO maybe can more efficiency  by tensor operator !? --> no , cause each sub-tensor have different size 
        path_log_mask = self.get_terminal_mask_route() 
        effective_route = list() 
        for batch_idx in range(self.batch_size): 
            effective_route.append(self.path_log[batch_idx][path_log_mask[batch_idx]])
        return effective_route

    def get_fulfill_rate(self): 
        return self.batchwise_fulfill_rate.mean() 
    
    
    def get_terminal_state(self):
        return self.terminal_list 
    
    def get_initilize_Batch(self): 
        return self.batch 

    def get_SplitedRoute(self): 
        batchwise_SplitedRoute = list() 
        for batch_idx in range(self.batch_size): batchwise_SplitedRoute.append(self.path_log_split(self.path_log[batch_idx]))
        return batchwise_SplitedRoute
    
    def get_DecisionSequence(self): 
        """
            只有使用在inference , 利用permutation tensor得到每一個batch的決策順序 + vehicle_charateristic來得到
            每個batch的路徑對應到的type   , 即由0-> v-1各自對應哪一個type
        """
        assert self.training == False , "get DS only support inference mode  !"
        # self.vehicle_capacity shape : vehicle_num --> 每一台車的屬性 , 整個batch共用
        type_dict, have_seen,i =dict(), set() , 0
        for charater in self.vehicle_capacity: 
            if charater not in have_seen :
                have_seen.add(charater)
                type_dict[charater] = i
                i += 1 
        # type_dict : {1:0.1 , 2:0.15 ... }  type:capacity pairs
        # self.permutation_tensor shape :  batch_size , vehicle_num --> 每一個batch的順序
        batchwise_decision_sequence = list() 
        # 除了DE,mix以外 , 因為我在模型inference的時候車輛set都是依照順序的,所以相當於permutation tensor是一個torch.arange 
        permutation_tensor = self.permutation_tensor if self.stateEQ_mode in ["DE","mix"] else torch.arange(self.vehicle_num).unsqueeze(0).expand(size=(self.batch_size , self.vehicle_num)) 
        for batch_idx , sequence in enumerate(permutation_tensor.tolist()): 
            decision_sequence = list() 
            for vehicle_ith in sequence:
                decision_sequence.append(
                    type_dict[ self.vehicle_capacity[vehicle_ith] ] 
                    )
            batchwise_decision_sequence.append(decision_sequence)
        # type dict以capacity為key , 代表每一個capacity對應哪一個type的車輛
        # batchwise_decision_sequence : batch x vehicle_num , 注意這是雙層迴圈 , 即每一個batch決策的順序以type表示 , 
        # print(f"Debug type dict : {type_dict}")
        # print(f"permutation tensor : {permutation_tensor}")
        # print(f"Debug batchwise -decision sequence : {batchwise_decision_sequence}")
        # print(f"--- \n\n --- ")
        return type_dict , batchwise_decision_sequence
        
        
        
    
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

    def get_ModelRoute(self,index): 
        """ 
            use for plot the node-edge diagram 
        """
        vehicle_routes = self.get_effective_route()[index]
        vehicle_routes = self.path_log_split(vehicle_routes) 
        assert len(vehicle_routes) == self.vehicle_num , "Route does not match the number of vehicle"
        Model_Predict_route = [None] * self.vehicle_num 
        # 在DE , mix模式需要用permutation-tensor做校正 , 取出vehicle-routes對應的部份依序排入Model_Predict_Route
        if self.stateEQ_mode in ["DE" , "mix"] : 
            
            permutation_reference = self.permutation_tensor[index].squeeze() 
            for i , path in enumerate(vehicle_routes): 
                # print(f"The {i}-th route is vehicle {permutation_reference[i]}")
                # 對照DE,第i條路徑是由 self.permutation_tensor[i] 車輛所走
                Model_Predict_route[permutation_reference[i]] = [node.item() for node in path]        
        else : 
            for i , path in enumerate(vehicle_routes): 
                Model_Predict_route[i] = [node.item() for node in path]
        return Model_Predict_route

    def calculate_splited_tour_length(self,batch_idx , splited_route): 
        path_distance_cost = torch.zeros(self.vehicle_num)
        assert len(splited_route) == self.vehicle_num 
        with torch.no_grad(): 
            for vehicle , path in enumerate(splited_route): 
                init_x , init_y = self.init_pos[batch_idx][vehicle] 
                first_x , first_y , ret = self.static_state[batch_idx][path[0]]
                distance = self.batch.node_route_cost[batch_idx][path[:-1],path[1:]] 
                # path_distance_cost[vehicle] = ( torch.sum(distance) + sqrt(  pow((init_x-first_x),2) + pow((init_y-first_y),2)  )) / self.vehicle_velocity[vehicle]
                path_distance_cost[vehicle] = ( torch.sum(distance) + sqrt(  pow((init_x-first_x),2) + pow((init_y-first_y),2)  )) / self.Fleet_state[batch_idx,vehicle,3]
        return path_distance_cost    
    
    def calculate_fulfill_rate(self,batch_idx,path): 
        return torch.sum(self.static_state[batch_idx,path,2]) / self.total_demand[batch_idx]
    
    def reward_training(self) : 
        effective_route = self.get_effective_route()
        batchwise_objective_value = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device)
        self.bachwise_fulfill_rate = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device )
        for batch_idx in range(self.batch_size): 
            fulfill_rate = self.calculate_fulfill_rate(batch_idx=batch_idx , path=effective_route[batch_idx])
            self.bachwise_fulfill_rate[batch_idx] = fulfill_rate
            splited_route = self.path_log_split(effective_route[batch_idx])
            distance_of_All_vehicle = self.calculate_splited_tour_length(batch_idx=batch_idx , splited_route=splited_route)
            minSum_tour = torch.sum(distance_of_All_vehicle) 
            batchwise_objective_value[batch_idx] = exp( 5*(1-fulfill_rate))* (minSum_tour + self.penality_coeifficient)
        return batchwise_objective_value 
    
    def reward_inference(self) :   
        effective_route = self.get_effective_route()
        batchwise_objective_value = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device)
        self.batchwise_fulfill_rate = torch.zeros(size=(self.batch_size,1),dtype=torch.float32,device=self.device )
        for batch_idx in range(self.batch_size): 
            fulfill_rate = self.calculate_fulfill_rate(batch_idx=batch_idx,path=effective_route[batch_idx])
            self.batchwise_fulfill_rate[batch_idx] = fulfill_rate 
            splited_route = self.path_log_split(effective_route[batch_idx])
            distance_of_All_vehicle = self.calculate_splited_tour_length(batch_idx=batch_idx , splited_route=splited_route)
            minSum_tour = torch.sum(distance_of_All_vehicle) 
            batchwise_objective_value[batch_idx] = minSum_tour
        return batchwise_objective_value , self.batchwise_fulfill_rate
        
    
    def calculate_PMPO_training(self , batchwise_objective_value , mode="mean"):
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
    
    
    def DecisionEquivalentTransform(self,origin_batch:Batch):
        """ 
            Expand the origin batch to batch-size , 
            define the permutation tensor for decision-equivalent transform 
        """
        assert self.batch_size % self.DE_transform_type == 0 , "DE transform type not match !"
        origin_batch = origin_batch.to_data_list()      
        # 回到data list後的長度 = batch_size//DE-type , 也代表有batch—size//DE-type不同組data
        # 假設原batch size=16, DE-type =8 , 則origin batch回到data list會有2組data , 
        # 04-10 , 修改疊加順序 , 修正reward計算錯誤 
        # befix :  origin_batch = Batch.from_data_list(  origin_batch * self.DE_transform_type   )        
        
        origin_batch = Batch.from_data_list( [data for data in origin_batch for i in range(self.DE_transform_type)]                            )
        
        self.permutation_tensor = torch.stack(
            [ torch.randperm( self.vehicle_num ) for i in range(self.batch_size)  ] , dim=0
        ) 
        return origin_batch
    
        
    def calculate_DE_training(self,batchwise_objective_value , mode="mean"): 
        assert self.batch_size % self.DE_transform_type == 0 , "Calculate DE training reward error"    
        mean = True if mode == "mean" else False 
        DE_reward = torch.zeros(size=(self.batch_size,1) , dtype=torch.float32 , device=self.device)
        if mean : 
            for i in range(self.batch_size // self.DE_transform_type): 
                DE_reward[ i*self.DE_transform_type : (i+1)*self.DE_transform_type  ] = torch.mean(
                    batchwise_objective_value[ i*self.DE_transform_type : (i+1)*self.DE_transform_type  ]
                )
        else: 
            for i in range(self.batch_size // self.DE_transform_type): 
                DE_reward[ i*self.DE_transform_type : (i+1)*self.DE_transform_type  ] = torch.min(
                    batchwise_objective_value[ i*self.DE_transform_type : (i+1)*self.DE_transform_type  ]
                )
        return DE_reward 
    
    def calculate_DE_inference(self,batchwise_objective_value ): 
        assert self.batch_size % self.DE_transform_type == 0 , "Calculate DE training reward error"    
        DE_reward = torch.zeros(size=(self.batch_size,1) , dtype=torch.float32 , device=self.device)
        DE_fulfill = torch.zeros(size=(self.batch_size,1) , dtype=torch.float32 , device=self.device)
        score = torch.exp( 5*(1-self.batchwise_fulfill_rate)  ) * ( batchwise_objective_value + self.penality_coeifficient)
        for i in range(self.batch_size // self.DE_transform_type): 
            ret , index = torch.min(score[i*self.DE_transform_type : (i+1)*self.DE_transform_type] , dim=0)
            DE_reward[i*self.DE_transform_type : (i+1)*self.DE_transform_type] = batchwise_objective_value[ (i*self.DE_transform_type) + index ]
            DE_fulfill[i*self.DE_transform_type : (i+1)*self.DE_transform_type] = self.batchwise_fulfill_rate[ (i*self.DE_transform_type)+ index ]
        
        return DE_reward , DE_fulfill
    
    
    def calculate_mix_training(self,batchwise_objective_value , mode="mean"): 
        assert self.batch_size % (self.DE_transform_type*8) == 0 , "calculate mix training error"
        mean = True if mode == "mean" else False 
        mix_reward = torch.zeros(size=(self.batch_size, 1) ,dtype=torch.float32 , device=self.device)
        interval_size  = self.DE_transform_type * 8 
        if mean : 
            for i in range(self.batch_size // interval_size):  
                mix_reward[ i * interval_size : (i+1) * interval_size ] = torch.mean(
                    batchwise_objective_value[ i * interval_size : (i+1) * interval_size ] 
                )
        else : 
            for i in range(self.batch_size //  interval_size ): 
                mix_reward[ i * interval_size : (i+1) * interval_size ] = torch.min(
                    batchwise_objective_value[ i * interval_size : (i+1) * interval_size ] 
                ) 
        return mix_reward 

  
    def calculate_mix_inference(self,batchwise_objective_value ): 
        assert self.batch_size % (self.DE_transform_type*8) == 0 , "calculate mix training error"
        mix_reward = torch.zeros(size=(self.batch_size, 1) ,dtype=torch.float32 , device=self.device)
        mix_fulfill = torch.zeros(size=(self.batch_size,1),dtype=torch.float32 , device=self.device ) 
        interval_size  = self.DE_transform_type * 8 
        score = torch.exp( 5*(1-self.batchwise_fulfill_rate)  ) * ( batchwise_objective_value + self.penality_coeifficient)
        for i in range(self.batch_size // interval_size):  
            ret , index = torch.min(score[ i * interval_size : (i+1) * interval_size ] , dim=0)
            mix_reward[ i * interval_size : (i+1) * interval_size ] = batchwise_objective_value[ (i * interval_size)+index ]
            mix_fulfill[ i * interval_size : (i+1) * interval_size ] = self.batchwise_fulfill_rate[(i*interval_size)+index ]
        return mix_reward , mix_fulfill
    
    
    
    def Calculate_Objective_byDist(self,Dist_matrix , best_index=None): 
        """
            For testing on the Dataset , must be use the distance matrix to calculate the objective
        """
        # splited_route = self.get_SplitedRoute()[best_index]    

        #  get a list of effect path_log , len of list == batch_size
        effective_route = self.get_effective_route() 
        batchwise_splite_route = list() 
        for batch_idx in range(self.batch_size):
            batchwise_splite_route.append(self.path_log_split(effective_route[batch_idx]))
        
        if isinstance(best_index , torch.Tensor ) :
            best_index = best_index.item()
            splited_route = batchwise_splite_route[best_index]
            assert len(splited_route) == self.vehicle_num
            best_distance = 0 
            for vehicle , path in enumerate(splited_route): 
                vehicle_distance = Dist_matrix[0][path[0]]
                # vehicle_distance = 0 
                for step in range(len(path)-1) :  
                    vehicle_distance += Dist_matrix[path[step]][path[step+1]]  
                #print(f"{vehicle}-th Vehicle  V:{self.Fleet_state[best_index,vehicle,3]} ")
                best_distance += vehicle_distance / self.Fleet_state[best_index,vehicle,3]
        else : 
            best_score = float("inf")
            for batch_idx in range(self.batch_size):   
                splited_route = batchwise_splite_route[batch_idx] 
                assert len(splited_route) == self.vehicle_num  
                total_distance = 0 
                for vehicle , path in enumerate(splited_route): 
                    vehicle_distance = Dist_matrix[0][path[0]]
                    for step in range(len(path)-1):  vehicle_distance += Dist_matrix[path[step]][path[step+1]]
                    total_distance += vehicle_distance / self.Fleet_state[batch_idx , vehicle , 3]
                score = torch.exp( 5*(1-self.batchwise_fulfill_rate[batch_idx])  ) * ( total_distance + self.penality_coeifficient)
                
                if score < best_score: 
                    best_score = score
                    best_distance = total_distance 
                    best_index = batch_idx 
        # 檢查路徑
        # best_route = batchwise_splite_route[best_index]
        # best_route = [ node.item() for sub_route in best_route for node in sub_route  ] 
        # print(f"Debug visited route : {sorted(best_route)}") 
        # vistied_node = set(best_route)  
        # all_node = set(list(range(101)))
        
        
        return best_distance , best_index 


    def RandomCharateristic(self , sample_size ): 
        probs = torch.tensor( [ 0.35,0.25 , 0.25, 0.15   ])
        capacity_ranges = [(0.25, 0.45), (0.55, 0.65), (0.75, 0.85), (0.95, 1)]
        num_range = len(capacity_ranges)
        categorical_dist = torch.distributions.Categorical(probs = probs) 
        range_indices = categorical_dist.sample( (sample_size , self.vehicle_num)  )
        capacity = torch.zeros(size=(sample_size, self.vehicle_num) )
        for i in range(num_range) : 
            mask = range_indices == i 
            low , high = capacity_ranges[i]
            range_capacities = torch.rand( size=(sample_size  , self.vehicle_num)).mul(high-low).add(low)
            capacity = torch.where(mask , range_capacities , capacity)
        velocity =  ( (-2/3)  * capacity ) + (32/30)
        return capacity.unsqueeze(1).to(self.device) , velocity.unsqueeze(1).to(self.device)    
        