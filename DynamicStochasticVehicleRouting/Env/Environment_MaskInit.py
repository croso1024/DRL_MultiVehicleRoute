""" 
2023-04-25 Dynamic & Stochastic Version 
本版本主要是支援隨機起始位置的訓練 , 以及包含隨機路徑長度的reward計算 

2023-05-05 解決模型在隨機節點上但不會在第一步前往所在節點的問題 , 

        這一部份工作同時考慮"training-mode" 以及 "inference mode" ,但為了各種scheme容易使用 ,
        將原先在Training+Random pos mode時 , 使用的完全隨機節點更改為和 Inference+Random pos mode相同 , 使用編號1-(v+1)的節點. 
        
    -- scheme 1. 
        透過在reset的時候去mask掉 1~(vehicle_num+1)的節點  , 並且將path_log的長度減少vehicle-num個 , 最終在path_log中不會出現1-(v+1)的節點 
        , 但在tour_length計算上基本上不影響 , 仍是由init-pos前往path_log上下一個地點 
    -- scheme 2.
        另一種解法需要在每一次切車時 , 使用該車編號去找其對應節點 , 並直接添加到path_log,以及mask掉 , 麻煩在於經過DE後這件事情不太容易

    最終使用scheme 1. 修改部份如下: 
        1. path_log 長度更改
        2. 初始化init-pos現在不分training or inference 都是使用1-(v+1)節點
        3. update_mask在reset模式下 , 增加mask掉1-(v+1)的節點 
        4. reset時將1-(v+1)節點have request設置為0 
        5. (移除)每一台車的capacity在經過DE打亂順序前先減掉自己所在節點的demand (移除,會影響到cumsum, 而且這可以用假設已經完成取代)
        6. 計算fulfill_rate的部份加回1-(v+1)的demand , 修改在calculate_fulfill_rate的函式
        7. 修正計算tour-length在init->path[0] 沒有使用到帶std的matrix這個bug 
"""
import torch 
from torch_geometric.utils import to_dense_batch , mask_to_index 
from torch_geometric.data import Batch 
from utils.KNN_transform import KNNGraph_transformer , RadiusGraph_transformer
from random import randint  , random  
from math import sqrt ,exp

""" 
    Parameter setting: 
    1. Vehicle_capacity : 使用list指定必須搭配inference mode否則raise error , 其餘使用Normal distribution產生 
    2. vehicle_pos_mode: Depot為標準的所有車輛由depot出發 , Random在training時會產生完全隨機節點 , 在inference時是固定使用 編號1 ~ vehicle_num+1的節點 , coordinate mode尚未完成 

"""



class DSVRP_Environment : 
    def __init__(   self , 
                    batch_size , 
                    node_nums , 
                    batch_data , 
                    vehicle_num = 1 , 
                    vehicle_capacity : None or list = None,
                    vehicle_pos_mode : str or list = "Random" ,  
                    StateEQ = "PMPO" ,
                    DE_transform_type = 8, 
                    training = True , 
                    graph_transform: dict = None ,
                    device = 'cpu' , 
                ): 
        self.batch_size = batch_size     
        self.num_nodes = node_nums 
        self.vehicle_num = vehicle_num 
        self.vehicle_pos_mode = vehicle_pos_mode
        #self.path_length = node_nums + vehicle_num - 1 
        # 0505 -> mask out the 1~(v+1) node in the reset , so the length of the path log will decrease vehicle_num 
        self.path_length = node_nums - 1 
        self.stateEQ_mode = StateEQ
        self.DE_transform_type = DE_transform_type
        # setup capacity vector 
        if type(vehicle_capacity) == list and len(vehicle_capacity) == self.vehicle_num : 
            if training : raise RuntimeError("list charateristic only support inference mode")
            self.vehicle_capacity = vehicle_capacity
        else: 
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
            assert self.batch_size % (DE_transform_type*8) == 0 ,"Batch size ERROR when mix mode"
            self.batch = self.DecisionEquivalentTransform(batch_data)
            self.usePMPO = True 
        elif self.stateEQ_mode == "PMPO": 
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
            
        # setup the stateEQ reward function , &only use in training 
        if self.stateEQ_mode == "DE" : 
            self.StateEQ_reward_function = self.calculate_DE_training
        elif self.stateEQ_mode == "PMPO": 
            self.StateEQ_reward_function = self.calculate_PMPO_training
        elif self.stateEQ_mode == "mix": 
            self.StateEQ_reward_function = self.calculate_mix_training

        self.static_state , ret = to_dense_batch(self.batch.x , self.batch.batch) 
        self.dynamic_state = torch.zeros(size=(self.batch_size , self.num_nodes ,2) , dtype=torch.float32 ,device=self.device)
        # Fleet [pos_x , pos_y , remaining ,vehicle-number(by cumulative capacity), on-decision , complete-decision ]
        # Vehicle [pos_x , pos_y , remaining ,vehicle-number(by cumulative capacity) ,cumulative-percentage ]
        self.Fleet_state = torch.zeros(size=(self.batch_size , self.vehicle_num, 6) , dtype=torch.float32, device=self.device)
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
        # sup_reuse_mask : the fake mask use in the batch terminate 
        self.sup_reuse_mask = torch.zeros(size=(self.num_nodes,1),device=self.device).bool() 
        # FulFill rate 
        self.total_demand : torch.Tensor
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
        self.static_state[:,self.depot_index,2] = 0 
        #self.dynamic_state[:,1:,1] = 1  #除了depot index都把have request開為1
        # 0505 , No request in depot & 1~(vehicle+1) node 
        self.dynamic_state[:, self.vehicle_num+1: , 1 ] = 1 
        # self.dynamic_state[:,self.depot_index,1] = 0 
        self.dynamic_state[:,self.depot_index,0] = 1 
        self.total_demand = torch.sum(self.static_state[:,:,2] , dim=1 , keepdim=True)     
        
         
        if self.stateEQ_mode == "mix" : random_num = 8 * self.DE_transform_type  # only batch//(8*DEtype) independant data 
        elif self.usePMPO :random_num = 8                                                        # only batch//8 independant data 
        elif self.stateEQ_mode=="DE" : random_num = self.DE_transform_type                       # only batch // DEtype independant data 
        else : random_num = 1                                                                    # every datas are independant
        
        if self.vehicle_pos_mode == "Depot": 
            # 關閉掉Depot mode , 沒有測試過怕會有異常 
            raise RuntimeError("Not support depot mode for MaskInit environment")
            self.init_pos = torch.gather(
                self.static_state , dim=1 , index = (
                    torch.zeros(size=(self.batch_size,self.vehicle_num,2) , dtype=torch.long ,device=self.device) 
                )
            )
        elif self.vehicle_pos_mode == "Random": 
            """ 
            # Random + training-mode : 用在訓練的部份 , 完全隨機產生節點
            if self.training : 
                random_index = torch.repeat_interleave(
                    input = torch.randint(low=1,high=self.num_nodes ,
                                        size=(self.batch_size//random_num , self.vehicle_num),
                                        device=self.device) , 
                    repeats = torch.tensor( 
                        [random_num for i in range(int(self.batch_size//random_num))] , dtype=torch.long , 
                        device = self.device),
                    dim = 0 
                )
            """
            #  Random + inference-mode : 就固定使用 1 - (vehicle_num+1)的節點 , 用在validation
            #  0505 : 為了解決初始化位置問題 , 現在無論training or inference都使用 1-(vehicle_num+1)的節點
            random_index = torch.repeat_interleave(
                input = torch.arange(1,self.vehicle_num+1 , dtype=torch.long , device=self.device).unsqueeze(0)
                .expand(size=(self.batch_size//random_num , self.vehicle_num)),
                repeats = torch.tensor(
                    [random_num for i in range(int(self.batch_size//random_num))] , dtype=torch.long ,
                    device=self.device ),
                dim=0
            )
            
            self.init_pos = torch.gather(
                    self.static_state , dim = 1 , 
                    index = random_index.unsqueeze(2).expand(size=(self.batch_size , self.vehicle_num,2))
                )
            
        else : 
            #TODO 尚未完成, 即指定vehicle coordinate直接初始化 
            # 故這邊做完還需要從env-Test開始測試 , 應該可以仿照random的作法去expand後call gather
            raise RuntimeError("Specify the vehicle pos has not yet  complete !")
           
        self.decision_vehicle = torch.zeros(size=(self.batch_size,1),dtype=torch.long,device=self.device)
        self.Fleet_state[:,:,:2] = self.init_pos.clone()   
       

        if self.training : 
            # 最終的Random capacity shape : ( self.batch_size//random_num , 1 ,self.vehicle_num )
            # 這邊使用Gaussian來設定代表中間狀態的車隊
            random_capacity = torch.normal(size=(self.batch_size//random_num , self.vehicle_num) ,
                                           mean=0.75 , std=0.15 , dtype=torch.float32 , device=self.device)
            random_capacity = torch.clip(random_capacity,min=0 , max=1).unsqueeze(1)
            # print(f'Debug random capacity : {random_capacity} , {random_capacity.shape}')
            
            # PMPO first , then DE 
            if self.stateEQ_mode == "mix": 
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([random_num for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
            elif self.stateEQ_mode == "DE": 
                
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([self.DE_transform_type for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
            elif self.stateEQ_mode == "PMPO": 
                random_capacity = torch.repeat_interleave(
                    input=random_capacity , repeats=torch.tensor([8 for i in range(int(self.batch_size/random_num))],dtype=torch.long,device=self.device) , dim=0 
                )
     
            self.total_capacity = torch.sum(random_capacity , dim=2 , keepdim=True)
            self.Fleet_state[self.sup_batchwise_indexing,:,2] = random_capacity
    

        else : 
            # 在非training的時候 , 車輛的capacity都是一整個batch共用 , 所以這邊用一般的sum
            self.total_capacity = torch.ones(size=(self.batch_size,1,1),dtype=torch.float32 ,device=self.device) * sum(self.vehicle_capacity)
            self.Fleet_state[:,:,2] = torch.tensor(self.vehicle_capacity ,dtype=torch.float32,device=self.device)
        # 到這一步 , random initial-pos和capacity都已經被放到self.batch_size : , : 的fleet state中 , 且不同的EQ所對應到的同一台車有一樣的pos,capacity 
        # 設定完所有車輛相關的parameters, 打亂順序做DE
        #print(f"Debug before shuffle the DE : Fleet :\n{self.Fleet_state}\n Init-pos:\n{self.init_pos} ")
        if self.stateEQ_mode in ["DE","mix"] :

            self.Fleet_state = self.Fleet_state[self.sup_batchwise_indexing,self.premutation_tensor,:]
            self.init_pos = self.init_pos[self.sup_batchwise_indexing , self.premutation_tensor , :]
            
        #print(f"Debug after shuffle the DE : Fleet :\n{self.Fleet_state} \nInit-pos:\n{self.init_pos} ")
            
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 4 ] = 1 
        cumulative_capacity =  torch.cumsum( self.Fleet_state[:,:,2] , dim=1  )
        self.Fleet_state[:,:,3] = cumulative_capacity / self.total_capacity.squeeze(1)
        

        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim=1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,4)
        )
      
        self.vehicle_vector = torch.concat(
            (self.vehicle_vector , self.cumulative_capacity/self.total_capacity) , dim=2    
        )
        self.observation = torch.concat(
            (self.static_state , self.dynamic_state) , dim= 2 
        ).view(self.batch_size*self.num_nodes , -1)
        
        
        # mask 移動到下面的理由只有 , 或許一開始車輛所在的地點是無法完成的
        # -- 0505  , mask out the 1~vehicle+1 node in the reset 
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
        self.Fleet_state[depot_token_index.squeeze(),self.decision_vehicle[depot_token_index].squeeze(),5] = 1 
        self.Fleet_state[:,:,4] = 0 
        self.decision_vehicle += depot_token 
        self.Fleet_state[self.sup_batchwise_indexing , self.decision_vehicle , 4 ] = 1 
        
        self.vehicle_vector = torch.gather(
            self.Fleet_state , dim = 1 , index = self.decision_vehicle.unsqueeze(2).expand(self.batch_size,1,4).long()
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
            # 0505 , mask out the 1~vehicle_num node 
            self.mask[:,1:self.vehicle_num+1 , :] = True 
            
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
        assert self.training == False , "get DS only support inference mode "
        # self.vehicle_capacity shape : vehicle_num --> 每一台車的屬性 , 整個batch共用
        type_dict, have_seen,i =dict(), set() , 0
        for charater in self.vehicle_capacity: 
            if charater not in have_seen :
                have_seen.add(charater)
                type_dict[charater] = i
                i += 1 
        batchwise_decision_sequence = list() 
        permutation_tensor = self.premutation_tensor if self.stateEQ_mode in ["DE","mix"] else torch.arange(self.vehicle_num).unsqueeze(0).expand(size=(self.batch_size , self.vehicle_num)) 
        for batch_idx , sequence in enumerate(permutation_tensor.tolist()): 
            decision_sequence = list() 
            for vehicle_ith in sequence:
                decision_sequence.append(
                    type_dict[ self.vehicle_capacity[vehicle_ith] ] 
                    )
            batchwise_decision_sequence.append(decision_sequence)
        # 在DSVRP , capacity不像HVRP那樣有固定幾個set, 因此此處大部分情況會是每一個capacity都對應一個type
        # 因此batchwise_decision_sequence就可得到每一車的各自路徑 ,
        # TODO 到時候實做會需要搭配self.init_pos來提供給視覺化工具 ,
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

    def calculate_splited_tour_length(self,batch_idx , splited_route): 
        # Change the distance-matrix in graph-data to the node-route-cost-std ,
        path_distance_cost = torch.zeros(self.vehicle_num)
        assert len(splited_route) == self.vehicle_num 
        if self.stateEQ_mode in ["DE","mix"] : 
            with torch.no_grad(): 
                for vehicle , path in enumerate(splited_route): 
                    # get the init_node by the permutation tensor 
                    # ( +1, because vehicle 0 -> node 1 , torch.randperm(vehicle): 0~(vehicle-1) -> node 1 -> node-vehicle )
                    init_node = self.premutation_tensor[batch_idx][vehicle] + 1
                    distance = self.batch.node_route_cost_std[batch_idx][path[:-1],path[1:]] 
                    path_distance_cost[vehicle] = torch.sum(distance) + self.batch.node_route_cost_std[batch_idx][init_node][path[0]]
        else: 
            with torch.no_grad() : 
                for vehicle , path in enumerate(splited_route): 
                    init_node = vehicle+1 
                    distance = self.batch.node_route_cost_std[batch_idx][path[:-1],path[1:]] 
                    path_distance_cost[vehicle] = torch.sum(distance) + self.batch.node_route_cost_std[batch_idx][init_node][path[0]]

        return path_distance_cost    
    
    def calculate_fulfill_rate(self,batch_idx,path): 
        
        return  ( torch.sum(self.static_state[batch_idx, 1:self.vehicle_num+1 , 2]) +
                 torch.sum(self.static_state[batch_idx,path,2]) )/ self.total_demand[batch_idx]
            

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
        # 差別在於batchwise-objective-value的計算不同 , 以及會return fulfill
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
        origin_batch = Batch.from_data_list( [data for data in origin_batch for i in range(self.DE_transform_type)]                            )
        
        self.premutation_tensor = torch.stack(
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
    
"""
    def Calculate_Objective_byDist(self,Dist_matrix , best_index=None): 
        
            # For testing on the Dataset , must be use the distance matrix to calculate the objective
            # Only use in benchmark on MTSP , HVRP 
        
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
"""