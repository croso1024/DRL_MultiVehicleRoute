"""
    2023-02-20 
    完整板的TSP資料產生 , 將node資訊在生成時直接產生取代使用固定json file , 
    省下reindex麻煩和潛在的錯誤可能 , 用於Reinforcement learning training 
    
    2023-03-10 
    加入PMPO的資料產生 , 新增包括
    1. PMPO_node_argument : 用來產生變化後的節點群 
    2. getInstance_PMPO_batch : 利用變化後的節點群產出包含8筆graph-data的list 
    3. getInstance_BatchPMPO_MultiProcess : 輸入8的倍數的batchsize , 產出指定數量的 ,由8-argument data組成的batch ,
    4. Instance_to_PMPO_batch : 將一個instance轉為8個PMPO變化並傳回包含這8筆graph-data的list
    
    2023-03-21 
    擴展為CVRP data generator , 主要是加入node demand的維度
    2023-04-04 
    加入PMPO single process / validation single process /dataset_to_PMPO_singleProcess 的版本 , 嘗試先避開knn卡死問題
"""

import numpy as np 
# from numpy.random import uniform
from random import uniform ,betavariate , randint , normalvariate,random 
from torch_geometric.data import Data , Batch  
from torch_geometric.transforms import Distance
from torch_geometric.loader import DataLoader 
import torch 
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Pool 
from time import time 
from math import sqrt
from tqdm import tqdm 



class Instance(Data): 
    def __cat_dim__(self, key: str, value, *args, **kwargs) :
        if key in ['y' , 'node_route_cost' , 'node_route_cost_std'] : return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 

class Node(object): 
    
    def __init__(self,x,y): 
        self.x = x 
        self.y = y 
    def node_attribute(self)->tuple: 
        return (self.x , self.y) 

class Demand_Node(object): 
    
    def __init__(self,x,y , demand=None): 
        self.x =x 
        self.y=y 
        #self.demand =  max(0.01 ,(betavariate(alpha =2,beta=5) /4) ) if demand == None else demand 
        self.demand = max(0.01,normalvariate(mu = 0.07 ,sigma=0.02)) if demand == None else demand 
    def node_attribute(self)->tuple : 
        return (self.x,self.y,self.demand)
        

    

class DSVRP_DataGenerator: 
    
    def __init__(self,workers=1 ,batch_size=32 ,node_num=20 , mode='training') :
        #print(f"Initialize the TSP Data generator \n-- {workers} workers\n--batch_size:{batch_size} \n--node_num:{node_num}") 
              
        self.workers = workers 
        self.batch_size = batch_size 
        self.node_num = node_num 
    ########## Basic function  ##################
    def getInstance(self,ret=None):
        # step1 . Generate the node data on the fly 

        # Distribution_sample = random() 
        # if Distribution_sample > 0.15 : 
        #     nodes =[ Demand_Node(uniform(0,1),uniform(0,1))  for i in range(self.node_num) ] 
        # else : 
        #     nodes = self.RandomClusteringNode()
        nodes = [ Demand_Node(uniform(0,1),uniform(0,1))  for i in range(self.node_num) ] 

        # step2 . Get the node features 
        node_features = self.node_feature(nodes) 
        Distance_matrix ,Stochastic_Distance_matrix ,edge_index , edge_attr = self.node_DistanceMatrix(nodes) 
        # print(f"Debug Distance matrix :\n{Distance_matrix}\n")
        return  Instance(
            x = torch.tensor(node_features,dtype=torch.float32) ,
            node_route_cost = torch.tensor(Distance_matrix,dtype=torch.float32) ,
            node_route_cost_std = torch.tensor(Stochastic_Distance_matrix , dtype= torch.float32) , 
            edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous() , 
            edge_attr = torch.tensor(edge_attr , dtype =torch.float32) 
        )
        

    
    def node_feature(self,nodes:Node):
        node_feature = np.zeros( (self.node_num,3) , dtype=np.float32 )
        for ith , node in enumerate(nodes):   
            node_feature[ith] = node.node_attribute()
        # print(f"Debug  node feature : {node_feature}")
        return node_feature 
    
    def node_DistanceMatrix(self,nodes): 
        # As the input of the ortools , only include the mean distance 
        Distance_matrix = np.zeros( (self.node_num,self.node_num) , dtype=np.float32 )
        # Sampled from mean distance & std , use to calculate the cost
        Stochastic_Distance_matrix = np.zeros((self.node_num  , self.node_num) , dtype=np.float32) 
        # Randomness matrix 
        #randomness_factor_matrix = np.random.uniform(low=-0.05, high=0.25, size=(self.node_num, self.node_num)) 
        randomness_factor_matrix = np.random.uniform(low=0.05, high=0.25, size=(self.node_num, self.node_num)) 
        randomness_factor_matrix = (randomness_factor_matrix + randomness_factor_matrix.T) / 2
        
        edge_index ,edge_attr = list() , list()
        
        for u , src_node in enumerate(nodes) : 

            for v , dst_node in enumerate(nodes) : 
                # determinstic distance between 2 node 
                distance = DSVRP_DataGenerator.distance_calculate(src_node,dst_node)
                # after sampled with some randomness 
                # maximum standard deviation is 0.2 * distance 
                randomness_factor = randomness_factor_matrix[u][v]
                stochastic_distance = max(normalvariate(mu=distance , sigma= randomness_factor * distance) , 0.75*distance)
                #stochastic_distance  = distance*(1+randomness_factor)
                Distance_matrix[u][v] = distance
                Stochastic_Distance_matrix[u][v] = stochastic_distance
                edge_index.append((u,v))
                # edge_attr.append(distance)
                # only provide the randomness factor to the model 
                edge_attr.append(randomness_factor)


        #print(f"Debug : Distance matrix : {Distance_matrix}")                
  
        #print(f"Debug : Stochastic Distance matrix : {Stochastic_Distance_matrix}")        
        # print(f"Debug edge index : {edge_index}")
        # print(f"Debug edge attr : {edge_attr}")
        return Distance_matrix ,Stochastic_Distance_matrix , edge_index , edge_attr
    
            
    @staticmethod 
    def distance_calculate(node1, node2): 
        return sqrt(  ((node1.x-node2.x)**2)  + ((node1.y-node2.y)**2)  )
    
    ########## Function for Multi Process ##################
    def getInstance_Batch(self,ret=None):
        return Batch.from_data_list([self.getInstance() for i in range(self.batch_size)]) 
        
    """ 
        2023-03-10 PMPO - instance generator , 8-x argumentation ,
        Use single thread to argument the node position  , 
        注意這個版本的PMPO , 雖然是單獨計算edge_index , edge_attr ,
        但本質的node順序是相同的 , 可能多一步打亂會更好 , 
        此function return一個長度為8的list , 裝著8筆data 
    """
    def getInstance_PMPO_batch(self,ret=None):
        # step1. Coordinate argumentation  --> 取得PMPO node list[i][j] i代表該node的變化 ,j為第幾個node
        PMPO_node_list = [ PMPO_node_argument( init_x = uniform(0,1) , init_y = uniform(0,1) ) for i in range(self.node_num) ] 
        # print(f"Debug PMPO node list : {len(PMPO_node_list)}")
        assert len(PMPO_node_list) == self.node_num and len(PMPO_node_list[0]) == 8  , "PMPO dimenstion error"

        PMPO_data = list()

        # Covert the PMPO node first dim to different node , second dim is argument
        PMPO_node_list = list(zip(*PMPO_node_list))
        # 0504 , use the first set of nodes to calculate the Stochastic-DM , edge_attr 
        Distance_matrix ,Stochastic_Distance_matrix , edge_index , edge_attr = self.node_DistanceMatrix(PMPO_node_list[0]) 
        
        for argument_th , nodes  in enumerate(PMPO_node_list): 
            # print(f"Debug nodes : {[node.coordinate() for node in nodes]}")
            node_feature = self.node_feature(nodes) 
            # 0504 - move this line outside the for-loop , cause those features are same in PMPO-dataset
            # Distance_matrix ,Stochastic_Distance_matrix , edge_index , edge_attr = self.node_DistanceMatrix(nodes) 
            PMPO_data.append( 
                Instance(
                    x = torch.tensor(node_feature , dtype=torch.float32) , 
                    node_route_cost = torch.tensor(Distance_matrix , dtype= torch.float32)  , 
                    node_route_cost_std = torch.tensor(Stochastic_Distance_matrix , dtype=torch.float32) , 
                    edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
                    edge_attr = torch.tensor(edge_attr , dtype=torch.float32) 
                )
            )
           
            
        assert len(PMPO_data) == 8 , "PMPO data dimension error "
        return PMPO_data
    
    def getInstance_Batch_MultiProcess(self,dataset_size=10):
        # dataset_size : How many batch per dataset  
        input_args = [i for i in range(dataset_size)] 
        pool = Pool(self.workers) 
        return pool.map(self.getInstance_Batch , input_args) 
    
    def getInstance_Batch_SingleProcess(self,dataset_size=10): 
        return [self.getInstance_Batch() for i in range(dataset_size)]
    
    
    def getValidationset_SingleProcess(self,dataset_size=10) :    
        dataset = [self.getInstance() for i in range(dataset_size)]
        return DataLoader(  dataset  , batch_size=self.batch_size , shuffle=False )
    
    
    def getValidationset_MultiProcess(self,dataset_size=10) :    
        input_args = [ i for i in range(dataset_size)] 
        pool =Pool(self.workers) 
        return DataLoader(pool.map(  self.getInstance , input_args  )  , batch_size=self.batch_size , shuffle=False )

    def getInstance_BatchPMPO_SingleProcess(self,dataset_size=10): 
        """ 
            2023-0404 , try to avoid the multiprocess stuck error cause by KNN -graph . 
        """
        assert self.batch_size % 8 == 0 , "batch size not meet the PMPO requirement !"
        # getInstance_PMPOBatch will return a list have 8-graph-data 
        dataset = list() 
        for i in range(dataset_size):  # create a single batch
            batch = list() 
            for j in range(self.batch_size//8) : 
                batch.extend( self.getInstance_PMPO_batch()  ) # 8-equivalent data * (self.batch_size//8) == self.batch_size
            dataset.append( Batch.from_data_list(batch) )
        return dataset
        
    def getInstance_BatchPMPO_MultiProcess(self,dataset_size=10) : 
        """ 
            假設batch size = 16 , dataset_size = 10  , 
            因為getInstance_PMPO_batch 一次是返回包含8組等效的graph-data  ,
            所以為了要湊 10個batch_size為16的batch , 需要call 10 * (batch_size//8) 次getInstance_PMPO_batch
            最後的output為list of Batch 
        """
        assert self.batch_size % 8 == 0 , " batch size not meet the PMPO !"
        input_args = [i for i in range(dataset_size*self.batch_size//8)]
        pool = Pool(self.workers) 
        PMPO_dataset = pool.map(self.getInstance_PMPO_batch , input_args)
        # PMPO_dataset : ( dataset_size*batch_size/8 ) 組 8筆等效data
        # print(f"Debug : {len(PMPO_dataset)}")
        dataset = list() 
        for i in range(dataset_size): 
            batch = list() 
            for data_list in PMPO_dataset[i*self.batch_size//8 : (i+1)*self.batch_size//8]: 
                batch.extend(data_list)
            dataset.append( Batch.from_data_list(batch) )

        return dataset 
                
    def Instance_to_PMPO(self, instance): 
        """ 
            Conver the single instance to 8-argument PMPO format , 
            利用instance的position 產出PMPO_node_list ,接著和getInstance_PMPO_batch內容相同

            注意因為是Capacity版本的Instance , 故instance.x 為node_num x 3 
        """
        # print(f"Debug node pos : {instance.x}")
        PMPO_node_list = [ PMPO_node_argument(init_x=x , init_y=y ,demand=demand) for x ,y ,demand in instance.x ]
        assert len(PMPO_node_list[0]) == 8 , "PMPO dimension error"
        PMPO_data = list() 
        # Covert the PMPO node first dim to different node , second dim is argument
        # [ [N1_Type1,N1_Type2...] ,[N2_Type1,N2_Type2...]  ]
        PMPO_node_list = list(zip(*PMPO_node_list))
        # [ [N1_Type1,N2_Type1 .. ] ,[N1_Type2,N2_Type2 .. ] ]
        """ 
            0504 - 發現說load-dataset時call instance to PMPO會造成std matrix和edge_attr被重新sample ,
            這個問題只在DSVRP要處理 , 這邊直接拿原始的std矩陣和edge_attr來建立instance即可
        """
        for argument_th , nodes  in enumerate(PMPO_node_list): 
            # print(f"Debug nodes : {[node.coordinate() for node in nodes]}")
            node_feature = self.node_feature(nodes) 
            Distance_matrix ,Stochastic_Distance_matrix , edge_index , edge_attr = self.node_DistanceMatrix(nodes) 
            PMPO_data.append( 
                Instance(
                    x = torch.tensor(node_feature , dtype=torch.float32) , 
                    node_route_cost = torch.tensor(Distance_matrix , dtype= torch.float32)  , 
                    #node_route_cost_std = torch.tensor(Stochastic_Distance_matrix , dtype=torch.float32),
                    node_route_cost_std = instance.node_route_cost_std.clone().detach() ,
                    edge_index = torch.tensor(edge_index , dtype=torch.long).t().contiguous() , 
                    #edge_attr = torch.tensor(edge_attr , dtype=torch.float32) 
                    edge_attr =instance.edge_attr.clone().detach() , 
                )
            )
        
            
        assert len(PMPO_data) == 8 , "PMPO data dimension error "
        # 回傳一個list , 內部有8組等效的Graph-data 
        return PMPO_data
    
    
    def dataset_to_PMPOdataset_SingleProcess(self,validation_set,maximum_batch_size=None): 
        PMPO_dataset = list() 
        if not maximum_batch_size : 
            for batch_data in tqdm(validation_set): 
                instance_list = batch_data.to_data_list() # 包含了batch_size筆data , 我們要拿出每一筆data去轉成PMPO-batch
                for i in range(8): 
                    PMPO_batch = list() 
                    for instance in instance_list[ i*(self.batch_size//8) : (i+1)*(self.batch_size//8)  ]  : 
                        PMPO_batch.extend(self.Instance_to_PMPO(instance))
                    PMPO_dataset.append( Batch.from_data_list(PMPO_batch ) )
        else:             
            # 0413 -add maximum batch-size setting , 
            # 原始的作法是將一個個graph-data拿出來 , 組合回設定的batch_size , 這邊就是改成允許組成更大的batch
            # 這一行確保maximum_batch_size不能過大 , 否則會打亂charater_set 
            assert maximum_batch_size <= self.batch_size * 8 , "maximum batch-size is  8 * batch_size ! ,to avoid VCVV error"
            assert type(maximum_batch_size) == int and maximum_batch_size % 8 == 0 , "maximum batch-size not match the PMPO"
            if self.batch_size * 8 <= maximum_batch_size:  #原始batch * 8 還小於maximum --> 直接弄成一個大batch 
                for batch_data in tqdm(validation_set): 
                    instance_list = batch_data.to_data_list()  
                    PMPO_batch = list() 
                    for instance in instance_list: PMPO_batch.extend(self.Instance_to_PMPO(instance))
                    PMPO_dataset.append(Batch.from_data_list(PMPO_batch))
            else : 
                segament_size = (self.batch_size*8) // maximum_batch_size
                for batch_data in tqdm(validation_set): 
                    instance_list = batch_data.to_data_list() 
                    for j in tqdm(range(segament_size)): 
                        PMPO_batch = list() 
                        for instance in instance_list[  j * (self.batch_size//segament_size) : (j+1)*(self.batch_size//segament_size)    ]: 
                            PMPO_batch.extend(self.Instance_to_PMPO(instance)) 
                        PMPO_dataset.append(Batch.from_data_list(PMPO_batch))
                    
        return PMPO_dataset 
    
    def dataset_to_PMPOdataset(self,validation_set): 
        # 每個batch_data裝有batch_size筆data , 
        # 而每次我們需要取batch_size//8個instance , 透過轉PMPO後回到batch_size 成為新的batch 
        # 假設validation set為64 , batch為16  ,一共放著4個batch , 經過以下的處理後
        # 會變成32個batch , 直覺上的資料量變為8倍 
        # 在4個worker的狀況下 , 最好 dataset_size % (4*batch_size) = 0 比較好 
        #-------------------------------------
        PMPO_dataset = list()
        pool = Pool(self.workers) 
        
        for batch_data in validation_set: 
            instance_list = batch_data.to_data_list() 
            for i in tqdm(range(8)): 
                PMPO_batch = pool.map( self.Instance_to_PMPO ,instance_list[ i*(self.batch_size//8)  : (i+1)*(self.batch_size//8) ] )
                # PMPO_batch [ [8x graph-data] x self.batch_size//8   ]
                #PMPO_batch = [ instance for i in range(8)  for instance in PMPO_batch[i] ]
                PMPO_batch = [ instance[i] for instance in PMPO_batch for i in range(8) ]
                # PMPO_batch = [instance for instance in PMPO_batch[i] for i in range(len(PMPO_batch))]
                PMPO_dataset.append(Batch.from_data_list(PMPO_batch))
        
        return PMPO_dataset
    
    def RandomClusteringNode(self,center_var=0.07 ): 
        nodes = [] 
        centers = [] 
        number_of_clustering = randint(1, self.node_num//7)
        #print(f"number_of_clustering:{number_of_clustering}")
        # 先產生節點群的中心(注意他們不是實際的節點) 
        for i in range( number_of_clustering ) : 
            x,y = uniform(0.2,0.8) , uniform(0.2,0.8)
            #print(f"Center : {x},{y}")
            centers.append((x,y)) 
        for i in range(self.node_num) : 
            cluster_ith = randint(0 , number_of_clustering-1)
            node_x = normalvariate( centers[cluster_ith][0] , center_var )
            node_y = normalvariate( centers[cluster_ith][1] , center_var )
            node = Demand_Node(node_x , node_y) 
            nodes.append(node) 
        return nodes 
    
def PMPO_node_argument(init_x , init_y , demand=None ):  
    
    node_1 = Demand_Node(init_x , init_y , demand)
    node_2 = Demand_Node(init_y , init_x , demand) 

    node_3 = Demand_Node(1-init_x , init_y, demand)
    node_4 = Demand_Node(init_x , 1-init_y, demand) 

    node_5 = Demand_Node(init_y , 1-init_x, demand)
    node_6 = Demand_Node(1-init_y , init_x, demand)
    
    node_7 = Demand_Node(1-init_x , 1-init_y, demand)
    node_8 = Demand_Node(1-init_y , 1-init_x, demand)
    
    return [node_1 , node_2 , node_3 , node_4 ,node_5 , node_6 , node_7 , node_8 ]
