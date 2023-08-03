""" 
    2023-05-13 實做用來繪製節點-路徑圖的功能 , 
    接受模型所輸出的每一台車輛對應的路線以及該instnace去繪製路線並儲存在特定資料夾內
"""


import networkx as nx 
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from utils.MTSPGenerator import MTSP_DataGenerator


class RouteVisulizer : 
    
    
    Color_map = ["red","orange","yellow","green","blue","indigo","purple","grey","peru","darksage"
                 ,"dodgerblue","crimson","sandybrown","darked","khaki","lime","royalblue","navy","darkslateblue"
                 ,"violet","orchid","teal","olive"]    
    node_size = 100
    lebel_font_size = 6
    edge_width = 2
    arrow_size =12 
    
    
    def __init__(self,node_num , vehicle_num  , method:str ,capacity:bool,
                 show:bool,store:bool ,store_path="./model/MultiTravelingSalesmanProblem/plot/"
                ): 
        self.node_num = node_num 
        self.vehicle_num = vehicle_num 
        self.method = method 
        self.show = show 
        self.store = store 
        self.store_path = store_path
        self.capacity = capacity
        
    """ 
        注意此版本的Visulizer在初始化的時候不使用batch_size , 換句話說所有的圖都是instance by instance 
    """
    def Visualize(self, fig_id , instance , routes , objective_value ): 

        if self.capacity : assert instance.x.shape[-1] == 3 ,"Node feature dimension is not 3 "
        else : assert instance.x.shape[-1] == 2 ,"Node feature dimension is not 2"
        assert len(routes) == self.vehicle_num , "Given routes number not match the vehicle-num"

        objective_text = f"objective : {round(objective_value,2)}" if objective_value else " "
        title = f"{self.method}-{self.node_num} Nodes & {self.vehicle_num} vehicles " + objective_text 
        
        if self.capacity : 
            node_pos_dict = {i:node.tolist()[:2] for i , node in enumerate(instance.x)}
            node_label =  {i:node.tolist()[2] for i , node in enumerate(instance.x)}
            
        else : 
            node_pos_dict = {i:node.tolist() for i , node in enumerate(instance.x)}
            node_label = {i:i for i in range(self.node_num)}
        # Start create base-graph 
        networkx_graph = to_networkx(instance , to_undirected= False , remove_self_loops= True) 
        
        
        
        node_color = "darkorange"        
        
        
         
        # plot other node 
        nx.draw_networkx_nodes(
            networkx_graph ,
            pos = {i:node_pos_dict[i] for i in range(1,self.node_num)} ,
            nodelist = list(range(1,self.node_num)) , 
            node_size = self.node_size ,
            node_color = node_color, 
            node_shape = "o",
        )
        
        # plot depot node
        nx.draw_networkx_nodes(
            networkx_graph ,
            pos = {0:node_pos_dict[0]} ,
            nodelist = [0],
            node_size = 3* self.node_size ,
            node_color = 'black', 
            node_shape = "s",
        )
        
        
        # add node number 
        #nx.draw_networkx_labels(
        #    networkx_graph , pos = node_pos_dict , font_size = self.lebel_font_size , labels = node_label
        #)
        
        legend_handles =[] 
        legend_labels = [] 
        for i  in range(self.vehicle_num): 
            legend_labels.append(f"Vehicle {i+1}")
            legend_handles.append(
                Line2D(
                    [],[] , color=self.Color_map[i], marker="." , linestyle="-" ,
                    markersize = 15 , linewidth= 2 ,
                )
            )
            plt.legend(handles=legend_handles , labels=legend_labels,loc="upper right")
        
        
        
        # draw the route planning result 
        for ith_vehicle_path , path in enumerate(routes): 
            # Add depot index in the begin of route. 
            path = [0] + [node.item() for node in path]  
            edge_list = [( path[i] ,path[i+1]  ) for i in range(len(path)-1)]
            nx.draw_networkx_edges( 
                networkx_graph , 
                pos = node_pos_dict , 
                edgelist = edge_list , 
                edge_color = self.Color_map[ith_vehicle_path] , 
                width = self.edge_width,
                arrows = True , 
                arrowsize = self.arrow_size , 
            )
            
        plt.title(title) 
        if self.store: 
            file_name = self.store_path + f"{self.method}/"+  str(fig_id) +f"V{self.vehicle_num}" +".png"
            #plt.savefig(file_name)
            print(f"Save fig : {file_name} ")
        if self.show :
            plt.show() 
        else : 
            plt.close()
        
        construct = True
        if construct: 
                # plot other node 
            nx.draw_networkx_nodes(
                networkx_graph ,
                pos = {i:node_pos_dict[i] for i in range(1,self.node_num)} ,
                nodelist = list(range(1,self.node_num)) , 
                node_size = self.node_size ,
                node_color = node_color, 
                node_shape = "o",
            )
            
            # plot depot node
            nx.draw_networkx_nodes(
                networkx_graph ,
                pos = {0:node_pos_dict[0]} ,
                nodelist = [0],
                node_size = 3* self.node_size ,
                node_color = 'black', 
                node_shape = "s",
            )
            
            # draw the route planning result 
            for ith_vehicle_path , path in enumerate(routes): 
                # Add depot index in the begin of route. 
                path = [0] + [node.item() for node in path]  
                for  i in range(len(path)-1) : 
                    e_temp = [(path[i],path[i+1])]
                    nx.draw_networkx_edges(
                        networkx_graph , 
                        pos = node_pos_dict , 
                        edgelist = e_temp , 
                        edge_color = self.Color_map[ith_vehicle_path] , 
                        width = self.edge_width,
                        arrows = True , 
                        arrowsize = self.arrow_size ,
                    )
                    file_name = self.store_path + f"{self.method}/"+"cons/" + str(fig_id)+"/" +f"V{self.vehicle_num}"+f"v{ith_vehicle_path}" +f"construct{i+1}" +".png"
                    plt.savefig(file_name)
                    print(f"Save fig : {file_name} ")
                    
                # edge_list = [( path[i] ,path[i+1]  ) for i in range(len(path)-1)]
                # nx.draw_networkx_edges( 
                #     networkx_graph , 
                #     pos = node_pos_dict , 
                #     edgelist = edge_list , 
                #     edge_color = self.Color_map[ith_vehicle_path] , 
                #     width = self.edge_width,
                #     arrows = True , 
                #     arrowsize = self.arrow_size , 
                # )
            
        
            
    def Batch_plot(self, batch  ,batch_routes , batch_objective):
        
        for ith , instance in enumerate(batch.to_data_list()): 
            self.Visualize(fig_id=ith+1 ,  
                           instance=instance , 
                           routes = batch_routes[ith], 
                           objective_value=batch_objective[ith].item() , 
                           ) 
        
        
if __name__ == "__main__":
    node_num = 10
    batch_size = 4 
    ig = MTSP_DataGenerator(
        workers= 1, batch_size=batch_size , node_num= node_num 
    )
    batch = ig.getInstance()
    visulizer = RouteVisulizer(
        node_num=node_num , 
        vehicle_num=3 , 
        capacity=False , 
        store = False , 
        show = True , 
        method="manual" , 
        store_path="./model/MultiTravelingSalesmanProblem/plot/"
    )

    import torch 
    route = [ [2,1,0] , [3,6,7,5,0] , [4,8,9,0] ] 
    route = [ torch.tensor(single_vehicle_route) for single_vehicle_route in route] 
    visulizer.Visualize(fig_id=0 , instance=batch , routes = route , objective_value=10)
            
        
        
        
        
        

    
    
    