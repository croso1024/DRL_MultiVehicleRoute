""" 
    0510 專用於OSM地圖繪製的功能 , 使用RoutingSimulator的每一個決策時刻紀錄產生一連串決策地圖 
"""
import osmnx as ox 
import networkx as nx 
import folium  , numpy as np 
from tqdm import tqdm 
from osmnx.utils_graph import get_route_edge_attributes
from RealData.location_set import loc_dict_encode 
import copy 


    
class OSMRoute_plotter: 
    
    
    
    
    def __init__(self,config , BaseGraph,map="NTU"): 

        self.Color_table =  ["red", "orange","yellow","green","blue","purple"] 
        self.loc_table = loc_dict_encode[map] 
        self.location_set = { int(i) : self.loc_table[i]   for i in self.loc_table.keys() }

        self.vehicle_num = config["vehicle_num"]
        self.total_step = 2 + config["total_update_times"] 
        self.BaseGraph = BaseGraph 
        self.known_request = set()
        self.new_request = set() 
        
    def plotting(self, RoutingSimulator_log:dict, text:str = None) : 
        Current_Request_log = RoutingSimulator_log["Current_Request"]
        Complete_Request_log  = RoutingSimulator_log["Complete_Request"]
        Vehicle_SDR = RoutingSimulator_log["Vehicle_SDR"] 
        ModelPredict_Route = RoutingSimulator_log["Model_Predict"]
        Vehicle_Routes = RoutingSimulator_log["Vehicle_Routes"]
        Done_vehicles = RoutingSimulator_log["Done_Vehicle"]
        
        
        assert len(Current_Request_log) == self.total_step ,"Logger size inconsistent error"
        assert len(Complete_Request_log) == self.total_step ,"Logger size inconsistent error"
        assert len(Vehicle_SDR) == self.total_step ,"Logger size inconsistent error"
        assert len(ModelPredict_Route) == self.total_step ,"Logger size inconsistent error"
        assert len(Vehicle_Routes) == self.total_step ,"Logger size inconsistent error"
        assert len(Done_vehicles) == self.total_step ,"Logger size inconsistent error"
        # 設定初始狀態的request都為已知 ,避免都是new-icon
        self.known_request = set(Current_Request_log[0])

        self.init = True 
        for timeFrame in tqdm(range(self.total_step)) : 
            self.plot_single_timeFrame(
                Current_Request=Current_Request_log[timeFrame],
                Complete_Request=Complete_Request_log[timeFrame], 
                SDR=Vehicle_SDR[timeFrame],
                Model_Predict=ModelPredict_Route[timeFrame]  ,
                Vehicle_Routes=Vehicle_Routes[timeFrame],
                Done_vehicles = Done_vehicles[timeFrame],
                text = str(timeFrame) , 
            )
            # 代表已經過初始狀態
            self.init = False
            
    
    
    def plot_single_timeFrame(self,Current_Request,Complete_Request,SDR,Model_Predict,Vehicle_Routes,Done_vehicles,text): 
        """ 
            使用特定時刻的log內容去進行html map的繪製 ,
            Step0. 更新new-request ,current-request , complete-request
            Step1. 在地圖上使用顏色1去繪製Current Request的marker 
            Step2. 在地圖上使用顏色2去繪製Complete Request的marker 
            Step3. 依據vehicle-Pos使用Icon畫出車輛目前所在位置,
                注意Pos內每一部車的形式為 ( Src , Dst , Ratio ) ,或著(0 , 0 , 0)代表初始狀態or該車已經結束
                另外需要一個function使用Src,Dst,Ratio去計算車輛所在的地圖位置 
            Step4. 繪製出目前每一台車完成的路線 
            Step5. 繪製出當前時刻模型Predict的路線 
        """
        self.fmap = folium.Map(
            location = self.location_set[0] , zoom_start=15 , 
        )
        self.folium_graph = ox.plot_graph_folium(
            self.BaseGraph , graph_map=self.fmap , popup_attribute="name" , title="NTU Campus" , weight=0.2 , zoom=5
        )
        # 新出現的request會等於這一時刻拿到的Request 減去已知的Request 
        self.new_request = Current_Request- self.known_request
        # 繪製新出現的Request 
        self.MarkerPlot(self.new_request , property="new")
        #print(f"Debug new Request : \n{self.new_request}")
        # 繪製當下的已知的Request , 要扣除掉New Request 
        self.MarkerPlot(Current_Request - self.new_request , property="uncomplete")
        #print(f"Debug Current Request : \n{Current_Request}")
        # 繪製當下已經完成的Request 
        self.MarkerPlot(Complete_Request , property="complete")
        #print(f"Debug Complete Request: \n{Complete_Request}")
        
        # 依據中斷點繪製車輛位置 , 同時返回SDR定位函式所給的各個車輛的位置 , 使路線能夠順利連接
        vehicle_position = self.Vehicle_Plot(SDR)
        print(f"Debug Predict Routes: \n{Model_Predict}")
        self.RoutePlot(Model_Predict ,vehicle_position ,Done_vehicles , complete=False) 
        print(f"Debug Vehicle Routes: \n{Vehicle_Routes}")
        self.RoutePlot(Vehicle_Routes ,vehicle_position, Done_vehicles ,complete=True) 
        
        self.known_request = Current_Request
        
        self.folium_graph.save("./model/DynamicStochasticVehicleRouting/test"+text+".html")
        print(f"Save complete")
        
    def LocateVehiclePos(self,Src,Dst,Ratio) -> tuple :
        # 如果Src是0 (初始狀態 or 該車輛已經回到depot) 則直接return depot的座標 
        if Src == 0 : 
            return self.location_set[0]
        # 輸入起點座標、目標座標、已經走的路徑比例
        # 返回目前位置的座標
        # 計算出發點和目標點之間的最短路徑
        Src = self.location_set[Src] 
        Dst = self.location_set[Dst]
        origin_node = ox.nearest_nodes(self.BaseGraph, Src[1], Src[0])
        target_node = ox.nearest_nodes(self.BaseGraph, Dst[1],Dst[0])
        #route = nx.shortest_path(self.BaseGraph, origin_node, target_node, weight='length' )
        route = ox.shortest_path(self.BaseGraph , origin_node , target_node)
        route_edge = get_route_edge_attributes(self.BaseGraph , route) 
        total_distance = sum(edge['length'] for edge in route_edge)
  
        target_length = total_distance * Ratio
        
        # 找到距離target_length處到達的節點
        current_length = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if current_length + self.BaseGraph[u][v][0]['length'] > target_length:
                # offset = ox.distance.along_linestring(self.BaseGraph[u][v]['geometry'], target_length - current_length)
                # nearest_node = ox.nearest_nodes(self.BaseGraph ,offset)
                x = (self.BaseGraph.nodes[u]["x"] + self.BaseGraph.nodes[v]["x"])/2 
                y = (self.BaseGraph.nodes[u]["y"] + self.BaseGraph.nodes[v]["y"])/2 
                nearest_node = (x,y)
                # print(f"Nearest node: {nearest_node}")
                break
            current_length += self.BaseGraph[u][v][0]['length']

        return (nearest_node[1] , nearest_node[0])

    def Vehicle_Plot(self,Vehicle_SDR): 
        # Store the result of LocateVehiclePos , use for connect the prediction & complete route 
        Vehicle_pos = list() 
        for i , (Src,Dst,Ratio) in enumerate(Vehicle_SDR): 
            customize_icon =  folium.CustomIcon(icon_image="./model/DynamicStochasticVehicleRouting/icon/vehicle_"+self.Color_table[i]+".png", icon_size=(60, 60))
            location = self.LocateVehiclePos(Src=Src, Dst=Dst, Ratio= Ratio)
            Vehicle_pos.append(location)
            self.folium_graph.add_child(
                child = folium.Marker(
                    location = location , 
                    icon = customize_icon,
                )
            )
        return Vehicle_pos 
    
    
    def MarkerPlot(self,node_set,property):
        if property == "new": 
            file_name = "new.png"
        elif property == "complete": 
            file_name = "complete.png"
        elif property == "uncomplete": 
            file_name = "uncomplete.png"
            
        for i , node in enumerate(node_set): 
            if not node == 0 :
                customize_icon = folium.CustomIcon(icon_image='./model/DynamicStochasticVehicleRouting/icon/'+file_name, icon_size=(20, 20)) 
            else : 
                customize_icon = folium.CustomIcon(icon_image='./model/DynamicStochasticVehicleRouting/icon/warehouse.png', icon_size=(60, 60)) 
            self.folium_graph.add_child(
                child = folium.Marker(
                    location = self.location_set[node] , 
                    icon = customize_icon , 
                )
            )

    def RoutePlot(self, Routes, vehicle_pos , done_vehicle ,complete): 
        """ 
            --完成的路徑使用實現 , Predict的結果使用虛線 
            每一台車的路徑的繪製步驟 : 
            Step1. 先將Route中的節點編號轉換回節點座標(注意此時只有Customer node對應的OSM節點) , 透過OSM地圖取出route 
            Step2. 從上一步得到的每一條路徑 , 取出路徑中所有的節點對應的OSM地圖節點  
            Step3. 畫出路徑 , 依據完成狀況決定虛線實線
        """        
        for ith_vehicle , path in enumerate(Routes): 
            # Step0. 如果該車已經完成了 , 就不繪製其Predict的路線了 
            if done_vehicle[ith_vehicle] and not complete : continue
            print(f"Debug {ith_vehicle} \ path:{path}\n")
            
            # Step1. 將代表customer -node的座標找到OSM-nearest , 透過ox.shortest-path連接這些節點(取得中間節點)  
            Customer_node_OSM = [] 
            Route_list = [] 
            for node in path :  
                node = self.location_set[node]
                Customer_node_OSM.append( ox.nearest_nodes(self.BaseGraph , node[1], node[0] ) )
            
            
            # Step2. 根據要繪製的路線種類 , 連接線段與車輛位置 , 同時產出所有路徑點給folium.Polyline
            vehicle_pos_node = ox.nearest_nodes(self.BaseGraph , vehicle_pos[ith_vehicle][1] , vehicle_pos[ith_vehicle][0] )
            # 如果是要繪畫已經完成的路徑, 則路徑會落後於車輛當前的位置 , 要補上 已經完成路徑的最後一個節點(customer node)到當前位置的路徑 
            if complete :     

                complete_customer_to_current_pos_route = ox.shortest_path(self.BaseGraph , Customer_node_OSM[-1] , vehicle_pos_node )

                for i in range(len(Customer_node_OSM) -1) : 
                    route = ox.shortest_path(self.BaseGraph , Customer_node_OSM[i] , Customer_node_OSM[i+1] )
                    Route_list.append(route) 

                Route_list.append(complete_customer_to_current_pos_route) 
                
            # 在繪畫模型預測的結果的時候 , 預測的起點可能會在實際位置之前 , 要補上車輛當前位置到預測起點這一段在最前方 
            else : 
                current_pos_to_predict_customer_route = ox.shortest_path(self.BaseGraph , vehicle_pos_node , Customer_node_OSM[0])

                Route_list.append(current_pos_to_predict_customer_route) 

                for i in range(len(Customer_node_OSM) -1) : 
                    route = ox.shortest_path(self.BaseGraph , Customer_node_OSM[i] , Customer_node_OSM[i+1] )
                    Route_list.append(route) 
                
            Route_coordinates = [ [(self.BaseGraph.nodes[node]["y"] ,self.BaseGraph.nodes[node]["x"] ) for node in route]  for route in Route_list  ]
            for ith , route_coordinate in enumerate(Route_coordinates): 
                self.folium_graph.add_child(
                    child = folium.PolyLine(
                        locations= route_coordinate  , weight=5 , color=self.Color_table[ith_vehicle] , dash_array = None if complete else "5,20"
                    )
                )
                
                
                
                
                
                
            # """ 
            #     把車輛pos也用這邊畫 , 直接畫在實際路線的最後一個點上
            # """
            
            # if  (complete  and Route_coordinates ):  
                
            #     customize_icon =  folium.CustomIcon(icon_image="./model/DynamicStochasticVehicleRouting/icon/vehicle_"+self.Color_table[ith_vehicle]+".png", icon_size=(60, 60))
            #     self.folium_graph.add_child(
            #             child = folium.Marker(
            #                 location = Route_coordinates[-1][-1],
            #                 icon = customize_icon,
            #             )
            #         ) 
            
            # if self.init : 
            #     customize_icon =  folium.CustomIcon(icon_image="./model/DynamicStochasticVehicleRouting/icon/vehicle_"+self.Color_table[ith_vehicle]+".png", icon_size=(60, 60))
            #     self.folium_graph.add_child(
            #             child = folium.Marker(
            #                 location = self.location_set[0],
            #                 icon = customize_icon,
            #             )
            #         ) 
        
        

