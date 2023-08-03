import osmnx as ox 
import folium  
from location_set import loc_dict_encode 
import matplotlib.pyplot as plt 


location_set = loc_dict_encode["TAIPEI"]
basegraph = ox.load_graphml("./model/DynamicStochasticVehicleRouting/Taipei7500D.graphml")

ox.plot_graph(basegraph)
plt.show()

fmap = folium.Map(
    location = location_set["0"] , zoom_start=15 , 
)
foliumMap = ox.plot_graph_folium(
    basegraph , graph_map=fmap , popup_attribute="name" , title="TAIPEI", weight=0.2 ,
    zoom=5 
)

# 加入節點位置 

for i , (loc_name , loc_coor) in enumerate(location_set.items()): 
    foliumMap.add_child(
        child = folium.Marker(
            location = loc_coor , 
            icon = folium.CustomIcon(
                icon_image="./model/DynamicStochasticVehicleRouting/icon/placeholder.png",
                icon_size=(25,25)
            )
        )
    )

foliumMap.save("./model/DynamicStochasticVehicleRouting/0612_test.html")  
print("Done")