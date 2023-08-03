import osmnx as ox 
import matplotlib.pyplot as plt 
import folium

base_graph = ox.graph_from_point((25.0416,121.5438) ,dist=7500  , simplify=True, network_type="drive")



# 可以進一步簡化道路網格
# simplified_graph = ox.simplify_graph(base_graph)
# base_graph = ox.load_graphml("../model/DynamicStochasticVehicleRouting/Taipei7500D.graphml")

ox.plot_graph(base_graph)
plt.show()

# ox.save_graphml(base_graph , "./model/DynamicStochasticVehicleRouting/Taipei7500B.graphml")