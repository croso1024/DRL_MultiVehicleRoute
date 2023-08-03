from torch_geometric.utils import contains_isolated_nodes
from utils.KNN_transform import RadiusGraph_transformer , KNNGraph_transformer
from utils.CVRPGenerator import CVRP_DataGenerator 
from torch_geometric.utils import  to_networkx
import  matplotlib.pyplot as plt
import networkx as nx 
ig = CVRP_DataGenerator(
    workers=1 , batch_size=1 , node_num=50
)
r = 0.2
k = 1
# Rt = RadiusGraph_transformer(r=r)
# Kn = KNNGraph_transformer(k=k)
# instance = ig.getInstance()

# instance = Kn.batch_transform(instance) 
# print(instance.num_edges)
# g1 = to_networkx(instance) 
# nx.draw(g1)
# plt.grid()
# plt.show()



# instance2 = Rt.batch_transform(instance)
# print(instance.num_edges)
# g2 = to_networkx(instance2)
# nx.draw(g2)
# plt.grid() 
# plt.show()

Kn = KNNGraph_transformer(k=k) 
isolated=0 
for i in range(100): 
    instance = ig.getInstance()
    instance = Kn.batch_transform(instance)
    if contains_isolated_nodes(instance.edge_index ,num_nodes=50): isolated+=1 

print(isolated)