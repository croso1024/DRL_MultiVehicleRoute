import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空图形对象
G = nx.Graph()

# 添加节点
nodes = [1, 2, 3, 4]  # 节点序列
G.add_nodes_from(nodes)

# 绘制节点
pos = nx.spring_layout(G)  # 为节点设置布局
nx.draw_networkx_nodes(G, pos)

# 可以添加其他绘制节点的设置，如颜色、标签等

# 显示图形
# plt.show()

# 添加边
edges1 = [(1, 2), (2, 3)]  # 第一组边序列
edges2 = [(3, 4), (4, 1)]  # 第二组边序列

# 绘制第一组边
nx.draw_networkx_edges(G, pos, edgelist=edges1)

# 可以添加其他绘制边的设置，如颜色、线条样式等

# 显示图形
# plt.show()

# 保存图像
plt.savefig("graph1.png")

# 清除边
nx.draw_networkx_edges(G, pos, edgelist=[])

# 绘制第二组边
nx.draw_networkx_edges(G, pos, edgelist=edges2)

# 保存图像
plt.savefig("graph2.png")

# 清除边
nx.draw_networkx_edges(G, pos, edgelist=[])

# 显示图形
plt.show()
