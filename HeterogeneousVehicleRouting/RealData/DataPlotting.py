"""
    用來畫出真實資料的節點分佈 , 採用Isomap或著MDS
"""
from Data2Graph import GraphDataParse
from DataParser import cParse , BrandaoParse
import glob , os 
import  matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

def get_txt_files(folder_path):
    # 使用glob模块获取该路径下所有副檔名为.txt的文件列表
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    # 使用列表推导式从每个文件路径中提取文件名并返回
    return [os.path.basename(f) for f in txt_files]


path = "/home/croso1024/python_code/GNN/GNNRoute/model/HeterogeneousVehicleRouting/RealData/"
listname = get_txt_files(path)

for map in listname : 
    map = map.split(".")[0] 
    if map[0] == "c" : 
        origin_coordinate ,*ret = cParse(map)
        normalized_coordinate , demand_list , vehicle_set , instance_info = GraphDataParse(
            instance_type="c" , instance_name= map 
        )
    else : 
        origin_coordinate ,*ret = BrandaoParse(map)
        normalized_coordinate , demand_list , vehicle_set , instance_info = GraphDataParse(
            instance_type="brandao" , instance_name= map 
        )

    # # 將 x 座標和 y 座標分別存放在兩個 list 中
    # origin_coordinate = (node-id , x, y ,demand )
    x_origin = [node[1] for node in origin_coordinate]
    y_origin = [node[2] for node in origin_coordinate]
    # normalized_coordinate = (x , y )
    x_coords = [node[0] for node in normalized_coordinate]
    y_coords = [node[1] for node in normalized_coordinate]
    capacity_list = [vehicle[0] for vehicle in vehicle_set]
    velocity_list = [vehicle[1] for vehicle in vehicle_set]
    
    plt.rcParams.update({'font.size': 10})

    # 創建一個2x2的GridSpec，第一行有兩個子圖，第二行只有一個子圖
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    cmap = plt.get_cmap('RdYlBu_r')
    # cmap2 = plt.cm.get_cmap('RdYlBu')
    # 畫出節點圖
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_coords, y_coords ,c=demand_list ,cmap=cmap)
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Node Map')

    # 畫出demand的直方圖
    # ax2 = fig.add_subplot(gs[0,1])
    # ax2.hist(demand_list, bins=20 ,color=cmap(demand_list))
    # ax2.set_xlabel('Demand')
    # ax2.set_ylabel('Count')
    # ax2.set_title('Demand Distribution')

    ax2 = fig.add_subplot(gs[0,1])
    n, bins, patches = ax2.hist(demand_list, bins=20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = cmap((bin_centers - min(demand_list)) / (max(demand_list) - min(demand_list)))
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', c)
    ax2.set_xlabel('Demand')
    ax2.set_ylabel('Count')
    ax2.set_title('Demand Distribution')



    # 畫出vehicle-capacity, velocity的直方圖
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(capacity_list, bins=20, alpha=0.5, label='Vehicle Capacity', color='blue')
    ax3.hist(velocity_list, bins=20, alpha=0.5, label='Vehicle Velocity', color='red')
    ax3.set_xlabel('Vehicle Capacity')
    ax3.set_ylabel('Count')
    ax3.set_title('Capacity(B) , Velocity(R)  Distribution')
    
    # 畫出原始的節點位置圖
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(x_origin, y_origin ,c=demand_list ,cmap=cmap)
    ax4.grid()
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Origin node Map')



    # 調整子圖間的間距
    gs.update(wspace=0.3, hspace=0.4)
    fig.suptitle(map)
    # 保存圖片
    file_name = path + map + ".png"
    plt.savefig(file_name)
    plt.close()


