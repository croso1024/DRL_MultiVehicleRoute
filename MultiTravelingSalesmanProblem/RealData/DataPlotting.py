from RealData.Data2Graph import GraphDataParse 
from RealData.DataParser import tspParse 
# from Data2Graph import GraphDataParse 
# from DataParser import tspParse 

import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import glob , os 

def get_tsp_files(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*.tsp'))
    return [os.path.basename(f) for f in txt_files]


if __name__ =="__main__" : 
    path = "/home/croso1024/python_code/GNN/GNNRoute/model/MultiTravelingSalesmanProblem/RealData/" 
    listname = get_tsp_files(path) 

    for map in listname : 
        map = map.split(".")[0]
        
        origin_coordinate, ret = tspParse(map)
        normalized_coordinate , instance_info = GraphDataParse(map)
        
        x_coords =  [node[1] for node in origin_coordinate]
        y_coords =  [node[2] for node in origin_coordinate]
        
        normalized_x_coords = [node[0] for node in normalized_coordinate]
        noramlized_y_coords = [node[1] for node in normalized_coordinate]
        plt.rcParams.update({'font.size': 10})
        
        fig = plt.figure(figsize=(8,4)) 
        gs = GridSpec(1,2 ,width_ratios=[1,1])
        cmap = plt.get_cmap('RdYlBu_r')
        ax1 = fig.add_subplot(gs[0,0])
        ax1.scatter(x_coords , y_coords ,c=range(len(origin_coordinate)), cmap=cmap) 
        ax1.grid() 
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Origin Data')
        
        ax2 = fig.add_subplot(gs[0,1])
        ax2.scatter(normalized_x_coords , noramlized_y_coords ,c=range(len(origin_coordinate)), cmap=cmap) 
        ax2.grid() 
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Normalized Data')

        fig.suptitle(map) 
        # plt.show() 
        file_name = path+map+".png"
        plt.savefig(file_name)
        plt.close()