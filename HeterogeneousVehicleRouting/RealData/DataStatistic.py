""" 
    Use to statistical calculate the data distribution of real data 
"""
from RealData.DataParser import cParse , BrandaoParse 
import os
import matplotlib.pyplot as plt 
import glob

def get_txt_files(folder_path):
    # 使用glob模块获取该路径下所有副檔名为.txt的文件列表
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    # 使用列表推导式从每个文件路径中提取文件名并返回
    return [os.path.basename(f) for f in txt_files]



if __name__ == "__main__": 
        
    listname = get_txt_files("/home/croso1024/python_code/GNN/GNNRoute/model/HeterogeneousVehicleRouting/RealData/")
    print(listname)

    for map in listname:
        map = map.split(".")[0]
        
        if map[0] == "c" : 
            nodes , vehicle_set ,instance_info =  cParse(map)
        else : 
            nodes , vehicle_set , instance_info = BrandaoParse(map)
        demand_list = [node[3] for node in nodes ] 
        print(demand_list)
        vehicle_capacity = [vehicle[0] for vehicle in vehicle_set]
        plt.title(map) 
        # plt.subplot(2,0)
        # plt.hist(vehicle_capacity , bins=20)
        # plt.subplot(2,1) 
        plt.hist(demand_list , bins=20) 
        plt.show() 
        

