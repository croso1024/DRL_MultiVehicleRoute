""" 
    2023-04-14 : Create fix Validation Dataset 
    此Dataset為Random generate , 利用batch-size=1的DataLoader保存為.pt 
    當要使用的時候 , 取出dataloader並將內部資料轉回list , 即可重新以不同Batch_size建立Dataloader  
    另一個要解決的問題是Capacity , Velocity的設定 , 也需要固定下來
    
    -- scheme . 
        目前的想法是C , V的資料是綁在Batch上的 , 因此可能第一步產生資料就要設定好該Data-loader的Batch size了 , 
        如果是這樣 , 那就是一開始直接設定Batch_size , 每一個Batch配套一組C,V , 
        Load function則是可以依照inference method去做調整 , 但因為env初始化只能設定一組CV , 因此Greedy可能就會Batch_size太小這樣
        至於PMPO, DE , MIX等則是可能會需要拆分Batch比較容易處理 
    
    工作項目 : 
        1. 建立基準測試Dataset產生器 , 設定原始Batch-size , Dataset-size , 而 Dataset-size // Batch_size 即為共有幾組C,V
        2. 建立Reload function , 根據inference mode,以及給定的maximum batch-size來產生batch-list , 每一個Batch都包含了C,V設定 
"""


""" 
    2023-09-26 加入為了Jouranl版本的 , 平均化Vehicle charateristic , 對於車輛速度與容量直接採用倒數關係
"""

import torch ,sys
sys.path.append("./model/HeterogeneousVehicleRouting/utils")
from CVRPGenerator import CVRP_DataGenerator 
from torch_geometric.loader import DataLoader 
from random import uniform
import numpy as np 
from tqdm import tqdm 
import argparse 



def CreateDataset(dataset_size,batch_size, node_num, vehicle_num , heterogeneous=False, journal=False) : 
    assert dataset_size % batch_size == 0 , "Dataset-size cannot be divided by Batch-size"
    # Step1. Create Data-list
    Generator = CVRP_DataGenerator(workers=1 , batch_size=batch_size , node_num=node_num )
    Dataset_instance = [ Generator.getInstance() for i in tqdm(range(dataset_size)) ]
    Dataset = DataLoader(Dataset_instance , batch_size=batch_size ,pin_memory=False)  
    # Step2. Setup the capaicty , velocity configuration
    charater_set_number = dataset_size // batch_size  
    
    if journal : 
        charateristic_function = RandomCharateristic_journal
        c2v = lambda c : 1.4 - c 
        
    else : 
        charateristic_function = RandomCharateristic_single
        c2v = lambda c : ((-2/3)*c) + (32/30)
            
    if heterogeneous : 
        capacity =  [   charateristic_function(vehicle_num=vehicle_num)[0]   for i in range(charater_set_number)] 
        velocity = [   [  c2v(c) for c in sub_capacity_list] for sub_capacity_list in capacity ]
    else :         
        capacity = [[1 for v in range(vehicle_num)] for i in range(charater_set_number)]
        velocity = [[1 for v in range(vehicle_num)] for i in range(charater_set_number)]
    # print(f"Debug capacity : {capacity}\n")
    # print(f"Debug velocity : {velocity}\n")
    CV_setting = torch.tensor([capacity ,velocity] , dtype=torch.float32) 
    CV_setting = torch.permute(CV_setting , dims=(1,0,2))
    # print(f"Debug CV setting : {CV_setting.shape}")    
    print(CV_setting)
    print(f"Mean capacity :{ torch.tensor(capacity,dtype=torch.float32).mean() }")
    print(f"Mean velocity :{torch.tensor(velocity,dtype=torch.float32).mean()}")
    
    if heterogeneous:
        Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}-he"
    else : 
        Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}" 
    torch.save(Dataset , "./model/HeterogeneousVehicleRouting/Dataset/"+"Data-"+Dataset_name+".pt") 
    torch.save(CV_setting, "./model/HeterogeneousVehicleRouting/Dataset/"+"CV-"+Dataset_name+".pt") 



def LoadDataset(dataset_size,batch_size,node_num,vehicle_num,heterogeneous =False
                , maximum_batch_size = None, PMPO=False):
    """ 
        用來載入保存好的Dataset 與CV設定 , 在此可以調整原先的Batch大小(只能增加不能減少,但增加是指同一個batch的擴充即PMPO) , 
        主要有兩種Case : 
        1. PMPO(最多會將原先Dataset的Batch-size變為8倍 ,最高到maximum-batch-size)
        2. DE (沿用dataset設定的batch-size,由環境自行去增加倍數 ,與maximum-batch-size無關)
        3. Mix(最多會將原先Dataset的Batch-size變為8倍 , 最高到maximum-batch-size,再由環境自行增加倍數)
        --> 最終輸出會是一個tuple-list [ (batch1,cv_setting1) , (batch2,cv_setting2) ... ]    
    """
    if heterogeneous:
        Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}-he"
    else : 
        Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}" 
        
    Dataset = torch.load("./model/HeterogeneousVehicleRouting/Dataset/Data-" + Dataset_name + ".pt")
    CV_setting  = torch.load("./model/HeterogeneousVehicleRouting/Dataset/CV-" + Dataset_name + ".pt")
    assert len(Dataset) == len(CV_setting) , f"Not consistency of CV and dataset error { len(Dataset)} , {len(CV_setting)}"
    # Case1. PMPO --> 以原始batch-size重新初始化一個Generator , call 轉PMPO-dataset來load回data
    if PMPO:
        # Step1. 轉dataset為PMPO dataset  
        if  maximum_batch_size:assert maximum_batch_size >= batch_size , "Error , maximum batchsize < origin_batch_size"
        else :  maximum_batch_size = batch_size
        PMPO_generator = CVRP_DataGenerator(workers=1, batch_size=batch_size, node_num=node_num)
        PMPO_dataset = PMPO_generator.dataset_to_PMPOdataset_SingleProcess(Dataset,maximum_batch_size=maximum_batch_size)
        # Step2. 把PMPO_batch list與CV-setting結合
        number_of_PMPO_batch = (dataset_size*8) // maximum_batch_size # 計算轉PMPO後Batch數量
        charateristic_set_number = number_of_PMPO_batch // len(CV_setting)  # 計算一組CV_setting要給幾個Batch使用
        Validation_Dataset = [] 
        for i in range(number_of_PMPO_batch): 
            Validation_Dataset.append((  PMPO_dataset[i] , CV_setting[ i // charateristic_set_number]  ))
    # Case2. 正常的Greedy , 或著OR-tools Dataset形式 , 直接串起Dataset和CV setting即可 
    else : 
        Validation_Dataset = [] 
        for i , origin_batch in enumerate(Dataset): 
            Validation_Dataset.append((origin_batch , CV_setting[i])) 

    return Validation_Dataset



def RandomCharateristic_single(vehicle_num): 
    probs = torch.tensor( [ 0.35 , 0.25, 0.25 , 0.15  ])
    capacity_ranges = [(0.25, 0.45), (0.55, 0.65), (0.75, 0.85), (0.95, 1)]
    num_range = len(capacity_ranges)
    categorical_dist = torch.distributions.Categorical(probs = probs) 
    range_indices = categorical_dist.sample( (1,vehicle_num) )
    capacity = torch.zeros(size= (1,vehicle_num) ) 
    for i in range(num_range) : 
        mask = range_indices == i 
        low , high = capacity_ranges[i]
        range_capacities = torch.rand( (1,vehicle_num) ).mul(high-low).add(low)
        capacity = torch.where(mask , range_capacities , capacity)
    velocity =  ( (-2/3)  * capacity ) + (32/30)
    return capacity.squeeze().tolist() , velocity.squeeze().tolist()    


def RandomCharateristic_journal(vehicle_num) : 
    
    capacity = torch.rand( (1,vehicle_num) ).mul(1-0.4).add(0.4)
    
    return capacity.squeeze().tolist() , None


if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d","--dataset_size" , type=int , default=256)
    argParser.add_argument("-b","--batch_size" , type=int , default=32)
    argParser.add_argument("-n","--node_num" , type=int , default=50)
    argParser.add_argument("-v","--vehicle_num" , type=int , default=5)
    argParser.add_argument("-he","--heterogeneous" , action="store_true",default=False)
    argParser.add_argument("-j","--journal" , action="store_true",default=False)
    
    arg = argParser.parse_args() 
    dataset_size = arg.dataset_size 
    batch_size = arg.batch_size 
    node_num = arg.node_num 
    vehicle_num = arg.vehicle_num 
    heterogeneous = arg.heterogeneous 
    journal = arg.journal 

    CreateDataset(dataset_size=dataset_size, batch_size=batch_size,
                  node_num=node_num,  vehicle_num=vehicle_num , heterogeneous=heterogeneous ,journal=journal)