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


import torch ,sys
sys.path.append("./model/MultiTravelingSalesmanProblem/utils")
from MTSPGenerator import MTSP_DataGenerator 
from torch_geometric.loader import DataLoader 
from random import uniform
import numpy as np 
from tqdm import tqdm 
import argparse 

def CreateDataset(dataset_size,batch_size, node_num, vehicle_num ) : 
    assert dataset_size % batch_size == 0 , "Dataset-size cannot be divided by Batch-size"
    # Step1. Create Data-list
    Generator = MTSP_DataGenerator(workers=1 , batch_size=batch_size , node_num=node_num)
    Dataset_instance = [ Generator.getInstance() for i in tqdm(range(dataset_size)) ]
    Dataset = DataLoader(Dataset_instance , batch_size=batch_size ,pin_memory=False)  
    Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}" 
    torch.save(Dataset , "./model/MultiTravelingSalesmanProblem/Dataset/"+"Data-"+Dataset_name+".pt") 



def LoadDataset(dataset_size,batch_size,node_num,vehicle_num, maximum_batch_size = None, PMPO=False ):
    """  
        用來載入保存好的Dataset 與CV設定 , 在此可以調整原先的Batch大小(只能增加不能減少,但增加是指同一個batch的擴充即PMPO) , 
        主要有兩種Case : 
        1. PMPO(最多會將原先Dataset的Batch-size變為8倍 ,最高到maximum-batch-size)
        2. DE (沿用dataset設定的batch-size,由環境自行去增加倍數 ,與maximum-batch-size無關)
        3. Mix(最多會將原先Dataset的Batch-size變為8倍 , 最高到maximum-batch-size,再由環境自行增加倍數)
        --> 最終輸出會是一個tuple-list [ (batch1,cv_setting1) , (batch2,cv_setting2) ... ]    
    """
    Dataset_name = f"D{dataset_size}-B{batch_size}-N{node_num}-V{vehicle_num}" 
        
    Dataset = torch.load("./model/MultiTravelingSalesmanProblem/Dataset/Data-" + Dataset_name + ".pt")
    # Case1. PMPO --> 以原始batch-size重新初始化一個Generator , call 轉PMPO-dataset來load回data
    if PMPO:
        # Step1. 轉dataset為PMPO dataset  
        if  maximum_batch_size:assert maximum_batch_size >= batch_size , "Error , maximum batchsize < origin_batch_size"
        else :  maximum_batch_size = batch_size
        PMPO_generator = MTSP_DataGenerator(workers=1, batch_size=batch_size, node_num=node_num)
        PMPO_dataset = PMPO_generator.dataset_to_PMPOdataset_SingleProcess(Dataset,maximum_batch_size=maximum_batch_size)
        number_of_PMPO_batch = (dataset_size*8) // maximum_batch_size # 計算轉PMPO後Batch數量
        Validation_Dataset = [] 
        for i in range(number_of_PMPO_batch): 
            Validation_Dataset.append(  PMPO_dataset[i] )
    # Case2. 正常的Greedy , 或著OR-tools Dataset形式 , 直接串起Dataset和CV setting即可 
    else : 
        Validation_Dataset = [] 
        for i , origin_batch in enumerate(Dataset): 
            Validation_Dataset.append( origin_batch ) 
    return Validation_Dataset

if __name__ == "__main__": 
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d","--dataset_size" , type=int , default=256)
    argParser.add_argument("-b","--batch_size" , type=int , default=32)
    argParser.add_argument("-n","--node_num" , type=int , default=50)
    argParser.add_argument("-v","--vehicle_num" , type=int , default=5)
    arg = argParser.parse_args() 
    dataset_size = arg.dataset_size 
    batch_size = arg.batch_size 
    node_num = arg.node_num 
    vehicle_num = arg.vehicle_num 

    CreateDataset(dataset_size=dataset_size, batch_size=batch_size,
                  node_num=node_num,  vehicle_num=vehicle_num  )