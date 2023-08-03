import matplotlib.pyplot as plt 
import torch
from random  import random
# # 設定參數

    
    

if __name__ == "__main__": 
    
    # tc  , tv = [] , [] 
    # statics = [0,0,0,0]
    # for i in range(50): 
    #     c , v = RandomCharateristic_single()
    #     tc.extend(c)
    
    # for c in tc : 
    #     if c>=0.1 and c <=0.25: statics[0] += 1
    #     elif c>=0.3 and c<=0.5 : statics[1] += 1
    #     elif c>=0.65 and c<=0.8 : statics[2] += 1
    #     elif c>=0.9 and c<=1 : statics[3] += 1
         
    # plt.hist(tc , bins=50)
    # plt.show()
    # print(statics)

    # c,v = RandomCharateristic(sample_size=1000) 
    # print(torch.mean(c))
    # c,v = RandomCharateristic(sample_size=1000) 
    # print(torch.mean(c))
    # c,v = RandomCharateristic(sample_size=1000) 
    # print(torch.mean(c))
    from random import betavariate 
    tt = [max(0.01 ,(betavariate(alpha =2,beta=5) /4) ) for i in range(1000) ]
    plt.hist(tt , bins=30)
    
    print( torch.tensor(tt).mean())
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    