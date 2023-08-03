




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
# 假設您的數據以(mean, std)的tuple形式保存在列表中
# data_list = [(10.0, 2.0), (8.0, 1.5), (12.0, 3.0)]
data_list = [] 
files=  ["N50V5_Greedy","N50V5_POMO","N50V5ORGD1","N50V5ORGL1","N50V5ORTS1"]
for file in files: 
    data = np.load("./model/MultiTravelingSalesmanProblem/experiment_data/"+file+".npy") 
    data : np.ndarray
    data_list.append( ( data.mean() , data.std()  ))
# 提取平均值和標準差
means = [data[0] for data in data_list]
stds = [data[1] for data in data_list]

# 設置柱的位置
positions = np.arange(len(data_list))

# 繪製柱狀圖
plt.bar(positions, means, yerr=stds, align='center', alpha=0.7)

# 添加誤差線和水平線
for i, data in enumerate(data_list):
    mean, std = data
    plt.plot([i, i], [mean-std, mean+std], color='red', linewidth=2)
    plt.plot([i-0.1, i+0.1], [mean-std, mean-std], color='red', linewidth=2)
    plt.plot([i-0.1, i+0.1], [mean+std, mean+std], color='red', linewidth=2)

# 設置X軸刻度和標籤
plt.xticks(positions, [f'Group {i+1}' for i in range(len(data_list))])

# 設置Y軸刻度間隔和格式
y_major_locator = MultipleLocator(0.5)  # 設置刻度間隔為0.5
plt.gca().yaxis.set_major_locator(y_major_locator)
plt.gca().yaxis.set_tick_params(which='both', labelsize=8)  # 設置刻度標籤大小

# 添加網格線
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# 顯示圖形
plt.show()