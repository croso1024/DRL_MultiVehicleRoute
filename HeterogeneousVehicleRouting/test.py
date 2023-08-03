import  numpy as np


# a = [2,3,4,5,6,7]

# np.save("testttt.npy" , np.array(a)   )
a = np.load("./model/HeterogeneousVehicleRouting/HVRP_DE_n55_v8.npy").tolist() 
print(a)