import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt
from  random  import uniform
from scipy.stats import gamma 

# #define three Gamma distributions
# # average_distance = [10 , 20 , 30]
# average_distance = [0.05 ,0.1, 0.15]
# x = np.linspace(0, 0.3, 100)

# for distance in average_distance : 
   
#     # variance = uniform(0.03,0.15) * distance
#     # shape =  distance**2 / variance
#     # scale = variance / distance 
#     # plt.plot(x , stats.gamma.pdf(x ,a = shape , scale = scale ) , label=f"Distance : {distance}")
#     variance = uniform(0.07,0.2)
#     print(variance)
#     plt.plot(x , stats.gamma.pdf(x ,a = distance/variance
#                                  , scale = variance) , label=f"Distance : {distance}")


# k=2 
# theta = 1.5 
# random_number = gamma.rvs(a=k,scale=theta)
# print(random_number)
# #add legend
# plt.legend()

# #display plot
# plt.show()

distance = 0.4

a = list()
for i in range(1000): 
    variance = uniform(0.02 , 0.06) 
    a.append( max(gamma.rvs(a=distance/variance , scale=variance),0.5*distance)) 
plt.hist(a ,  bins=100)
print(np.array(a).mean())
plt.show()