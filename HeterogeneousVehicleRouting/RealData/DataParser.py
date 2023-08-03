import matplotlib.pyplot as plt 
from random import shuffle

def BrandaoParse(file_name): 
    with open("./model/HeterogeneousVehicleRouting/RealData/"+file_name+".txt" , "r") as f : 
        lines = f.readlines() 
    n = int(lines[0].strip()) + 1 # node num 
    nodes = []
    num_vehicle_types  = int(lines[2*n+1].strip())
    vehicle_set =  [] 
    maxcapacity = max([ float(lines[i].strip().split()[0]) for i in range(2*n+2,2*n+2+num_vehicle_types)] )
    # print(f"Maximum capacity : {maxcapacity}")
    total_capacity = 0 
    total_demand = 0
    bound = float(lines[-1].strip())
    # print(f"bound : {bound}")
    for i in range(2*n+2 , 2*n+2+num_vehicle_types): 
        vehicle_info = lines[i].strip().split() 
        capacity = float(vehicle_info[0]) / maxcapacity
        velocity =  1/ float(vehicle_info[2])
        num_vehicles = int(vehicle_info[4])
        vehicle_charateristic = (capacity , velocity)
        #print(f"{i}-th vehicle : {vehicle_charateristic}")
        vehicle_set.extend( [vehicle_charateristic] * num_vehicles  )
        total_capacity += capacity * num_vehicles
        
    for i in range(1, n+1): 
        node_info = lines[i].strip().split() 
        node_id = int(node_info[0])
        x = float(node_info[1])
        y = float(node_info[2]) 
        demand = int(lines[i+n].strip().split()[1]) / maxcapacity
        total_demand += demand 
        node = (node_id , x , y ,demand) 
        #print(f" {i}-th node : {node}")
        nodes.append(node)  
    
    print(f"Total C : {total_capacity} , Total D : {total_demand}")
    return nodes , vehicle_set  , {"file_name":file_name,"node_num":len(nodes) ,"vehicle_num":len(vehicle_set),
                                   "bound":bound,"vehicle_types":num_vehicle_types}


def cParse(file_name): 
    with open("./model/HeterogeneousVehicleRouting/RealData/"+file_name+".txt" , "r") as f : 
        lines = f.readlines() 
    n = int(lines[0].strip()) + 1  # node num 
    nodes = []
    num_vehicle_types  = int(lines[n+1].strip())
    vehicle_set =  [] 
    maxcapacity = max([ float(lines[i].strip().split()[0]) for i in range(n+2,n+2+num_vehicle_types)] )
    total_capacity = 0 
    total_demand = 0
    print(f"Maximum capacity : {maxcapacity}") 
    print(lines[-1])
    bound = float(lines[-1].strip())
    print(f"bound : {bound}")
    for i in range(n+2 , n+2+num_vehicle_types): 
        vehicle_info = lines[i].strip().split() 
        capacity = float(vehicle_info[0])  / maxcapacity
        velocity =  1/ float(vehicle_info[2])
        num_vehicles = int(vehicle_info[4])
        vehicle_charateristic = (capacity , velocity)
        #print(f"{i}-th vehicle : {vehicle_charateristic}")
        vehicle_set.extend( [vehicle_charateristic] * num_vehicles  )
        total_capacity += capacity * num_vehicles
        
        
    for i in range(1, n+1): 
        node_info = lines[i].strip().split() 
        node_id = int(node_info[0])
        x = float(node_info[1])
        y = float(node_info[2]) 
        demand = float(node_info[3]) / maxcapacity
        total_demand += demand 
        node = (node_id , x , y ,demand) 
        #print(f" {i}-th node : {node}")
        nodes.append(node)  
        
    print(f"Total C : {total_capacity} , Total D : {total_demand}")
    return nodes , vehicle_set , {"file_name":file_name,"node_num":len(nodes) ,"bound":bound,
                                  "vehicle_num":len(vehicle_set),"vehicle_types":num_vehicle_types}

if __name__ == "__main__":

    nodes , vehicles = BrandaoParse("brandaoN5hd")
    # nodes , vehicles = cParse("c75_18hd")
    #print(f"nodes:{nodes}")
    print(f"--\n\n")
    print(f"vehicles : {vehicles}")
    shuffle(vehicles)
    print(f"vehicles : {vehicles}")
    # demand_list = [node[3] for node in nodes]
    # plt.hist(demand_list , bins=30)
    # plt.show()