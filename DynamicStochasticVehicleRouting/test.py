from RealData.RoutingSimulator import RoutingSimulator


vehicle_num = 2
simulator_environment = RoutingSimulator(
    init_Request_num=6 , 
    vehicle_num=vehicle_num , 
    init_capacity=1 , 
    maximum_Request_num=12,
    step_duration=1 , 
    total_update_times=1, 
    PMPO=False 
)
RoadNetwork  = simulator_environment.reset()
print(f"RoadNetwork:{RoadNetwork}\n" )

while 1 : 
    vehicle_routes = list()
    for i in range(vehicle_num):
        route = str(input(f"Input route for vehicle {i} use a,b,c ")).split(",")  
        route = [int(node) for node in route]
        print(f"Input route for vehicle {i} :{route}")
        vehicle_routes.append(route)
    
    RoadNetwork = simulator_environment.step(vehicle_routes=vehicle_routes)
    print(f"RoadNetwork:{RoadNetwork}\n" )
    
        