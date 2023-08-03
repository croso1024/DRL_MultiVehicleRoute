from ortools.constraint_solver import pywrapcp , routing_enums_pb2

class ORtools_DSVRP(object): 
    
     
    support_algorithms = {
        "GREEDY_DESCENT" : routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT, 
        "GUIDED_LOCAL_SEARCH" : routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING" : routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING , 
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH , 
        "GENERIC_TABU_SERACH" : routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH , 
    }
    
    
    #GUIDED_LOCAL_SEARCH
    def __init__(self,dist_matrix ,dist_matrix_std ,demand_vector , vehicle_num , capacity_vector
                 , start_pos , depot_index=0, algo="GREEDY_DESCENT",time_limit=None):

        self.dist_matrix = dist_matrix
        self.dist_matrix_std = dist_matrix_std 
        
        assert demand_vector[depot_index] == 0 ,'Depot demand must be zero'
        assert type(capacity_vector) == list and len(capacity_vector) == vehicle_num , "Capacity setting error"
        assert type(start_pos) == list and len(start_pos) == vehicle_num , "Start pos setting error"
        self.algorithms = algo 
        self.demand_vector = demand_vector 
        self.num_vehicle = vehicle_num
        self.depot = depot_index 
        self.depot_list = [depot_index] * vehicle_num
        self.capacity_vector = capacity_vector
        self.start_pos = start_pos
        self.time_limit = time_limit
        
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.dist_matrix) , self.num_vehicle  , self.start_pos , self.depot_list
        )
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        self.transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)
        self.demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.demand_callback)

        self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)
        
        self.routing.AddDimension(
            self.transit_callback_index , 
            0 , 
            300000 , 
            True , 
            "Distance" 
        )
        
        self.routing.AddDimensionWithVehicleCapacity(
            self.demand_callback_index,
            0 , 
            self.capacity_vector , 
            True  , 
            "Capacity"
        )

        penality = 10000000
        for node in range(1,len(self.dist_matrix)): 
            self.routing.AddDisjunction([self.manager.NodeToIndex(node)] , penality)        
        
          
        self.search_parameter = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameter.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        self.search_parameter.local_search_metaheuristic = (
            # routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            self.support_algorithms[self.algorithms]
            # routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
        if time_limit : 
            self.search_parameter.time_limit.FromSeconds(time_limit)
        
        
    def distance_callback(self,from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index) 
        return self.dist_matrix[from_node][to_node]
    
    def demand_callback(self , index): 
        node = self.manager.IndexToNode(index) 
        return self.demand_vector[node]
    
    def print_solution(self,solution): 
        print(f"start print")
        drop_nodes = "Drpped nodes: "
        for node in range(self.routing.Size()): 
            if self.routing.IsStart(node) or self.routing.IsEnd(node): continue 
            if solution.Value(self.routing.NextVar(node)) == node : 
                drop_nodes += " {}".format(self.manager.IndexToNode(node))
        print(f"Dropped node : {drop_nodes}\n")

        total_distance = 0 
        total_load = 0 
        route_path = [] 
        for vehicle_id in range(self.num_vehicle): 
            vehicle_route = list()
            index = self.routing.Start(vehicle_id)
            print(f"Debug Start({vehicle_id}) -> {index}")
            plan_output = "Route for vehicle {}:\n".format(vehicle_id)
            route_distance = 0 
            route_load = 0 
            while not self.routing.IsEnd(index): 
                
                node_index = self.manager.IndexToNode(index)
                route_load += self.demand_vector[node_index]
                #plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                plan_output += " {0} ->" .format(node_index)
                # record the path manual
                vehicle_route.append(node_index)

                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                distance = self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                print(f"Distance from {previous_index} to {index} is {distance}")
                route_distance += distance
            # record the path manual 
            vehicle_route.append(self.manager.IndexToNode(index)) 
            route_path.append(vehicle_route)

            plan_output += ' {0} Load({1})\n'.format(self.manager.IndexToNode(index),
                                                    route_load)
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            plan_output += 'Load of the route: {}\n'.format(route_load)
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print('Total distance of all routes: {}m'.format(total_distance))
        print('Total load of all routes: {}'.format(total_load))
        print(f"Debug route path : {route_path}")
    def solution_cost(self , solution): 
        total_distance = 0 
        total_load = 0 
        for vehicle_id in range(self.num_vehicle): 
            index = self.routing.Start(vehicle_id) 
            route_distance = 0 
            route_load = 0 
            while not self.routing.IsEnd(index): 
                node_index = self.manager.IndexToNode(index) 
                route_load += self.demand_vector[node_index]  
                previous_index = index 
                index = solution.Value(self.routing.NextVar(index))  
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index , index , vehicle_id
                )
            total_distance += route_distance 
            total_load += route_load 
        fulfill_rate = total_load / sum(self.demand_vector)
        return total_distance , fulfill_rate

    def solution_by_Route(self,solution):
        """ 
            Collect the route of or-tools then calculate the objective manually
        """
        total_distance = 0 
        total_load = 0 
        # Step 1. collect the route path for all vehicle & calculate the fulfill rate
        Route_path = list() 
        for vehicle_id in range(self.num_vehicle): 
            vehicle_path = list() 
            index = self.routing.Start(vehicle_id) 
            route_distance = 0 
            route_load = 0 
            while not self.routing.IsEnd(index) : 
                node_index = self.manager.IndexToNode(index) 
                route_load += self.demand_vector[node_index]  
                vehicle_path.append(node_index) 
                index = solution.Value(self.routing.NextVar(index)) 
                #route_distance += self.routing.GetArcCostForVehicle(previous_index , index , vehicle_id)
            vehicle_path.append(self.manager.IndexToNode(index)) 
            Route_path.append(vehicle_path)
            total_load += route_load
        fulfill_rate = total_load / sum(self.demand_vector)
        
        #print(f"Debug Route_path : {Route_path}" )
        # Step 2. calculate the route cost by the path 
        for i , path in enumerate(Route_path): 
            distance_of_vehicle = 0
            for step in range(len(path)-1) : 

                #distance =  self.dist_matrix[ path[step]  ][ path[step+1]  ]
                distance =  self.dist_matrix_std[ path[step]  ][ path[step+1]  ]

                #print(f"Vehicle {i} - from {path[step]} to {path[step+1]} is {distance}")
                distance_of_vehicle += distance
            #print(f"\n  Distance of vehicle {i} : {distance_of_vehicle}  \n")
            total_distance += distance_of_vehicle
        #print(f"Total Distance : {total_distance}")
        return total_distance , fulfill_rate
    
    def get_solution_RouteOnly(self,solution): 
        """
            直接拿solution_by_Route的上半 ,但要注意進入RoutingSimulator的,不需要給path上的第一個點,
            即在OR-tools時會是node額外加上vehicle , 即將每一個path的第一個點砍掉
        
        """
        
        Route_path = list() 
        for vehicle_id in range(self.num_vehicle): 
            vehicle_path = list() 
            index = self.routing.Start(vehicle_id) 
            route_load = 0 
            while not self.routing.IsEnd(index) : 
                node_index = self.manager.IndexToNode(index) 
                route_load += self.demand_vector[node_index]  
                vehicle_path.append(node_index) 
                index = solution.Value(self.routing.NextVar(index)) 
                #route_distance += self.routing.GetArcCostForVehicle(previous_index , index , vehicle_id)
            vehicle_path.append(self.manager.IndexToNode(index)) 
            Route_path.append(vehicle_path)
        Route_path = [path[1:] for path in Route_path]
        # print(f"Get Route : \n{Route_path}\n")
        return Route_path 
    
    def solve(self,route_only=False ): 
        solution = self.routing.SolveWithParameters(self.search_parameter) 
        if solution: 
            # return  self.solution_cost(solution)
            # self.print_solution(solution)
            # print("\n\n ---- \n\n")
            if route_only : 
                return self.get_solution_RouteOnly(solution)
            else :  
                return self.solution_by_Route(solution)
                
            
        else : 
            print("no solution found !")
            return 0 , 0 
        


if __name__ == "__main__": 

    dist  = [   [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
            [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
            [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
            [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
            [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
            [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
            [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
            [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
            [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
            [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
            [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
            [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
            [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0], ]
    
    
    from random import random ,randint
    demand_vector = [0]+[randint(10,25) for i in range(len(dist)-1)]
    
    print(demand_vector,"\n")
    
    CVRP_instance = ORtools_DSVRP(
        dist_matrix=dist , 
        demand_vector=demand_vector , 
        vehicle_num= 3  , 
        capacity_vector=[40,70,150] , 
        start_pos = [4,5,6] ,
        depot_index= 0 , 
        time_limit=3
    )
    CVRP_instance.solve()