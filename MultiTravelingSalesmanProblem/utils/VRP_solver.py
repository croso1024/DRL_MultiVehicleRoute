
from ortools.constraint_solver import pywrapcp , routing_enums_pb2

class ORtools_VRP(object): 
    
    support_algorithms = {
        "GREEDY_DESCENT" : routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT, 
        "GUIDED_LOCAL_SEARCH" : routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING" : routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING , 
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH , 
        "GENERIC_TABU_SERACH" : routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH , 
    }
    
    def __init__(self,dist_matrix , vehicle_num =1 , depot_index =0 , time_limit = None ,return_total=False,algo="GUIDED_LOCAL_SEARCH"):
        self.dist_matrix = dist_matrix 
        self.num_vehicle = vehicle_num 
        self.depot = depot_index 
        self.return_total = return_total
        
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.dist_matrix) , 
            self.num_vehicle , 
            self.depot , 
        )
        
        self.routingModel = pywrapcp.RoutingModel(self.manager) 
        self.transit_callback_index = self.routingModel.RegisterTransitCallback(self.distance_callback_func)
        self.routingModel.SetArcCostEvaluatorOfAllVehicles(
            self.transit_callback_index
        )
        
        self.routingModel.AddDimension(
            self.transit_callback_index , 
            0 , 
            50000000 , 
            True , 
            "Distance"
        )
        
        self.distance_dimension = self.routingModel.GetDimensionOrDie("Distance")
        self.distance_dimension.SetGlobalSpanCostCoefficient(100)
        self.search_parameter = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameter.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        self.search_parameter.local_search_metaheuristic = (
            #routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            self.support_algorithms[algo]
        )
        if time_limit : 
            self.search_parameter.time_limit.FromSeconds(time_limit) 
            
    def distance_callback_func(self,u,v): 
        src = self.manager.IndexToNode(u) 
        dst = self.manager.IndexToNode(v) 
        return self.dist_matrix[src][dst]
    
    
    def solution_tours(self , solution): 
        #print(f"Objective : {solution.ObjectiveValue()}")

        max_route_distance = 0 
        total_distance = 0 
        for vehicle_id in range(self.num_vehicle): 
            index = self.routingModel.Start(vehicle_id)
            route_distance = 0 
            while not self.routingModel.IsEnd(index) : 
                previous_index = index 
                index = solution.Value(self.routingModel.NextVar(index))
                route_distance += self.routingModel.GetArcCostForVehicle(
                    previous_index , index , vehicle_id
                )
            total_distance+=route_distance    
            max_route_distance = max(route_distance , max_route_distance) 
        if self.return_total : 
            return max_route_distance ,total_distance
        else : 
            return max_route_distance 
            
    def print_solution(self, solution):
        """Prints solution on console."""
        #print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        total_distance = 0 
        for vehicle_id in range(self.num_vehicle):
            index = self.routingModel.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not self.routingModel.IsEnd(index):
                plan_output += ' {} -> '.format(self.manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(self.routingModel.NextVar(index))
                distance =  self.routingModel.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
                print(f"Distance from {previous_index} to {index} is {distance}")
                route_distance += distance
            plan_output += '{}\n'.format(self.manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            print(plan_output)
            total_distance+= route_distance
            max_route_distance = max(route_distance, max_route_distance)
        print('Maximum of the route distances: {}m'.format(max_route_distance))
        # if self.return_total : 
        #     return max_route_distance , total_distance
        # else :
        #     return max_route_distance
    def solve(self): 
        solution = self.routingModel.SolveWithParameters(self.search_parameter)
        if solution : 
            #self.print_solution(solution)
            return self.solution_tours(solution) 
            #cost = self.print_solution(solution)
        else : 
            print("No solution found ! ") 
            raise RuntimeError("Can't find the solution !")


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


    VRP_instance = ORtools_VRP(dist ,vehicle_num=3, time_limit=2) 
    print(VRP_instance.solve())
