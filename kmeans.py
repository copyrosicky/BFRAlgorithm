import math
from collections import defaultdict
import random
class KMeans:
    def __init__(self, clusterCnt, dimensions):
        self.clusterCnt = clusterCnt
        self.dimensions = dimensions

    def get_distance_from_point(self, current_point, point):
        dist = 0
        for a, b in zip(current_point, point):
            dist += (a - b)**2
        return math.sqrt(dist)

    def pickKPointsAsClusters(self, inputDataMap, relevant_points):
        first_point = random.randint(0, len(relevant_points) - 1)
        selected_points = set()
        selected_points.add(first_point)
        while len(selected_points) < self.clusterCnt:
            global_maxi = float('-inf')
            global_point = None
            for index in relevant_points:
                dimensions = inputDataMap[index]
                current_point = dimensions
                local_mini = float('inf')
                if index in selected_points:
                    continue 
                for point in selected_points:
                    selected_point_tuple = inputDataMap[point]
                    dist = self.get_distance_from_point(current_point, selected_point_tuple)
                    if dist < local_mini:
                        local_mini = dist
                if local_mini > global_maxi:
                    global_maxi = local_mini
                    global_point =  index
            selected_points.add(global_point)
        return selected_points

    def placePointsInNearestCentroid(self, selected_clusters, inputDataMap, relevant_points):
        res = defaultdict(list)
        for point in relevant_points:
            dimensions = tuple(inputDataMap[point])
            local_mini = math.inf
            cluster_id = None
    
            for (i, point_tuple) in selected_clusters.items():
                dist = self.get_distance_from_point(dimensions, point_tuple)
                if dist < local_mini:
                    local_mini = dist
                    cluster_id = i
            res[cluster_id].append(point)

        return res

    def init_n_d_vector(self, point_n_d: list, num_dimensions: int):
        n_dimensional_vector = []
        for i in range(num_dimensions):
            n_dimensional_vector.append(point_n_d[i])
        return n_dimensional_vector

    def updateLocationsOfKCentroids(self, cid_to_points_map, inputDataMap):
        centroid_locations = defaultdict(tuple)
        for (index, points_list) in cid_to_points_map.items():
            num_dimensions = self.dimensions
            n_dimensional_vector = [0] *  num_dimensions
            for point in points_list:
                point_n_d = inputDataMap[point]
                for (i, point_dimensional_value) in enumerate(point_n_d):
                    n_dimensional_vector[i] += point_dimensional_value
            for i in range(num_dimensions):
                n_dimensional_vector[i] = n_dimensional_vector[i] / len(points_list)
            centroid_locations[index] = tuple(n_dimensional_vector)
        return centroid_locations

    def getClusters(self, inputDataMap, relevant_points):
        selected_clusters = list(self.pickKPointsAsClusters(inputDataMap, relevant_points))
        centroids = defaultdict(tuple)
        for (i, point) in enumerate(selected_clusters):
            centroids[i] = tuple(inputDataMap[point])
        prev_assignment_map = None
        cnt = 1
        while True:
            
            cnt += 1
            if cnt > 6:
                break
            point_assignment_map = self.placePointsInNearestCentroid(centroids, inputDataMap, relevant_points)
            if prev_assignment_map:
                no_movement = True
                for key in prev_assignment_map:
                    if prev_assignment_map[key] != point_assignment_map[key]:
                        no_movement = False
                        break
                if no_movement:
                    break
            new_centroids = self.updateLocationsOfKCentroids(point_assignment_map, inputDataMap)
            centroids = new_centroids
            prev_assignment_map = point_assignment_map


        return (centroids, point_assignment_map)