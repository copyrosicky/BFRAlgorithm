from collections import defaultdict
from math import ceil, sqrt
import sys
import os
from kmeans import KMeans
import math
from itertools import combinations
import json

class Homework:
    def set_num_dimensions(self, dimensions):
        self.num_dimensions = dimensions

    def get_num_dimensions(self):
        return self.num_dimensions
    
    def extractDimensions(self, subset):
        recordData = subset[0].rstrip().split(",")[1:]
        return len(recordData)

    def getIndexAndData(self, subset):
        input_data_map = defaultdict(list)
        for line in subset:
            line = line.rstrip()
            comma_separated_data = line.split(",")
            i = int(comma_separated_data[0])
            dimensions = [float(item) for item in comma_separated_data[1:]]
            input_data_map[i] = dimensions
        return input_data_map
    
    def generate_ds_stats(self, point_assignment_map, input_data_map, filtered_keys=None):
        res = defaultdict(tuple)
        for key in point_assignment_map:
            if filtered_keys != None:
                if key not in filtered_keys:
                    continue
            N = len(point_assignment_map[key])
            sum_i = [0] * self.get_num_dimensions()
            sum_sq = [0] * self.get_num_dimensions()
            for point in point_assignment_map[key]:
                for (i, value) in enumerate(input_data_map[point]):
                    sum_i[i] += value
                    sum_sq[i] += (value * value)
            res[key] = (N, sum_i, sum_sq)
        return res

    def get_mahalanobis_dist(self, point, centroid, cs_stats):
        """
        Args:
            point (Tuple): (p_1, p_2, .... p_d)
            centroid (Tuple): (c_1, ... c_d)
            cs_stats (Tuple): (N, sum vector, sum_square_vector)
        """
        N = cs_stats[0]
        sum_vector = cs_stats[1]
        sum_sq_vector = cs_stats[2]
        num_dimens = len(sum_vector)
        dist = 0

        for i in range(num_dimens):
            variance_i = (sum_sq_vector[i] / N) - ((sum_vector[i] / N) ** 2)
            std_dev = math.sqrt(variance_i)
            if std_dev != 0:
                dist += (((point[i] - centroid[i]) / std_dev) ** 2)

        return math.sqrt(dist)

    def update_cluster_stats(self, N, sum_vector, sum_sq_vector, num_dimens, point_tuple):
        for it in range(num_dimens):
            sum_vector[it] += point_tuple[it]
            sum_sq_vector[it] += (point_tuple[it] * point_tuple[it])
        return (N + 1, sum_vector, sum_sq_vector)
    
    def update_cluster_centroid(self, N, sum_vector):
        arr = []
        for sum_i in sum_vector:
            arr.append(sum_i / N)
        return tuple(arr)

    def merge_old_cs_with_new_cs(self, cs_centroids_map, cs_stats_map, cs_assignment_map, updated_cs_centroids_map, updated_cs_stats_map, updated_cs_assignment_map):

        merged_cs_centroids_map = defaultdict(tuple)
        merged_cs_assignment_map = defaultdict(list)
        merged_cs_stats_map = defaultdict(tuple)
        ind = 0
        for key in updated_cs_centroids_map.keys():
            merged_cs_assignment_map[ind] = updated_cs_assignment_map[key]
            merged_cs_centroids_map[ind] = updated_cs_centroids_map[key]
            merged_cs_stats_map[ind] = updated_cs_stats_map[key]
            ind += 1

        for key in cs_centroids_map.keys():
            merged_cs_assignment_map[ind] = cs_assignment_map[key]
            merged_cs_centroids_map[ind] = cs_centroids_map[key]
            merged_cs_stats_map[ind] = cs_stats_map[key]
            ind += 1

        return (merged_cs_centroids_map, merged_cs_stats_map, merged_cs_assignment_map)


    def merge_cs_and_rs_to_ds(self, cs_centroids_map, cs_stats_map, cs_assignment_map, rs, extracted_rs_map, ds_centroids_map, ds_stats_map, ds_assignment_map):
        unassigned_rs = []
        unassigned_cs = defaultdict(list)
        # Assign RS to DS.
        for rs_id in rs:
            point_tuple = extracted_rs_map[rs_id]
            d = len(point_tuple)
            local_mini = float('inf')
            closest_ds_id = None
            for (ds_centroid_id, ds_centroid_tuple) in ds_centroids_map.items():
                dist = self.get_mahalanobis_dist(point=point_tuple, centroid=ds_centroid_tuple, cs_stats=ds_stats_map[ds_centroid_id])
                if dist < local_mini:
                    local_mini = dist
                    closest_ds_id = ds_centroid_id
            if closest_ds_id:
                    # 1. Assign point to cluster
                    cluster_stat = ds_stats_map[closest_ds_id]
                    ds_assignment_map[closest_ds_id].append(rs_id)
                    # 2. Update Cluster Stats
                    ds_stats_map[closest_ds_id] = self.update_cluster_stats(\
                        N=cluster_stat[0],\
                        sum_vector=cluster_stat[1],\
                        sum_sq_vector=cluster_stat[2],\
                        num_dimens=d,\
                        point_tuple=point_tuple)

                    ds_centroids_map[closest_ds_id] = self.update_cluster_centroid(\
                        N=ds_stats_map[closest_ds_id][0],\
                        sum_vector=ds_stats_map[closest_ds_id][1])    
            else:
                unassigned_rs.append(rs_id)            

        for cs_centroid_id in cs_centroids_map:
            centroid_cs = cs_centroids_map[cs_centroid_id]
            local_mini = float('inf')
            closest_ds_id = None
            for (ds_centroid_id, ds_centroid_tuple) in ds_centroids_map.items():
                dist = self.get_mahalanobis_dist(point=centroid_cs,centroid=ds_centroid_tuple,cs_stats=ds_stats_map[ds_centroid_id])
                if dist < local_mini:
                    local_mini = dist
                    closest_ds_id = ds_centroid_id
            
            if closest_ds_id:
                points_list = cs_assignment_map[cs_centroid_id]
                for point in points_list:
                    ds_assignment_map[closest_ds_id].append(point)
                merged_stats = self.get_merged_centroid_stats(ds_stats_map[closest_ds_id], cs_stats_map[cs_centroid_id])
                merged_centroid = self.get_merged_centroid(N=merged_stats[0], sum_vector=merged_stats[1])
                ds_stats_map[closest_ds_id] = merged_stats
                ds_centroids_map[closest_ds_id] = merged_centroid
            else:
                unassigned_cs[cs_centroid_id] = cs_assignment_map[cs_centroid_id]
        
        return (unassigned_rs, unassigned_cs)

    def get_merged_centroid(self, N, sum_vector):
        res = []
        for sum_i in sum_vector:
            res.append(sum_i / N)
        return tuple(res)

    def get_merged_centroid_stats(self, centroid_1, centroid_2):
        n_1 = centroid_1[0]
        sum_vector_1 = centroid_1[1]
        sumsq_vector_1 = centroid_1[2]
        n_2 = centroid_2[0]
        sum_vector_2 = centroid_2[1]
        sumsq_vector_2 =centroid_2[2]
        d = len(sum_vector_1)

        new_sum_vector = []
        new_sumsq_vector = []
        for i in range(d):
            new_sum_vector.append(sum_vector_1[i] + sum_vector_2[i])
            new_sumsq_vector.append(sumsq_vector_1[i] + sumsq_vector_2[i])

        return (n_1 + n_2, new_sum_vector, new_sumsq_vector)

    def merge_cs_clusters(self, cs_centroids_map, cs_stats_map, cs_assignment_map):
            
        is_merge_found = True
        while is_merge_found:
            is_merge_found = False
            cs_keys = list(cs_centroids_map.keys())
            # Merge all distinct pairs
            merged = set()
            all_possible_merges = []
            for tup in combinations(cs_keys, 2):
                sorted_tuple = sorted(tup)
                c1_id = sorted_tuple[0]
                c2_id = sorted_tuple[1]
                centroid_1 = cs_centroids_map[c1_id]
                d = len(centroid_1)
                centroid_2 = cs_centroids_map[c2_id]
                centroid_1_stats = cs_stats_map[c1_id]

                dist = self.get_mahalanobis_dist(centroid_2, centroid_1, centroid_1_stats)
                if dist < sqrt(d):
                    all_possible_merges.append((c1_id, c2_id, dist))
            new_centroids_map = defaultdict(tuple)
            new_assignment_map = defaultdict(list)
            new_stats_map = defaultdict(tuple)
            new_centroid_id = 0
            all_possible_merges.sort(key=lambda x: (x[2]))
            for (centroid_1, centroid_2, dist) in all_possible_merges:
                if centroid_1 in merged or centroid_2 in merged:
                    continue
                is_merge_found = True
                merged.add(centroid_1)
                merged.add(centroid_2)
                
                merged_stats = self.get_merged_centroid_stats(cs_stats_map[centroid_1], cs_stats_map[centroid_2])
                merged_centroid = self.get_merged_centroid(N=merged_stats[0], sum_vector=merged_stats[1])

                new_centroids_map[new_centroid_id] = merged_centroid
                new_stats_map[new_centroid_id] = merged_stats
                new_assignment_map[new_centroid_id] = cs_assignment_map[centroid_1] + cs_assignment_map[centroid_2]
                new_centroid_id += 1
            
            if not is_merge_found:
                # No CS Clusters to merge.
                break

            for centroid_id in cs_keys:
                if centroid_id in merged:
                    continue
                new_centroids_map[new_centroid_id] = cs_centroids_map[centroid_id]
                new_stats_map[new_centroid_id] = cs_stats_map[centroid_id]
                new_assignment_map[new_centroid_id] = cs_assignment_map[centroid_id]
                new_centroid_id += 1

            cs_centroids_map = new_centroids_map
            cs_stats_map = new_stats_map
            cs_assignment_map = new_assignment_map
        
        return (cs_centroids_map, cs_stats_map, cs_assignment_map)

    def add_new_points(self, input_data_map, ds_centroids_map, ds_cluster_stats_map, ds_assignment_map, cs_centroids_map, cs_stats_map, cs_assignment_map):
        """
        Args:
            input_data_map (dict): Point id -> Point N-Dimensional Feature Tuple
            ds_centroids_map (dict): cluster_id -> tuple of centroid dimensions
            ds_cluster_stats_map (dict): cluster_id -> tuple(N, Sum_i, sum_sq)
            ds_assignment_map (dict): cluster_id -> list of points
            cs_centroids_map (dict): Compressed set cluster_id -> tuple of CS Centroid dimensions
            cs_stats_map (dict): Compressed set cluster_id -> tuple(N, Sum_i vector , Sum squares vector)
            cs_assignment_map (dict): Compressed set cluster id -> List of points 
        """
        rs = set()
        for point in input_data_map:
            added_to_ds = False
            added_to_cs = False
            point_tuple = input_data_map[point]
            d = len(point_tuple)
            nearest_ds_id = None
            local_mini = float('inf')
            for (centroid_id, centroid_tuple) in ds_centroids_map.items():
                cluster_stat = ds_cluster_stats_map[centroid_id]
                dist = self.get_mahalanobis_dist(point_tuple, centroid_tuple, cluster_stat)
                if dist < 2 * math.sqrt(d):
                    if dist < local_mini:
                        local_mini = dist
                        nearest_ds_id = centroid_id

            if nearest_ds_id != None:
                # 1. Assign point to cluster
                added_to_ds = True
                ds_assignment_map[nearest_ds_id].append(point)
                cluster_stat = ds_cluster_stats_map[nearest_ds_id]
                # 2. Update Cluster Stats
                ds_cluster_stats_map[nearest_ds_id] = self.update_cluster_stats(\
                    N=cluster_stat[0],\
                    sum_vector=cluster_stat[1],\
                    sum_sq_vector=cluster_stat[2],\
                    num_dimens=d,\
                    point_tuple=point_tuple)

                ds_centroids_map[nearest_ds_id] = self.update_cluster_centroid(\
                    N=ds_cluster_stats_map[nearest_ds_id][0],\
                    sum_vector=ds_cluster_stats_map[nearest_ds_id][1])
            
            if not added_to_ds:
                local_mini = float('inf')
                nearest_cs_id = None
                for (centroid_id, centroid_tuple) in cs_centroids_map.items():
                    cs_stat = cs_stats_map[centroid_id]
                    dist = self.get_mahalanobis_dist(point_tuple, centroid_tuple, cs_stat)
                    if dist < 2 * math.sqrt(d):
                        if dist < local_mini:
                            local_mini = dist
                            nearest_cs_id = centroid_id
                
                if nearest_cs_id != None:
                    added_to_cs = True
                    cs_stat = cs_stats_map[nearest_cs_id]
                    # 1. Assign point to Compressed Set
                    cs_assignment_map[nearest_cs_id].append(point)
                    # 2. Update Compressed Set Stats
                    cs_stats_map[nearest_cs_id] = self.update_cluster_stats(\
                        N=cs_stat[0],\
                        sum_vector=cs_stat[1],\
                        sum_sq_vector=cs_stat[2],\
                        num_dimens=d,\
                        point_tuple=point_tuple)

                    cs_centroids_map[nearest_cs_id] = self.update_cluster_centroid(\
                        N=cs_stats_map[nearest_cs_id][0], \
                        sum_vector=cs_stats_map[nearest_cs_id][1])
            
            if not added_to_ds and not added_to_cs:
                rs.add(point)
        
        return rs

    def writeToOutputFile(self, output_file_path, ds_assignment_map, rs_list, cs_assignment_map):
        res = {}
        for cluster_id, points_list in ds_assignment_map.items():
            for point in points_list:
                res[str(point)] = cluster_id
        
        for rs_id in rs_list:
            res[str(rs_id)] = -1
        
        for cluster_id, points_list in cs_assignment_map.items():
            for point in points_list:
                res[str(point)] = -1

        with open(output_file_path, "w+") as of:
            of.write(json.dumps(res))
        of.close()

    def writeToIntermediateFile(self, round_id, ds_assignment_map, cs_assignment_map, rs, intermediate_file_path):
        num_clusters_ds = len(ds_assignment_map.keys())
        
        num_points_ds = 0
        for points_list in ds_assignment_map.values():
            num_points_ds += len(points_list)
        
        num_clusters_cs = len(cs_assignment_map.keys())

        num_points_cs = 0
        for points_list in cs_assignment_map.values():
            num_points_cs += len(points_list)
        
        num_points_rs = len(rs)

        with open(intermediate_file_path, "a") as of:
            if round_id == 1:
                of.write("round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained\n")
            data = [str(round_id), str(num_clusters_ds), str(num_points_ds), str(num_clusters_cs), str(num_points_cs), str(num_points_rs)]
            of.write((",".join(data))+"\n") 
        of.close()

    def run_clustering_on_rs(self, n_clusters, extracted_rs_map, cs_centroids_map, cs_stats_map, cs_assignment_map, rs):
        kmeans_obj = KMeans(3 * n_clusters, self.get_num_dimensions())

        updated_cs_centroids_map, updated_cs_assignment_map = kmeans_obj.getClusters(extracted_rs_map, rs)
        updated_rs = []
        updated_cs_cluster_ids = set()
        for cluster_id, points_list in updated_cs_assignment_map.items():
            if len(points_list) > 1:
                updated_cs_cluster_ids.add(cluster_id)
            else:
                updated_rs.extend(points_list)
        rs = updated_rs

        updated_cs_assignment_map = {key: updated_cs_assignment_map[key] for key in updated_cs_cluster_ids}
        updated_cs_centroids_map = {key: updated_cs_centroids_map[key] for key in updated_cs_cluster_ids}
        updated_cs_stats = self.generate_ds_stats(updated_cs_assignment_map, extracted_rs_map, updated_cs_cluster_ids)

        (cs_centroids_map, cs_stats_map, cs_assignment_map) = self.merge_old_cs_with_new_cs(cs_centroids_map=cs_centroids_map,\
                cs_stats_map=cs_stats_map,\
                cs_assignment_map=cs_assignment_map,\
                updated_cs_centroids_map=updated_cs_centroids_map,\
                updated_cs_stats_map=updated_cs_stats,\
                updated_cs_assignment_map=updated_cs_assignment_map)

        return (cs_centroids_map, cs_stats_map, cs_assignment_map)
    
    def main(self):
        input_dir = sys.argv[1]
        n_clusters = int(sys.argv[2])
        cluster_results_path = sys.argv[3]
        intermediate_results_path = sys.argv[4]
        extracted_rs_map = defaultdict(tuple)
        num_files = len(os.listdir(input_dir))

        for index, file_path in enumerate(sorted(os.listdir(input_dir))):
            input_file_path = os.path.join(input_dir, file_path)
            
            with open(input_file_path) as csv_file:
                content = csv_file.readlines()
                
                content_size = len(content)
                if index == 0:
                    subset_size = ceil(content_size * 0.2)
                    subset = content[:subset_size]
                    rest = content[subset_size:]
                    dimensions = self.extractDimensions(subset)
                    self.set_num_dimensions(dimensions=dimensions)
                    # Step 2: 
                    input_data_map = self.getIndexAndData(subset)
                    kmeans_obj = KMeans(3 * n_clusters, self.get_num_dimensions())
                    ds_centroids, ds_assignment_map = kmeans_obj.getClusters(input_data_map, list(input_data_map.keys()))
                    rs = []
                    common = []
                    for val_list in ds_assignment_map.values():
                        if len(val_list) < 2:
                            rs.extend(val_list)
                        else:
                            common.extend(val_list)
                    
                    kmeans_obj2 = KMeans(n_clusters, self.get_num_dimensions())
                    ds_centroids, ds_assignment_map = kmeans_obj2.getClusters(input_data_map, common)
                    ds_stats_map = self.generate_ds_stats(ds_assignment_map, input_data_map)
                    # Define things in case step 5 does not go through. Lack of RS. 
                    cs_centroids_map = defaultdict(tuple)
                    cs_assignment_map = defaultdict(list)
                    cs_stats_map = defaultdict(tuple)

                    # # Step 5:
                    if rs and len(rs) >= 3 * n_clusters:
                        kmeans_obj3 = KMeans(3 * n_clusters, self.get_num_dimensions())
                        cs_centroids_map, cs_assignment_map = kmeans_obj3.getClusters(input_data_map, rs)
                        cs_cluster_ids = set() # Only cluster ids with more than one points.
                        rs = []
                        for cluster_id, points_list in cs_assignment_map.items():
                            if len(points_list) > 1:
                                cs_cluster_ids.add(cluster_id)
                            else:
                                rs.extend(points_list)
                        cs_stats_map = self.generate_ds_stats(cs_assignment_map, input_data_map, cs_cluster_ids)

                    # # Step 7: Load Rest of file
                    rest_data_map = self.getIndexAndData(rest)
                    new_rs = self.add_new_points(input_data_map=rest_data_map,\
                        ds_centroids_map=ds_centroids,\
                        ds_cluster_stats_map=ds_stats_map,\
                        ds_assignment_map=ds_assignment_map,\
                        cs_centroids_map=cs_centroids_map,\
                        cs_assignment_map=cs_assignment_map,\
                        cs_stats_map=cs_stats_map)
                    rs.extend(new_rs)
                    for rs_id in rs:
                        if rs_id in input_data_map:
                            extracted_rs_map[rs_id] = input_data_map[rs_id]
                        elif rs_id in rest_data_map:
                            extracted_rs_map[rs_id] = rest_data_map[rs_id]
                    if rs and (len(rs) >= 3  * n_clusters):
                        (cs_centroids_map, cs_stats_map, cs_assignment_map) = self.run_clustering_on_rs(n_clusters=n_clusters, \
                            extracted_rs_map=extracted_rs_map,\
                            cs_centroids_map=cs_centroids_map,\
                            cs_stats_map=cs_stats_map,\
                            cs_assignment_map=cs_assignment_map,\
                            rs = rs)

                    (cs_centroids_map, cs_stats_map, cs_assignment_map) = self.merge_cs_clusters(cs_centroids_map=cs_centroids_map, \
                            cs_stats_map=cs_stats_map, \
                            cs_assignment_map=cs_assignment_map)
                    
                    self.writeToIntermediateFile(round_id=index + 1, \
                        ds_assignment_map=ds_assignment_map, \
                        cs_assignment_map=cs_assignment_map, \
                        rs=rs, \
                        intermediate_file_path=intermediate_results_path)
                else:
                    new_data_map = self.getIndexAndData(content)
                    new_rs = self.add_new_points(input_data_map=new_data_map,\
                        ds_centroids_map=ds_centroids,\
                        ds_cluster_stats_map=ds_stats_map,\
                        ds_assignment_map=ds_assignment_map,\
                        cs_centroids_map=cs_centroids_map,\
                        cs_assignment_map=cs_assignment_map,\
                        cs_stats_map=cs_stats_map)
                    rs.extend(new_rs)
                    for rs_id in new_rs:
                        extracted_rs_map[rs_id] = new_data_map[rs_id]
                    if rs and len(rs) >= 3  * n_clusters:
                        (cs_centroids_map, cs_stats_map, cs_assignment_map) = self.run_clustering_on_rs(n_clusters=n_clusters, \
                            extracted_rs_map=extracted_rs_map,\
                            cs_centroids_map=cs_centroids_map,\
                            cs_stats_map=cs_stats_map,\
                            cs_assignment_map=cs_assignment_map,\
                            rs = rs)

                    (cs_centroids_map, cs_stats_map, cs_assignment_map) = self.merge_cs_clusters(cs_centroids_map=cs_centroids_map, \
                            cs_stats_map=cs_stats_map, \
                            cs_assignment_map=cs_assignment_map)
                    if index != num_files - 1:
                        self.writeToIntermediateFile(round_id=index + 1, \
                            ds_assignment_map=ds_assignment_map, \
                            cs_assignment_map=cs_assignment_map, \
                            rs=rs, \
                            intermediate_file_path=intermediate_results_path)

        (unassigned_rs, unassigned_cs) = self.merge_cs_and_rs_to_ds(cs_centroids_map=cs_centroids_map,\
            cs_stats_map=cs_stats_map, \
            cs_assignment_map=cs_assignment_map, \
            rs=rs, \
            extracted_rs_map=extracted_rs_map, \
            ds_centroids_map=ds_centroids,\
            ds_stats_map=ds_stats_map,\
            ds_assignment_map=ds_assignment_map)

        self.writeToIntermediateFile(round_id=num_files, \
            ds_assignment_map=ds_assignment_map, \
            cs_assignment_map=unassigned_cs,\
            rs=unassigned_rs,
            intermediate_file_path=intermediate_results_path)
        self.writeToOutputFile(output_file_path=cluster_results_path, ds_assignment_map=ds_assignment_map,rs_list=unassigned_rs, cs_assignment_map=unassigned_cs)


if __name__ == "__main__":
    h = Homework()
    h.main()