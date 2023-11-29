import pickle
import numpy as np
import torchvision
import distances as ds

from collections import deque
#For any gien 'point' in the enitire 'data' finds and returns a list of its neighbors bases on 'epsilon'
def range_query(point, data, eps):
    neighbors = []
    for i, data_point in enumerate(data):
        if ds.euclidean_distance(point, data_point) <= eps:
            neighbors.append(i)
            
    return neighbors

def expand_cluster(data, labels, point_index, neighbors, cluster_id, eps, min_samples, core_points):
    #Start by including the initial point(core point) in the current cluster
    labels[point_index] = cluster_id
    i=0
    while i<(len(neighbors)):
        neighbor = neighbors[i]
        
        #If the neighboring point is unvisited then just change its cluster_id to current cluster id(include it in the cluster)
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id
        
        #If the point is univisited first include in the current cluster(labels[neighbor] = cluster_id)
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            
            #Then find out if it's a core point. If it is a core point then include its neighbors in the current neighbors
            #list so that these points will also be included in the cluster
            new_neighbors = range_query(data[neighbor], data, eps)
            if len(new_neighbors) >= min_samples:
                core_points.append(neighbor)
                neighbors.extend(new_neighbors)
                
        i+=1

# not used, this approach is scrapped
def dbscan_basic_approach(data, min_samples, eps):
    core_points = []
    #labels is a list of same length as of data points that we have. For each data point the label assigned will be 
    #initially 0(stands for unvisited)
    data = np.array(data)

    labels = [0]*data.shape[0]
    
    #Cluster id for current cluster(first cluster initialized as 0)
    cluster_id = 0
    
    for i in range(data.shape[0]):
        #if the point if already visited then either a noise point(cluster_id = -1) or belonngs to some cluster(cluster_id=k)
        #So let it go
        if labels[i] != 0:
            continue
            
        #list of neighbors for current point
        neighbors = range_query(data, data[i], eps)
        
        #If the neighborhood of the current point contains less points than min_samples then mark it as noise(labels[i] = -1)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            #IT'S A CORE POINT, include it in current cluster and expand the cluster beginning with this core point
            core_points.append(i)
            cluster_id+=1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples, core_points)
    
    return labels, core_points

class FastDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.visited = set()
        self.data = None
        # create a distance matrix for compute and save all the distances at the start of algorithm only
        self.distance_matrix = None
        #Core Points
        self.core_points = []

    def fit(self, data):
        self.data = np.array(data)
        # mark all data as outlier initially
        self.labels = np.full(len(data), fill_value=-1, dtype=int)
        # precaution
        self.visited.clear()
        self.distance_matrix = self.compute_distance_matrix()

        cluster_id = 0

        for point_id in range(len(data)):
            if point_id not in self.visited:
                self.visited.add(point_id)
                neighbors = self.find_neighbors(point_id)
                # print(len(neighbors))
                # Points will keep getting marked as noise 
                # till we reach a point where this criteria is met, 
                # then from there on out we expand the cluster
                if len(neighbors) < self.min_samples:
                    self.labels[point_id] = -1  # Mark as noise
                else:
                    # assign it to new cluster
                    cluster_id += 1
                    #Core Points
                    self.core_points.append(point_id)
                    self.expand_cluster(point_id, neighbors, cluster_id, self.core_points)

        return self.labels, self.core_points

    def expand_cluster(self, point_id, neighbors, cluster_id, core_points):
        self.labels[point_id] = cluster_id
        neighbors_queue = deque(neighbors)
        # Visit Each of the neighbours identified above
        while neighbors_queue:
            current_point_id = neighbors_queue.popleft()
            if current_point_id not in self.visited:
                self.visited.add(current_point_id)
                new_neighbors = self.find_neighbors(current_point_id)

                if len(new_neighbors) >= self.min_samples:
                    #Core Points
                    core_points.append(current_point_id)
                    neighbors_queue.extend(new_neighbors)

            if self.labels[current_point_id] == -1:
                self.labels[current_point_id] = cluster_id

    def find_neighbors(self, point_id):
        return np.where(self.distance_matrix[point_id] < self.eps)[0]

    def compute_distance_matrix(self):
        data = np.array(self.data)
        num_points = len(data)
        distance_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance_matrix[i, j] = ds.euclidean_distance(data[i], data[j])

        return distance_matrix


def fast_db_scan(data, min_samples, eps):
    fast_dbscan = FastDBSCAN(eps, min_samples)
    labels, core_points = fast_dbscan.fit(data)
    return labels, core_points
