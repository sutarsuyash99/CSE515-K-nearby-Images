import numpy as np
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import Mongo.mongo_query_np as query
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import utils
import distances
from scipy.spatial.distance import pdist, squareform
import cProfile
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

class Task2():
    def __init__(self) -> None:
        ''' Using FC Layer Feature descriptors for this '''
        self.feature_model = "fc_layer"
        self.labelled_data = utils.get_image_categories()
        self.odd_images = utils.get_odd_iamges(self.feature_model, self.labelled_data)
        self.feature_descriptors = query.get_entire_collection(self.feature_model)
        self.labels = utils.get_labels()
        self.dataset,_  = utils.initialise_project()
        self.grouped_data = {}
        self.grouped_data_original_imageId = {}
        self.number_of_clusters = 0
        self.min_eps=0.0001
        self.max_eps=40
        self.min_neighbours = 2
        self.max_neighbours = 6
        self.combined_clusters = {}
        self.best_clusters = None
        self.best_eps = None
        self.highest_clusters_formed_till_now = 0
        self.neighbours = 0
        # data = []
        for item in self.feature_descriptors:
            label = item['label']
            # data.append(item['feature_descriptor'])
            self.grouped_data.setdefault(label, []).append(item['feature_descriptor'])
            self.grouped_data_original_imageId.setdefault(label, []).append(item['imageID'])
            

    def dbscan_logic(self, min_eps=None, max_eps=None, neighbours=None, image_vectors=None, label=None):
        mid = round((min_eps + max_eps) / 2, 4)
        diff = round(max_eps - min_eps, 4)
        found = False
        if(min_eps < max_eps and (diff > 0.1)):
            dbscan = DBSCAN(eps=mid, min_samples=neighbours)
            clusters = dbscan.fit(image_vectors)
            unique, counts = np.unique(clusters.labels_, return_counts=True)
            
            if len(unique) == self.number_of_clusters+1:
                print(f"{label}: {mid} : {neighbours}: {len(unique)}")
                # Best cluster criteria = minimum number of outliers i.e min of counts of 0th index
                return clusters, mid, True
            
            if len(unique) > self.highest_clusters_formed_till_now:
                self.highest_clusters_formed_till_now = len(unique)
                self.best_clusters = clusters
                self.eps = mid
                self.neighbours = neighbours

            result, epsilon, found = self.dbscan_logic(mid, max_eps, neighbours, image_vectors, label)
            if found:
                return result, epsilon, found
            
            result, epsilon, found = self.dbscan_logic(min_eps, mid, neighbours, image_vectors, label)
            if found:
                return result, epsilon, found
        
        return self.best_clusters,self.best_eps,False

    def get_number_of_clusters(self):
        print('\tEnter Number of Cluseters (c) : ')
        c = utils.int_input()
        self.number_of_clusters = c

    def mds_call(self, data):
        mds = MDS(n_components=2)
        data_2d = mds.fit_transform(data)
        return data_2d

    def visualize_clusters(self, data, labels, title):
        data_2d = self.mds_call(data)

        unique_labels = np.unique(labels)

        plt.figure(figsize=(8, 6))
        for label in unique_labels:
            cluster_points = data_2d[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

        plt.title(title)
        plt.legend()
        plt.show()

    def visualize_thumbnails(self, label_vectors, labels, original_image_ids):
        data_2d = self.mds_call(label_vectors)
        # actual_indices = [x * 2 for x in original_image_ids]
        pil_images = []
        fig, ax = plt.subplots()
        for i in original_image_ids:
            pil_images.append(np.array(self.dataset[i][0]))

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        for label in unique_labels:
            cluster_points = data_2d[labels == label]

            for point, image in zip(cluster_points, pil_images):
                imagebox = OffsetImage(image, zoom=0.1)  # Adjust the zoom factor as needed
                ab = AnnotationBbox(imagebox, point, frameon=False, pad=0)
                ax.add_artist(ab)
            
        x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
        y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])

        # Adjust x and y axes accordingly
        plt.xlim(x_min - 1, x_max + 1)
        plt.ylim(y_min - 1, y_max + 1)

        ax.set_title('Image Thumbnails in Scatter Plot')
        plt.show()
        # print(original_image_ids)

    def write_to_file(self):
        file_name = 'clusters_data.pkl'
        torch.save(self.combined_clusters, file_name)
    
    def process_individual_label(self, label_vectors, original_image_ids, selected_label):
        print(selected_label)
        found_combination = False
        neighbours = self.min_neighbours
        self.best_clusters = None
        while neighbours < self.max_neighbours and not found_combination:
            clusters, eps, found_combination = self.dbscan_logic(self.min_eps, self.max_eps, neighbours, label_vectors, selected_label)
            
            if found_combination:
                self.combined_clusters[selected_label] = {
                    "core_components_": clusters.components_,
                    "core_sample_indices": clusters.core_sample_indices_,
                    "labels": clusters.labels_,
                    "original_image_ids": original_image_ids,
                    "eps": eps,
                    "neighbours": neighbours
                }
            # increase the number of neighbors
            neighbours += 1

        if not found_combination:
            self.combined_clusters[selected_label] = {
                "core_components_": self.best_clusters.components_,
                "core_sample_indices": self.best_clusters.core_sample_indices_,
                "labels": self.best_clusters.labels_,
                "original_image_ids": original_image_ids,
                "eps": self.eps,
                "neighbours": self.neighbours
            }

        print(f"label = {selected_label}: total_clusters formed = {self.highest_clusters_formed_till_now}")
        # Print Cluster distribution
        unique, counts = np.unique(self.best_clusters.labels_, return_counts=True)
        print(unique)
        print(counts)
        self.best_clusters = None
        self.eps = None
        self.neighbours = 0
        self.highest_clusters_formed_till_now = 0
        
        # Plot the graph
        # self.visualize_clusters(label_vectors, self.combined_clusters[selected_label]["labels"], selected_label)
        # plot the thumbnails
        # print(self.combined_clusters[selected_label])
        # self.visualize_thumbnails(label_vectors, self.combined_clusters[selected_label]["labels"], self.combined_clusters[selected_label]["original_image_ids"])

    def specific_label(self):
        label_number = utils.get_user_input_label()
        selected_label = self.labels[label_number]
        
        # self.get_number_of_clusters()

        label_vectors = self.grouped_data[selected_label]
        original_image_ids = self.grouped_data_original_imageId[selected_label]
        self.process_individual_label(label_vectors, original_image_ids, selected_label)

    def show_all(self):
        ''' process all '''
        # self.get_number_of_clusters()
        for label, image_vectors in self.grouped_data.items():
            original_image_ids = self.grouped_data_original_imageId[label]
            self.process_individual_label(image_vectors, original_image_ids, label)

        self.write_to_file()

    def read_file(self):
        file_data = torch.load("clusters_data.pkl")
        return file_data

    def predict_labels(self):
        # Get all odd images
        # read file
        file_data = self.read_file()
        
        true_labels_odd = []
        predicted_labels_odd = []
        for odd_image in tqdm(self.odd_images, desc="Processing Odd Images", unit="image"):
            closest_distance = None
            closest_label = None
            true_labels_odd.append(odd_image["label"])
            for label, value in file_data.items():
                image = odd_image["feature_descriptor"]
                for vector in value["core_components_"]:
                    distance = distances.euclidean_distance(vector, image)
                    if closest_distance is None or distance < closest_distance:
                        closest_distance = distance
                        closest_label = label
            predicted_labels_odd.append(closest_label)

        print(true_labels_odd)
        print(predicted_labels_odd)

        precision = precision_score(true_labels_odd, predicted_labels_odd, average='weighted')
        recall = recall_score(true_labels_odd, predicted_labels_odd, average='weighted')
        f1 = f1_score(true_labels_odd, predicted_labels_odd, average='weighted')
        accuracy = accuracy_score(true_labels_odd, predicted_labels_odd)

        print(f"precision = {precision}")
        print(f"recall = {recall}")
        print(f"f1 = {f1}")
        print(f"accuracy = {accuracy}")
            


    def execute(self):
        self.get_number_of_clusters()
        print("-"*25, 'MENU', '-'*25)
        print('Select your option:\
            \n\n\
            \n 1. Select Specific Label\
            \n 2. Process All Labels \
            \n 3. Predict Labels \
            \n\n')
        
        option = utils.int_input()
        match option:
            case 1: self.specific_label()
            case 2: self.show_all()
            case 3: self.predict_labels()


task2 = Task2()
task2.execute()
# if __name__ == "__init__":
#     task2 = Task2()
#     task2.execute()