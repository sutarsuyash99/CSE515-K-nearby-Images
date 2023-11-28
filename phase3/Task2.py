import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import Mongo.mongo_query_np as query
import utils
import distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
from tqdm import tqdm
import DBScan
import inherent_dimensionality

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
        self.min_eps=0.00001
        self.max_eps=40
        self.min_neighbours = 2
        self.max_neighbours = 6
        self.combined_clusters = {}
        self.best_clusters = None
        self.best_eps = None
        self.highest_clusters_formed_till_now = 0
        self.neighbours = 0
        self.min_outliers_count = 5000
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
        if(min_eps < max_eps and (diff > 0.001)):
            labels, core_points = DBScan.fast_db_scan(image_vectors, neighbours, mid)
            unique, counts = np.unique(labels, return_counts=True)
            current_outliers = 0 if unique[0] == -1 else counts[0]
            if len(unique) == self.number_of_clusters+1:
                print(f"{label}: {mid} : {neighbours}: {len(unique)}")
                # Best cluster criteria = minimum number of outliers i.e min of counts of 0th index
                clusters = {}
                clusters["core_sample_indices_"] = core_points
                clusters["labels_"] = labels
                components = [image_vectors[i] for i in core_points]
                clusters["components_"] = components

                return clusters, mid, True
            
            if len(unique) >= self.highest_clusters_formed_till_now:
                # if current_outliers <= self.min_outliers_count:
                self.highest_clusters_formed_till_now = len(unique)
                clusters = {}
                clusters["core_sample_indices_"] = core_points
                clusters["labels_"] = labels
                components = [image_vectors[i] for i in core_points]
                clusters["components_"] = components
                self.best_clusters = clusters
                self.eps = mid
                self.neighbours = neighbours
                self.min_outliers_count = current_outliers
            
            if len(unique) > self.number_of_clusters+1:
                # number of cluster created are higher than required, that means, eps is too small
                # look in the higher values of eps
                # print("increase eps")
                result, epsilon, found = self.dbscan_logic(mid, max_eps, neighbours, image_vectors, label)
                if found:
                    return result, epsilon, found
            elif len(unique) < self.number_of_clusters+1:
                # number of cluster created are lower than required, that means, eps is too high
                # look in the lower values of eps
                # print("decrease eps")
                result, epsilon, found = self.dbscan_logic(min_eps, mid, neighbours, image_vectors, label)
                if found:
                    return result, epsilon, found

        return self.best_clusters,self.best_eps,False

    def get_number_of_clusters(self):
        print('\tEnter Number of Cluseters (c) : ')
        c = utils.int_input()
        self.number_of_clusters = c

    def mds_call(self, label, data):
        # mds = MDS(n_components=2, random_state=0)
        # data_2d = mds.fit_transform(data)
        # print(data_2d.shape)
        # print(f"type of data_2d = {type(data_2d)}")
        data = np.array(data)
        data_2d = inherent_dimensionality.mds(label, data, 2)
        # print(data_2d.shape)
        # print(f"type of data_2d = {type(data_2d)}")
        return data_2d

    def visualize_clusters(self, data_2d, labels, title):

        unique_labels = np.unique(labels)

        plt.figure(figsize=(8, 6))
        for label in unique_labels:
            cluster_points = data_2d[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_thumbnails(self, data_2d, labels, original_image_ids):
        # actual_indices = [x * 2 for x in original_image_ids]
        pil_images = []
        fig, ax = plt.subplots()
        for i in original_image_ids:
            pil_images.append(np.array(self.dataset[i][0]))

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        for label in unique_labels:
            cluster_points = data_2d[labels == label]
            # print(type(cluster_points))
            for point, image in zip(cluster_points, pil_images):
                imagebox = OffsetImage(image, zoom=0.1)  # Adjust the zoom factor as needed
                # print(point)
                # print(type(point))
                ab = AnnotationBbox(imagebox, point, frameon=False, pad=0)
                ax.add_artist(ab)
            
        x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
        y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])

        # Adjust x and y axes accordingly
        plt.xlim(x_min - 1, x_max + 1)
        plt.ylim(y_min - 1, y_max + 1)
        plt.tight_layout()
        ax.set_title('Image Thumbnails in Scatter Plot')
        plt.show()

    def write_to_file(self):
        file_name = 'clusters_data.pkl'
        torch.save(self.combined_clusters, file_name)
    
    def process_individual_label(self, label_vectors, original_image_ids, selected_label):
        print(selected_label)
        found_combination = False
        neighbours = self.min_neighbours
        self.best_clusters = None
        while neighbours < self.max_neighbours and not found_combination:
            # print(f"neighbours = {neighbours}")
            clusters, eps, found_combination = self.dbscan_logic(self.min_eps, self.max_eps, neighbours, label_vectors, selected_label)
            # print(f"found_combination = {found_combination}")
            if found_combination:
                print(f"label = {selected_label}: eps = {eps} total_clusters formed = {self.number_of_clusters + 1}")
                unique, counts = np.unique(clusters["labels_"], return_counts=True)
                print(unique)
                print(counts)
                self.combined_clusters[selected_label] = {
                    "core_components_": clusters["components_"],
                    "core_sample_indices": clusters["core_sample_indices_"],
                    "labels": clusters["labels_"],
                    "original_image_ids": original_image_ids,
                    "eps": eps,
                    "neighbours": neighbours
                }
            # increase the number of neighbors
            neighbours += 1

        if not found_combination:
            print(f"label = {selected_label}: eps = {self.eps} total_clusters formed = {self.highest_clusters_formed_till_now}")
            unique, counts = np.unique(self.best_clusters["labels_"], return_counts=True)
            print(unique)
            print(counts)
            self.combined_clusters[selected_label] = {
                "core_components_": self.best_clusters["components_"],
                "core_sample_indices": self.best_clusters["core_sample_indices_"],
                "labels": self.best_clusters["labels_"],
                "original_image_ids": original_image_ids,
                "eps": self.eps,
                "neighbours": self.neighbours
            }

        self.best_clusters = None
        self.eps = None
        self.neighbours = 0
        self.highest_clusters_formed_till_now = 0


    def visualization_options(self):
        choice = -1
        
        while choice != 2:
            print('\
                \n\n\
                \n 1. Choose a label to visualize it \
                \n 2. Return to main menu \
                \n\n')
        
            choice = utils.int_input()
            file_data = utils.read_file("clusters_data.pkl")

            match choice:
                case 1:
                    label_number = utils.get_user_input_label()
                    selected_label = self.labels[label_number]

                    label_vectors = self.grouped_data[selected_label]
                    data_2d = self.mds_call(selected_label, label_vectors)
                    if len(self.combined_clusters.keys()) > 0:
                        # Plot the graph
                        plt1 = self.visualize_clusters(data_2d, self.combined_clusters[selected_label]["labels"], selected_label)
                        # plot the thumbnails
                        plt2 = self.visualize_thumbnails(data_2d, self.combined_clusters[selected_label]["labels"], self.combined_clusters[selected_label]["original_image_ids"])
                    else:
                        # Plot the graph
                        self.visualize_clusters(data_2d, file_data[selected_label]["labels"], selected_label)
                        # plot the thumbnails
                        self.visualize_thumbnails(data_2d, file_data[selected_label]["labels"], file_data[selected_label]["original_image_ids"])
                    
                case 2:
                    return
    
    def show_all(self):
        ''' process all '''
        self.get_number_of_clusters()
        for label, image_vectors in self.grouped_data.items():
            original_image_ids = self.grouped_data_original_imageId[label]
            self.process_individual_label(image_vectors, original_image_ids, label)
            
        self.write_to_file()
        self.visualization_options()

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
                image = np.array(image)
                eps = value["eps"]
                for vector in value["core_components_"]:
                    vector = np.array(vector)
                    distance = distances.euclidean_distance(vector, image)
                    # If closest is not set or current is less than closest, as well as distance must be less than eps
                    if distance < eps and (closest_distance is None or distance < closest_distance):
                        closest_distance = distance
                        closest_label = label
            predicted_labels_odd.append(closest_label)

        true_labels_odd_id = []
        predicted_labels_odd_id = []
        label_map = {}

        none_indexes = [i for i, val in enumerate(predicted_labels_odd) if val is None]
        none_count = len(none_indexes)

        # Remove None values from predicted_labels_odd
        predicted_labels_odd = [val for val in predicted_labels_odd if val is not None]

        # Remove corresponding values from true_labels_odd
        true_labels_odd = [val for i, val in enumerate(true_labels_odd) if i not in none_indexes]

        for idx, label in enumerate(self.labels):
            label_map[label] = idx  
        for true_label in true_labels_odd:
            true_labels_odd_id.append(label_map[true_label])

        for predicted_label in predicted_labels_odd:
            predicted_labels_odd_id.append(label_map[predicted_label])
        
        true_labels_odd_id = np.array(true_labels_odd_id)
        predicted_labels_odd_id = np.array(predicted_labels_odd_id)
        #Test 
        precision, recall, f1, accuracy  = utils.compute_scores(true_labels_odd_id, predicted_labels_odd_id, avg_type=None, values=True)
        print(len(precision))
        #Display results
        utils.print_scores_per_label(self.dataset, precision, recall, f1, accuracy,'DBSCAN')
        print(f"Could not Predict Labels for {none_count}")


    def execute(self):
        option = -1
        while option != 4:
            print("-"*25, 'MENU', '-'*25)
            print('Select your option:\
                \n\n\
                \n 1. Create Clusters for all labels \
                \n 2. Predict Labels \
                \n 3. Visualize only \
                \n 4. Exit \
                \n\n')
            
            option = utils.int_input()
            match option:
                case 1: self.show_all()
                case 2: self.predict_labels()
                case 3: self.visualization_options()


task2 = Task2()
task2.execute()
# if __name__ == "__init__":
#     task2 = Task2()
#     task2.execute()