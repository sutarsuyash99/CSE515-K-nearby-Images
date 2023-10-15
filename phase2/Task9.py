# Task 9 
# (a) a label l, (b) a user selected latent semantics, and (c) positive integer k, 
# identifies and lists k most likely matching labels, along with their scores, under the selected latent space.
import utils
import label_vectors
import os
import numpy as np
import torch
import dimension_reduction
from sklearn.metrics.pairwise import cosine_similarity
from Mongo.mongo_query_np import get_all_feature_descriptor_for_label, get_all_feature_descriptor
import distances as ds
class task9:
    def __init__(self) -> None:
        self.labels = utils.get_labels()
        self.image_labels = utils.get_image_categories()
        pass  
        
    def select_latent_semantics(self, path):
        # show available files
        print("Available Latent Semantics : ")
        # file_names = os.listdir(path)
        # for i in range(len(file_names)):
        #     file_name = file_names[i]
        #     if file_name != ".gitkeep":
        #         print(f"\t {i} : {file_name}")
        
        # # print("\n\t Select Latent Semantic")
        # semantic_option = utils.int_input()
        # selected_latent_semantic = file_names[semantic_option]
        # print("selected_latent_semantic ",  selected_latent_semantic)

        # find which feature space it is using
        feature_models = utils.feature_model
        feature_models_values = list(feature_models.values())
        
        for feature_space in feature_models_values:
            feature_space = feature_space.strip().lower()
            path = path.lower()
            if feature_space in path:
                detected_feature_space = feature_space
                break  # Exit the loop as soon as a match is found
        print(f"detected feature space = {detected_feature_space}")

        return path, detected_feature_space

    def find_closest_labels_with_label_reference(self, latent_semantics, semantic_vector, k):
        distances = []
        for i in range(len(latent_semantics)):
            distances.append(ds.cosine_similarity(semantic_vector.flatten(), latent_semantics[i]))
        indexed_list = list(enumerate(distances))
        sorted_list = sorted(indexed_list, key=lambda x: x[1])
        output_list = sorted_list[- (k+1):].copy()
        output_list.reverse()
        
        for i in range(0, len(output_list)):
            print(f"label = {self.labels[output_list[i][0]]}, similarity_score = {output_list[i][1]}")
        return

    def find_closest_labels_with_image_reference(self, latent_semantics, semantic_vector, k):
        distances = []
        for i in range(len(latent_semantics)):
            distances.append(ds.cosine_similarity(semantic_vector.flatten(), latent_semantics[i]))
        indexed_list = list(enumerate(distances))
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

        # print(sorted_list)
        labels_of_each_indices = []
        for i in range(0, len(sorted_list)):
            label = self.image_labels[sorted_list[i][0]*2]
            labels_of_each_indices.append(label)
        seen = set()
        count_unique_labels = 0
        for i in range(0, len(labels_of_each_indices)):
            label = labels_of_each_indices[i]
            score = sorted_list[i][1]
            if label not in seen:
                count_unique_labels += 1
                seen.add(label)
                print(f"label = {label}, similarity_score = {score}") 
                if count_unique_labels == k+1:
                    break
        return

    def LS1(self, selected_label, path, k):
        selected_latent_semantic, detected_feature_space = self.select_latent_semantics(path)
        # get particular labels vector
        cur_label_fv = label_vectors.label_fv_kmediods(selected_label, detected_feature_space)

        # load Entire Model Space
        model_space = get_all_feature_descriptor(detected_feature_space)
        
        # find closest image
        closest_image = label_vectors.label_image_distance_using_cosine(len(model_space), cur_label_fv, model_space, 1)
        closest_image_id = closest_image[0][0]

        latent_semantics = torch.load(path)
        semantic_vector = latent_semantics[closest_image_id]

        self.find_closest_labels_with_image_reference(latent_semantics, semantic_vector, k)
        return
    
    def LS2(self,selected_label, path, k):
        selected_latent_semantic, detected_feature_space = self.select_latent_semantics(path)
        index = self.labels.index(selected_label)
        latent_semantics = torch.load(path)
        label_weights = latent_semantics[1][2]
        semantic_vector = label_weights[index]
        self.find_closest_labels_with_label_reference(label_weights, semantic_vector, k)
        return
    
    def LS3(self,selected_label, path, k):
        selected_latent_semantic, detected_feature_space = self.select_latent_semantics(path)
        # cur_label_fv = label_vectors.label_fv_kmediods(selected_label, detected_feature_space)
        index = self.labels.index(selected_label)
        
        latent_semantics = torch.load(path)
        semantic_vector = latent_semantics[index]

        self.find_closest_labels_with_label_reference(latent_semantics, semantic_vector, k)
        return
    
    def LS4(self,selected_label, path, k):
        selected_latent_semantic, detected_feature_space = self.select_latent_semantics(path)

        # get particular labels vector
        cur_label_fv = label_vectors.label_fv_kmediods(selected_label, detected_feature_space)

        # load Entire Model Space
        model_space = get_all_feature_descriptor(detected_feature_space)
        
        # find closest image
        closest_image = label_vectors.label_image_distance_using_cosine(len(model_space), cur_label_fv, model_space, 1)
        closest_image_id = closest_image[0][0]

        latent_semantics = torch.load(path)
        semantic_vector = latent_semantics[closest_image_id]

        self.find_closest_labels_with_image_reference(latent_semantics, semantic_vector, k)
        return

    def menu(self):
        # # Select a label
        
        # labels = utils.print_labels()
        print("Input Label Number")
        label_num = utils.int_input()
        selected_label = self.labels[label_num] # 0 index
        print("\t selected label" , selected_label)
        
        # Show latent_features and give option to select [LS1, LS2, LS3, etc.]
        path, option = utils.get_user_input_latent_semantics()
        print(f"selected latent feature is = {path}")

        k = utils.get_user_input_k()

        match option:
            case 1: self.LS1(selected_label, path, k)
            case 2: self.LS2(selected_label, path, k)
            case 3: self.LS3(selected_label, path, k)
            case 4: self.LS4(selected_label, path, k)
            case default : 
                print("Invalid Input")
                return

if __name__ == "__main__":
    temp = task9()
    temp.menu()
