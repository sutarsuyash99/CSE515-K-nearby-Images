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

    def find_closest_LS3(self, latent_semantics, semantic_vector, k):
        similarities = cosine_similarity(latent_semantics, [semantic_vector])
        distances = 1 - similarities

        sorted_indices = np.argsort(distances, axis=0).flatten()
        unique_elements = []
        for i in range (0, len(sorted_indices)):
            unique_elements.append((self.labels[i], similarities[i]))
            if len(unique_elements) == k+1:
                    break
        print(unique_elements)
        return

    def find_closest_labels_with_image_reference(self, latent_semantics, semantic_vector, k):
        similarities = cosine_similarity(latent_semantics, [semantic_vector])
        distances = 1 - similarities

        sorted_indices = np.argsort(distances, axis=0).flatten()
        labels_of_each_indices = []
        for i in sorted_indices:
            label = self.image_labels[i*2]
            labels_of_each_indices.append(label)

        seen = set()
        unique_elements = []
        for i in range(0, len(labels_of_each_indices)):
            item = labels_of_each_indices[i]
            score = similarities[i][0]
            if item not in seen:
                unique_elements.append((item, score))
                seen.add(item)
                if len(unique_elements) == k+1:
                    break
        print(unique_elements)
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
        return
    
    def LS3(self,selected_label, path, k):
        selected_latent_semantic, detected_feature_space = self.select_latent_semantics(path)
        # cur_label_fv = label_vectors.label_fv_kmediods(selected_label, detected_feature_space)
        index = self.labels.index(selected_label)
        
        latent_semantics = torch.load(path)
        semantic_vector = latent_semantics[index]

        self.find_closest_LS3(latent_semantics, semantic_vector, k)
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

        latent_semantics = torch.load(path+"/"+selected_latent_semantic)
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
