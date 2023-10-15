import utils
import label_vectors
from label_vectors import label_fv_kmediods
import Mongo.mongo_query_np as mongo_query
import distances 
import numpy as np
import os
import glob
from tqdm import tqdm
import utils
import torch
from distances import cosine_distance


def query_label_image_top_k(k : int, feature_space_name :str, fs_id : int, label_name :str, label_id : int) -> list :
    
    
    #Function to use for label feature descriptor 
    label_fd_function_to_use = label_fv_kmediods
    
    #Get the label feature descriptor using the function
    label_feature_descriptor = label_fd_function_to_use(label_name, feature_space_name)

    #Get images for that feature space 
    image_feature_descriptors = mongo_query.get_all_feature_descriptor(feature_space_name)

    #distance function
    distance_function_to_use = utils.select_distance_function_for_model_space(fs_id)
    
    #top k
    top_distances = []
    for i in range(len(image_feature_descriptors)):
        distance = distance_function_to_use(
            label_feature_descriptor.flatten(),
            image_feature_descriptors[i].flatten(),
        )
        top_distances.append((distance, i * 2))
    top_distances.sort()

    top_k = []
    for i in range(k):
        top_k.append((top_distances[i][1], top_distances[i][0]))

    print("-" * 20)
    for i in top_k:
        #print(f"ImageId: {i[0]}, Distance: {i[1]}")
        print(f"ImageId: {i[0]}")
    print("-" * 20)
    
    
    #While printing diplay even but send normal 
    top_ids = [ int(t[0]/2) for t in top_k ]
    
    return top_ids    
    



def query_label_image_top_k_ls(k : int, feature_space_name : str, fs_id : int, label_name : str, label_id : int, latent_space : int, latent_sematic : str, labelled_images : dict ) -> list :


    
    def compute_closet_distance(cur_label_vector, all_vectors, k) -> list:
            distances = []
            for i in range(len(all_vectors)):
                cur_distance = cosine_distance(
                    cur_label_vector.flatten(), all_vectors[i].flatten()
                )
                distances.append((cur_distance, i))
            distances.sort()
    
            top_k = []
            for i in range(k):
                top_k.append((distances[i][1], distances[i][0]))
            return top_k


    def get_closet_image(cur_label_vector):
        all_image_vectors = mongo_query.get_all_feature_descriptor(
            utils.feature_model[5]
        )
        closest = compute_closet_distance(cur_label_vector, all_image_vectors, 1)
        print(f"Most similar image with distance: {closest}")
        return closest[0][0]

    #label name and id 
    label_index_selected = label_id
    label_selected = label_name
    labelled_images = labelled_images
    
    #Function to use to get the label feature descriptor
    label_fd_function_to_use = label_fv_kmediods
    
    # create and load label vector for current label in question
    # always selected fc layer (gives the best result)
    # Get label feature vector for seed label 
    cur_label_fv = label_fd_function_to_use(label_name, utils.feature_model[5]
        )
    
    #Get file if available
    file  = utils.get_saved_model_files(feature_space_name, latent_space, latent_sematic)
    
   
    latent_model_space = torch.load(file)    
    
    top_k_distances = []
    
    match latent_space:
        case 1:
            # get closest image for each latent_space
            closest_image_id = get_closet_image(cur_label_fv)
            print(f"moving to fc layer --> {closest_image_id * 2}")
            print(
                f"label range: {labelled_images[label_selected][0]} - {labelled_images[label_selected][-1]}"
            )
            # LS1 - compare cur_label_fv with latent_model_space
            # use latent_model_space
            # get top k images
            closest_image_vector = latent_model_space[closest_image_id // 2]
            print(f"Moving to latent space {latent_space} retrieving top {k} images")
            top_k_distances = compute_closet_distance(
                closest_image_vector, latent_model_space, k
            )
        case 2:
            # get closest image for each latent_space
            closest_image_id = get_closet_image(cur_label_fv)

            print(f"moving to fc layer --> {closest_image_id * 2}")
            print(
                f"label range: {labelled_images[label_selected][0]} - {labelled_images[label_selected][-1]}"
            )
           
            print(
                f"Moving to latent space {latent_space} retrieving top {k} images ... image weights distribution"
            )
            latent_model_space = latent_model_space[1][0]
            closest_image_vector = latent_model_space[closest_image_id // 2]
            top_k_distances = compute_closet_distance(
                closest_image_vector, latent_model_space, k
            )
        case 4:
            # get closest image for each latent_space
            closest_image_id = get_closet_image(cur_label_fv)
            print(f"moving to fc layer --> {closest_image_id * 2}")
            print(
                f"label range: {labelled_images[label_selected][0]} - {labelled_images[label_selected][-1]}"
            )
            print(closest_image_id)
            print(f"Moving to latent space {latent_space} retrieving top {k} images")
            closest_image_vector = latent_model_space[closest_image_id // 2]
            top_k_distances = compute_closet_distance(
                closest_image_vector, latent_model_space, k
            )
        case 3:
            
            # get closest label for latent space
            # loop over entire db and get top k images
            '''
            closest_label_index_for_selected_label = compute_closet_distance(
                latent_model_space[label_index_selected], latent_model_space, k
            )
            '''
            closest_label_index_for_selected_label = [label_id]
            
            '''
            print("-" * 25)
            print("Found top k label")
            for i in closest_label_index_for_selected_label:
                print(i)
            print("-" * 25)
            
            # going in FC layer
            print("Found k labels now going in FC layer")
            '''
            collection_name_in_consideration = utils.feature_model[5]
            top_k_distances = []
            all_features = mongo_query.get_all_feature_descriptor(
                collection_name_in_consideration
            )
            
            
            for i in closest_label_index_for_selected_label:
                cur_label = label_name
                cur_ls_model = label_fd_function_to_use(
                    cur_label, collection_name_in_consideration
                )

                # find 1 closest image
                top_k_distances_1 = compute_closet_distance(
                    cur_ls_model, all_features, 1
                )[0]
                print(
                    f"For label {cur_label} found closest image: {top_k_distances_1[0] * 2} with distance: {top_k_distances_1[1]}"
                )
                top_k_distances.append(top_k_distances_1)
            print("-" * 25)
        case default:
            print("yeh toh bada toi hai!!")

    top_k_distances = [(id * 2, distance) for id, distance in top_k_distances]
    
  
    print(top_k_distances)
        
    #While printing diplay even but send normal 
    top_ids = [ int(t[0]/2) for t in top_k_distances ]
    
    return top_ids    
      




