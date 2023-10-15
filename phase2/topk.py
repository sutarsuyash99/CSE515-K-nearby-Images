import utils
from label_vectors import label_fv_kmediods
import Mongo.mongo_query_np as mongo_query
import distances 
import numpy as np
import os
import glob
from tqdm import tqdm


def query_label_image_top_k(k, feature_space_name, fs_id, label_name, label_id ):
    
    label_fd_function_to_use = label_fv_kmediods
    
    #Get the label feature descriptor 
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
        print(f"ImageId: {i[0]}, Distance: {i[1]}")
    print("-" * 20)
    
    
    #While printing diplay even but send normal 
    top_ids = [ int(t[0]/2) for t in top_k ]
    
    return top_ids    
    
    
    
'''    
def query_label_image_top_k_ls(k, feature_space_name, fs_id, label_name, label_id ):
    
    label_fd_function_to_use = label_fv_kmediods
    
    #Get the label feature descriptor 
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
        print(f"ImageId: {i[0]}, Distance: {i[1]}")
    print("-" * 20)
    
    #While printing diplay even but send odd 
    top_ids = [ t[0]/2 for t in top_k ]
    print(top_ids)
    return top_ids    
    



#query_label_image_top_k(2,'color_moment', 1 , 'Faces', 0)
'''