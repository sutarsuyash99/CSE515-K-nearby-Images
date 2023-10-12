from typing import TypedDict
import heapq

import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from distances import cosine_similarity
from utils import get_user_selected_feature_model

def label_fv_init(label: str, labelled_images: TypedDict, feature_space):
    cur = labelled_images[label]
    if len(cur) == 0:
        raise KeyError
    # else: print(cur[0], cur[-1])
    even_cur = [num for num in cur if num%2 == 0]

    # change this to 
    label_features = []
    for i in range(len(even_cur)):
        label_features.append(feature_space[even_cur[i]])
    label_features = np.asarray(label_features)

    return label_features

def label_fv_kmediods(label: str, labelled_images: TypedDict, feature_shape):
    '''
    create label feature vector using the kmediods value of the contained data
    return that feature vector
    '''
    try:
        label_features = label_fv_init(label, labelled_images, feature_shape)
        if label_features.ndim > 2:
            og_shape = label_features.shape
            new_shape = (og_shape[0], np.prod(og_shape[1:]))
            label_features = label_features.reshape(new_shape)
        kmedoids = KMedoids(n_clusters=1).fit(label_features).cluster_centers_
        # print(kmedoids.shape, type(kmedoids))
        return kmedoids
    except KeyError:
        return None


def label_image_distance_using_cosine(max_len: int, label_feature_vectors, dict_all_feature_vectors, k: int):
    distances = []
    for i in tqdm(range(max_len)):
        distances.append(cosine_similarity( label_feature_vectors.flatten(), np.asarray(dict_all_feature_vectors[i]).flatten() ))
    top_k = heapq.nlargest(k, enumerate(distances), key=lambda x: x[1])
    # print(top_k)
    return top_k

# this function is responsible to create feature vector for each label
def create_labelled_feature_vectors(labelled_images):
    '''
    labelled_images from map -> label_id : [images_ids]
    output format: [dict[key: label, value: feature_vector_label], model_space]
    '''    
    model_space, _ = get_user_selected_feature_model()
    if model_space is not None:
        labelled_feature_vectors = {}
        for key in labelled_images.keys():
            # combine all feature vectors into one for label -- labelled embedding
            # add your min, mean, max code for label here
            labelled_feature_vectors[key] = label_fv_kmediods(key, labelled_images, model_space)
        return (labelled_feature_vectors, model_space)
    else: return None