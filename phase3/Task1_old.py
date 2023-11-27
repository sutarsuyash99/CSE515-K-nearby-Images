import numpy as np
import utils
import Mongo.mongo_query_np as mongo
import dimension_reduction as dr
import torch 
import os


def get_image_vectors_and_label_ids(feature_model : str) -> np.ndarray :

    even_image_vectors = mongo.get_all_feature_descriptor(feature_model)
    even_image_vectors = utils.convert_higher_dims_to_2d(even_image_vectors)

    even_image_label_ids = np.zeros(len(even_image_vectors))    
    for i in range(len(even_image_vectors)) :
        _ , even_image_label_ids[i] = dataset[i*2]

    odd_image_vectors = utils.get_odd_image_feature_vectors(feature_model)
    if odd_image_vectors is None :
        return
    odd_image_vectors = utils.convert_higher_dims_to_2d(odd_image_vectors)
    odd_image_label_ids = np.zeros(len(odd_image_vectors))
    for i in range(len(odd_image_vectors)) :
        _ , odd_image_label_ids[i] = dataset[i*2+1]   


    return  even_image_vectors, even_image_label_ids, odd_image_vectors, odd_image_label_ids


def get_image_vectors_by_label(label_feature_model : str, even_image_vectors : np.ndarray = None, even_image_label_ids : np.ndarray = None ) -> np.ndarray :

    #image vectors associated with a label :  of array  101 * n * 1000
    if even_image_vectors is None and even_image_label_ids is None :

        print(f"Needs image vectors and label ids")            

    else :

        unique_label_ids = np.unique(even_image_label_ids.astype(int))
        map_image_vectors_by_label = { label_id : [] for label_id in unique_label_ids }
        for label_id, image_vector in zip(even_image_label_ids, even_image_vectors):
            map_image_vectors_by_label[label_id].append(image_vector)
        image_vectors_by_label = [None] * len(map_image_vectors_by_label)
        
        for label_id, image_vectors in map_image_vectors_by_label.items() :
            #print(label_id)
            image_vectors_by_label[label_id] = np.vstack(image_vectors)
        
        return image_vectors_by_label





def get_label_wise_latent_semantics(k : int, even_image_vectors_by_label : list) -> list :


    '''
    Return the latent semantics for all the labels either from saved file or by calculating.
    Calculate latent semantics for all the label matrices created from even images and saves it.
    '''
    filename = f"{k}_label_wise_latent_semantics.pkl"

    ### TEMP ###
    temp_file1 = f"{k}_label_wise_latent_semantics_non_center.pkl"
    temp_file2 = f"{k}_label_wise_latent_semantics_auto_center.pkl"
    temp_file3 = f"{k}_label_wise_latent_semantics_auto_non_center.pkl"

    if os.path.exists(filename) :
        print(f"Label wise latent semantics with k value {k} already exists...\n")
        label_wise_latent_semantics = torch.load(filename)
    else :
        label_wise_latent_semantics = [] 
        for label_id, label_vectors in enumerate(even_image_vectors_by_label) :
            print(f"Calulating latent semantics for label id {label_id} .....")
            U,_,_ = dr.svd(label_vectors, k)
            label_wise_latent_semantics.append(U)

        #Saving for future use
        torch.save(label_wise_latent_semantics, filename)
        
        ### TEMP ###
        label_wise_latent_semantics_non_center = [] 
        label_wise_latent_semantics_auto_center = [] 
        label_wise_latent_semantics_auto_non_center = [] 
        for label_id, label_vectors in enumerate(even_image_vectors_by_label) :
            print(f"Calulating latent semantics for label id {label_id} .....")
            U,_,_ = dr.svd(label_vectors, k, False)
            label_wise_latent_semantics_non_center.append(U)
            

            #By default non center 
            U,_,_ = np.linalg.svd(label_vectors,full_matrices=False)
            label_wise_latent_semantics_auto_non_center.append(U[:,:k])
            


            mean = np.mean(label_vectors, axis=0)
            label_vectors = label_vectors - mean
            U,_,_ = np.linalg.svd(label_vectors,full_matrices=False)
            label_wise_latent_semantics_auto_center.append(U[:,:k])
            

        torch.save(label_wise_latent_semantics_non_center, temp_file1)
        torch.save(label_wise_latent_semantics_non_center, temp_file3)
        torch.save(label_wise_latent_semantics_auto_center, temp_file2)

    return label_wise_latent_semantics

def get_label_wise_latent_semantic_representives(label_wise_latent_semantics : list) -> list :

    label_wise_latent_semantic_representives = []
    print(f"Getting label representatives from label latent semantics...\n")
    for label_features in label_wise_latent_semantics :
        label_wise_latent_semantic_representives.append(utils.label_fv_kmediods(label_features))

    return label_wise_latent_semantic_representives

def get_odd_image_vectors_latent_semantics(k : int, odd_image_vectors : np.ndarray) -> np.ndarray :

    filename = f"{k}_odd_images_latent_semantics.pkl"
    if os.path.exists(filename) :
        print(f"Latent semantics of odd images with k value {k} already exists...\n")
        odd_image_vectors_latent_semantics = torch.load(filename)
    else :
        #Odd images latent semantics :
        print(f"Calculating latent semantics for odd images...")
        odd_image_vectors_latent_semantics,_,_ = dr.svd(odd_image_vectors,k)
        torch.save(odd_image_vectors_latent_semantics, filename)
    return odd_image_vectors_latent_semantics


def get_predictions(label_wise_latent_semantic_representives : np.ndarray, odd_image_vectors_latent_semantics : np.ndarray) -> np.ndarray :
    
    #Classifying by calculating distances 
    print(f"Calculating distances...")
    distance_matrix = utils.cosine_distance_matrix(np.vstack(label_wise_latent_semantic_representives), odd_image_vectors_latent_semantics)
    print(distance_matrix.shape)

    odd_image_predicted_label_ids = np.argmin(distance_matrix, axis=0)

    return odd_image_predicted_label_ids

def test_and_print(odd_image_label_ids : np.ndarray, odd_image_predicted_label_ids : np.ndarray)  :

    #Test 
    precision, recall, f1, accuracy  = utils.compute_scores(odd_image_label_ids, odd_image_predicted_label_ids, avg_type=None, values=True)

    #Display results
    utils.print_scores_per_label(dataset, precision, recall, f1, accuracy,'Task 1')

#Test if all even images svd is different than label wise :


#1 a. collect images for each label and apply dimensionality reduction or b.reverse c.create a label matrix and do dr d. invdividual label dr 


#2. Create a label vectors -> 1. kmediods or 2. Mean 

#3. Get odd vectors
#3 a. apply same dimensionality reduction

#4. compare odd images to label vector and assign top resutl using distance measure

#5. test scores 
#6. display scores 


############### MAIN ##################
print(f"Enter the value of k for obtaining latent semantics : ")
k = utils.int_input()

#option 5 fc layer 
option = 5
dataset, labelled_images = utils.initialise_project()

even_image_vectors, even_image_label_ids, odd_image_vectors, odd_image_label_ids = get_image_vectors_and_label_ids(utils.feature_model[option])
even_image_vectors_by_label = get_image_vectors_by_label(utils.label_feature_model[option], even_image_vectors, even_image_label_ids)
label_wise_latent_semantics = get_label_wise_latent_semantics(k, even_image_vectors_by_label)
label_wise_latent_semantic_representives = get_label_wise_latent_semantic_representives(label_wise_latent_semantics)
odd_image_vectors_latent_semantics =  get_odd_image_vectors_latent_semantics(k, odd_image_vectors)
odd_image_predicted_label_ids = get_predictions(label_wise_latent_semantic_representives, odd_image_vectors_latent_semantics)

print(odd_image_label_ids)
print(odd_image_predicted_label_ids)
test_and_print(odd_image_label_ids, odd_image_predicted_label_ids)

#Label vectors not in the DB 
#print(len(label_vectors))