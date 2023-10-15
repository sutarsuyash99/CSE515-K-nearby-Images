import torch
import utils
import dimension_reduction as dr
import numpy as np
import distances as ds
import Mongo.mongo_query_np  as monogo_query
from tqdm import tqdm

class task6:
    def __init__(self) -> None:
        pass
    def image_image_ls(self):
        """This programs is to get the input from the user and calculate the LS3 using the chosen latent semnatics"""
        print("*"*25 + " Task 6 "+ "*"*25)
        print("Please select from below mentioned options")
        data, feature, dbName = utils.get_user_selected_feature_model()
        k = utils.get_user_input_k()
        ls_option = utils.get_user_selected_dim_reduction()

        # Using the data provided by the utils function get_user_selected_feature_model
        # data = monogo_query.get_entire_collection(utils.feature_model[feature])
        

        final_matrix = np.zeros((len(data),len(data)))

        # Gets the appropriet distance function that will be used for the feature model selected
        distance_function_to_use = utils.select_distance_function_for_model_space(feature)

        # Generate image image similarity matrix
        print("\nGenerating Image - Image similarity matrix under the given feature space - \n")
        for j in tqdm(range(len(data))):
            for i in range(len(data)):
                distances = distance_function_to_use(data[j].flatten(), data[i].flatten())
                final_matrix[j, i ] = distances
            
        print(final_matrix.shape)
        
        path =  str("./LatentSemantics/LS4/image_image_matrix/image_image_similarity_matrix_" + utils.feature_model[feature]) + ".pkl"
        torch.save(final_matrix, path)
        print("\n Similarity Matrix output file is saved at - " + path+ "\n")

        V = final_matrix
        # W is matrix with M x K dimension matrix with latent semantics
        match ls_option:
            case 1: W, _ , _ = dr.svd(V, k)
            case 2: W, _  = dr.nmf_als(V, k)
            case 3: W = dr.lda(k, V)
            case 4: W = dr.K_means(k, V)
            case default: print('No matching input was selected')
        
        path =  str("./LatentSemantics/LS4/LS4_" + utils.feature_model[feature]) + "_" + str(utils.latent_semantics[ls_option]) + "_" + str(k) + ".pkl"
        torch.save(W, path)
        print("\nLatent Semantics output file is saved at - " + path)
        data = W

        # Printing the ImageID weight pairs in decreasing order
        utils.print_decreasing_weights(data, "ImageID")
        print("\n ............. Exiting Task6 ............. \n")

if __name__ == "__main__":
    temp = task6()
    temp.image_image_ls()
    # V = torch.load("LatentSemantics/LS4/image_image_layer3_SVD_10.pkl")
    # W, _ , _ = dr.svd(V, 10)
    # dataset, _ = utils.initialise_project()
    # cat = dataset.categories
    # print(cat[10])


    
