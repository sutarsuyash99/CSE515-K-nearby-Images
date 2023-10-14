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

        data = monogo_query.get_entire_collection(utils.feature_model[feature])

        final_matrix = np.zeros((len(data),len(data)))

        distance_function_to_use = utils.select_distance_function_for_model_space(feature)

        for j in tqdm(range(len(data))):

            distances = []
            query= np.array(data[j]["feature_descriptor"])
            for i in range(len(data)):
                vec = np.array(data[i]["feature_descriptor"])
                if  utils.feature_model[feature] in ["color_moment","hog","avgpool","layer3","fc_layer"]:
                    distances.append(distance_function_to_use(query.flatten(), vec.flatten()))
            final_matrix[j, : ] = distances
        print(final_matrix.shape)
        
        path =  str("./LatentSemantics/LS4/image_image_" + utils.feature_model[feature]) + "_" + str(utils.latent_semantics[ls_option]) + "_" + str(k) + ".pkl"
        torch.save(final_matrix, path)
        print("Output file is saved with name - " + path)

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
        print("Output file is saved with name - " + path)
        data = W

        # Printing the ImageID weight pairs in decreasing order
        utils.print_decreasing_weights(data, "ImageID")
        print("Exiting Task6 .............")

if __name__ == "__main__":
    temp = task6()
    temp.image_image_ls()
    # V = torch.load("LatentSemantics/LS4/image_image_layer3_SVD_10.pkl")
    # W, _ , _ = dr.svd(V, 10)
    # dataset, _ = utils.initialise_project()
    # cat = dataset.categories
    # print(cat[10])


    
