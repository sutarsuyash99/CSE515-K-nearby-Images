import torch
import utils
import dimension_reduction as dr
import numpy as np
import distances

featues = {1: "",}
class task3:
    def __init__(self) -> None:
        pass
    def k_latent_semantics(self):
        """This programs is to get the input from the user and calculate the LS1 using the chosen latent semnatics"""
        print("*"*25 + " Task 3 "+ "*"*25)
        print("Please select from below mentioned options")
        data, feature, dbName = utils.get_user_selected_feature_model()
        k = utils.get_user_input_k()
        ls_option = utils.get_user_selected_dim_reduction()

        # V = np.array([data[key] for key in range(0, 8677, 2)])
        V = np.array(data)
        match ls_option:
            case 1: W, _, _ = dr.svd(V, k)
            case 2: W, _ = dr.nmf_als(V, k)
            case 3: W = dr.lda(V, k)
            case 4: W = dr.K_means(k, V)
            case default: print('No matching input was selected')
        
        path =  str("./LatentSemantics/LS1/LS1_" + utils.feature_model[feature]) + "_" + str(utils.latent_semantics[ls_option]) + "_" + str(k) + ".pkl"
        torch.save(W, path)
        print("Output file is saved with name - " + path)

if __name__ == "__main__":
    temp = task3()
    temp.k_latent_semantics()
    # data = torch.load("avgpool_layer_NNMF_1.pkl")
    # # val = distances.cosine_similarity(data[0],data[4000])
    # val = distances.euclidean_distance(data[0],data[2])
    # print(val)
    # # df = pd.DataFrame(data)
    # # print(df)
    # # val  = df.sort_values(by = [0], ascending=False)
    # # print(val)
    # # print(data[736])
