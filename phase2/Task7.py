import torch
import utils
import dimension_reduction as dr
import numpy as np
import distances as ds
import Mongo.mongo_query_np  as monogo_query
from sortedcollections import OrderedSet
from resnet_50 import resnet_features
import torchvision
import label_vectors


class task7:
    def __init__(self) -> None:
        pass

    def distance_function(self, query_vector, data, k ):
        """
        Runs the distance function in loop gets you the K top images
        """
        distances = []
        for i in range(len(data)):
            distances.append(ds.cosine_distance(query_vector.flatten(), data[i]))
        indexed_list = list(enumerate(distances))

        # Sorting the the list 
        sorted_list = sorted(indexed_list, key=lambda x: x[1])

        output_list = sorted_list[: (k+1)].copy()
        # Revese the list becasue we are using cosine similarity
        # output_list.reverse()
        

        return output_list
    
    def get_closest_image_id (self, input_image_vector, db_data):
        """
        This function Quries that DB and gets the entire feature space for FC, then it finds the closest EVEN
        image id using cosine distance in a loop
        Input - input_image_vector
        Output - Closest EVEN image ID
        """

        # Get the entire dataset to find the closest even image
        
        output = self.distance_function(input_image_vector, db_data, 1)
        print(output)
        closest_image_id = output[0][0]
        next_id = output[1][0]
        return closest_image_id, next_id

    def image_in_image_out(self):
        """This program takes input from the user for ImageId, 
        K and Latent Semantics to be used and displys K most similar images"""
        
        dataset, labelled_image = utils.initialise_project()

        print("*"*25 + " Task 7 "+ "*"*25)
        print("Please select from below mentioned options")

        # Get input from user for ImageID
        image_id, img = utils.get_user_input_internalexternal_image()

        # Get input for ls
        path , option = utils.get_user_input_latent_semantics()

        data = torch.load(path)
        # Take input K
        k = utils.get_user_input_k()

        # If input image is external image calculate the feature vectos of the image (FC Layer)
        if image_id == -1:
            print("\nCalculating feature descriptor for external Image (FC Layer) \n")
            resnet = resnet_features()
            resnet.run_model(img)
            input_image_vector = resnet.resnet_fc_layer()

            db_data = monogo_query.get_all_feature_descriptor(utils.feature_model[5])
            print("Getting the closest even ImageId in the DB which is similar to the external Image \n")
            closest_image_id, _ = self.get_closest_image_id(input_image_vector, db_data)
            closest_image_id = closest_image_id*2
            print(f"Closest Even imageid - {closest_image_id} using FC Layer \n")
        
        # If input image is Odd image calculte the feature vectos of the image (FC Layer)
        elif not image_id % 2 == 0:
            print("\nCalculating feature descriptor for Odd ImageId not in database (FC Layer)\n")
            img, _  = dataset[image_id]
            resnet = resnet_features()
            resnet.run_model(img)
            input_image_vector = resnet.resnet_fc_layer()

            db_data = monogo_query.get_all_feature_descriptor(utils.feature_model[5])
            print("Getting the closest even ImageId in the DB which is similar to the odd ImageID provided \n")
            closest_image_id, _ = self.get_closest_image_id(input_image_vector, db_data)
            closest_image_id = closest_image_id*2
            print(f"Closest Even imageid - {closest_image_id} using FC Layer \n")


        # If image is even then fetch the vectors from db
        else:
            print("\nEven imageid provided fetching the feature descriptor from DB \n")
            input_image_vector = monogo_query.get_feature_descriptor(utils.feature_model[5], image_id)
            closest_vector = input_image_vector
            closest_image_id = image_id

        # Startin with LS specific computation 
        if option in [1,4]:
            print("LS1/4 were selected using the latent spaces to find the top K images \n")
            ls_vector  = data[int(closest_image_id/2)]
            output = self.distance_function(ls_vector,data,k)

        elif option == 2:
            print("LS2 was selected using CP decompose Image factors to find top K images \n")
            ls_vector  = data[1][0][int(closest_image_id/2)]
            cp_data = data[1][0]
            output = self.distance_function(ls_vector,cp_data,k)
            
        else:
            print("LS3 was selected using closeset label of the image provided \n")
            # IF LS3 get the closest Label and then get the nearest label of that label with Latent Semantics
            all_labels  = np.array(label_vectors.get_all_label_feature_vectors(labelled_image))

            closest_label, _ = self.get_closest_image_id(input_image_vector, all_labels)
            print(f"Label vector selected has the index{closest_label} - Name - {dataset.categories[closest_label]} \n")
            _ , closest_label_ls = self.get_closest_image_id(data[closest_label], data)

            db_data = monogo_query.get_all_feature_descriptor(utils.feature_model[5])
            output = self.distance_function(all_labels[closest_label_ls],db_data,k)


            
        final_output = [(x * 2, y) for x, y in output]
        print("Output - (ImageId, Cosine Distance)"+ "\n")
        print(final_output)
        path = path.replace(f'./LatentSemantics/LS{option}/','')
        utils.display_k_images_subplots(dataset,final_output, f"Using LS {option} with file {path}", img)

        print("Exiting Task7 .............")


if __name__ == "__main__":
    temp = task7()
    temp.image_in_image_out()
    # dataset, _ = utils.initialise_project()
    # print(utils.name_for_label_index(dataset=dataset, index=94 ))

