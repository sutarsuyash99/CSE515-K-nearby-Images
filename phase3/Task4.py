import utils
import distances as d
import numpy as np
from Mongo import mongo_query_np
from resnet_50 import *
from dimension_reduction import *
import os
import pickle
import math
import json
import warnings

warnings.filterwarnings("ignore")
resnet = resnet_features()

class Task4a:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()
        self.data_matrix = mongo_query_np.get_all_feature_descriptor("fc_layer")
        
    def runTask4a(self):
        print("*"*25 + " Task 4a "+ "*"*25)
        print("Please enter the number of hashes:")
        num_hashes = int(utils.int_input(10))
        print("Please enter the number of layers:")
        num_layers = int(utils.int_input(10))

        return self.LSH(num_hashes, num_layers)

    
    def LSH(self, num_hashes, num_layers):
        print("Using the ResNet50 FC Layer visual model reducing the dimensions to 256")
        sparsity_percentage=0.4
        w = (num_layers//num_hashes) * int(math.sqrt(len(self.data_matrix))) * 2
        # w = 101

        hyperplanes = [np.random.randn(num_hashes, self.data_matrix.shape[1]) for _ in range(num_layers)]
        mask = np.random.rand(num_hashes, self.data_matrix.shape[1]) > sparsity_percentage

        for i in range(len(hyperplanes)):
            hyperplanes[i][mask] = 0


        random_projections = [np.zeros((num_hashes, self.data_matrix.shape[0])) for _ in range(num_layers)]

        for layer in range(len(random_projections)):
            for hash_function in range(len(random_projections[layer])):
                for i in range(len(random_projections[layer][hash_function])):
                    random_projections[layer][hash_function][i] = np.dot(hyperplanes[layer][hash_function], self.data_matrix[i])

        divisor = []
        for i in random_projections:
            length = np.ptp(i, axis=1)
            divisor.append(length/w)


        neighbouring_index = [[[0 for _ in range(len(random_projections[0]))] for _ in range(len(random_projections))] 
                    for _ in range(len(random_projections[0][0]))]

        for layer in range(len(random_projections)):
            for hash_function in range(len(random_projections[layer])):
                for i in range(len(random_projections[layer][hash_function])):
                    neighbouring_index[i][layer][hash_function] = random_projections[layer][hash_function][i] // divisor[layer][hash_function]
        print("\nLSH index structure has been created in memory\n")
        file_hashcode = "hash_codes.pkl"
        file_hyperplanes = "hyperplanes.pkl"
        file_divisor = "divisor.pkl"

        with open(file_hashcode, 'wb') as file:
            pickle.dump(neighbouring_index, file)

        with open(file_hyperplanes, 'wb') as file:
            pickle.dump(hyperplanes, file)
        
        with open(file_divisor, 'wb') as file:
            pickle.dump(divisor, file)


        return neighbouring_index, divisor, hyperplanes
    

class Task4b:

    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()
        self.data_matrix = mongo_query_np.get_all_feature_descriptor("fc_layer")
        

    def runTask4b(self, imageID = None, query_vector = []):
        print("*"*25 + " Task 4b "+ "*"*25)

        file_hashcode = "hash_codes.pkl"
        file_hyperplanes = "hyperplanes.pkl"
        file_divisor = "divisor.pkl"

        with open(file_hashcode, 'rb') as file:
            hash_codes = pickle.load(file)

        with open(file_hyperplanes, 'rb') as file:
            hyperplanes = pickle.load(file)

        with open(file_divisor, 'rb') as file:
            divisor = pickle.load(file)

        if len(query_vector) == 0 or imageID == None:
            imageID = utils.get_user_input_image_id()
            img, _ = self.dataset[imageID]
            run_model = resnet.run_model(img)
            query_vector = resnet.resnet_fc_layer()



        neighbouring_index = self.approx_images(imageID, query_vector, hash_codes, hyperplanes, divisor)

        print("Number of considered images: ", len(neighbouring_index))
        print("Indices of images: ", sorted(neighbouring_index))
        print("Please enter the number of relevant images (T):")
        k = utils.int_input(10)
        similar_images = self.knn(query_vector, imageID, k, neighbouring_index)
        data = {
            'query_image': imageID,
            'neighbour_images': list(neighbouring_index),
            't_similar_images': list(similar_images)
        }

        json_data = json.dumps(data, indent=2)

        with open('4b_output.json', 'w') as json_file:
            json_file.write(json_data)
            json_file.flush()
        utils.display_k_images_subplots(self.dataset, similar_images, f"{k} most relevant images using LSH index structure")

        return query_vector
    

    def generate_hash_code(self, imageID, query_vector, hyperplanes, divisor):

        random_projections = [[]]

        random_projections = [[] for _ in range(len(hyperplanes))]

        for layer in range(len(hyperplanes)):
            for hash_function in range(len(hyperplanes[layer])):
                random_projections[layer].append(np.dot(hyperplanes[layer][hash_function], query_vector))

        neighbouring_index = [[0 for _ in range(len(random_projections[0]))] for _ in range(len(random_projections))]

        for layer in range(len(random_projections)):
            for hash_function in range(len(random_projections[layer])):
                neighbouring_index[layer][hash_function] = random_projections[layer][hash_function] // divisor[layer][hash_function]

        return neighbouring_index

        


    def approx_images(self, imageID, query_vector, hash_codes, hyperplanes, divisor):

        query_hash_code = self.generate_hash_code(imageID, query_vector, hyperplanes, divisor)
        layers_output = {}
        neighbouring_index = set()
        sum = 0
        count = 0
        for i in range(len(hash_codes)):    
            for j in range(len(hash_codes[i])):
                for m, n in zip(query_hash_code[j], hash_codes[i][j]):
                    sum += abs(m - n)
                    count += 1
        sum = sum//count
        print(sum//count)
        for i in range(len(hash_codes)):
            for j in range(len(hash_codes[i])):
                flag = 0
                for m, n in zip(query_hash_code[j], hash_codes[i][j]):
                    # print("m,n:", i, j, m-n)
                    if abs(m - n) > (sum*2)//4:
                        flag += 1

                if flag < len(hash_codes[0][0])//3: neighbouring_index.add(i*2)
        neighbouring_index.add(imageID)

        # data = {
        #     'query_image': imageID,
        #     'neighbour_images': list(neighbouring_index)
        # }

        # json_data = json.dumps(data, indent=2)

        # with open('4b_output.json', 'w') as json_file:
        #     json_file.write(json_data)


        lsh_set = []
        for i in neighbouring_index:
            lsh_set.append(i//2)

        data_pkl = {}
        for i, data_point in enumerate(self.data_matrix):
            if i in lsh_set:
                data_pkl[i] = data_point

        file_path = "considered_set_vectors.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(data_pkl, file)

        return neighbouring_index

    def knn(self, query_vector,img, k, neighbouring_index):
    
        lsh_set = []
        for i in neighbouring_index:
            lsh_set.append(i//2)

        img = img//2
        distances = []
        for i, data_point in enumerate(self.data_matrix):
            if i in lsh_set:
                distance = d.cosine_distance(query_vector, data_point)
                distances.append((i, distance))
        
        distances.sort(key=lambda x: x[1])
        similar_images = [(index*2, dist) for index, dist in distances[:k]]
        return similar_images
    

if __name__ == '__main__':
    
    task4a = Task4a()
    hash_codes, divisor, hyperplanes = task4a.runTask4a()

    task4b = Task4b()
    task4b.runTask4b()