from Mongo.mongo_connection import *
from Mongo.mongo_query import *
from Mongo.push_data_to_mongodb import *
import utils
from utils import initialise_project
from Image_color_moment import color_moments
from get_hist_og import histogram_of_oriented_gradients
from resnet_50 import resnet_features
from tqdm import tqdm

color = color_moments()
hog = histogram_of_oriented_gradients()
resnet = resnet_features()

class Task0a():

    def __init__(self) -> None:
        self.feature_dict_avgpool = {}
        self.feature_dict_layer3 = {}
        self.feature_dict_fc_layer = {}
        self.feature_dict_color_moments = {}
        self.features_dict_hog = {}
        self.softmax = {}
        self.feature_list = []
        self.dataset, self.labelled_images = initialise_project()
        pass

    def custom_feature_extraction(self, i, image):

        color_moments_feature = color.color_moments_fn(image)
        hog_features = hog.compute_hog(image)

        # To compute we need to re inintialize the object again - reason for defining here
        resnet = resnet_features()

        # To run the resnet50 model
        run_model = resnet.run_model(image)
        avgpool_features = resnet.resnet_avgpool(image)
        layer3_features = resnet.resnet_layer3(image)
        fc_layer_features = resnet.resnet_fc_layer(image)
        softmax_features = resnet.apply_softmax()

        
        # Saving the outputs of all the layers in the resnet model
        self.feature_dict_avgpool[i] = avgpool_features
        self.feature_dict_layer3[i] = layer3_features
        self.feature_dict_fc_layer[i] = fc_layer_features
        self.softmax[i]= softmax_features

        # Saving the outputs of HOG and Color moments features
        self.feature_dict_color_moments[i] = color_moments_feature
        self.features_dict_hog[i] = hog_features
            
        return avgpool_features
    
    # Creates a dict in form of the document stored in collection
    def add_to_map(self, data, i):
        _, _, label = utils.img_label_and_named_label_for_query_int(self.dataset, i)
        data = data.tolist()
        map = {
            "imageID": i,
            "label": label or "unknown",
            "feature_descriptor": data
            }
        return map
    
    # Inserts the data into selected collection
    def add_to_database(self, collection_name, data):
        db = mongo_connection.get_database()
        collection = db[collection_name]
        result = collection.insert_one(data)
        print("Inserted document ID:", result.inserted_id)

    # Truncates the given collection
    def empty_collections(self, collection_name):
        db = mongo_connection.get_database()
        collection = db[collection_name]
        collection.delete_many({})

    # stores all the vectors to the database
    def store_all_feature_vectors(self, Pickle_Flag = False):

        self.empty_collections('color_moment')
        self.empty_collections('hog')
        self.empty_collections('avgpool')
        self.empty_collections('layer3')
        self.empty_collections('fc_layer')
        self.empty_collections('resnet_final')

        if Pickle_Flag:
            
            labelled_data = utils.get_image_categories()
            # avgpool vectors
            data = combine_data("avgpool_vectors.pkl", labelled_data)
            upsert_data("avgpool", data)
            # Color_moments_vectors.pkl
            data = combine_data("Color_moments_vectors.pkl", labelled_data)
            upsert_data("color_moment", data)
            # fc_layer_vectors.pkl
            data = combine_data("fc_layer_vectors.pkl", labelled_data)
            upsert_data("fc_layer", data)
            # HOG_vectors.pkl
            data = combine_data("HOG_vectors.pkl", labelled_data)
            upsert_data("hog", data)
            # layer3_vectors.pkl
            data = combine_data("layer3_vectors.pkl", labelled_data)
            upsert_data("layer3", data)
            # resnet_vectors.pkl
            data = combine_data("resnet_vectors.pkl", labelled_data)
            upsert_data("resnet_final", data)
            
        else:
            
            
            total_image = len(self.dataset)
            for i in tqdm(range(0, total_image, 2), desc= 'Running feature extraction on all models'):
                image, _ = self.dataset[i]
                
                self.custom_feature_extraction(i, image)

                map = self.add_to_map(self.feature_dict_color_moments[i], i)
                self.add_to_database('color_moment', map)
                map = self.add_to_map(self.features_dict_hog[i], i)
                self.add_to_database('hog', map)
                map = self.add_to_map(self.feature_dict_avgpool[i], i)
                self.add_to_database('avgpool', map)
                map = self.add_to_map(self.feature_dict_layer3[i], i)
                self.add_to_database('layer3', map)
                map = self.add_to_map(self.feature_dict_fc_layer[i], i)
                self.add_to_database('fc_layer', map)
                map = self.add_to_map(self.softmax[i], i)
                self.add_to_database('resnet_final', map)

            
        print("The image feature descriptors have been stored in Database")

if __name__ == '__main__':
    task = Task0a()
    task.store_all_feature_vectors()