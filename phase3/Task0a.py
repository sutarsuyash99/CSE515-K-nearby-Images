import Mongo.mongo_query_np as mongo_query
import utils
import inherent_dimensionality

class Task0a:
    def all_data(self):
        # Assumption - using FC layer
        int_model_option = 5
        model_option = utils.feature_model[int_model_option]
        # Load all data
        all_feature_vectors = mongo_query.get_all_feature_descriptor(model_option)
        # call functions -- call here in the below format
        k = inherent_dimensionality.PCA(all_feature_vectors)
        # print inherent dimensionality
        print(f"Inherent Dimensionality for all feature vectors: {k}")

if __name__ == '__main__':
    task0a = Task0a()
    task0a.all_data()