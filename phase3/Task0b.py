import Mongo.mongo_query_np as mongo_query
import utils
import inherent_dimensionality

class Task0b:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()

    def all_data(self):
        # Assumption - using FC layer
        int_model_option = 5
        model_option = utils.feature_model[int_model_option]
        # Load all data
        for i in range(len(self.labelled_images)):
            # cur_index = i
            label_selected = self.dataset.categories[i]
            print(f"Input provided: {i} => {label_selected}")
            # get all data for label
            label_images = mongo_query.get_all_feature_descriptor_for_label(model_option, label_selected)
            # call functions -- call here in the below format
            k = inherent_dimensionality.PCA(label_images)
            # print inherent dimensionality    
            print(f"For Label: {label_selected} at index: {i}, inherent dimensionality: {k}")
        


if __name__ == '__main__':
    task0b = Task0b()
    task0b.all_data()