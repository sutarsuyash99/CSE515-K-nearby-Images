import utils
from distances import cosine_distance
import label_vectors
from Mongo.mongo_query_np import get_feature_descriptor
from resnet_50 import resnet_features
from Image_color_moment import color_moments
from get_hist_og import histogram_of_oriented_gradients

class Task2a:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()
    
    def image_query_top_k(self):
        print("*"*25, 'SUB-MENU', '*'*25)
        print('\n\nSelect your option')

        labelled_feature_vectors, model_space, feature_space_selected = label_vectors.create_labelled_feature_vectors(self.labelled_images)
        imageId, img = utils.get_user_input_internalexternal_image()

        feature_space_name_selected = utils.feature_model[feature_space_selected]
        k = utils.get_user_input_k()
        feature_vector_imageId = get_feature_descriptor(
            feature_space_name_selected, imageId
        )

        if feature_vector_imageId is None:
            if imageId != -1:
                img, _ = self.dataset[imageId]
            if feature_space_name_selected in {"avgpool", "layer3", "fc_layer"}:
                resnet = resnet_features()
                resnet.run_model(img)
                if feature_space_name_selected == utils.feature_model[3]:
                    feature_vector_imageId = resnet.resnet_avgpool()
                elif feature_space_name_selected == utils.feature_model[4]:
                    feature_vector_imageId = resnet.resnet_layer3()
                else:
                    feature_vector_imageId = resnet.resnet_fc_layer()
            elif feature_space_name_selected == utils.feature_model[1]:
                cm = color_moments()
                feature_vector_imageId = cm.color_moments_fn(img)
            else:
                hog = histogram_of_oriented_gradients()
                feature_vector_imageId = hog.compute_hog(img)
        else: 
            img, _ = self.dataset[imageId]

        utils.display_image_og(img)
        
        distances = []
        for i in labelled_feature_vectors.keys():
            cur_distance = cosine_distance( feature_vector_imageId.flatten(), labelled_feature_vectors[i].flatten() )
            distances.append((cur_distance, i))
        distances.sort()
        
        top_k = []
        for i in range(k):
            top_k.append((distances[i][1], distances[i][0]))
        print(top_k)

if __name__ == '__main__':
    task2a = Task2a()
    task2a.image_query_top_k()