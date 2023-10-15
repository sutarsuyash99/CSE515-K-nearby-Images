import utils
import label_vectors
from Mongo.mongo_query_np import get_feature_descriptor
from distances import kl_divergence
from resnet_50 import resnet_features

class Task2b:
    # Implement a program which, given (a) a query imageID or image 
    # file and (b) positive integer k, identifies and lists k most 
    # likely matching labels, along with their scores, under the RESNET50 
    # neural network model.
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()

    def resnet_50_image_label_topk(self):
        """
        Function that accepts imageId / external image, runs resnet_50 final 
        layer over everything and retrieves top k similar images
        Parameters: None
        Returns: None
        """
        print("*"*25, 'SUB-MENU', '*'*25)
        print('\n\nSelect your option')

        labelled_feature_vectors, _, _ = label_vectors.create_labelled_feature_vectors(self.labelled_images, True)
        imageId, img = utils.get_user_input_internalexternal_image()
        
        k = utils.get_user_input_k()
        feature_vector_imageId = get_feature_descriptor(
            'resnet_final', imageId
        )

        if feature_vector_imageId is None:
            print('Computing values -- on demand')
            if imageId != -1:
                img, _ = self.dataset[imageId]
            resnet = resnet_features()
            resnet.run_model(img)
            feature_vector_imageId = resnet.apply_softmax()
        else: 
            img, _ = self.dataset[imageId]

        #displaying user requested image
        utils.display_image_og(img)

        distances = []
        for i in labelled_feature_vectors.keys():
            # TODO: switch to KL distance
            cur_distance = kl_divergence( feature_vector_imageId.flatten(), labelled_feature_vectors[i].flatten() )
            distances.append((cur_distance, i))
        distances.sort()
        
        top_k = []
        for i in range(k):
            top_k.append((distances[i][1], distances[i][0]))
        
        print('\n\n','-'*50)
        print(top_k)
        print('-'*50, '\n\n')

if __name__ == '__main__':
    task2b = Task2b()
    task2b.resnet_50_image_label_topk()