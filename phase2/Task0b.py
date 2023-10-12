import utils
from Mongo.mongo_query_np import get_feature_descriptor
from distances import cosine_distance
from resnet_50 import resnet_features
from Image_color_moment import color_moments
from get_hist_og import histogram_of_oriented_gradients


class Task0b:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()

    def image_image_distance(self):
        """This programs is image-image distance function and display k top matches"""
        print("*" * 25 + " Task 0b " + "*" * 25)
        print("Please select from below mentioned options")
        # take inputs imageId, external image, feature space, k

        # ImageId, PILImage gets user inputs and handles exception if any
        imageId, img = utils.get_user_input_internalexternal_image()
        (
            model_space,
            feature_space_selected,
        ) = utils.get_user_selected_feature_model()
        feature_space_name_selected = utils.feature_model[feature_space_selected]
        k = utils.get_user_input_k()
        feature_vector_imageId = get_feature_descriptor(
            feature_space_name_selected, imageId
        )

        if feature_vector_imageId is None:
            # construct metric for that image by running model space on the image provided
            print("Constructing vector for image -- on demand")
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
            # TODO: show this in same subplot
            utils.display_image_og(pil_img=img)
        else:
            # if image is in database, increment k by 1 and show all images in same subplot
            k = k + 1

        # display top k image from DB
        distances = []
        for i in range(len(model_space)):
            distance = cosine_distance(
                feature_vector_imageId.flatten(), model_space[i].flatten()
            )
            distances.append((distance, i))
        distances.sort()
        # display image
        print(distances[0], distances[1], distances[-2], distances[-1])
        top_k = []
        for i in range(k):
            top_k.append((distances[i][1], distances[i][0]))
        print('\n\n','-'*50)
        print(top_k)
        print('-'*50, '\n\n')

        utils.display_k_images_subplots(self.dataset, top_k, "Top K images")


if __name__ == "__main__":
    task0b = Task0b()
    task0b.image_image_distance()
