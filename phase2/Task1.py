import numpy as np

import utils
from label_vectors import create_labelled_feature_vectors


class Task1:
    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()

    def query_image_top_k(self):
        print("=" * 25, "SUB-MENU", "=" * 25)
        labelled_feature_vectors, model_space, option = create_labelled_feature_vectors(
            self.labelled_images
        )
        k = utils.get_user_input_k()
        cur_label_index = utils.get_user_input_label()
        cur_label = self.dataset.categories[cur_label_index]
        print(f"Input provided: {cur_label_index} => {cur_label}")

        if cur_label not in labelled_feature_vectors:
            print("Improper Input label provided")
            return

        top_k = utils.compute_distance_query_image_top_k(
            k, labelled_feature_vectors, model_space, cur_label, option
        )

        utils.display_k_images_subplots(self.dataset, top_k, "Closest to Label")


if __name__ == "__main__":
    task1 = Task1()
    task1.query_image_top_k()
