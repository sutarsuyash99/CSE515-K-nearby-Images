import utils
from label_vectors import create_labelled_feature_vectors
import distances


class Task1:
    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()

    def query_image_top_k(self):
        print("=" * 25, "SUB-MENU", "=" * 25)
        labelled_feature_vectors, _, _ = create_labelled_feature_vectors(
            self.labelled_images
        )
        k = utils.get_user_input_k()
        cur_label_index = utils.get_user_input_label()
        cur_label = self.dataset.categories[cur_label_index]
        print(f"Input provided: {cur_label_index} => {cur_label}")
        if cur_label not in labelled_feature_vectors:
            print("Improper Input label provided")
            return

        top_distances = []
        for key in labelled_feature_vectors.keys():
            distance = distances.cosine_distance(
                labelled_feature_vectors[cur_label].flatten(),
                labelled_feature_vectors[key].flatten(),
            )
            top_distances.append((distance, key))
        top_distances.sort()

        index_name_map = {}
        for i in range(len(self.dataset.categories)):
            index_name_map[self.dataset.categories[i]] = i

        print("-" * 20)
        for i in range(k + 1):
            key = top_distances[i][1]
            print(f"{key} ({index_name_map[key]}) -- {top_distances[i][0]}")
        print("-" * 20)


if __name__ == "__main__":
    task1 = Task1()
    task1.query_image_top_k()
