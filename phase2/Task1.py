import utils
from label_vectors import create_labelled_feature_vectors
import distances

class Task1:
    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()

    def query_image_top_k(self):
        print("=" * 25, "SUB-MENU", "=" * 25)
        labelled_feature_vectors, model_space, _ = create_labelled_feature_vectors(
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
        cur_label_fv = labelled_feature_vectors[cur_label]

        for i in range(len(model_space)):
            distance = distances.cosine_distance(
                cur_label_fv.flatten(),
                model_space[i].flatten(),
            )
            top_distances.append((distance, i * 2))
        top_distances.sort()

        top_k = []
        for i in range(k):
            top_k.append((top_distances[i][1], top_distances[i][0]))
        
        print("-" * 20)
        for i in top_k: print(f'ImageId: {i[0]}, Distance: {i[1]}')
        print("-" * 20)

        utils.display_k_images_subplots(self.dataset, top_k, 'Closest to Label')


if __name__ == "__main__":
    task1 = Task1()
    task1.query_image_top_k()
