import numpy as np

import utils
import Mongo.mongo_query_np as mongo_query
import classifiers


class Task3:
    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()

    def ppr_classifier(
        self,
        number_clusters: int,
        option: int,
        label_vectors: np.ndarray,
        input_image_vector: np.ndarray,
        B: float,
    ) -> list:
        connections = 2
        id_rank = classifiers.ppr_classifier(
            connections, number_clusters, label_vectors, input_image_vector, option, B
        )

        name_rank = [
            (utils.name_for_label_index(self.dataset, i[0]), i[1]) for i in id_rank
        ]
        return name_rank

    def ppr_classifier_img_img(
        self,
        number_clusters: int,
        option: int,
        image_vectors: np.ndarray,
        img_id: int,
        label_vectors: np.ndarray,
    ) -> list:
        connections = 5
        id_rank = classifiers.ppr_classifier_using_image_image(
            connections,
            number_clusters,
            image_vectors,
            img_id,
            option,
            self.dataset,
            self.labelled_images,
            label_vectors,
        )

        name_rank = [
            (utils.name_for_label_index(self.dataset, i[0]), i[1]) for i in id_rank
        ]
        return name_rank

    def print_labels(self, result: list) -> None:
        print("-" * 40)
        for i in result:
            print(i)

    def run_classifiers(self):
        print("=" * 25, "MENU", "=" * 25)

        # Assumption: The model runs with fc_layer
        option = 5

        # Load label vectors
        label_vectors = mongo_query.get_label_feature_descriptor(
            utils.label_feature_model[option]
        )
        image_vectors = mongo_query.get_all_feature_descriptor(
            utils.feature_model[option]
        )

        # Load Image vectors - take image input
        imgId = utils.get_user_input_image_id()
        input_image_vector = mongo_query.get_feature_descriptor(
            utils.feature_model[option], imgId
        )

        if input_image_vector is None:
            print(f"Image not in DB: {imgId}")

            top_k = utils.get_closest_image_from_db_for_image(
                imgId, image_vectors, option, 1, self.dataset
            )
            closest_index = top_k[0][0]
            input_image_vector = mongo_query.get_feature_descriptor(
                utils.feature_model[option], closest_index
            )
            print(
                f"Closest image index: {closest_index} with feature shape: {input_image_vector.shape}"
            )

        number_clusters = utils.get_user_input_numeric_common(10, "Top m")
        # Damping factor : Probability for random walk and random jump
        # (1-B) -> Probability of random walk , B -> Probability of random jump or Seed Jump
        # By convention between 0.8 and 0.9
        B = utils.get_user_input_numeric_common(0.15, "damping factor")

        option = utils.get_user_selection_classifier()
        # 1 -> kNN
        # 2 -> Decision Tree
        # 3 -> PPR
        t3 = Task3()
        res = None
        print("-" * 40)
        match option:
            case 3:
                res = task3.ppr_classifier(
                    number_clusters, option, label_vectors, input_image_vector, B
                )
                # res = task3.ppr_classifier_img_img(
                #     number_clusters, option, image_vectors, imgId, label_vectors
                # )

        t3.print_labels(res)


if __name__ == "__main__":
    task3 = Task3()
    task3.run_classifiers()
