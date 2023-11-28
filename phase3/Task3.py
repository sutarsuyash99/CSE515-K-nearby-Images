import numpy as np
import utils
import Mongo.mongo_query_np as mongo_query
import classifiers

class Task3:
    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()
    
    def knn_init(self):
        '''
        Case 1 : kNN or mNN
        Takes m as value of neighbors to compare.
        1. Gets best label for all the odd images 
        2. Calculates the precision, recall, f1 per label and accuracy for whole classifier using odd images as test set
        '''

        ### Even Images ### 
        even_image_vectors = mongo_query.get_all_feature_descriptor('fc_layer')
        even_image_vectors = utils.convert_higher_dims_to_2d(even_image_vectors)

        even_image_label_ids = np.zeros(len(even_image_vectors))
        for i in range(len(even_image_vectors)) :
            _ , even_image_label_ids[i] = self.dataset[i*2]


        ### Odd Images ### Load from pickle
        odd_image_vectors = utils.get_odd_image_feature_vectors('fc_layer')
        if odd_image_vectors is None :
            return
        odd_image_vectors = utils.convert_higher_dims_to_2d(odd_image_vectors)

        odd_image_label_ids = np.zeros(len(odd_image_vectors))
        for i in range(len(odd_image_vectors)) :
            _ , odd_image_label_ids[i] = self.dataset[i*2+1]

        #Normalize if required
        #Best results : euclidean without normalization, euclidean L2, Cosine without normalization, Cosine L2  - All same 
        even_image_vectors = utils.l2_normalization(even_image_vectors)
        odd_image_vectors = utils.l2_normalization(odd_image_vectors)

        #Get user input for m 
        print(f"\nEnter the number of neigbours to consider \"m\" : ")
        m = utils.int_input()

        #Calculate mNN using classifier
        mnn_classifier = classifiers.kNN_classifier(m, metric='cosine', algorithm='brute')
        mnn_classifier.kNN_fit(even_image_vectors, even_image_label_ids)
        odd_image_predicted_label_ids = mnn_classifier.kNN_predict(odd_image_vectors)

        #Map the predicted labels from index to id
        map_odd_image_id_predicted_label = { (index*2)+1 : label_id for index, label_id in enumerate(odd_image_predicted_label_ids)}

        #Test 
        precision, recall, f1, accuracy  = utils.compute_scores(odd_image_label_ids, odd_image_predicted_label_ids, avg_type=None, values=True)
        
        #Display results
        utils.print_scores_per_label(self.dataset, precision, recall, f1, accuracy,'m-NN')
        while True :
            user_input = utils.get_user_input_odd_image_id_looped(self.dataset)
            if user_input == 'x' :
                return
            else :
                predicted_label_id = map_odd_image_id_predicted_label[user_input]
                predicted_label = self.dataset.categories[predicted_label_id]
                print(f"The predicted label for image id - {user_input} is : {predicted_label}")

    def tree_init(self):
        classifiers.tree_init()
    
    def ppr_init(self):
        classifiers.ppr_init()

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

    def print_labels(self, result: list) -> None:
        print("-" * 40)
        for i in result:
            print(i)


    def run_classifiers(self):
        print("=" * 25, "MENU", "=" * 25)

        option = utils.get_user_selection_classifier()
        # 1 -> kNN
        # 2 -> Decision Tree
        # 3 -> PPR
        res = None
        print("-" * 40)
        match option:

            case 1: 
                self.knn_init()
            case 2:
                self.tree_init()
            case 3:
                self.ppr_init()


if __name__ == "__main__":
    task3 = Task3()
    task3.run_classifiers()
