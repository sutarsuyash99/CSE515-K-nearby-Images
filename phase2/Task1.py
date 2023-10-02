from utils import initialise_project, int_input, display_k_images_subplots
from label_vectors import create_labelled_feature_vectors, label_image_distance_using_cosine

class Task1:
    def __init__(self):
        self.dataset, self.labelled_images = initialise_project()
        
    def query_image_top_k(self):
        print("="*25, 'SUB-MENU', '='*25)
        labelled_feature_vectors, model_space = create_labelled_feature_vectors(self.labelled_images)
        print('\n\nSelect your sub-option:\
            \nEnter value of K:\n')
        k = int_input(10)
        print('\n\nEnter the label format:')
        cur_label = input()
        if cur_label not in labelled_feature_vectors:
            print('Improper Input label provided')
        else:
            top_k = label_image_distance_using_cosine(len(self.dataset), labelled_feature_vectors[cur_label], model_space, k)
            print(top_k)
            display_k_images_subplots(self.dataset, top_k, 'Query-Images distances')

task1 = Task1()
task1.query_image_top_k()