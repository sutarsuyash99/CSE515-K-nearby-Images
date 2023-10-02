import heapq

from utils import initialise_project, int_input, display_image_og
from distances import cosine_similarity
from label_vectors import create_labelled_feature_vectors

class Task2a:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = initialise_project()
    
    def image_query_top_k(self):
        print("="*25, 'SUB-MENU', '='*25)
        labelled_feature_vectors, model_space = create_labelled_feature_vectors(self.labelled_images)
        print('\n\nSelect your sub-option:\
            \nEnter value of Image Id:\n')
        image_id = int_input(0)
        if not(image_id >= 0 and image_id <= len(self.dataset)):
            print('Improper input provided')
            return
        print('\n\nEnter the value of top k querys to find:')
        k = int_input(10)
        distances = []

        display_image_og(self.dataset[image_id][0])

        for i in labelled_feature_vectors.keys():
            cur_distance = cosine_similarity( model_space[image_id].flatten(), labelled_feature_vectors[i].flatten() )
            heapq.heappush(distances, (-1*cur_distance, i))
        top_k = []
        # print(distances)
        while k >= 0:
            cur_distance, key = heapq.heappop(distances)
            top_k.append((-1*cur_distance, key))
            k-=1
        print(top_k)

if __name__ == '__main__':
    task2a = Task2a()
    task2a.image_query_top_k()