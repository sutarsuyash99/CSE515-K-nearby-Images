import heapq

from label_vectors import create_labelled_feature_vectors
from utils import int_input, initialise_project
from distances import cosine_similarity
from Task1 import Task1
from Task2a import Task2a

# we can remove this, refactor this further down the line
dataset, labelled_images = initialise_project()

def query_query_top_k():
    # distance formula using cosine formula
    print("="*25, 'SUB-MENU', '='*25)
    labelled_feature_vectors, model_space = create_labelled_feature_vectors(labelled_images)
    print('\n\nSelect your sub-option:\
          \nEnter value of K:\n')
    k = int_input(10)
    k+=1
    print('\n\nEnter the label format:')
    cur_label = input()
    if cur_label not in labelled_feature_vectors:
        print('Improper Input label provided')
    else:
        cur_label_feature_vector = labelled_feature_vectors[cur_label]
        distances = []
        for i in labelled_feature_vectors.keys():
            cur_distance = cosine_similarity( cur_label_feature_vector.flatten(), labelled_feature_vectors[i].flatten())
            heapq.heappush(distances, (-1*cur_distance, i))
        top_k = []
        while(k >= 0):
            cur_distance, id = heapq.heappop(distances)
            top_k.append((-1*cur_distance, id))
            k-=1
        print(top_k)

def task1_main():
    task1 = Task1()
    task1.query_image_top_k()

def task2_main():
    task2a = Task2a()
    task2a.image_query_top_k()

option = -1
while option != 0:
    print("-"*25, 'MENU', '-'*25)
    print('Select your option:\
        \n\n\
        \n1. Label - Image distance\
        \n2. Image - Label distance\
        \n3. Label - Label distance\
        \n0. Quit\
        \n\n')
    option = int_input()
    # I made a typo to name functions like where query should be 'label'
    # too lazy to correct
    match option:
        case 0: print('Exiting...')
        case 1: task1_main()
        case 2: task2_main()
        case 3: query_query_top_k()
        case default: print('No matching input was found')