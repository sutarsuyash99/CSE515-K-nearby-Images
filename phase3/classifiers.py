import numpy as np

import pagerank
import utils
from scipy import stats

# remember to add model_space in param
def ppr_classifier(
    connections: int,
    number_cluster: int,
    label_vectors: np.ndarray,
    img_features: np.ndarray,
    option: int,
    B: float
) -> list:
    """
    PPR based classifier - recommender in disguise
    link: https://medium.com/eni-digitalks/a-simple-recommender-system-using-pagerank-4a63071c8cbf
    """
    # compute Image-Label scores
    # image_label_sm = utils.generate_image_label_similarity_matrix(5,5)
    # compute Label-label scores
    label_label_sm = pagerank.generate_matrix_cosine_similarity(label_vectors, label_vectors)

    G = pagerank.create_graph(label_label_sm, connections)
    N = len(list(G.nodes()))
    T = pagerank.create_stochastic_transition(G)

    # create seed set
    # TODO: try generate seeds using ppr with bipartite graph (weighted pagerank)
    S = utils.get_closest_label_for_image(label_vectors, img_features, option, 1)
    # print(S)

    # Teleportation matrix : set the probabilities for jumping to seeds instead of random jumps
    # Set to 1/number_of_seeds for the seed image ids, rest are set to 0
    E = np.zeros((N, 1))
    for seed in S:
        E[seed[0]] = 1 / len(S)

    # Final matrix : transformation matrix combining all probabilities
    # Personalized pagerank formula where T is the transition matrix probability of B and E is seed jumps with probability 1-B i.e about every 5-6 walks, since B is between 0.8 to 0.9
    A = ((1 - B) * T) + (B * E)

    # Rank vector : Maintains the personalized page ranks, Initialized with probability 1/N for all nodes
    R = np.full((N, 1), 1 / N)
    # PPR based random walk - power method
    R = pagerank.power_iteration_rank(A, R)
    print('sum of R', np.sum(R))

    # return 'c' most important labels
    # Sort and return the ids and scores :
    return pagerank.get_top_rankings(R, number_cluster)

def ppr_classifier_using_image_image(
    connections: int,
    number_cluster: int,
    image_vectors: np.ndarray,
    img_id: int,
    option: int,
    dataset,
    labelled_images,
    label_vectors: np.ndarray
) -> list:
    """
    PPR based classifier - recommender in disguise
    link: https://medium.com/eni-digitalks/a-simple-recommender-system-using-pagerank-4a63071c8cbf
    """
    # compute Image-Label scores
    # image_label_sm = utils.generate_image_label_similarity_matrix(5,5)
    # compute Label-label scores
    image_image_sm = utils.generate_matrix(option, image_vectors, image_vectors)

    # create bipartite graph
    G = pagerank.create_graph(image_image_sm, connections)
    N = len(list(G.nodes()))
    T = pagerank.create_stochastic_transition(G, N, connections)

    # Damping factor : Probability for random walk and random jump
    # (1-B) -> Probability of random walk , B -> Probability of random jump or Seed Jump
    # By convention between 0.8 and 0.9
    B = 0.15

    # create seed set
    label_id = utils.get_closest_label_for_image(label_vectors, image_vectors[img_id//2], option, 1)
    all = labelled_images[label_id[0]]
    S = []
    for i in all: 
        if i % 2 == 0: 
            print(f'Image Id: {i}')
            S.append(i//2)
    print(S)

    # Teleportation matrix : set the probabilities for jumping to seeds instead of random jumps
    # Set to 1/number_of_seeds for the seed image ids, rest are set to 0
    E = np.zeros((N, 1))
    for seed in S:
        E[seed] = 1 / len(S)

    # Final matrix : transformation matrix combining all probabilities
    # Personalized pagerank formula where T is the transition matrix probability of B and E is seed jumps with probability 1-B i.e about every 5-6 walks, since B is between 0.8 to 0.9
    A = ((1 - B) * T) + (B * E)

    # Rank vector : Maintains the personalized page ranks, Initialized with probability 1/N for all nodes
    R = np.full((N, 1), 1 / N)
    # PPR based random walk - power method
    R = pagerank.power_iteration_rank(A, R)

    # return 'c' most important labels
    # Sort and return the ids and scores :
    rankings = pagerank.get_top_rankings(R, image_vectors.shape[0])

    myset = set()
    label_ranking = []
    for i in rankings:
        if number_cluster == 0: 
            break
        _, label = dataset[i[0]]
        if label in myset: continue
        else:
            myset.add(label)
            label_ranking.append((label, i[1]))
            number_cluster-=1
    
    return label_ranking




#Need to add split function to split data if required
class kNN_classifier :

    VALID_METRIC = ['euclidean','cosine']
    VALID_ALGORITHM = ['brute']

    def __init__(self, k : int, metric : str,  algorithm : str) :
        
        self.k = k

        if metric not in kNN_classifier.VALID_METRIC :
            raise ValueError(f"Metric not available!")
        if algorithm not in kNN_classifier.VALID_ALGORITHM :
            raise ValueError(f"Algorithm not available!")
        
        self.metric =  metric
        self.algorithm = algorithm



    def kNN_fit(self, data_matrix : np.ndarray, class_matrix : np.ndarray ) :

        '''
        Loads training data
        '''
        if len(data_matrix) != len(class_matrix) :
            raise ValueError(f"The data and class length don't match")
        
        self.data_matrix = data_matrix
        self.class_matrix = class_matrix
        print(f"Data and class loaded....")



    def kNN_predict(self, test_data_matrix : np.ndarray) -> np.ndarray :
        
        '''
        Predicts based on the training data, k, metric and algo and returns predicted class in an array
        '''
        self.test_data_matrix = test_data_matrix
        self.prediction_class_matrix = np.zeros(len(test_data_matrix))

        match self.metric :
            case 'cosine' :

                #Distance matrix create, Rows : DataMatrix, Columns : TestMatrix
                self.distance_matrix = utils.cosine_distance_matrix(self.data_matrix, self.test_data_matrix)
                #print(self.distance_matrix.shape)
            
            case 'euclidean' :

                #Distance matrix create, Rows : DataMatrix, Columns : TestMatrix
                self.distance_matrix = utils.euclidean_distance_matrix(self.data_matrix, self.test_data_matrix)
                #print(self.distance_matrix.shape)
            
        #Based on k get the closest k ids in the data_matrix for each in the test matrix column wise
        k_smallest_distances_index = np.argsort(self.distance_matrix, axis=0)[:self.k, :]
        
        #Based on the k_smallest_distances_index, get class values.
        predicted_class_values = self.class_matrix[k_smallest_distances_index]
        
        #Calculate the most frequent class i.e column wise mode
        predictions, _ = stats.mode(predicted_class_values, axis=0)
        
        return predictions
    

    def train_test_split(self, data_matrix : np.ndarray, class_matrix : np.ndarray, train_size : float = None, test_size : float = None, random_state : int = None, stratify : bool = False) -> np.ndarray :
    
        '''
        Splits the data and class arrays into two arrays [train and test ] as per the options.
        Only train or test size  argument can be set at once. Other is derived.
        Random State allows randomness with same random seed.
        Stratify allows to maintain the distribution as per the original data. Usefull for imbalanced data. 
        '''

        #Check if length of arrays equal 
        if len(data_matrix) != len(class_matrix) :
            raise ValueError(f"Data matrix and class matrix need to be of same length")

        #Check if both are set or other options
        if train_size != None and test_size != None :
            raise ValueError(f"Can only set train size of test size")
        elif train_size == None and test_size == None :
            print(f"Split by default size : Train - 75% and Test - 25%")
            train_size = 0.75
            test_size = 1 - train_size
        elif test_size != None :
            train_size = 1 - test_size
        else :    
            test_size = 1 - train_size


        #Get split_size 
        split_size = int(len(data_matrix) * train_size)
        '''
        match case :

            case 0 :
                
                #case 0 : random_state and stratify not set 
                data_train, data_test = data_matrix[:split_size], data_matrix[split_size:]
                class_train, class_test = class_matrix[:split_size], class_matrix[split_size:]

            case 1 :

                #case 1 : random state is set 
                #Shuffle and need to set seed so that randomness is deterministic i.e same every time for same data fpr same random state value
                random_indices = np.random.permutation(len(data_matrix), random_state=random_state)

                
                train_random_indices = random_indices[:split_size]
                test_random_indices = random_indices[split_size:]
                data_train, data_test = data_matrix[train_random_indices], data_matrix[test_random_indices]
                class_train, class_test =  class_matrix[train_random_indices], class_matrix[test_random_indices]
            
            case 2 :

                #case 2 : stratify is set tp true
                #calculate the number of unique classes and count of each class
                unique_classes, counts = np.unique(class_matrix, return_counts=True)
                min_class_count = min(counts)

                #dictionary mapping for each class number of elements
                train_size_per_class = {}
                for cls, count in zip(unique_classes, counts):
                    train_size_per_class[cls] = int(train_size * count)
        '''
                