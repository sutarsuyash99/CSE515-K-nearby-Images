import numpy as np

import pagerank
import utils


# remember to add model_space in param
def ppr_classifier(
    connections: int,
    number_cluster: int,
    label_vectors: np.ndarray,
    img_features: np.ndarray,
    option: int
) -> list:
    """
    PPR based classifier - recommender in disguise
    link: https://medium.com/eni-digitalks/a-simple-recommender-system-using-pagerank-4a63071c8cbf
    """
    # compute Image-Label scores
    # image_label_sm = utils.generate_image_label_similarity_matrix(5,5)
    # compute Label-label scores
    label_label_sm = utils.generate_matrix(option, label_vectors, label_vectors)

    # create bipartite graph
    G = pagerank.create_graph(label_label_sm, connections)
    N = len(list(G.nodes()))
    T = pagerank.create_stochastic_transition(G, N, connections)

    # Damping factor : Probability for random walk and random jump
    # (1-B) -> Probability of random walk , B -> Probability of random jump or Seed Jump
    # By convention between 0.8 and 0.9
    B = 0.45

    # create seed set
    S = utils.get_closest_label_for_image(label_vectors, img_features, option, 5)
    print(S)

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


