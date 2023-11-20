import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

import distances

def generate_matrix_cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Function to generate Matrix of distances using selected
    model space (and corresponding distance)
    """
    distance_matrix = np.zeros((f1.shape[0], f2.shape[0]))

    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            distance = distances.cosine_similarity(f1[i].flatten(), f2[j].flatten())
            distance_matrix[i, j] = distance

    # print("Matrix", distance_matrix.shape)
    # print(distance_matrix)
    return distance_matrix

def create_graph(label_label_similarity: np.ndarray, connections: int) -> nx.DiGraph:
    B = nx.MultiDiGraph(parallel=True)

    B.add_nodes_from(range(label_label_similarity.shape[0]))

    # print(f"There are in total {len(B.nodes)} in Graph")
    # print("-" * 35)
    # print(B.nodes)

    for index, row in enumerate(tqdm(label_label_similarity)):
        mapping = {key: value for key, value in enumerate(row) if key != index}
        sorted_mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)

        # print(sorted_mapping)
        for i in range(connections):
            connect_node, connect_node_weight = sorted_mapping[i]
            B.add_edge(index, connect_node, weight=connect_node_weight)
    
    # pos = nx.spring_layout(B)
    # nx.draw(B, pos, with_labels=True, font_weight='bold')
    # plt.show()

    return B


def create_stochastic_transition(G: nx.Graph) -> np.ndarray:
    """
    Takes a graph as input and returns a transition matrix with probabilities for outgoing nodes as output
    """

    T = nx.adjacency_matrix(G, weight='weight').T
    # convert T to probablistic ndarray
    T = T.toarray()
    den = T.sum(axis=0, keepdims=True)
    # print('Den has NAN?', np.isnan(den).any())
    pT = T / den
    return pT


def power_iteration_rank(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Takes Rank vector R and A the transformation matrix and iterates the dot product of A ans R until R converges and returns R
    """

    # Counter
    iteration = 1

    # Epsilon value for measuring convergence error
    eps = 1e-6
    while True:
        R_last = R

        # Iterating A and R's dot product
        R = A @ R

        # Absolute change in all components in R. The maximum of that change, if less than epsilon then convergence is reached.
        change = max(np.max(np.abs(R_last - R)), 1e-14)

        # Convergence
        if change < eps:
            print(f"Matched on iteration {iteration}")
            break

        iteration += 1

    R = R.flatten()

    return R


def get_top_rankings(R: np.ndarray, c: int) -> list:
    top_m_ids = np.argsort(R)[::-1]

    a = []
    for i in top_m_ids:
        if len(a) > c:
            break
        a.append(i)
    top_m_ids = a
    top_m_scores = R[top_m_ids]

    rankings = [(index, scores) for index, scores in zip(top_m_ids, top_m_scores)]

    # print("-" * 35)
    # print(rankings)
    return rankings
