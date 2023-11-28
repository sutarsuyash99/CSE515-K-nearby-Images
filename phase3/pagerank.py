import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

import distances
import utils
import Mongo.mongo_query_np as mongo_query
import os
import torch


class Pagerank:
    def __init__(self) -> None:
        self.option = 5
        self.number_clusters = utils.get_user_input_numeric_common(10, "Top m")
        # Damping factor : Probability for random walk and random jump
        # (1-B) -> Probability of random walk , B -> Probability of random jump or Seed Jump
        # By convention between 0.8 and 0.9
        self.B = utils.get_user_input_numeric_common_float(0.15, "damping factor")
        self.dataset, self.labelled_images = utils.initialise_project()
        (
            self.even_image_vectors,
            self.even_image_label_ids,
            self.odd_image_vectors,
            self.odd_image_label_ids,
        ) = self.load_image_vectors_and_label_data()
        self.image_image_similarity = self.make_similarity_matrices()
        # create Graph
        self.G = self.create_graph(self.image_image_similarity, self.number_clusters)
        self.N = len(list(self.G.nodes()))
        # create stochastic
        self.T = self.create_stochastic_transition(self.G)

    def create_ppr(self) -> dict:
        dic = {}

        for i in self.labelled_images:
            # do ppr inside 
            S = [(ii//2) for ii in self.labelled_images[i] if ii%2 == 0]
            E = np.zeros((self.N, 1))
            for seed in S:
                E[seed] = 1 / len(S)

            A = ((1 - self.B) * self.T) + (self.B * E)
            # Rank vector : Maintains the personalized page ranks, Initialized with probability 1/N for all nodes
            R = np.full((self.N, 1), 1 / self.N)
            # PPR based random walk - power method
            R = self.power_iteration_rank(A, R)

            rank = self.get_top_rankings(R, self.number_clusters)
            dic[i] = [imgId for imgId, _ in rank]
        
        print(dic)

        return dic

    def start_ppr(self):
        # TODO: change 1 to actual imgId
        pprPath = (
            "pprPath_"
            + str(round(self.B * 100, 2))
            + "_"
            + str(self.number_clusters)
            + ".pkl"
        )
        dic = None
        if not os.path.isfile(pprPath):
            print(
                "No existing model was found with the exact params...\n\
                \nCreating one right now"
            )
            dic = self.create_ppr()
            torch.save(dic, pprPath)
        else:
            print("Pre-existing model was found, using that")
            dic = torch.load(pprPath)

        if dic is None:
            return
        
        odd_label_ids = []
        for i in range(len(self.odd_image_vectors)):
            label_index = self.computed_weighted_label(self.odd_image_vectors[i], dic)
            odd_label_ids.append(label_index)

        precision, recall, f1, accuracy = utils.compute_scores(
            self.odd_image_label_ids, odd_label_ids, avg_type=None, values=True
        )

        # Display results
        utils.print_scores_per_label(
            self.dataset, precision, recall, f1, accuracy, "PPR based classifier"
        )
    
    def computed_weighted_label(self, considered_image: np.ndarray, dic: dict) -> int:
        res = []
        for key in dic.keys():
            count = 0
            for val in dic[key]:
                count+= distances.cosine_distance(self.even_image_vectors[val].flatten(), considered_image.flatten())
            res.append(((count/len(dic[key])), key ))
        res.sort()
        label = res[0][1]
        # print(f'Closest label: {label} to image')
        label_index = self.dataset.categories.index(label)
        return label_index

    def load_image_vectors_and_label_data(
        self,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        even_image_vectors = mongo_query.get_all_feature_descriptor(
            utils.feature_model[self.option]
        )
        # convert them to 2d -> flatten
        even_image_vectors = utils.convert_higher_dims_to_2d(even_image_vectors)

        # get label for even vectors
        even_image_label_ids = np.zeros(len(even_image_vectors))
        for i in range(len(even_image_vectors)):
            _, even_image_label_ids[i] = self.dataset[i * 2]

        # load odd vectors from pkl file
        odd_image_vectors = utils.get_odd_image_feature_vectors("fc_layer")
        if odd_image_vectors is None:
            return
        odd_image_vectors = utils.convert_higher_dims_to_2d(odd_image_vectors)

        # get actual values of odd image labels
        odd_image_label_ids = np.zeros(len(odd_image_vectors))
        for i in range(len(odd_image_vectors)):
            _, odd_image_label_ids[i] = self.dataset[i * 2 + 1]

        return (
            even_image_vectors,
            even_image_label_ids,
            odd_image_vectors,
            odd_image_label_ids,
        )

    def make_similarity_matrices(self) -> np.ndarray:
        # TODO: LOAD image-label similarity matrix from pkl

        # image_label_similarity
        image_image_similarity = self.generate_matrix_cosine_similarity(
            self.even_image_vectors, self.even_image_vectors
        )
        print(f"Generated Image-label similarity {image_image_similarity.shape}")
        
        return image_image_similarity

    def generate_matrix_cosine_similarity(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> np.ndarray:
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

    def create_graph(
        self, similarity_matrix: np.ndarray, connections: int
    ) -> nx.DiGraph:
        B = nx.MultiDiGraph(parallel=True)

        B.add_nodes_from(range(similarity_matrix.shape[0]))

        # print(f"There are in total {len(B.nodes)} in Graph")

        for index, row in enumerate(similarity_matrix):
            mapping = {key: value for key, value in enumerate(row) if key != index}
            sorted_mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)

            for i in range(connections):
                connect_node, connect_node_weight = sorted_mapping[i]
                B.add_edge(
                    index,
                    connect_node,
                    weight=connect_node_weight,
                )

        # self.draw_graph(B)
        return B

    def draw_graph(self, G: nx.Graph) -> None:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight="bold")
        plt.show()

    def create_stochastic_transition(self, G: nx.Graph) -> np.ndarray:
        """
        Takes a graph as input and returns a transition matrix with probabilities for outgoing nodes as output
        """
        T = nx.adjacency_matrix(G, weight="weight").T
        # convert T to probablistic ndarray
        T = T.toarray()
        den = T.sum(axis=0, keepdims=True)
        pT = T / den
        return pT

    def power_iteration_rank(self, A: np.ndarray, R: np.ndarray) -> np.ndarray:
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
                # print(f"Matched on iteration {iteration}")
                break

            iteration += 1

        R = R.flatten()

        return R

    def get_top_rankings(self, R: np.ndarray, c: int) -> list:
        top_m_ids = np.argsort(R)[::-1][:c]

        # a = []
        # for i in top_m_ids:
        #     if len(a) > c:
        #         break
        #     a.append(i)
        # top_m_ids = a
        top_m_scores = R[top_m_ids]
        # with np.printoptions(threshold=sys.maxsize):
        # print(top_m_ids.shape, top_m_scores.shape)

        rankings = [(index, scores) for index, scores in zip(top_m_ids, top_m_scores)]

        # print("-" * 35)
        # print(rankings)
        return rankings


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
