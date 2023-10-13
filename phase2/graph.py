import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from tqdm import tqdm 


def create_similarity_graph(matrix : np.ndarray, connections) -> nx.Graph : 
    
    '''
    Takes a 2D numpy square matrix array as matrix and creates a graph G(V,E) such as a node is pointing 
    to vi and vj i.e nodes with higest similarity scores.
    Also stores the similarity score as edge weights.
    Returns the graph
    '''
    
    #Validate input
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input should be a 2D numpy square matrix array")
    if matrix.ndim != 2:
        raise ValueError("Input should be a 2D numpy square matrix array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input should be a 2D numpy square matrix array")
    if connections >= len(matrix):
        raise ValueError("Connections should be less than the shape of matrix")

    
    #Number of outgoing edges per node 
    connections = connections  
    
    #Create an empty graph
    G = nx.MultiDiGraph(parellel=True)
    
    #Add nodes equal to the max shape of matrix
    shape = matrix.shape
    nodes = max(shape)
    G.add_nodes_from(range(nodes))
    
    print(f"Creating similarity graph with {len(matrix)} node vertices and {connections} outgoing connected edges per vertex")
    
    #For every node find the next two node based on the score
    for index, row in enumerate(tqdm(matrix)):
    
        #Create a mapping of a nodes in a row with their scores
        mapping = { key:value for key,value in enumerate(row) if key != index }
        sorted_mapping = sorted(mapping.items(), key=lambda x : x[1], reverse=True)
        
        #For every node add connections based on the score and number of connections provided
        for i in range(connections):
            connect_node,connect_node_weight = sorted_mapping[i]
            G.add_edge(index, connect_node,weight=connect_node_weight)
        
    
    print("Similarity graph generated...")
    
    #Return graph
    return G



def draw_graph(G : nx.Graph) -> None :
    
    '''
    Take a nx.Graph object as input and plots it
    '''

    if not isinstance(G, nx.Graph) :
        raise ValueError("Input should be a valid networkx graph")
    elif len(G.nodes()) > 100 :
        raise ValueError("Draw supported for upto 100 nodes")
        
    nx.draw(G, with_labels=True)
    plt.show()

        