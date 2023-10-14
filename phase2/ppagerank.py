import numpy as np
import networkx as nx
from graph import create_similarity_graph
from tqdm import tqdm
       

def create_transition_matrix(G : nx.Graph, N : int, connections: int) -> np.ndarray :
    
    '''
    Takes a graph as input and returns a transition matrix with probabilities for outgoing nodes as output
    '''
    
    number_of_nodes = N
    connections = connections
    
    T = np.zeros((number_of_nodes, number_of_nodes))
    print("Generating a transition matrix from similarity graph....")
    
    #Get the outgoing nodes for each node from graph and update the values 
    for node in tqdm(G.nodes()):
        outgoing_edge_nodes = list(G.successors(node))
        #print(f"Successors of {node} are {outgoing_edge_nodes}")
        
        for i in outgoing_edge_nodes:
            T[i][node] = 1/connections
    
    return T


def eigen_rank(A: np.ndarray, R : np.ndarray) -> np.ndarray :

    '''
    Since, new Rank vector is such that  R = AR which is in eigen form with lambda=1, we can calculate eigen decomposition and select the eigen vector corresponding to eigenvalue 1
    
    Takes rank vector and transformation matrix as input and return principal eigenvector normalized
    '''
    
    eigenvalues, eigenvectors = np.linalg.eig(A)

    larget_eigenvalue_index = np.argsort(eigenvalues)[-1]
    principal_eigenvalue = eigenvalues[larget_eigenvalue_index]
    principal_eigenvector = eigenvectors[:,larget_eigenvalue_index].real
    
    #Rank vector is the principal eigenvector to score from 0 to 1
    R = np.abs(principal_eigenvector)/(np.sum(np.abs(principal_eigenvector)))
    
    return R
    
def power_iteration_rank(A: np.ndarray, R : np.ndarray) -> np.ndarray :

    '''
    Takes Rank vector R and A the transformation matrix and iterates the dot product of A ans R until R converges and returns R
    '''

    #Counter
    iteration = 1
    
    #Epsilon value for measuring convergence error
    eps = 1e-9
    while True:
        R_last = R
        
        #Iterating A and R's dot product
        R = A @ R
        
        #Absolute change in all components in R. The maximum of that change, if less than epsilon then convergence is reached.
        change = np.max(np.abs(R_last - R))
        
        #Convergence 
        if change < eps :
            print(f"Matched on iteration {iteration}")
            break 
        iteration += 1 
    
    R = R.flatten()
    
    return R
    
    

def pagerank(distance_matrix : np.ndarray, label_representatives : list, m : int, n : int, method='power') -> list :
    
    '''
    Pending : 1.What should be label representatives 
              2.Image ids need to be converted to even as the input database is only even images
              3.Should the label representative also be returned among m, because it will always have highest pagerank
              4.Seed noise reduction from professors paper
    '''

    '''
    Input : 
    similarity_matrix - NxN similarity matrix 2D numpy array, Ex: image-image similarity matrix
    label_representatives - List of seed nodes for biasing. Ex: image-id/ids that represent the label in matrix/graph
    m - No of similar nodes to return. Ex : image-ids in the graph whose pagerank is highest
    n - No of outgoing edges per vertex/node to consider while calculating pagerank
    
    Output :
    m image-ids similar to the label representatives with their score sorted in descending
    
    Takes a similarity_matrix and converts it into a Graph. Performs a personalized page rank with restarts on the Matrix created from Graph as well as the bias.
    '''
              
    #Validating input 
    if not isinstance(distance_matrix, np.ndarray) :
        raise ValueError("Matix should be a 2D square numpy array")
    elif distance_matrix.ndim != 2 :
        raise ValueError("Matrix should be a 2D square numpy array")
    elif not isinstance(label_representatives, list) :
        raise ValueError("Label representatives should be list of image id/ids")
    elif not isinstance(m, int) or not isinstance(n, int) :
        raise ValueError("Number of Images - m and Number of edges - n should be integers")
    

    #The matrix saved in task 6 saves distances
    similarity_matrix = np.ones(distance_matrix.shape)
    similarity_matrix = similarity_matrix - distance_matrix
              
    #Damping factor : Probability for random walk and random jump
    #B -> Probability of random walk , (1-B) -> Probability of random jump or Seed Jump
    #By convention between 0.8 and 0.9 
    B = 0.85
    
    
    #Create a graph from the similarity matrix where each vertex has n edges w.r.t similarity scores
    connections = n
    G = create_similarity_graph(similarity_matrix,connections)
    
    #Number of nodes in the graph
    N = len(list(G.nodes()))
    
    #Creating transition matrix from Graph : a column stochastic probability distribution
    #Column values add to one, and values for outgoing nodes are set to 1/number of edges
    T = create_transition_matrix(G, N, connections)

   
    #Seed set is label representative image id/ids
    S = sorted(label_representatives)
    
    
    #Teleportation matrix : set the probabilities for jumping to seeds instead of random jumps
    #Set to 1/number_of_seeds for the seed image ids, rest are set to 0
    E = np.zeros((N,1))
    for seed in S :
        E[seed] = 1/len(S)
    
    
    #Final matrix : transformation matrix combining all probabilities
    #Personalized pagerank formula where T is the transition matrix probability of B and E is seed jumps with probability 1-B i.e about every 5-6 walks, since B is between 0.8 to 0.9
    A = (B * T) + ((1-B) * E)
    
    
    #Rank vector : Maintains the personalized page ranks, Initialized with probability 1/N for all nodes
    R = np.full((N,1),1/N)
   
    #Calculating pageranks :
    print("\n"*2)
    print("Calculating pagerank...")
    
    '''
    Method 1 : Eigen decomposition
    Observation : Slower / More expensive
    '''
    if method == 'eigen' :
        R = eigen_rank(A, R)
    
    '''
    Method 2 : Power iteration
    Observation : Much faster
    '''    
    if method == 'power' :
        R = power_iteration_rank(A, R)
        
    
    
    #Sort and return the ids and scores :
    top_m_ids = np.argsort(R)[::-1][:m]
    top_m_scores = R[top_m_ids]
    
    rankings = [ (index, scores) for index, scores in zip(top_m_ids, top_m_scores) ]
    
    return rankings 

