# Add all the dimension reduction algotihms here
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA
from normalisation import *
from utils import convert_higher_dims_to_2d


def svd(data_matrix : np.ndarray, k=None, center=True ) -> np.ndarray :
    
    '''
    Single Value Decomposition : Dimensionality reduction using eigen decomposition.
    
    Takes data matrix a 2D numpy ndarray as input. Ex : imageID x feature descriptor. 
    Performs the SVD dimensionality reduction using eigen decomposition.
    Returns the left factor matrix U , core matrix S and right factor matrix V^T
    Where D = USV^T and k is the desired latent semantics.
    D [mxn] = U [mxk] S [kxk] V^T [kxn]
    
    By default gives all the latent features unless k is provided.
    '''
    
    #Check if data_matrix is a 2D np array
    if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2:
        raise ValueError("Input data matrix should be a 2D numpy array")

    #Datamatrix : images * feature_descriptor matrix
    D = data_matrix
    
    #Center data matrix by subracting column means
    if center :
        D = D - np.mean(D, axis=0)
   
    #Datamatrix transpose
    DT = np.transpose(D)

    #Create a symmetric matrix for decomposition
    DTD = DT @ D
    
    #Eigen decomposition for V 
    eigenvalues, eigenvectors = np.linalg.eig(DTD)
    
    #Eigen decomposition, np.linalg.eig does not return sorted values 
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:,sorted_indices]

    #Calculate the right factor matrix VT
    V = sorted_eigenvectors
    VT = V.T
    
    #Calculate the core matrix S 
    singular_values = np.sqrt(sorted_eigenvalues) 
    S = np.diag(singular_values)
    SI = np.linalg.inv(S)
    
    
    '''
    Derive U from S and V https://cs.fit.edu/~dmitra/SciComp/Resources/singular-value-decomposition-fast-track-tutorial.pdf
    Since V is orthonormal V^T is V Inverse and D = USV^T 
    To get U we can multiply both sides composition of V and S^I
    U = DVS^i
    Gives SVD where If D is mxn then U is mxk S is kxk and VT is kxn
    '''

    #Calculate the left factor matrix U
    U = D @ V @ SI
    

    #Return k latent sematics along with the truncated matrices 
    if k != None : 
        if k <= len(singular_values) :
            return U[:,:k], S[:k,:k], VT[:k,:]
        else  :
            raise ValueError("k is higher than the discovered latent features")
    return U,S,VT




def nmf_als(V, K, iteration=200, tol=1, alpha=0.01):
    """
    Non-negative Matrix Factorization (NMF) using Alternating Least Squares (ALS) method.

    Parameters:
        V (numpy.ndarray): The input data matrix of shape (m, ...).
        K (int): Desired number of latent semantics
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to stop the iterations.
        alpha (float): Regularization parameter.

    Returns:
        W (numpy.ndarray): The factorization matrix of shape (m, rank).
        H (numpy.ndarray): The factorization matrix of shape (rank, n).
    """
    # convert higher dims to 2d
    V = convert_higher_dims_to_2d(V)

    # Shape of the original vector
    m, n = V.shape

    # normalise here to [0-1]
    normalisation = Normalisation()
    V = normalisation.train_normalize_min_max(V)

    # Initialize W and H with random non-negative values
    np.random.seed(0)
    W = np.random.rand(m, K)
    H = np.random.rand(K, n)

    for iter in tqdm(range(iteration)):
        # Update H 
        H = H * (np.dot(W.T, V)) / (np.dot(np.dot(W.T, W), H))
        # Update W 
        W = W * (np.dot(V, H.T) ) / (np.dot(W, np.dot(H, H.T)) )

        # Error or distance from the original matrix , NMF for very large matrix have distance > 300
        residual = np.linalg.norm(V - np.dot(W, H))
        

        # Check for convergence
        if residual < tol:
            break
    print(residual)
    return W, H

def lda(k: int, data_collection: np.ndarray) -> np.ndarray:
    """
    returns reduced matrix using sklearn's LDA inbuilt function
    Parameters: 
        k (int): Desired input for dimensions required
        data_collections (numpy.ndarray): The input data matrix of shape (m, ...)
    Returns:
        reduced_data (numpy.ndarray): Data matrix of shape (m, k)
    source code reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
    inbuilt-method: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/base.py#L888
    explanation: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
    """
    # converting data_collection from multi dimensions to 2 dimensions
    data_collection = convert_higher_dims_to_2d(data_collection)
    
    # normalise value to [0,1]
    normalisation = Normalisation()
    data_collection = normalisation.train_normalize_min_max(data_collection)
    
    # there is something weird, for k = 1, currently raising error
    if k == 1: 
        # every value comes out as [1.]
        raise ValueError

    lda_model = LDA(n_components=k, max_iter=10, random_state=42, learning_method='batch')
    # print(train_data.shape[0] == len(train_label))
    reduced_data = lda_model.fit_transform(data_collection)
    print(f'Reducing {data_collection.shape} => {reduced_data.shape}')
    return reduced_data

