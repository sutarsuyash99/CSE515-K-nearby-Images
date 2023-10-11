# Add all the dimension reduction algotihms here
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA
from normalisation import *
from utils import convert_higher_dims_to_2d


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
