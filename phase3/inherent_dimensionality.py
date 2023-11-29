import numpy as np
import utils
import torch
import os

# All code related to inherent dimensionality here...
# Include graph plots if possible

def PCA(data):
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices].real
    eigenvectors = eigenvectors[:, sorted_indices]
    mean = np.mean(eigenvalues)
    for val in range(len(eigenvalues)):
        # PCA where mean is greater than the 
        if eigenvalues[val] < mean:
            # return val, eigenvectors
            return val, eigenvectors[:, :val]
    # return eigenvalues, eigenvectors

def mds(label, X, N, learning_rate = 0.001, num_iterations = 300):
    """
    Multidimensional Scaling for dimension reduction - 
    Function takes the three input parameters - label, data matrix to be reduced and number of reduced dimensions 
    Two optional parameters - 
        learning rate of gradient descent - (Default: 0.001) The change of learning rate could cause major changes in outputs.
        number of iterations - (Default: 300) The higher number of iterations could provide better results in many cases, however
                                            computational cost and time increases. 

    """
    def calculate_pairwise_distances(X):
        """
        Fucntion to claculate the pairwise distances of matrix
        """
        n = X.shape[0]
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]

        return distances

    def compute_stress(distances, Y):
        """
        Calculation of stress on basis of the original distances
        """
        n = distances.shape[0]
        stress = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                og_distance = distances[i, j]
                stress += (og_distance - np.linalg.norm(Y[i] - Y[j])) ** 2

        return stress

    def gradient_descent(Y, distances, learning_rate, num_iterations, prev_stress, convergence_threshold=1e-6):
        """
        Gradient Descent function-
        Input: A numpy array to perform distance conservation on, distances in original data, Number of iterations, stess for previous epoch
            , convergence criteria
        Output: Y closer the original data, and stress

        """
        result = Y.copy()
        for iteration in range(num_iterations):
            gradient = np.zeros_like(Y)
            n = Y.shape[0]
            stress = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    actual_distance = distances[i, j]
                    pairwise_distance = np.linalg.norm(Y[i] - Y[j])
                    factor = 2 * (actual_distance - pairwise_distance) / (pairwise_distance + 1e-12)
                    gradient[i] += factor * (Y[i] - Y[j])
                    gradient[j] += factor * (Y[j] - Y[i])
                    stress += (actual_distance - pairwise_distance) ** 2

            if np.isnan(stress):
                raise ValueError("NaN value")

            if iteration > 0:
                relative_change = abs((stress - prev_stress) / prev_stress)
                if relative_change < convergence_threshold:
                    break

            Y -= learning_rate * gradient

            if stress < prev_stress:
                prev_stress = stress
                result = Y.copy()

            # print(f"Iteration: {iteration}, Stress: {stress}")

        return result, prev_stress

    def normalize_data(X):
        """
        Noramlization function-
        Input: numpy array to be normalized
        Output: normalized numpy array
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / (std + 1e-8)  
        
        return normalized_X



    file_path = "reduced_2D.pkl"

    # If saved reduced data exists, then return the data
    if os.path.isfile(file_path):
        data = torch.load(file_path)
        data = data[label]
        Y = np.array(data)
        return Y

    # Set the seed
    np.random.seed(99)

    # Initialise the random matrix using the clusters created in DBScan
    dataset, _ = utils.initialise_project()
    label_name = dataset.categories[label]
    file_path = 'clusters_data.pkl'
    data = torch.load(file_path)
    data = data[label_name]["labels"]
    Y = [[x,x] for x in data]
    Y = np.array(Y)
    Y = Y.astype(float)
    random_values = np.random.uniform(-0.1, 0.1, Y.shape)
    Y = Y + random_values
    Y = normalize_data(Y)
    
    n = X.shape[0]  

    # If the clusters formed are not valid, generate it randomly
    if Y.shape[0] != n:
        Y = np.random.randn(n, N)

    # Y, _, __ = dr.svd_old(X)

    # Y = Y[:, :N]

    # Normalize the data
    X = normalize_data(X)
    distances = calculate_pairwise_distances(X)

    stress = float('inf')

    # for dimension in range(1, N + 1):
    Y, stress = gradient_descent(Y, distances, learning_rate, num_iterations, stress)
    print(f"Dimension: {N}, Stress: {stress}")
    return Y



def classical_mds(X, N=2):
    """
    Mteric/ Classical Multidimensional Scaling for dimension reduction - 
    Function takes the two input parameters - data matrix to be reduced and number of reduced dimensions 
    """

    
    def normalize_data(X):
        """
        Noramlization function-
        Input: numpy array to be normalized
        Output: normalized numpy array
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / (std + 1e-8)  
        
        return normalized_X

    X = normalize_data(X)

    # Similarity matrix calculation
    X = X @ X.T

    n = len(X)

    # Double centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ X @ H

    # Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sorting eigenvectors on the basis of eigenvalues
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Reducing the eigenvectors to N 
    eigenvalues = eigenvalues[:N]
    eigenvectors = eigenvectors[:, :N]

    # Getting the reduced data
    eigenvalues = np.abs(eigenvalues)
    X_low_dimensional = eigenvectors @ np.diag(np.sqrt(abs(eigenvalues)))

    return X_low_dimensional