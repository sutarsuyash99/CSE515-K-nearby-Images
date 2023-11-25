import numpy as np
import distances as d


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
            return val
    # return eigenvalues, eigenvectors

def mds(X, N, learning_rate = 0.1, num_iterations = 50):
    """
    Multidimensional Scaling for dimension reduction - 
    Function takes the two input parameters - data matrix to be reduced and number of reduced dimensions 
    Two optional parameters - 
        learning rate of gradient descent - (Default: 0.1) The change of learning rate could cause major changes in outputs.
        number of iterations - (Default: 50) The higher number of iterations could provide better results in many cases, however
                                            computational cost and time increases. 

    """
    def calculate_pairwise_distances(X):
    # Calculate pairwaise distances of the images in given feature space
        n = X.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = d.cosine_distance(X[i], X[j])
                distances[j, i] = distances[i, j]
        
        return distances

    def compute_stress(distances, Y):
    # compute stress between the original and the reduced matrix
        n = distances.shape[0]
        stress = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                og_distance = distances[i, j]
                stress += (og_distance - d.cosine_distance(Y[i], Y[j])) ** 2
        return stress


    def gradient_descent(Y, distances, learning_rate, num_iterations, prev_stress, convergence_threshold= 1e-6,):
    # Gradient Descent algorithm to get values with minimum stress
        result = Y
        for iteration in range(num_iterations):
            gradient = np.zeros_like(Y)
            n = Y.shape[0]
            for i in range(n):
                for j in range(i+1, n):
                    actual_distance = distances[i, j]
                    gradient[i, -1] += 2 * (actual_distance - d.cosine_distance(Y[i], Y[j])) * (Y[i, -1] - Y[j, -1])
                    gradient[j, -1] += 2 * (actual_distance - d.cosine_distance(Y[i], Y[j])) * (Y[j, -1] - Y[i, -1])
            
            
            stress = compute_stress(distances, Y - learning_rate * gradient)
            
            if iteration > 0:
                relative_change = abs((stress - prev_stress) / prev_stress)
                if relative_change < convergence_threshold:
                    break
            
            Y[:, -1] -= learning_rate * gradient[:, -1]

            if stress < prev_stress:
                prev_stress = stress
                result = Y
        return result, prev_stress
    

    n = X.shape[0]
    distances = calculate_pairwise_distances(X)
    max_stress = 0
    Y = np.random.rand(n, 1)
    stress = float('inf')
    for dimension in range(1, N + 1):
        Y, stress = gradient_descent(Y, distances, learning_rate, num_iterations, stress)
        if dimension == 1:
            max_stress = stress
        print(f"Dimension: {Y.shape}, Stress: {stress/max_stress}")
        
        random_column = np.random.rand(Y.shape[0], 1)
        Y = np.hstack((Y, random_column))
    
    return Y