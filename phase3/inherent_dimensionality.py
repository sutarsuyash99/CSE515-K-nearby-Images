import distances as ds
import numpy as np

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
            return val
    # return eigenvalues, eigenvectors