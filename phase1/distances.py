import numpy as np
from scipy.spatial import distance

def cosine_similarity(a, b):
    # return distance.cosine(a,b)
    cos_sim = (np.dot(a, b)) / ((np.linalg.norm(a)) * np.linalg.norm(b))
    return cos_sim

# dont send flattened
def mahalanobis(a,b):
    i,j,k = a.shape
    xx = a.reshape(i, j*k).T
    yy = b.reshape(i, j*k).T
    X = np.vstack([xx,yy])
    V = np.cov(X.T)
    delta = xx-yy
    VI = np.linalg.inv(V)
    distances = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
    # print(distances.shape)
    return np.sum(distances)

# this is same as euclidean, DO NOT USE
def mahalanobis_with_identity(a,b):
    covar = np.identity(len(a))
    return distance.mahalanobis(a,b,np.linalg.inv(covar))

def cross_correlation_distance(a,b):
    cross_correlation = np.correlate(a,b,'valid')
    return cross_correlation[0]

def manhattan_distance(a,b):
    return np.sum(np.abs(a,b))

def intersection_similarity(a,b):
    return np.sum(np.min(a,b))/np.sum(np.max(a,b))

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)