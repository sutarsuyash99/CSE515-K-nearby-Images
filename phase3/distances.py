import numpy as np

def cosine_similarity(a, b):
    # return distance.cosine(a,b)
    cos_sim = (np.dot(a, b)) / ((np.linalg.norm(a)) * np.linalg.norm(b))
    return cos_sim

def kl_divergence(a,b):
    # To ensure both vectors are valid probability distributions
    if not np.all(a >= 0) or not np.all(b >= 0) or not np.isclose(np.sum(a), 1.0) or not np.isclose(np.sum(b), 1.0):
        raise ValueError("Input vectors must be valid probability distributions")

    kl = np.sum(a * np.log(a / b))
    return kl

def cosine_distance(a,b):
    return 1 - cosine_similarity(a,b)

def cross_correlation_distance(a,b):
    cross_correlation = np.correlate(a,b,'valid')
    return cross_correlation[0]

def manhattan_distance(a,b):
    return np.sum(np.abs(a,b))

def intersection_similarity(a,b):
    return np.sum(np.min(a,b))/np.sum(np.max(a,b))

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)