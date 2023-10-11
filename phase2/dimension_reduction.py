# Add all the dimension reduction algotihms here
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA
import mongo_query
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import distances

def nmf_als(V, K, iteration=200, tol=1, alpha=0.01):
    """
    Non-negative Matrix Factorization (NMF) using Alternating Least Squares (ALS) method.

    Parameters:
        V (numpy.ndarray): The input non-negative data matrix of shape (m, n).
        K (int): Desired number of latent semantics
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to stop the iterations.
        alpha (float): Regularization parameter.

    Returns:
        W (numpy.ndarray): The factorization matrix of shape (m, rank).
        H (numpy.ndarray): The factorization matrix of shape (rank, n).
    """
    # Shape of the original vector
    m, n = V.shape

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
    '''
    returns reduced matrix using sklearn's LDA inbuilt function
    negative values do not work well with model => handle somehow
    source code reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
    inbuilt-method: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/base.py#L888
    explanation: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
    '''
    # converting data_collection from multi dimensions to 2 dimensions
    if data_collection.ndim >= 2:
        data_collection = data_collection.flatten()
    
    # there is something weird, for k = 1, currently raising error
    if k == 1: 
        # every value comes out as [1.]
        raise ValueError

    lda_model = LDA(n_components=k, max_iter=10, random_state=42, learning_method='batch')
    # print(train_data.shape[0] == len(train_label))
    reduced_data = lda_model.fit_transform(data_collection)
    print(f'Reducing {data_collection.shape} => {reduced_data.shape}')
    return reduced_data

def extractDistanceFeatures(X,C):
    X_r = []
    for i in range(C.shape[0]):
        ci = C[i,:]
        di = np.sum((X-ci)**2,axis=1)
        X_r.append(di)
    return np.array(X_r).T

def K_means(k, data_collection):
    print("Original Shape ", data_collection.shape)
    
    if data_collection.ndim >= 2:
        data_collection = data_collection.reshape(data_collection.shape[0], -1)
        print("Reshaped Data Shape ", data_collection.shape)
    
    # ------ TRIED With library
    # json_data = mongo_query.query_all("fc_layer")
    # all_labels = [json_data[key]["label"] for key in range(0,8677, 1)]
    # unique_labels = np.unique(all_labels)
    # print(unique_labels.shape)
    # feature_descriptors = [json_data[key]["feature_descriptor"] for key in range(0,8677, 2)]
    
    # kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # cluster_assignments = kmeans.fit_predict(data_collection)

    def initialize_centroids(data, k):
        np.random.seed(0) # IMP: to produce same sequence everytime
        indices = np.random.choice(len(data), k, replace=False)
        # selected 'k' random data points to be centroids
        centroids = data[indices]
        return centroids

    # Assign data point to the nearest centroid
    def assign_to_centroids(data, centroids):
        # Initialize an array to store the cluster assignments
        cluster_assignment = np.zeros(len(data), dtype=float)

        for i in range(len(data)): # for each data point
            min_distance = float('inf')
            for j in range(len(centroids)): # check with each centroid of cluster
                distance = distances.euclidean_distance(data[i], centroids[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster_assignment[i] = j  # Assign the data point to the nearest centroid
        return cluster_assignment

    # Update centroids as the mean of the assigned data points
    def update_centroids(data, cluster_assignment, k):
        new_centroids = np.zeros((k, data.shape[1]))
        for i in range(k):
            new_centroids[i] = np.mean(data[cluster_assignment == i], axis=0)
        return new_centroids
    
    # K-means clustering
    def k_means(data, k, max_iterations=300, n_init=10):
        best_centroids, best_inertia = None, float('inf')
        for _ in range(n_init):
            # This is the basic approach
            centroids = initialize_centroids(data, k)
            for _ in range(max_iterations):
                cluster_assignments = assign_to_centroids(data, centroids)
                new_centroids = update_centroids(data, cluster_assignments, k)
                if np.all(centroids == new_centroids):
                    # same centroid is getting created
                    print(f"same centroid formed, breaking loop")
                    break
                centroids = new_centroids
            # We will check for intertia here
            current_inertia = 0
            for i in range(k):
                # Calculate the squared Euclidean distance for data points in cluster i
                cluster_points = data[cluster_assignments == i]
                centroid = centroids[i]
                squared_distances = distances.euclidean_distance(cluster_points, centroid)
                current_inertia += np.sum(squared_distances)
                # print("current intertia = ", current_inertia)
            
            if current_inertia < best_inertia:
                best_centroids, best_inertia = centroids, current_inertia
        return best_centroids
    
    Centroids = k_means(data_collection, k)


    # C = kmeans.cluster_centers_
    # print("Cluster Shape = ", C.shape)
    # Using the Clusters find the latent semantic features 
    # X_r = extractDistanceFeatures(data_collection,C)
    X_r = extractDistanceFeatures(data_collection, Centroids)
    print("Latent Semantics Shape = ", X_r.shape)
    print(X_r)
    return X_r
    

    # clf = SVC().fit(X_r, feature_descriptors_label)
    # X_test_r = extractDistanceFeatures(feature_descriptors_test,C)

    # y_pred = clf.predict(X_test_r)

    # mat = confusion_matrix(feature_descriptors_test_label,y_pred)
    # sns.set()
    # sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
    #             xticklabels=np.unique(all_labels),yticklabels=np.unique(all_labels))
    # plt.xlabel("True label")
    # plt.ylabel("predicted label")
    # plt.show()
    return