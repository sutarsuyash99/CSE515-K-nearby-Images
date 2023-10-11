# Add all the dimension reduction algotihms here
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
import mongo_query
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

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
    # json_data = mongo_query.query_all("fc_layer")
    # all_labels = [json_data[key]["label"] for key in range(0,8677, 1)]
    # unique_labels = np.unique(all_labels)
    # print(unique_labels.shape)
    # feature_descriptors = [json_data[key]["feature_descriptor"] for key in range(0,8677, 2)]
    # feature_descriptors_label = [json_data[key]["label"] for key in range(0,8677, 2)]
    # feature_descriptors_test = [json_data[key]["feature_descriptor"] for key in range(1,8677, 100)]
    # feature_descriptors_test_label = [json_data[key]["label"] for key in range(1,8677, 100)]

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_assignments = kmeans.fit_predict(data_collection)


    C = kmeans.cluster_centers_
    print("Cluster Shape = ", C.shape)

    X_r = extractDistanceFeatures(data_collection,C)
    print("Latent Semantics Shape = ", X_r.shape)
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


    # imageID_weight_pairs = []
    # for i in range(k):
    #     cluster_centroid = C[i]
    #     cluster_weights = np.abs(cluster_centroid)
    #     cluster_indices = np.where(cluster_assignments == i)[0]

    #     # Iterate through the data points in the cluster and associate them with weights
    #     for index in cluster_indices:
    #         imageID_weight_pairs.append((index, cluster_weights))

    # # Sort the imageID-weight pairs by weights in descending order
    # imageID_weight_pairs.sort(key=lambda x: -x[1][1])

    # # Print or store the imageID-weight pairs
    # for i, (imageID, weight) in enumerate(imageID_weight_pairs, start=1):
    #     print(f"ImageID: {imageID}, Weight: {weight}")
    return