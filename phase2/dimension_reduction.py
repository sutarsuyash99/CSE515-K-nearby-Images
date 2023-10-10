# Add all the dimension reduction algotihms here
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA

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
