# Add all the dimension reduction algotihms here
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def lda(k: int, data_collection: np.ndarray, data_label: list) -> np.ndarray:
    '''
    returns reduced matrix using sklearn's LDA inbuilt function
    since LDA is supervised learning, we need data_label input size
    source code reference: https://scikit-learn.org/0.20/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis.fit
    inbuilt-method: https://github.com/scikit-learn/scikit-learn/blob/be7633dbe/sklearn/base.py#L440
    '''
    # converting data_collection from multi dimensions to 2 dimensions
    if data_collection.ndim >= 2:
        data_collection = data_collection.flatten()
    
    if data_collection.shape[0] == len(data_label):
        lda_model = LDA(solver='svd', n_components=k)
        reduced_data = lda_model.fit_transform(data_collection, data_label)
        print(f'Reducing {data_collection.shape} => {reduced_data.shape}')
        return reduced_data
    else: raise ValueError
