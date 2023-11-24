from sklearn import preprocessing, exceptions
import numpy

class Normalisation:
    def __init__(self) -> None:
        self.scalerMinMax = preprocessing.MinMaxScaler()
    
    """
    Normalize the vector in range [0,1]
    Parameters:
        X: numpy.ndarray of input vector
    Returns: 
        X': numpy.ndarray transformed vector
    relies on the the sklearn preprocessing library:
    read the docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    github: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/preprocessing/_data.py#L416
    """
    def train_normalize_min_max(self, X: numpy.ndarray) -> numpy.ndarray:
        self.scalerMinMax = self.scalerMinMax.fit(X)
        return self.scalerMinMax.transform(X)

    """
    use the scalerMinMax fitted values to transform new vector to range[0,1]
    use this to normalize new image or single image row
    Parameters:
        X: numpy.ndarray of input vector
    Returns: 
        X': numpy.ndarray transformed vector
    relies on the the sklearn preprocessing library:
    read the docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    github: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/preprocessing/_data.py#L500
    """
    def normalize_on_trained(self, X: numpy.ndarray) -> numpy.ndarray:
        try: 
            x = self.scalerMinMax.transform(X)
            return x
        except exceptions.NotFittedError:
            print(f'The scaler has not been fitted, try ** train_normalize_min_max ** before')
            return None