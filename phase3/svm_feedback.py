import numpy as np
from tqdm import tqdm

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_signed = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in tqdm(range(self.n_iters)):
            for index, x_i in enumerate(X):
                condition = y_signed[index] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_signed[index]))
                    self.b -= self.lr * y_signed[index]

    def return_weights_bias(self):
        return self.w, self.b
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return approx
    
    def get_result(self, x_tain, y_train, x_test):
        """Takes x_train, y_train and x_test returns predictions"""
        self.fit(x_tain, y_train)
        result = self.predict(x_test)
        return result

    def run_svm(self, condition_1, condition_2, data, feedback, task4b_index ):
        """Creates X_train, Y_train and X_test from conditions and feedback and task4b_index then runs the SVM"""

        x_train, y_train, x_test = [], [] , []
        x_train.extend([data[x] for x, key in feedback.items() if eval(condition_1)])
        x_train.extend([data[x] for x, key in feedback.items() if eval(condition_2)])

        y_train.extend([1 for x, key in feedback.items() if eval(condition_1)])
        y_train.extend([0 for x, key in feedback.items() if eval(condition_2)])

        x_test.extend([data[x] for x in task4b_index])

        x_test = np.array(x_test)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        r_i_output = self.get_result(x_train,y_train, x_test )
        r_i_index = [(x,y) for x,y in zip(task4b_index, r_i_output)]
        return r_i_index
    
# # Testing
# if __name__ == "__main__":

#     clf = SVM()
#     clf.fit(X_train, y_train_binary)
#     predictions = clf.predict(x_test)
#     print(enumerate(predictions))
#     for i in range(len(predictions)):
#         print(str(i)+ "The predictions for the value is  - " + str( predictions[i]))

  