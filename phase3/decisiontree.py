import numpy as np
import os
import torch

import utils
import Mongo.mongo_query_np as mongo_query
import dimension_reduction


class DecisionTree:
    def __init__(self) -> None:
        self.option = 5

        # Assumptions
        self.max_depth = 150

        # Load odd, even image and features
        self.dataset, self.labelled_images = utils.initialise_project()
        # self.image_vectors, self.image_label_ids = self.load_all_image_vectors()

        # if self.image_vectors is None:
        #     print("There is undefined error")
        #     return
        # (
        #     self.train_vectors,
        #     self.test_vectors,
        #     self.train_target,
        #     self.test_target,
        # ) = self.split_data_into_train_test(self.image_vectors, self.image_label_ids)
        (
            self.train_vectors,
            self.train_target,
            self.test_vectors,
            self.test_target,
        ) = self.get_split_train_test_data()

    def start_dt(self):
        decisionTreePath = (
            "decisionTree_"
            + str(self.max_depth)
            + ".pkl"
        )
        clf = None
        if not os.path.isfile(decisionTreePath):
            print(
                "No existing model was found with the exact params...\n\
                \nCreating one right now"
            )
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(self.train_vectors, self.train_target)
            torch.save(clf, decisionTreePath)
        else:
            print("Pre-existing model was found, using that")
            clf = torch.load(decisionTreePath)
        if clf is None:
            print("Encountered Error")
            return
        test_predicted_ids = clf.predict(self.test_vectors)
        print(f"The maximum depth is: {clf.max_depth}")

        # for i in range(len(test_predicted_ids)):
        #     print(f'Actual {self.test_target[i]} vs Predicted {test_predicted_ids[i]}')

        precision, recall, f1, accuracy = utils.compute_scores(
            self.test_target, test_predicted_ids, avg_type=None, values=True
        )

        # Display results
        utils.print_scores_per_label(
            self.dataset, precision, recall, f1, accuracy, "Decision Tree classifier"
        )

    def get_split_train_test_data(
        self,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        even_vectors = mongo_query.get_all_feature_descriptor(utils.feature_model[5])
        # U, sigma, VT = dimension_reduction.svd(
        #     even_vectors, self.dimensionsToReduce
        # )
        # reduced_even_vectors = np.dot(even_vectors, VT.T)

        odd_vectors = utils.get_odd_image_feature_vectors("fc_layer")
        # reduced_odd_vectors = np.dot(odd_vectors, VT.T)/sigma
        # print(reduced_even_vectors.shape, reduced_odd_vectors.shape)

        even_image_label_ids, odd_image_label_ids = [], []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if i % 2 == 0:
                even_image_label_ids.append(label)
            else:
                odd_image_label_ids.append(label)

        even_image_label_ids = np.asarray(even_image_label_ids, dtype=int)
        odd_image_label_ids = np.asarray(odd_image_label_ids, dtype=int)

        return (
            even_vectors,
            even_image_label_ids,
            odd_vectors,
            odd_image_label_ids,
        )

    def load_all_image_vectors(
        self,
    ) -> (np.ndarray, np.ndarray):
        image_vectors = utils.get_all_image_feature_vectors("fc_layer")
        if image_vectors is None:
            return
        image_vectors = utils.convert_higher_dims_to_2d(image_vectors)

        image_label_ids = np.zeros(len(image_vectors))
        for i in range(len(image_label_ids)):
            _, label = self.dataset[i]
            image_label_ids[i] = int(label)

        # reduce image_vectors to reduced dimensions
        image_vectors, _ = dimension_reduction.nmf_als(
            image_vectors, self.dimensionsToReduce
        )

        return image_vectors, image_label_ids

    def split_data_into_train_test(
        self, image_vectors, target
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        train_vectors, test_vectors, train_target, test_target = [], [], [], []
        for i in range(len(self.image_vectors)):
            if i % 2 == 0:
                train_vectors.append(image_vectors[i])
                train_target.append(target[i])
            else:
                test_vectors.append(image_vectors[i])
                test_target.append(target[i])

        train_vectors = np.asarray(train_vectors)
        test_vectors = np.asarray(test_vectors)
        train_target = np.asarray(train_target, dtype=int)
        test_target = np.asarray(test_target, dtype=int)

        # print(
        #     f"Train vectors shape: {train_vectors.shape} -> train target shape: {train_target.shape}\n\
        #         Test vectors shape: {test_vectors.shape} -> test target shape: {test_target.shape}"
        # )

        return train_vectors, test_vectors, train_target, test_target


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
