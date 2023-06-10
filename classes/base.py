import numpy as np
import pandas as pd
from collections import Counter


class KNeighbors():

    def __init__(self, k_neighbors=3):
        self.__neighbors = k_neighbors

    def __euclidian_distance(self, x, y):
        return np.linalg.norm(x - y)

    def __predict(self, event):
        neighbors = self.k_neighbors(event)
        class_label = [self._class_train[index] for index in neighbors]
        most_common = Counter(class_label).most_common(1)
        return most_common[0][0]

    def k_neighbors(self, event, distances=False):
        if not hasattr(self, '_features_train'):
            raise ValueError('First set the sample data with fit function')
        k_distances = [
            self.__euclidian_distance(event, train_data) for train_data in self._features_train]
        k_indices = np.argsort(k_distances, kind='quicksort')
        if distances:
            return distances, k_indices[:self.__neighbors]
        return k_indices[:self.__neighbors]

    def fit(self, feature_sample, class_sample):
        self._features_train = feature_sample
        self._class_train = class_sample

    def predict(self, features_predict):
        predictions = [self.__predict(
            features_data for features_data in features_predict)]
        return np.array(predictions)


class SMOTE():

    def __init__(self, amount, seed=None, k_neighbors=5,):
        self.__neighbors = k_neighbors
        self.__amount = amount
        self.__k_nn = KNeighbors(self.__neighbors)
        self.__random = np.random.default_rng(seed)

    def fit(self, feature_sample, class_sample=''):
        self._columns_name = feature_sample.columns
        self.__features = feature_sample.drop(columns=class_sample).values
        target = feature_sample[[class_sample]].values  # Es necesario?
        classes, count = np.unique(target, return_counts=True)
        self.__classes = classes
        self.__count_class = count
        self.__index_minority = np.argmin(count)
        minority_value = self.__classes[self.__index_minority]
        self.__features_minority = feature_sample[feature_sample[class_sample]
                                                  == minority_value].drop(columns=class_sample).values
        self.__k_nn.fit(self.__features, target)

    def resample(self):
        minority_value = self.__classes[self.__index_minority]
        t = self.__count_class[self.__index_minority]
        N = self.__amount
        if N < 100:
            t = (N / 100) * t
            N = 100
            # TODO: tomar de manera aleatorea t datos de __features
            # features_minority = ...
        N = (int)(N / 100)
        synthetic = []
        for feature_vector in self.__features_minority:
            knn = self.__k_nn.k_neighbors(feature_vector)
            new_sample = self.__populate(
                N, feature_vector, knn)
            new_sample = np.append(new_sample, minority_value)
            synthetic.append(new_sample)
        return pd.DataFrame(synthetic, columns=self._columns_name)

    def __populate(self, N, feature_vector, knn):
        synthetic_sample = None
        while N != 0:
            neighbor_index = self.__random.integers(len(knn), size=1)
            neighbor = self.__features[knn[neighbor_index[0]]]
            diff = neighbor - feature_vector
            gap = self.__random.random()
            synthetic_sample = np.round((feature_vector + (diff * gap)), 3)
            N -= 1
        return synthetic_sample
