import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class KNeighbors():

    def __init__(self, k_neighbors=3):
        self.__neighbors = k_neighbors

    def __euclidian_distance(self, event, classes):
        return np.linalg.norm(event - classes)

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
        return self

    def predict(self, features_predict):
        predictions = [self.__predict(
            features_data) for features_data in features_predict]
        return np.array(predictions)


class SMOTE():

    def __init__(self, seed=None, k_neighbors=5,):
        self.__neighbors = k_neighbors
        self.__k_nn = KNeighbors(self.__neighbors)
        self.__random = np.random.default_rng(seed)

    def fit(self, feature_sample, class_sample=''):
        self._columns_name = feature_sample.columns
        self.__features = feature_sample.drop(columns=class_sample).values
        target = feature_sample[[class_sample]].values
        classes, count = np.unique(target, return_counts=True)
        self.__classes = classes
        self.__count_class = count
        self.__index_minority = np.argmin(count)
        minority_value = self.__classes[self.__index_minority]
        self.__features_minority = feature_sample[feature_sample[class_sample]
                                                  == minority_value].drop(columns=class_sample).values
        self.__k_nn.fit(self.__features, target)

    def resample(self, amount):
        minority_value = self.__classes[self.__index_minority]
        t = self.__count_class[self.__index_minority]
        if amount < 100:
            t = (np.int32)((amount / 100) * t)
            amount = 100
            features_minority = self.__random.choice(
                self.__features_minority, size=t, replace=False)
        ratio = (np.int32)(amount / 100)
        synthetic = []
        for feature_vector in features_minority:
            knn = self.__k_nn.k_neighbors(feature_vector)
            new_sample = self.__populate(
                amount, feature_vector, knn)
            new_sample = np.append(new_sample, minority_value)
            synthetic.append(new_sample)
        return pd.DataFrame(synthetic, columns=self._columns_name)

    def __populate(self, ratio, feature_vector, knn):
        synthetic_sample = None
        while ratio != 0:
            neighbor_index = self.__random.integers(len(knn), size=1)
            neighbor = self.__features[knn[neighbor_index[0]]]
            diff = neighbor - feature_vector
            gap = self.__random.random()
            synthetic_sample = np.round((feature_vector + (diff * gap)), 3)
            ratio -= 1
        return synthetic_sample


class NaiveBayes():

    def fit(self, features, classes):
        n_samples, n_features = features.shape
        self._classes = np.unique(classes)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            features_c = features[classes == c]
            self._mean[idx, :] = features_c.mean(axis=0)
            self._var[idx, :] = features_c.var(axis=0)
            self._priors[idx] = features_c.shape[0] / float(n_samples)
        return self

    def predict(self, features):
        y_pred = [self._predict(event) for event in features]
        return np.array(y_pred)

    def _predict(self, event):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, event)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, event):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((event - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


class VIF():
    def __init__(self, n_features=-1):
        self.n_features = n_features

    def fit(self, feature_sample, target):
        self._target = target
        self._feature = feature_sample
        self.__features_names = feature_sample.columns

    def transform(self, *models, test_features=None, test_target=None):
        if self.n_features == -1:
            self.n_features = 1
        models_scores = {}
        models_names = [model.__class__.__name__ for model in models]
        for model, name in zip(models, models_names):
            model.fit(self._feature.values, self._target.values)
            models_scores[name] = []
        drop_feat = []
        while len(self.__features_names) - len(drop_feat) != self.n_features:
            selected_features = self._feature.drop(drop_feat, axis=1)
            selected_features_name = selected_features.columns.values
            vif_values = [variance_inflation_factor(
                selected_features.values, index_feat) for index_feat in range(selected_features.shape[1])]
            vif_values = np.round(vif_values, 2)
            index_sorted_vif = np.argsort(vif_values, kind='quicksort')
            if vif_values[index_sorted_vif[-1]] > 1:
                for model, name in zip(models, models_names):
                    models_scores[name].append(
                        f1_score(
                            test_target,
                            model.fit(selected_features.values, self._target.values).predict(
                                test_features.drop(drop_feat, axis=1).values),
                            average='weighted') * 100)
            drop_feat.append(selected_features_name[index_sorted_vif[-1]])

        return self._feature.drop(drop_feat, axis=1), models_scores
