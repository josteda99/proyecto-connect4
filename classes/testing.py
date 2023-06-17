import numpy as np
import pandas as pd
import sys


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def find_best_split(X, y):
    best_feature_index = None
    best_threshold = None
    best_gain = -np.inf
    initial_entropy = entropy(y)

    for feature_index in range(X.shape[1]):
        feature_values = X[:, feature_index]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:
            left_indices = feature_values <= threshold
            right_indices = feature_values > threshold

            left_entropy = entropy(y[left_indices])
            right_entropy = entropy(y[right_indices])

            information_gain = initial_entropy - \
                (left_entropy * np.sum(left_indices) +
                 right_entropy * np.sum(right_indices)) / len(y)

            if information_gain > best_gain:
                best_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


def build_decision_tree(X, y, max_depth):
    if len(np.unique(y)) == 1:
        # Si todas las instancias tienen la misma clase, retornar un nodo hoja
        return {'class': y[0]}

    if X.shape[1] == 0 or max_depth == 0:
        # Si no quedan características para dividir o se alcanza la profundidad máxima, retornar un nodo hoja con la clase mayoritaria
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return {'class': majority_class}

    best_feature_index, best_threshold = find_best_split(X, y)

    left_indices = X[:, best_feature_index] <= best_threshold
    right_indices = X[:, best_feature_index] > best_threshold

    left_subtree = build_decision_tree(
        X[left_indices], y[left_indices], max_depth - 1)
    right_subtree = build_decision_tree(
        X[right_indices], y[right_indices], max_depth - 1)

    return {'feature_index': best_feature_index, 'threshold': best_threshold,
            'left_subtree': left_subtree, 'right_subtree': right_subtree}


def predict(instance, tree):
    if 'class' in tree:
        return tree['class']

    feature_value = instance[tree['feature_index']]

    if feature_value <= tree['threshold']:
        return predict(instance, tree['left_subtree'])
    else:
        return predict(instance, tree['right_subtree'])


df = pd.read_csv(
    'data\Spam.csv',
    header=0,
    index_col=False)

columnas_deseadas = ["word_freq_make", "word_freq_address",
                     "word_freq_all", "word_freq_3d", "word_freq_our"]

X = df.drop("spam", axis=1).values
y = df["spam"].values
df = df.drop(
    columns=[col for col in df.columns if col not in columnas_deseadas])

tree = build_decision_tree(X, y, max_depth=7)

new_instance = np.array([0.5, 0.3, 0.2, 0.1, 0.6, 0.7, 0.2, 0.4, 0.3, 0.1, 0.4, 0.5, 0.2, 0.1,
                         0.3, 0.5, 0.6, 0.2, 0.4, 0.7, 0.2, 0.3, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2,
                         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                         0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         100])

prediction = predict(new_instance, tree)
print("Predicción:", prediction)
