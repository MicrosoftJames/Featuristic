from typing import List
import numpy as np
import pandas as pd
from featuristic.classification.naive_bayes_classifier import Distribution
from featuristic.classification import MixedTypeNaiveBayesClassifier


class FeaturisticClassifier:
    def __init__(self, distributions: List[Distribution]):
        self._classifier = self._initialize_classifier(distributions)

    @staticmethod
    def _initialize_classifier(distributions: List[Distribution]):
        classifier = MixedTypeNaiveBayesClassifier()
        data_slice_start = 0
        for distribution in distributions:
            classifier.add_classifier(distribution, slice(
                data_slice_start, data_slice_start + 1))
            data_slice_start += 1
        return classifier

    def fit(self, features: pd.DataFrame, Y):
        X = features.to_numpy().astype(float)

        if not len(X) == len(Y):
            raise ValueError("Data and labels must have the same length.")

        self._classifier.fit(X, Y)

    def predict(self, features: pd.DataFrame):
        X = features.to_numpy().astype(float)
        return self._classifier.predict(X)

    def calculate_entropy(self, features: pd.DataFrame):
        X = features.to_numpy().astype(float)
        proba = self._classifier.predict_proba(X)
        log_proba = self._classifier.predict_log_proba(X)
        return - np.sum(proba * log_proba, axis=1)

    def rank_by_uncertainty(self, features: pd.DataFrame):
        """Ranks features based on the uncertainty of their class predictions.
        This is done by calculating the entropy of the predicted probabilities
        for each feature and returning the indices of the features sorted by
        decreasing entropy.

        Args:
            features (pd.DataFrame): Features to rank.

        Returns:
            numpy array (shape=(n_features,)): Array of feature indices sorted by increasing certainty (decreasing entropy).
        """
        entropies = self.calculate_entropy(features)
        # Sort by decreasing entropy
        return np.argsort(entropies)[::-1]
