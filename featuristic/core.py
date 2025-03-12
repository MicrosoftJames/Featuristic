from typing import List
import numpy as np
from featuristic.classification.naive_bayes_classifier import Distribution
from featuristic.classification import MixedTypeNaiveBayesClassifier
from featuristic.features.feature import Feature


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

    @staticmethod
    def _convert_features_to_array(features):
        """
        Convert a list of features to a numpy array.
        """
        return np.array([f.values for f in features]).T

    @staticmethod
    def _validate_features(features):
        if not features:
            raise ValueError("No features provided.")

        if not all(isinstance(f, Feature) for f in features):
            raise ValueError("All items in features must be of type Feature.")

    def fit(self, features: List[Feature], Y):
        self._validate_features(features)
        X = self._convert_features_to_array(features)

        if not len(X) == len(Y):
            raise ValueError("Data and labels must have the same length.")

        self._classifier.fit(X, Y)

    def predict(self, features: List[Feature]):
        self._validate_features(features)
        X = self._convert_features_to_array(features)
        return self._classifier.predict(X)

    def calculate_entropy(self, features: List[Feature]):
        self._validate_features(features)
        X = self._convert_features_to_array(features)

        proba = self._classifier.predict_proba(X)
        log_proba = self._classifier.predict_log_proba(X)
        return - np.sum(proba * log_proba, axis=1)

    def rank_features_by_uncertainty(self, features: List[Feature]):
        """Ranks features based on the uncertainty of their class predictions.
        This is done by calculating the entropy of the predicted probabilities
        for each feature and returning the indices of the features sorted by
        decreasing entropy.

        Args:
            features (List[Feature]): List of features to rank.

        Returns:
            numpy array (shape=(n_features,)): Array of feature indices sorted by increasing certainty (decreasing entropy).
        """
        self._validate_features(features)
        entropies = self.calculate_entropy(features)
        # Sort by decreasing entropy
        return np.argsort(entropies)[::-1]
