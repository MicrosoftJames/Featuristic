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

    def fit(self, features: List[Feature], Y):
        X = self._convert_features_to_array(features)

        if not len(X) == len(Y):
            raise ValueError("Data and labels must have the same length.")

        self._classifier.fit(X, Y)

    def predict(self, features: List[Feature]):
        X = self._convert_features_to_array(features)
        return self._classifier.predict(X)
