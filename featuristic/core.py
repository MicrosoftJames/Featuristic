from typing import List
import numpy as np
import pandas as pd
from featuristic.classification.naive_bayes_classifier import Distribution
from featuristic.classification import MixedTypeNaiveBayesClassifier


class FeaturisticClassifier:
    """A wrapper class for the MixedTypeNaiveBayesClassifier.
    """

    def __init__(self, distributions: List[Distribution], class_prior=None):
        """Initializes the classifier with the given distributions and class prior.

        Args:
        distributions (List[Distribution]): A list of distributions to use for the classifier 
            where each distribution corresponds to a column in the data based on the column order.
        class_prior (array-like of shape (n_classes,), default=None): 
            Prior probabilities of the classes. If specified, the priors are not
            adjusted according to the data.
        """
        self._classifier = self._initialize_classifier(
            distributions, class_prior)

    @staticmethod
    def _initialize_classifier(distributions: List[Distribution], class_prior=None):
        classifier = MixedTypeNaiveBayesClassifier()
        data_slice_start = 0
        for distribution in distributions:
            classifier.add_classifier(distribution, slice(
                data_slice_start, data_slice_start + 1))
            data_slice_start += 1
        return classifier

    def fit(self, features: pd.DataFrame, Y):
        """Fits the classifier to the features and labels.
        Args:
            features (pd.DataFrame): The features to fit.
            Y (pd.Series): The labels to fit.
        """
        X = features.to_numpy().astype(float)

        if not len(X) == len(Y):
            raise ValueError("Data and labels must have the same length.")

        self._classifier.fit(X, Y)

    def predict(self, features: pd.DataFrame):
        """Predicts the labels for the given features.
        Args:
            features (pd.DataFrame): The features to predict.
        Returns:
            numpy array (shape=(n_samples,)): Array of predicted labels.
        """
        X = features.to_numpy().astype(float)
        return self._classifier.predict(X)

    def predict_proba(self, features: pd.DataFrame):
        """Predicts the class probabilities for the given features.
        Args:
            features (pd.DataFrame): The features to predict.
        Returns:
            numpy array (shape=(n_samples, n_classes)): Array of predicted class probabilities.
        """
        X = features.to_numpy().astype(float)
        return self._classifier.predict_proba(X)

    def calculate_entropy(self, features: pd.DataFrame):
        """Calculates the entropy of the predicted class probabilities for each feature.

        Args:
            features (pd.DataFrame): Features to calculate entropy for.
        Returns:
            numpy array (shape=(n_samples,)): Array of entropies for each feature.
        """
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
            numpy array (shape=(n_samples,)): Array of feature indices sorted by increasing certainty (decreasing entropy).
        """
        entropies = self.calculate_entropy(features)
        # Sort by decreasing entropy
        return np.argsort(entropies)[::-1]
