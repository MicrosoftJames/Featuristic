from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
from sklearn.naive_bayes import _BaseNB, BernoulliNB, GaussianNB, MultinomialNB

from featuristic.classification.mixed_type_naive_bayes import predict, predict_proba, predict_log_proba


class Distribution(Enum):
    """An enum to represent the different types of Naive Bayes classifiers."""
    MULTINOMIAL = 1
    BERNOULLI = 2
    GAUSSIAN = 3


@dataclass
class _NBClassifier():
    """A dataclass to hold the classifier settings for each column.

    Attributes:
        nb_classifier(_BaseNB): The classifier to use.
        data_slice(slice): The slice of the data to used for this classifier.
        expand_multinomial_col(bool): Whether to expand the multinomial column into two columns.
    """
    nb_classifier: _BaseNB
    data_slice: slice
    expand_multinomial_col: bool


class MixedTypeNaiveBayesClassifier():
    """A class that implements a Naive Bayes classifier that can handle mixed types of data.
    This class is a wrapper around the sklearn Naive Bayes classifiers and allows for
    different types of classifiers to be used for different columns of the data.
    """

    def __init__(self):
        self._classifier_settings: List[_NBClassifier] = []

    def add_classifier(self, distribution: Distribution, data_slice: slice, classifier_args: dict = {}, expand_multinomial_col=False, class_prior=None):
        """Adds a classifier to the list of classifiers.

        Args:
            distribution (Distribution): The type of distribution to use for the classifier.
            data_slice (slice): The slice of the data columns to used for this classifier.
            classifier_args (dict): The arguments to pass to the corresponding sklearn classifier.
            expand_multinomial_col (bool): Whether to expand the multinomial column into two columns.
            class_prior (array-like of shape (n_classes,), default=None):
                Prior probabilities of the classes. If specified, the priors are not
                adjusted according to the data.
        """
        if distribution == Distribution.MULTINOMIAL:
            nb_classifier = MultinomialNB(**classifier_args,
                                          class_prior=class_prior)

        elif distribution == Distribution.BERNOULLI:
            nb_classifier = BernoulliNB(**classifier_args,
                                        class_prior=class_prior)

        elif distribution == Distribution.GAUSSIAN:
            nb_classifier = GaussianNB(**classifier_args,
                                       priors=class_prior)

        self._classifier_settings.append(
            _NBClassifier(nb_classifier, data_slice, expand_multinomial_col))

    def _expand_proportions(self, col: np.ndarray) -> np.ndarray:
        """Turns a single column into two whereby the second column is one minus the first column."""
        return np.column_stack((col, 1-col))

    def _validate_slices(self, X):
        # Raise an error if the slices are not disjoint, i.e. raise an error if they overlap
        slice_set = set()
        for classifer_settings in self._classifier_settings:
            if not slice_set.isdisjoint(set(range(classifer_settings.data_slice.start, classifer_settings.data_slice.stop))):
                raise ValueError("The slices must be disjoint")
            slice_set.update(
                range(classifer_settings.data_slice.start, classifer_settings.data_slice.stop))

        if len(slice_set) != X.shape[1]:
            raise ValueError("The slices must cover all columns in X")

        if min(slice_set) != 0 or max(slice_set) != X.shape[1] - 1:
            raise ValueError(
                "The slices must start from 0 and end at the last column index")

    def fit(self, X, Y):
        """Fits the Naive Bayes classifiers to the data.
        Args:
            X (np.ndarray): The feature matrix.
            Y (np.ndarray): The target vector.
        """
        self._validate_slices(X)
        for classifer_settings in self._classifier_settings:
            X_slice = X[:, classifer_settings.data_slice]
            if classifer_settings.expand_multinomial_col:
                X_slice = self._expand_proportions(X_slice)
            classifer_settings.nb_classifier.fit(X_slice, Y)

    def predict(self, X) -> np.ndarray:
        """Predicts the class labels for the data.
        Args:
            X (np.ndarray): The feature matrix.
        Returns:
            np.ndarray: The predicted class labels.
        """
        return predict([cs.nb_classifier for cs in self._classifier_settings],
                       [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])

    def predict_proba(self, X) -> np.ndarray:
        """Predicts the class probabilities for the data.
        Args:
            X (np.ndarray): The feature matrix.
        Returns:
            np.ndarray: The predicted class probabilities.
        """
        return predict_proba([cs.nb_classifier for cs in self._classifier_settings],
                             [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])

    def predict_log_proba(self, X) -> np.ndarray:
        """Predicts the log-probabilities of the class labels for the data.
        Args:
            X (np.ndarray): The feature matrix.
        Returns:
            np.ndarray: The predicted log-probabilities of the class labels.
        """
        return predict_log_proba([cs.nb_classifier for cs in self._classifier_settings],
                                 [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])
