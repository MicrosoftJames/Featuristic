from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
from sklearn.naive_bayes import _BaseNB, BernoulliNB, GaussianNB, MultinomialNB

from featuristic.classification.mixed_type_naive_bayes import predict, predict_proba, predict_log_proba


class Distribution(Enum):
    MULTINOMIAL = 1
    BERNOULLI = 2
    GAUSSIAN = 3


@dataclass
class _NBClassifier():
    nb_classifier: _BaseNB
    data_slice: slice
    expand_multinomial_col: bool


class MixedTypeNaiveBayesClassifier():
    def __init__(self):
        self._classifier_settings: List[_NBClassifier] = []

    def add_classifier(self, classifier: Distribution, data_slice: slice, classifier_args: dict = {}, expand_multinomial_col=False):
        if classifier == Distribution.MULTINOMIAL:
            nb_classifier = MultinomialNB(**classifier_args)

        elif classifier == Distribution.BERNOULLI:
            nb_classifier = BernoulliNB(**classifier_args)

        elif classifier == Distribution.GAUSSIAN:
            nb_classifier = GaussianNB(**classifier_args)

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
        self._validate_slices(X)
        for classifer_settings in self._classifier_settings:
            X_slice = X[:, classifer_settings.data_slice]
            if classifer_settings.expand_multinomial_col:
                X_slice = self._expand_proportions(X_slice)
            classifer_settings.nb_classifier.fit(X_slice, Y)

    def predict(self, X) -> np.ndarray:
        return predict([cs.nb_classifier for cs in self._classifier_settings],
                       [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba([cs.nb_classifier for cs in self._classifier_settings],
                             [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])

    def predict_log_proba(self, X) -> np.ndarray:
        return predict_log_proba([cs.nb_classifier for cs in self._classifier_settings],
                                 [X[:, cs.data_slice] if not cs.expand_multinomial_col else self._expand_proportions(X[:, cs.data_slice]) for cs in self._classifier_settings])
