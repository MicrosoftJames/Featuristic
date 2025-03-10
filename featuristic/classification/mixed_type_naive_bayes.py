from typing import List

import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import _BaseNB


def predict_log_proba(bayesian_classifiers: List[_BaseNB], Xs: List[np.ndarray]):

    log_class_priors = [bc.class_log_prior_ if hasattr(bc, "class_log_prior_") else np.log(
        bc.class_prior_)for bc in bayesian_classifiers]

    if not all([np.allclose(log_class_priors[0], log_class_priors[i], atol=0.000001) for i in range(1, len(bayesian_classifiers))]):
        raise AttributeError("All classifiers must have the same priors")

    jlls = [bc._joint_log_likelihood(X)
            for bc, X in zip(bayesian_classifiers, Xs)]

    jll_all = np.sum(jlls, axis=0) - (len(jlls) - 1) * \
        log_class_priors[0]

    assert jll_all.shape == jlls[0].shape

    log_prob_x = logsumexp(jll_all, axis=1)
    # Shape: (n_samples, n_classes)
    return jll_all - np.atleast_2d(log_prob_x).T


def predict_proba(baysian_classifiers: List[_BaseNB], Xs: List[np.ndarray]):
    # Shape: (n_samples, n_classes)
    return np.exp(predict_log_proba(baysian_classifiers, Xs))


def predict(baysian_classifiers: List[_BaseNB], Xs: List[np.ndarray]):
    # Shape: (n_samples, n_classes)
    return np.argmax(predict_proba(baysian_classifiers, Xs), axis=1)
