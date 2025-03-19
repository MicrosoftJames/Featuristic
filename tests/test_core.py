import numpy as np
import pandas as pd
import pytest
from featuristic.classification.naive_bayes_classifier import Distribution
from featuristic.core import FeaturisticClassifier


def test_init_featuristic_classifier():
    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    assert classifier._classifier is not None
    assert len(classifier._classifier._classifier_settings) == 3


def test_fit_predict():
    # Create dummy data
    train_features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "feature3": [7.0, 8.0, 9.0]
    })

    Y = np.array([0, 1, 1])

    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    # Fit the classifier
    classifier.fit(train_features, Y)

    # Create dummy test data
    test_features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "feature3": [7.0, 8.0, 9.0]
    })

    # Predict
    predictions = classifier.predict(test_features)

    assert len(predictions) == len(Y)
    np.testing.assert_equal(predictions, [0, 1, 1])  # Will be same as Y


def test_invalid_fit():
    # Create dummy data
    train_features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "feature3": [7.0, 8.0, 9.0]
    }, index=["a", "b", "c"])

    Y = np.array([0, 1])

    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    # Fit the classifier and expect a ValueError
    with pytest.raises(ValueError):
        classifier.fit(train_features, Y)


def test_entropy_uniform_distribution():
    """Test entropy with uniform probability distribution (maximum entropy)."""
    features_train = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [2.0, 3.0]
    })

    Y_train = np.array([0, 1])

    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN, Distribution.GAUSSIAN])

    classifier.fit(features_train, Y_train)

    features_test = pd.DataFrame({
        "feature1": [1.5, 1.5],
        "feature2": [2.5, 2.5]
    })

    entropy = classifier.calculate_entropy(features_test)

    assert len(entropy) == 2  # Two samples, two lots of entropy values

    # Expected entropy for uniform distribution: -sum(0.5 * log(0.5)) = -log(0.5)
    expected_entropy = -np.log(0.5)  # For two samples
    np.testing.assert_almost_equal(entropy, expected_entropy, decimal=4)


def test_entropy_deterministic_distribution():
    """Test entropy with deterministic probability distribution (minimum entropy)."""
    test_features = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0]
    })

    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN, Distribution.GAUSSIAN])

    # Mock deterministic probability distribution (minimum entropy)
    def mock_predict_proba(X):
        # Return deterministic probabilities [1.0, 0.0] for all samples
        return np.array([[1.0, 0.0], [1.0, 0.0]])

    def mock_predict_log_proba(X):
        # Return log probabilities with small epsilon to avoid log(0)
        epsilon = 1e-10
        return np.array([[np.log(1.0), np.log(epsilon)], [np.log(1.0), np.log(epsilon)]])

    classifier._classifier.predict_proba = mock_predict_proba
    classifier._classifier.predict_log_proba = mock_predict_log_proba

    entropy = classifier.calculate_entropy(test_features)

    # Expected entropy for deterministic distribution should be close to 0
    np.testing.assert_almost_equal(entropy, 0, decimal=4)


def test_rank_features_by_uncertainty():
    # Create dummy data
    features_train = pd.DataFrame({
        "feature1": [1.0, 2.0, 4.0],
        "feature2": [4.0, 5.0, 8.0],
        "feature3": [7.0, 7.5, 20.0]
    })

    Y = np.array([1, 1, 0])

    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    # Fit the classifier
    classifier.fit(features_train, Y)

    # Create 3 test features - the first is the same as the training data (perfectly certain)
    # the second is a fractionally larger than the second training example (fractionally uncertain)
    # the third is halfway between the second and third training example (uncertain)
    features_test = pd.DataFrame({
        "feature1": [1.0, 2.2, 3.0],
        "feature2": [4.0, 5.5, 6.5],
        "feature3": [7.0, 8.0, (7.5 + 20.0) / 2]
    })

    # Rank features by uncertainty
    ranked_features_idx = classifier.rank_by_uncertainty(
        features_test)

    assert len(ranked_features_idx) == len(features_train)
    assert ranked_features_idx[2] == 0  # Most uncertain feature
    assert ranked_features_idx[1] == 1  # Second most uncertain
    assert ranked_features_idx[0] == 2  # Least uncertain feature
