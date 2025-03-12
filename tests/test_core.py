import numpy as np
import pytest
from featuristic.classification.naive_bayes_classifier import Distribution
from featuristic.core import FeaturisticClassifier
from featuristic.features.feature import Feature


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
    train_features = [
        Feature(name="feature1", values=[1.0, 2.0, 3.0]),
        Feature(name="feature2", values=[4.0, 5.0, 6.0]),
        Feature(name="feature3", values=[7.0, 8.0, 9.0])
    ]

    Y = np.array([0, 1, 1])

    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    # Fit the classifier
    classifier.fit(train_features, Y)

    # Create dummy test data
    test_features = [
        Feature(name="feature1", values=[1.0, 2.0, 3.0]),
        Feature(name="feature2", values=[4.0, 5.0, 6.0]),
        Feature(name="feature3", values=[7.0, 8.0, 9.0])
    ]

    # Predict
    predictions = classifier.predict(test_features)

    assert len(predictions) == len(Y)
    np.testing.assert_equal(predictions, [0, 1, 1])  # Will be same as Y


def test_invalid_fit():
    # Create dummy data
    train_features = [
        Feature(name="feature1", values=[1.0, 2.0, 3.0]),
        Feature(name="feature2", values=[4.0, 5.0, 6.0]),
        Feature(name="feature3", values=[7.0, 8.0, 9.0])
    ]

    Y = np.array([0, 1])

    # Initialize the classifier
    classifier = FeaturisticClassifier(
        distributions=[Distribution.GAUSSIAN,
                       Distribution.GAUSSIAN, Distribution.GAUSSIAN]
    )

    # Fit the classifier and expect a ValueError
    with pytest.raises(ValueError):
        classifier.fit(train_features, Y)


def test_validate_features():
    # Valid features should not raise an exception
    valid_features = [
        Feature(name="feature1", values=[1.0, 2.0]),
        Feature(name="feature2", values=[3.0, 4.0])
    ]
    FeaturisticClassifier._validate_features(
        valid_features)  # Should not raise

    # Test empty features list
    with pytest.raises(ValueError, match="No features provided."):
        FeaturisticClassifier._validate_features([])

    # Test non-Feature objects
    invalid_features = [
        Feature(name="feature1", values=[1.0, 2.0]),
        "not a feature"  # This is a string, not a Feature
    ]
    with pytest.raises(ValueError, match="All items in features must be of type Feature."):
        FeaturisticClassifier._validate_features(invalid_features)

    # Test None
    with pytest.raises(ValueError, match="No features provided."):
        FeaturisticClassifier._validate_features(None)
