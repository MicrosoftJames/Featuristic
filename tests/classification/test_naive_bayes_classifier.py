import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from featuristic.classification.mixed_type_naive_bayes import predict_log_proba, predict_proba
from featuristic.classification.naive_bayes_classifier import MixedTypeNaiveBayesClassifier, Distribution


@pytest.fixture
def gaussian_data():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    Y = np.random.randint(0, 2, 100)

    X_train = X[:80]
    Y_train = Y[:80]

    X_test = X[80:]
    Y_test = Y[80:]

    return X_train, Y_train, X_test, Y_test


@pytest.fixture
def bernoulli_data():
    np.random.seed(42)
    X = np.random.randint(0, 2, (100, 4))
    Y = np.random.randint(0, 2, 100)

    X_train = X[:80]
    Y_train = Y[:80]

    X_test = X[80:]
    Y_test = Y[80:]

    return X_train, Y_train, X_test, Y_test


@pytest.fixture
def multinomial_data():
    np.random.seed(42)
    X_feature_1 = np.random.randint(0, 3, (100, 4))
    X_feature_2 = np.random.randint(0, 3, (100, 4))
    Y = np.random.randint(0, 2, 100)

    X_feature_1_train = X_feature_1[:80]
    X_feature_2_train = X_feature_2[:80]

    X_feature_1_test = X_feature_1[80:]
    X_feature_2_test = X_feature_2[80:]

    Y_train = Y[:80]
    Y_test = Y[80:]

    return X_feature_1_train, X_feature_2_train, Y_train, X_feature_1_test, X_feature_2_test, Y_test


def test_predict_log_proba_all_gaussian(gaussian_data):
    X_train, Y_train, X_test, Y_test = gaussian_data

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    gnb_1 = GaussianNB()
    gnb_1.fit(X_train[:, :2], Y_train)

    gnb_2 = GaussianNB()
    gnb_2.fit(X_train[:, 2:], Y_train)

    single_gnb_log_proba = gnb.predict_log_proba(X_test)
    combined_gnb_log_proba = predict_log_proba(
        [gnb_1, gnb_2], [X_test[:, :2], X_test[:, 2:]])

    assert combined_gnb_log_proba.shape == single_gnb_log_proba.shape
    assert np.allclose(single_gnb_log_proba, combined_gnb_log_proba)


def test_predict_proba_all_gaussian(gaussian_data):
    X_train, Y_train, X_test, Y_test = gaussian_data

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    gnb_1 = GaussianNB()
    gnb_1.fit(X_train[:, :2], Y_train)

    gnb_2 = GaussianNB()
    gnb_2.fit(X_train[:, 2:], Y_train)

    single_gnb_proba = gnb.predict_proba(X_test)
    combined_gnb_proba = predict_proba(
        [gnb_1, gnb_2], [X_test[:, :2], X_test[:, 2:]])

    assert combined_gnb_proba.shape == single_gnb_proba.shape
    assert np.allclose(single_gnb_proba, combined_gnb_proba)


def test_predict_proba_all_bernoulli(bernoulli_data):
    X_train, Y_train, X_test, Y_test = bernoulli_data

    gnb = BernoulliNB()
    gnb.fit(X_train, Y_train)

    gnb_1 = BernoulliNB()
    gnb_1.fit(X_train[:, :2], Y_train)

    gnb_2 = BernoulliNB()
    gnb_2.fit(X_train[:, 2:], Y_train)

    single_gnb_proba = gnb.predict_proba(X_test)
    combined_gnb_proba = predict_proba(
        [gnb_1, gnb_2], [X_test[:, :2], X_test[:, 2:]])

    assert combined_gnb_proba.shape == single_gnb_proba.shape
    assert np.allclose(single_gnb_proba, combined_gnb_proba)


def test_predict_log_proba_all_bernoulli(bernoulli_data):
    X_train, Y_train, X_test, Y_test = bernoulli_data

    gnb = BernoulliNB()
    gnb.fit(X_train, Y_train)

    gnb_1 = BernoulliNB()
    gnb_1.fit(X_train[:, :2], Y_train)

    gnb_2 = BernoulliNB()
    gnb_2.fit(X_train[:, 2:], Y_train)

    single_gnb_log_proba = gnb.predict_log_proba(X_test)
    combined_gnb_log_proba = predict_log_proba(
        [gnb_1, gnb_2], [X_test[:, :2], X_test[:, 2:]])

    assert combined_gnb_log_proba.shape == single_gnb_log_proba.shape
    assert np.allclose(single_gnb_log_proba, combined_gnb_log_proba)


def test_predict_log_proba_multinomial_random_data(multinomial_data):
    X_feature_1_train, X_feature_2_train, Y_train, X_feature_1_test, X_feature_2_test, Y_test = multinomial_data

    mnb_1 = MultinomialNB()
    mnb_1.fit(X_feature_1_train, Y_train)

    mnb_2 = MultinomialNB()
    mnb_2.fit(X_feature_2_train, Y_train)

    mnb_1_log_proba = mnb_1.predict_log_proba(X_feature_1_test)
    mnb_2_log_proba = mnb_2.predict_log_proba(X_feature_2_test)

    combined_mnb_log_proba = predict_log_proba(
        [mnb_1, mnb_2], [X_feature_1_test, X_feature_2_test])

    assert combined_mnb_log_proba.shape == mnb_1_log_proba.shape == mnb_2_log_proba.shape


def test_predict_log_proba():
    X_a = np.array([
        [1, 2],
        [0, 1],
        [0, 2],
        [0, 1]])

    X_b = np.array([
        [4, 0],
        [2, 1],
        [3, 0],
        [1, 0]])

    Y = np.array([0, 0, 1, 1])

    x_a_test = np.array([[1, 2]])
    x_b_test = np.array([[3, 0]])

    # Laplace smoothing to prevent zero division error
    mnb_a = MultinomialNB(alpha=1e-30)
    mnb_a.fit(X_a, Y)

    # Laplace smoothing to prevent zero division error
    mnb_b = MultinomialNB(alpha=1e-30)
    mnb_b.fit(X_b, Y)

    proba = predict_proba([mnb_a, mnb_b], [x_a_test, x_b_test])

    # Using Laplace smoothing which means the result is not exactly 0 or 1
    np.testing.assert_allclose(proba[0][0], 1, atol=1e-5)
    np.testing.assert_allclose(proba[0][1], 0, atol=1e-5)

    # Second example that doesn't require Laplace smoothing
    X_a = np.array([
        [4, 6],
        [1, 3],
        [9, 1],
        [0, 1]])

    X_b = np.array([
        [3, 1],
        [1, 0],
        [2, 4],
        [0, 2]])

    Y = np.array([0, 0, 1, 1])

    x_a_test = np.array([[3, 5]])
    x_b_test = np.array([[2, 0]])

    mnb_a = MultinomialNB(alpha=0, force_alpha=True)
    mnb_a.fit(X_a, Y)

    mnb_b = MultinomialNB(alpha=0, force_alpha=True)
    mnb_b.fit(X_b, Y)

    proba = predict_proba([mnb_a, mnb_b], [x_a_test, x_b_test])

    # Hand cranked calculation
    P_Xa_Xb_y0 = (1/2)*((5/14) ** 3)*((9/14) ** 5)*(40320/720)*((4/5) ** 2)
    P_Xa_Xb_y1 = (1/2)*((9/11) ** 3)*((2/11) ** 5)*(40320/720)*((2/8) ** 2)
    normalizer = P_Xa_Xb_y0 + P_Xa_Xb_y1

    np.testing.assert_allclose(proba[0][0], P_Xa_Xb_y0/normalizer)
    np.testing.assert_allclose(proba[0][1], P_Xa_Xb_y1/normalizer)

    # Third example that mixes Bernoulli and Multinomial
    X_b = np.array([[1],
                    [0],
                    [1],
                    [0]])  # MLE of P(X_b|Y) is 1/2

    x_b_test = np.array([[1]])

    bnb_b = BernoulliNB()
    bnb_b.fit(X_b, Y)

    proba = predict_proba([mnb_a, bnb_b], [x_a_test, x_b_test])

    # Hand cranked calculation
    P_Xa_Xb_y0 = (1/2)*((5/14) ** 3)*((9/14) ** 5)*(40320/720)*(1/2)
    P_Xa_Xb_y1 = (1/2)*((9/11) ** 3)*((2/11) ** 5)*(40320/720)*(1/2)
    normalizer = P_Xa_Xb_y0 + P_Xa_Xb_y1

    np.testing.assert_allclose(proba[0][0], P_Xa_Xb_y0/normalizer)
    np.testing.assert_allclose(proba[0][1], P_Xa_Xb_y1/normalizer)


def test_specifying_priors_gaussian(gaussian_data):
    X_train, Y_train, X_test, Y_test = gaussian_data

    # Define priors
    gaussian_prior = [0.6, 0.4]

    # Initialize the classifier
    classifier = MixedTypeNaiveBayesClassifier()

    # Add Gaussian classifier with specified priors
    classifier.add_classifier(Distribution.GAUSSIAN, slice(
        0, 4), class_prior=gaussian_prior)

    # Fit the classifier
    classifier.fit(X_train, Y_train)

    # Perform predictions
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    # Assert predictions and probabilities are valid
    assert predictions.shape == Y_test.shape
    assert probabilities.shape == (X_test.shape[0], len(np.unique(Y_train)))

    # Ensure priors are correctly set
    assert np.allclose(
        classifier._classifier_settings[0].nb_classifier.class_prior_, gaussian_prior)


def test_specifying_priors_multinomial(multinomial_data):
    X_feature_1_train, X_feature_2_train, Y_train, X_feature_1_test, X_feature_2_test, Y_test = multinomial_data

    # Define priors
    multinomial_prior = [0.7, 0.3]

    # Initialize the classifier
    classifier = MixedTypeNaiveBayesClassifier()

    # Add Multinomial classifiers with specified priors
    classifier.add_classifier(Distribution.MULTINOMIAL, slice(
        0, 4), class_prior=multinomial_prior)
    classifier.add_classifier(Distribution.MULTINOMIAL, slice(
        4, 8), class_prior=multinomial_prior)

    # Combine features for testing
    X_train_combined = np.hstack((X_feature_1_train, X_feature_2_train))
    X_test_combined = np.hstack((X_feature_1_test, X_feature_2_test))

    # Fit the classifier
    classifier.fit(X_train_combined, Y_train)

    # Perform predictions
    predictions = classifier.predict(X_test_combined)
    probabilities = classifier.predict_proba(X_test_combined)

    # Assert predictions and probabilities are valid
    assert predictions.shape == Y_test.shape
    assert probabilities.shape == (
        X_test_combined.shape[0], len(np.unique(Y_train)))

    # Ensure priors are correctly set
    assert np.allclose(
        classifier._classifier_settings[0].nb_classifier.class_log_prior_, np.log(multinomial_prior))
    assert np.allclose(
        classifier._classifier_settings[1].nb_classifier.class_log_prior_, np.log(multinomial_prior))
