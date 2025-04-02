import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from featuristic.classification import Distribution, MixedTypeNaiveBayesClassifier


def test_add_classifier():
    mtc = MixedTypeNaiveBayesClassifier()
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2), {})
    mtc.add_classifier(Distribution.BERNOULLI, slice(2, 4), {})
    mtc.add_classifier(Distribution.GAUSSIAN, slice(4, 6), {})
    assert len(mtc._classifier_settings) == 3
    assert mtc._classifier_settings[0].nb_classifier.__class__ == MultinomialNB
    assert mtc._classifier_settings[1].nb_classifier.__class__ == BernoulliNB
    assert mtc._classifier_settings[2].nb_classifier.__class__ == GaussianNB


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


def test_predict_proba(gaussian_data):
    X_train, Y_train, X_test, Y_test = gaussian_data

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    mtc = MixedTypeNaiveBayesClassifier()

    mtc.add_classifier(Distribution.GAUSSIAN, slice(0, 2))
    mtc.add_classifier(Distribution.GAUSSIAN, slice(2, 4))

    mtc.fit(X_train, Y_train)

    single_gnb_log_proba = gnb.predict_log_proba(X_test)
    combined_gnb_log_proba = mtc.predict_log_proba(X_test)

    assert combined_gnb_log_proba.shape == single_gnb_log_proba.shape
    assert np.allclose(single_gnb_log_proba, combined_gnb_log_proba)

    mtc = MixedTypeNaiveBayesClassifier()

    mtc.add_classifier(Distribution.GAUSSIAN, slice(0, 2),
                       class_prior=np.array([0.5, 0.5]))
    mtc.add_classifier(Distribution.GAUSSIAN, slice(2, 4),
                       class_prior=np.array([0.4, 0.6]))

    mtc.fit(X_train, Y_train)

    with pytest.raises(AttributeError):
        mtc.predict_log_proba(X_test)


def test_predict(gaussian_data):
    X_train, Y_train, X_test, Y_test = gaussian_data

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    mtc = MixedTypeNaiveBayesClassifier()

    mtc.add_classifier(Distribution.GAUSSIAN, slice(0, 2))
    mtc.add_classifier(Distribution.GAUSSIAN, slice(2, 4))

    mtc.fit(X_train, Y_train)

    single_gnb_predict = gnb.predict(X_test)
    combined_gnb_predict = mtc.predict(X_test)
    assert combined_gnb_predict.shape == single_gnb_predict.shape


def test_with_expanded_proportions():
    X_a = np.array([
        [0.4, 0.1],
        [0.2, 0.4]])

    X_b = np.array([
        [0.1, 0.5],
        [0.3, 0.1]])
    Y = np.array([0, 1])

    mtc = MixedTypeNaiveBayesClassifier()
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2), classifier_args={
        "alpha": 0, "force_alpha": True}, expand_multinomial_col=True)
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(2, 4), classifier_args={
        "alpha": 0, "force_alpha": True}, expand_multinomial_col=True)
    mtc.fit(np.hstack([X_a, X_b]), Y)


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

    mtc = MixedTypeNaiveBayesClassifier()

    # Laplace smoothing to prevent zero division error
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(
        0, 2), classifier_args={"alpha": 1e-30})
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(
        2, 4), classifier_args={"alpha": 1e-30})

    mtc.fit(np.hstack([X_a, X_b]), Y)

    proba = mtc.predict_proba(np.hstack([x_a_test, x_b_test]))

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

    mtc = MixedTypeNaiveBayesClassifier()

    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2), classifier_args={
        "alpha": 0, "force_alpha": True})
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(2, 4), classifier_args={
        "alpha": 0, "force_alpha": True})

    mtc.fit(np.hstack([X_a, X_b]), Y)

    proba = mtc.predict_proba(np.hstack([x_a_test, x_b_test]))

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

    mtc = MixedTypeNaiveBayesClassifier()

    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2), classifier_args={
        "alpha": 0, "force_alpha": True})
    mtc.add_classifier(Distribution.BERNOULLI, slice(2, 3))

    mtc.fit(np.hstack([X_a, X_b]), Y)

    proba = mtc.predict_proba(np.hstack([x_a_test, x_b_test]))

    # Hand cranked calculation
    P_Xa_Xb_y0 = (1/2)*((5/14) ** 3)*((9/14) ** 5)*(40320/720)*(1/2)
    P_Xa_Xb_y1 = (1/2)*((9/11) ** 3)*((2/11) ** 5)*(40320/720)*(1/2)
    normalizer = P_Xa_Xb_y0 + P_Xa_Xb_y1

    np.testing.assert_allclose(proba[0][0], P_Xa_Xb_y0/normalizer)
    np.testing.assert_allclose(proba[0][1], P_Xa_Xb_y1/normalizer)


def test_invalid_slices():
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

    mtc = MixedTypeNaiveBayesClassifier()

    # Overlapping slices
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2))
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(1, 3))

    with pytest.raises(ValueError):
        mtc.fit(np.hstack([X_a, X_b]), Y)

    mtc = MixedTypeNaiveBayesClassifier()

    # Slice exceeds number of columns
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(0, 2))
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(2, 5))

    with pytest.raises(ValueError):
        mtc.fit(np.hstack([X_a, X_b]), Y)

    mtc = MixedTypeNaiveBayesClassifier()

    # Slice does not start from 0
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(-1, 2))
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(3, 5))

    with pytest.raises(ValueError):
        mtc.fit(np.hstack([X_a, X_b]), Y)


def test_validate_slices():
    X_a = np.array([
        [1, 2],
        [0, 1],
        [0, 2],
        [0, 1]])

    Y = np.array([0])

    mtc = MixedTypeNaiveBayesClassifier()

    # Valid slices
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(1, 2))
    mtc.add_classifier(Distribution.MULTINOMIAL, slice(2, 3))

    with pytest.raises(ValueError):
        mtc.fit(X_a, Y)
