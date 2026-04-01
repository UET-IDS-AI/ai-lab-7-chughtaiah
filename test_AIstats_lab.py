import numpy as np
import AI_stats_lab as A


def test_naive_bayes_mle_spam():
    priors, word_probs, prediction = A.naive_bayes_mle_spam()

    assert isinstance(priors, dict)
    assert isinstance(word_probs, dict)
    assert prediction in [0, 1]

    assert 0 in priors and 1 in priors
    assert abs(priors[0] + priors[1] - 1.0) < 1e-8

    assert 0 in word_probs and 1 in word_probs
    assert isinstance(word_probs[0], dict)
    assert isinstance(word_probs[1], dict)

    # Strong spam-related words should exist in spam vocabulary
    for word in ["win", "cash", "prize", "now"]:
        assert word in word_probs[1]

    # The test email should be classified as spam
    assert prediction == 1


def test_knn_iris():
    train_acc, test_acc, predictions = A.knn_iris(k=3, test_size=0.2, seed=0)

    assert isinstance(train_acc, float)
    assert isinstance(test_acc, float)
    assert isinstance(predictions, np.ndarray)

    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= test_acc <= 1.0

    assert predictions.ndim == 1
    assert len(predictions) == 30  # 20% of 150 iris samples

    # KNN should perform well on Iris
    assert train_acc > 0.90
    assert test_acc > 0.85
