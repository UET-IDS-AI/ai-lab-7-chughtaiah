"""
AIstats_lab_solution.py

Solution file for:
1. Naive Bayes spam classification using simple MLE
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    tokenized = [text.lower().split() for text in texts]
    vocabulary = sorted(set(word for doc in tokenized for word in doc))

    classes = np.unique(labels)

    priors = {}
    word_probs = {}

    for cls in classes:
        class_docs = [tokenized[i] for i in range(len(tokenized)) if labels[i] == cls]
        priors[cls] = len(class_docs) / len(texts)

        word_counts = {}
        total_words = 0

        for doc in class_docs:
            for word in doc:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words += 1

        probs = {}
        for word in vocabulary:
            probs[word] = word_counts.get(word, 0) / total_words

        word_probs[cls] = probs

    test_tokens = test_email.lower().split()

    scores = {}
    for cls in classes:
        score = priors[cls]
        for word in test_tokens:
            score *= word_probs[cls].get(word, 0.0)
        scores[cls] = score

    prediction = int(max(scores, key=scores.get))

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict_one(x, X_ref, y_ref, k):
        distances = np.array([euclidean_distance(x, x_ref) for x_ref in X_ref])
        nn_indices = np.argsort(distances)[:k]
        nn_labels = y_ref[nn_indices]

        values, counts = np.unique(nn_labels, return_counts=True)
        return values[np.argmax(counts)]

    train_predictions = np.array([predict_one(x, X_train, y_train, k) for x in X_train])
    test_predictions = np.array([predict_one(x, X_train, y_train, k) for x in X_test])

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
