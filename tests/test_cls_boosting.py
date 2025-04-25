import numpy as np
import pytest
from cls_boosting import BoostingClassifier

# === Shared Stats Function ===
def print_stats(y_true, y_pred, test_name):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"\n==== {test_name} ====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


def test_basic_fit_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    model = BoostingClassifier(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Basic Fit Predict")
    assert np.mean(preds == y) >= 0.6


def test_invalid_labels():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])  # invalid label
    model = BoostingClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_overfitting_behavior():
    X = np.array([[i] for i in range(10)])
    y = np.array([1 if i > 5 else 0 for i in range(10)])
    model = BoostingClassifier(n_estimators=200)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Overfitting Behavior")
    assert np.mean(preds == y) > 0.95


def test_random_data_accuracy():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = BoostingClassifier(n_estimators=30)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Random Data Accuracy")
    assert np.mean(preds == y) > 0.8


def test_non_linear_boundary():
    np.random.seed(0)
    X = np.random.rand(500, 5)
    y = (np.sin(3 * X[:, 0]) + np.cos(5 * X[:, 1]) > 1).astype(int)
    model = BoostingClassifier(n_estimators=100)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Non-Linear Feature Combination")
    assert np.mean(preds == y) > 0.75


def test_high_dimensional_sparse_signal():
    np.random.seed(42)
    X = np.random.randn(300, 50)
    y = (X[:, 7] > 0).astype(int)  # only feature 7 matters
    model = BoostingClassifier(n_estimators=100)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "High-Dimensional Sparse Signal")
    assert np.mean(preds == y) > 0.7


def test_noisy_labels():
    np.random.seed(0)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    flip_idx = np.random.choice(len(y), size=40, replace=False)
    y[flip_idx] = 1 - y[flip_idx]  # flip 20%
    model = BoostingClassifier(n_estimators=100)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Noisy Labels Test")
    assert 0.65 <= np.mean(preds == y) <= 0.85


def test_tree_quantity_sensitivity():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    model_few = BoostingClassifier(n_estimators=3)
    model_many = BoostingClassifier(n_estimators=100)
    model_few.fit(X, y)
    model_many.fit(X, y)
    pred_few = model_few.predict(X)
    pred_many = model_many.predict(X)
    print_stats(y, pred_few, "Few Trees")
    print_stats(y, pred_many, "Many Trees")
    assert np.mean(pred_many == y) > np.mean(pred_few == y)


def test_skewed_feature_scale():
    np.random.seed(1)
    X = np.random.randn(200, 2)
    X[:, 1] *= 1000  # skew scale
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = BoostingClassifier(n_estimators=50)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Skewed Feature Scale")
    assert np.mean(preds == y) > 0.7


def test_prediction_stability():
    X = np.random.randn(50, 2)
    y = np.random.randint(0, 2, size=50)
    model = BoostingClassifier(n_estimators=15, learning_rate=0.2)
    model.fit(X, y)
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    print_stats(y, preds1, "Prediction Stability")
    assert np.array_equal(preds1, preds2)
