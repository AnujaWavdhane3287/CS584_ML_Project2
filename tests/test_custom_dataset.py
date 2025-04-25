import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from cls_boosting import BoostingClassifier

# === Utility Function for Evaluation ===
def print_stats(y_true, y_pred, title):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"\n===== {title} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# ===== Test 1: Spiral Dataset =====
def generate_spiral_data(n_samples=150, noise=0.2):
    np.random.seed(42)
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi / 360)

    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X1 = np.hstack((d1x, d1y))
    y1 = np.zeros(n_samples)

    d2x = np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d2y = -np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X2 = np.hstack((d2x, d2y))
    y2 = np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2)).astype(int)
    return X, y

def test_spiral_dataset():
    X, y = generate_spiral_data()
    model = BoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Spiral Dataset")

# ===== Test 2: XOR Pattern =====
def test_xor_pattern():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])
    model = BoostingClassifier(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "XOR Pattern")

# ===== Test 3: Concentric Circles =====
def generate_circles(n=200, noise=0.1, factor=0.4):
    t = 2 * np.pi * np.random.rand(n)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1) + noise * np.random.randn(n, 2)
    inner = factor * np.stack([np.cos(t), np.sin(t)], axis=1) + noise * np.random.randn(n, 2)
    X = np.vstack([outer, inner])
    y = np.array([0]*n + [1]*n)
    return X, y

def test_circles():
    X, y = generate_circles()
    model = BoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Concentric Circles")

# ===== Test 4: Gaussian Blobs =====
def generate_blobs(n=150, std=2.0):
    mean0 = np.array([-2, -2])
    mean1 = np.array([2, 2])
    cov = np.eye(2) * std
    X0 = np.random.multivariate_normal(mean0, cov, n)
    X1 = np.random.multivariate_normal(mean1, cov, n)
    X = np.vstack((X0, X1))
    y = np.array([0]*n + [1]*n)
    return X, y

def test_gaussian_blobs():
    X, y = generate_blobs()
    model = BoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Gaussian Blobs")

# ===== Test 5: Noisy Linear Data =====
def test_noisy_linear():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] + 0.5 * np.random.randn(200) > 0).astype(int)
    model = BoostingClassifier(n_estimators=30, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Noisy Linear Data")

# ===== Test 6: Imbalanced Classes =====
def test_imbalanced_data():
    X_majority = np.random.randn(190, 2)
    y_majority = np.zeros(190)
    X_minority = np.random.randn(10, 2) + 2
    y_minority = np.ones(10)

    X = np.vstack((X_majority, X_minority))
    y = np.hstack((y_majority, y_minority))

    model = BoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    print_stats(y, preds, "Imbalanced Dataset")

# ===== Run All Tests =====
if __name__ == "__main__":
    test_spiral_dataset()
    test_xor_pattern()
    test_circles()
    test_gaussian_blobs()
    test_noisy_linear()
    test_imbalanced_data()
