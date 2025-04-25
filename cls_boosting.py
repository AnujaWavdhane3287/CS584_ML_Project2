import numpy as np

class DecisionStump:
    """
    A decision stump that splits data based on a single feature threshold.
    It predicts a constant value on each side of the threshold.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_error = float('inf')
        found_valid_split = False

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_output = residuals[left_mask].mean()
                right_output = residuals[right_mask].mean()

                predictions = np.where(left_mask, left_output, right_output)
                error = np.mean((residuals - predictions) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_output
                    self.right_value = right_output
                    found_valid_split = True

        if not found_valid_split:
            # Fallback when no valid split is found
            self.feature_index = 0
            self.threshold = X[:, 0].max() + 1  # Forces everything to left
            self.left_value = residuals.mean()
            self.right_value = residuals.mean()



    def predict(self, X):
        return np.where(X[:, self.feature_index] <= self.threshold,
                        self.left_value, self.right_value)


class BoostingClassifier:
    """
    Gradient Boosting Classifier for binary classification with decision stumps as weak learners.
    """
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []
        self.F0 = 0.0

    def _log_odds(self, y):
        pos = np.clip(np.sum(y == 1), 1, None)
        neg = np.clip(np.sum(y == 0), 1, None)
        return np.log(pos / neg)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Enforce binary labels (0 and 1 only)
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")

        y_trans = np.where(y == 1, 1, -1)
        self.F0 = self._log_odds(y)
        Fm = np.full(X.shape[0], self.F0)


        for _ in range(self.n_estimators):
            # Step 1: Compute pseudo-residuals (gradient of exponential loss)
            residuals = y_trans / (1 + np.exp(y_trans * Fm))

            # Step 2: Fit a weak learner (decision stump)
            stump = DecisionStump()
            stump.fit(X, residuals)

            # Step 3: Predict on current stump
            pred = stump.predict(X)

            # Step 4: Line search to find optimal gamma
            numer = np.dot(residuals, pred)
            denom = np.dot(pred, pred)
            gamma = numer / denom if denom != 0 else 0.0


            # Step 5: Update model
            Fm += self.learning_rate * gamma * pred

            # Step 6: Save model and its weight
            self.models.append(stump)
            self.model_weights.append(gamma)

    def predict_scores(self, X):
        X = np.array(X)
        Fm = np.full(X.shape[0], self.F0)
        for model, gamma in zip(self.models, self.model_weights):
            Fm += self.learning_rate * gamma * model.predict(X)
        return Fm

    def predict(self, X):
        scores = self.predict_scores(X)
        return (scores > 0).astype(int)
