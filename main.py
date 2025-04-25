import numpy as np
from cls_boosting import BoostingClassifier

def main():
    # Simple binary classification dataset
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([0, 0, 1, 1, 1])

    # Initialize and train the model
    model = BoostingClassifier(n_estimators=10, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_train)

    # Print results
    print("Training Data:")
    print(X_train)
    print("Actual Labels:   ", y_train)
    print("Predicted Labels:", predictions)

    # Optional: show accuracy
    accuracy = np.mean(predictions == y_train)
    print(f"Training Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
