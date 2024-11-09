import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(Model):
    """
    Logistic Regression model subclass that inherits from the base Model class.

    This model uses scikit-learn's LogisticRegression for classification.
    """
    _model: LogisticRegression = PrivateAttr()

    def __init__(self, hyperparameters: dict = None):
        """
        Initialize the LogisticRegressionClassifier with
        optional hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters
            for the LogisticRegression model.
        """
        super().__init__(name="logistic_regression",
                         asset_path="logistic_regression",
                         data=b"",
                         version="1.0.0",
                         hyperparameters=hyperparameters,
                         type="classification")
        self._model = LogisticRegression(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by fitting
        scikit-learn's LogisticRegression model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        self._model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Features to make predictions on.

        Returns:
            np.ndarray: Model predictions.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        return self._model.predict(X)
