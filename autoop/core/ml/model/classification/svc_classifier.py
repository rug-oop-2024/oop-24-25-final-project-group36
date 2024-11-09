import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.svm import SVC


class SupportVectorClassifier(Model):
    """
    Support Vector Classifier (SVC) subclass that inherits
    from the base Model class.

    This model uses scikit-learn's SVC for classification.
    """
    _model: SVC = PrivateAttr()

    def __init__(self, hyperparameters: dict = None):
        """
        Initialize the SupportVectorClassifier with optional hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters for the SVC model.
        """
        super().__init__(asset_path="svc_classifier",
                         data=b"",
                         version="1.0.0",
                         hyperparameters=hyperparameters,
                         type="classification")
        self._model = SVC(probability=True, **self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by
        fitting scikit-learn's SVC model.

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
