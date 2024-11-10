import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier model subclass that inherits
    from the base Model class.

    This model uses scikit-learn's RandomForestClassifier for classification.
    """
    _model: RandomForestClassifier = PrivateAttr()

    def __init__(self, hyperparameters: dict = None) -> None:
        """
        Initialize the RandomForestClassifierModel with
        optional hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters for the
            RandomForestClassifier model.
        """
        super().__init__(asset_path="random_forest_classifier",
                         data=b"",
                         version="1.0.0",
                         hyperparameters=hyperparameters,
                         type="classification")
        self._model = RandomForestClassifier(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by fitting
        scikit-learn's RandomForestClassifier.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
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
