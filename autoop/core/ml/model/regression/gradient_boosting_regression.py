import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingRegression(Model):
    """
    Gradient Boosting Regression model subclass
    that inherits from the base Model class.

    This model uses scikit-learn's GradientBoostingRegressor
    to perform regression.
    """

    _model: GradientBoostingRegressor = PrivateAttr()

    def __init__(self, hyperparameters: dict = None):
        """
        Initialize the GradientBoostingRegression
        model with optional hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters
            for the GradientBoostingRegressor model.
        """
        super().__init__(
            asset_path="gradient_boosting_regression",
            data=b"",
            version="1.0.0",
            hyperparameters=hyperparameters,
            type="regression",
        )
        self._model = GradientBoostingRegressor(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by fitting
        scikit-learn's GradientBoostingRegressor.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
        y = np.ravel(y)
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
