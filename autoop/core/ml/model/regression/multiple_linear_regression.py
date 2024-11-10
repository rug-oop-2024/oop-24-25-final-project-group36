import numpy as np
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model subclass.

    This model uses scikit-learn's LinearRegression to perform regression.
    """

    _model: LinearRegression = PrivateAttr()

    def __init__(self, hyperparameters: dict = None) -> None:
        """
        Initialize the MultipleLinearRegression model with
        required fields for the Artifact.

        Args:
            hyperparameters (dict): Hyperparameters for the model,
            passed to scikit-learn's LinearRegression.
        """
        super().__init__(
            asset_path="multiple_linear_regression",
            data=b"",
            version="1.0.0",
            hyperparameters=hyperparameters,
            type="regression",
        )

        self._model = LinearRegression(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by fitting
        scikit-learn's LinearRegression model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
        self._model.fit(X, y)
        self.parameters["coef_"] = self._model.coef_
        self.parameters["intercept_"] = self._model.intercept_
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
