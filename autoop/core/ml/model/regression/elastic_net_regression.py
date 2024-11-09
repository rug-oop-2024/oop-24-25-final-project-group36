import numpy as np
from sklearn.linear_model import ElasticNet
from autoop.core.ml.model import Model
from pydantic import PrivateAttr

class ElasticNetRegression(Model):
    """
    Elastic Net Regression model subclass that inherits from the base Model class.

    This model uses scikit-learn's ElasticNet to perform regression.
    """

    _model: ElasticNet = PrivateAttr()
    
    def __init__(self, hyperparameters: dict = None):
        """
        Initialize the ElasticNetRegression model with optional hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters for the ElasticNet model.
        """
        super().__init__(asset_path="elastic_net_regression",
                         data=b"",
                         version="1.0.0",
                         hyperparameters=hyperparameters,
                         type="regression")
        self._model = ElasticNet(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data by fitting scikit-learn's ElasticNet model.

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

   
