from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from autoop.core.ml.model import Model
from pydantic import PrivateAttr
import numpy as np


class ElasticNetRegression(Model):
    """
    Elastic Net Regression model subclass
    that inherits from the base Model class.
    """

    _model: ElasticNet = PrivateAttr()
    _scaler: StandardScaler = PrivateAttr()
    _grid_search: GridSearchCV = PrivateAttr(default=None)

    def __init__(self, hyperparameters: dict = None, param_grid: dict = None):
        """
        Initialize ElasticNetRegression with optional
        hyperparameters and a parameter grid for GridSearch.

        Args:
            hyperparameters (dict): Hyperparameters for ElasticNet.
            param_grid (dict): Parameter grid for GridSearchCV.
        """
        super().__init__(
            asset_path="elastic_net_regression",
            data=b"",
            version="1.0.0",
            hyperparameters=hyperparameters,
            type="regression",
        )
        self._param_grid = (
            param_grid
            if param_grid
            else {"alpha": [0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.5, 0.7, 0.9]}
        )
        self._model = ElasticNet(**self.hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using GridSearchCV for hyperparameter tuning.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
        if self._param_grid:
            grid_search = GridSearchCV(
                self._model,
                self._param_grid,
                cv=5,
                scoring="neg_mean_squared_error"
            )
            grid_search.fit(X, y)
            self._model = grid_search.best_estimator_

        else:
            self._model.fit(X, y)

        self.parameters["coef_"] = self._model.coef_
        self.parameters["intercept_"] = self._model.intercept_
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on scaled data.

        Args:
            X (np.ndarray): Features to make predictions on.

        Returns:
            np.ndarray: Model predictions.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        return self._model.predict(X)
