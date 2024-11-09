from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from autoop.core.ml.artifact import Artifact
from pydantic import Field


class Model(Artifact, ABC):
    """
    Base class for all models.

    Attributes:
        parameters (dict): Model parameters that are learned during training.
        hyperparameters (dict): Hyperparameters for model configuration.
        trained (bool): Indicates whether the model has been trained.
    """

    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    trained: bool = Field(default=False)

    def __init__(
        self,
        asset_path: str,
        name: Optional[str] = "Unnamed Model",
        type: Optional[str] = "model",
        data: bytes = b"",
        version: str = "1.0.0",
        hyperparameters: dict = None,
    ):
        """
        Initialize the model with artifact properties and hyperparameters.

        Args:
            asset_path (str): The path or identifier of the model artifact.
            data (bytes): The binary data representing the model (if any).
            version (str): Version of the model artifact.
            hyperparameters (dict): Hyperparameters for model configuration.
        """
        super().__init__(
            asset_path=asset_path,
            data=data,
            version=version,
            name=name,
            type=type
        )
        self.parameters = {}
        self.hyperparameters = (
            hyperparameters if hyperparameters is not None else {}
        )
        self.trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels/targets.
        """
        raise NotImplementedError(
            "Subclasses must implement the train method."
            )

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Features to make predictions on.

        Returns:
            np.ndarray: Model predictions.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method."
            )

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return self.trained
