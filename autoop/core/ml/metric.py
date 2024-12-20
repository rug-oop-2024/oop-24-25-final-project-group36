from abc import ABC, abstractmethod
from typing import Union

import numpy as np

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall"]

REGRESSION_METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]


def get_metric(name: str) -> "Metric":
    """
    Gets metric by its name.

    Args:
        name (str): The name of the metric.

    Returns:
        Metric: An instance of the requested metric class.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2_score":
        return R2Score()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the metric given ground truth and predictions.

        Args:
            y_true (np.ndarray or list): Ground truth values.
            y_pred (np.ndarray or list): Predicted values.

        Returns:
            float: The calculated metric value.
        """

    def __str__(self) -> str:
        """Return the name of the metric when converted to a string."""
        return self.__class__.__name__


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric for regression tasks.
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the Mean Squared Error between ground truth and predictions.
        Args:
            y_true (Union[np.ndarray, list]): Ground truth values.
            y_pred (Union[np.ndarray, list]): Predicted values.

        Returns:
            float: The Mean Squared Error value.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """
    Accuracy metric for classification tasks.
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the Accuracy between ground truth and predictions.
        Args:
            y_true (Union[np.ndarray, list]): Ground truth values.
            y_pred (Union[np.ndarray, list]): Predicted values.

        Returns:
            float: The Accuracy value.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_true == y_pred)


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error (MAE) for regression tasks.
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the Mean Absolute Error between ground truth and predictions.

        Args:
            y_true (Union[np.ndarray, list]): Ground truth values.
            y_pred (Union[np.ndarray, list]): Predicted values.

        Returns:
            float: The Mean Absolute Error value.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    """
    R^2 (Coefficient of Determination) for regression tasks.
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the R^2 score between ground truth and predictions.

        Args:
            y_true (Union[np.ndarray, list]): Ground truth values.
            y_pred (Union[np.ndarray, list]): Predicted values.

        Returns:
            float: The R^2 score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance) \
            if total_variance > 0 else 0.0


class Precision(Metric):
    """
    Precision metric for classification tasks.
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the Precision score between ground truth and predictions.

        Args:
            y_true (Union[np.ndarray, list]): Ground truth labels.
            y_pred (Union[np.ndarray, list]): Predicted labels.

        Returns:
            float: The precision score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        precision_scores = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            predicted_positives = np.sum(y_pred == cls)

            if predicted_positives == 0:
                precision_scores.append(0.0)
            else:
                precision_scores.append(true_positives / predicted_positives)

        return np.mean(precision_scores)


class Recall(Metric):
    """
    Recall metric for classification tasks..
    """

    def __call__(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate the Recall score between ground truth and predictions.

        Args:
            y_true (Union[np.ndarray, list]): Ground truth labels.
            y_pred (Union[np.ndarray, list]): Predicted labels.

        Returns:
            float: The recall score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        recall_scores = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            actual_positives = np.sum(y_true == cls)

            if actual_positives == 0:
                recall_scores.append(0.0)
            else:
                recall_scores.append(true_positives / actual_positives)

        return np.mean(recall_scores)
