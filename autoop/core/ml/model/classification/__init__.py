"""
This package contains implementation of various classification models.
This includes: Logistic Regression, Random Forest Classifier,
and Support Vector Classifier.

These models can be utilized for classification tasks
and are designed to integrate with the AutoML framework.
"""

from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressionClassifier,
)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForestClassifierModel,
)
from autoop.core.ml.model.classification.svc_classifier import (
    SupportVectorClassifier,
)

__all__ = [
    "LogisticRegressionClassifier",
    "RandomForestClassifierModel",
    "SupportVectorClassifier",
]
