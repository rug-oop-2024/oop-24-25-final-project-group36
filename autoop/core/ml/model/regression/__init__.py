"""
This package contains implementation of various regression models.
This includes: Multiple Linear Regression, Elastic Net Regression,
and Gradient Boosting Regression.

These models can be utilized for regression tasks
and are designed to integrate with the AutoML framework.
"""
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.elastic_net_regression import (
    ElasticNetRegression,
)
from autoop.core.ml.model.regression.gradient_boosting_regression import (
    GradientBoostingRegression,
)

__all__ = [
    "MultipleLinearRegression",
    "ElasticNetRegression",
    "GradientBoostingRegression",
]
