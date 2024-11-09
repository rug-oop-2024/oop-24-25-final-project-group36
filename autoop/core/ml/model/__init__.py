from autoop.core.ml.model.classification import (
    LogisticRegressionClassifier,
    RandomForestClassifierModel,
    SupportVectorClassifier,
)
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    ElasticNetRegression,
    GradientBoostingRegression,
    MultipleLinearRegression,
)

# List of available regression models
REGRESSION_MODELS = [
    "multiple_linear_regression",
    "elastic_net_regression",
    "gradient_boosting_regression",
]

# List of available classification models
CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest_classifier",
    "svc_classifier",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "multiple_linear_regression":
        return MultipleLinearRegression()
    elif model_name == "elastic_net_regression":
        return ElasticNetRegression()
    elif model_name == "gradient_boosting_regression":
        return GradientBoostingRegression()
    elif model_name == "logistic_regression":
        return LogisticRegressionClassifier()
    elif model_name == "random_forest_classifier":
        return RandomForestClassifierModel()
    elif model_name == "svc_classifier":
        return SupportVectorClassifier()
    else:
        raise NotImplementedError("To be implemented.")
