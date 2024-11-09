
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects feature types in the provided DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        List[Feature]: A list of Feature objects with detected types.
    """
    data = dataset.read()
    features = []
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'
        feature = Feature(name=column, type=feature_type)
        features.append(feature)
    return features
