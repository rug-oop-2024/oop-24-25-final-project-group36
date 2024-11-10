import pickle
from typing import List

import numpy as np
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """Pipeline class for managing the training and evaluation process."""
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """
        Initialize the Pipeline.

        Args:
            metrics (List[Metric]): List of metrics to evaluate the model.
            dataset (Dataset): Dataset used for training and evaluation.
            model (Model): Machine learning model.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): Target feature.
            split (float): Train-test split ratio.

        Raises:
            ValueError: If the model type and target
            feature type are incompatible.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == \
                "categorical" and model.type != "classification":
            raise ValueError(
                "Model type must be classification for \
                    categorical target feature"
            )
        if target_feature.type == "numerical" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """String representation of the Pipeline."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Get the model used in the pipeline.

        Returns:
            Model: The machine learning model used.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieve artifacts generated during pipeline execution.

        Returns:
            List[Artifact]: List of artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: dict) -> None:
        """
        Register an artifact by name.

        Args:
            name (str): Name of the artifact.
            artifact (dict): Artifact data.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocess features and prepare input and output vectors.

        Raises:
            ValueError: If preprocessing fails for a feature.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """Split data into training and testing sets."""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenate input vectors into a single array.

        Args:
            vectors (List[np.ndarray]): List of input vectors.

        Returns:
            np.ndarray: Concatenated array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model using the training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluate the model on the test set and store predictions."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Execute the pipeline.

        Returns:
            dict: Dictionary containing training and
            testing metrics, and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        predictions_train = self._model.predict(X_train)

        train_metrics_results = []
        for metric in self._metrics:
            metric_name = metric.__class__.__name__
            result_train = metric(predictions_train, Y_train)
            train_metrics_results.append((metric_name, result_train))

        X_test = self._compact_vectors(self._test_X)
        Y_test = self._test_y
        predictions_test = self._model.predict(X_test)

        test_metrics_results = []
        for metric in self._metrics:
            metric_name = metric.__class__.__name__
            result_test = metric(predictions_test, Y_test)
            test_metrics_results.append((metric_name, result_test))

        self._predictions = predictions_test

        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
            "predictions": self._predictions,
        }
