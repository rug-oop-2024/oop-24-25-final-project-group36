import streamlit as st
import pandas as pd
import pickle
import hashlib
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric, \
    CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model, \
    CLASSIFICATION_MODELS, REGRESSION_MODELS

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a \
    machine learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


# hina code below
def artifact_to_dataset(artifact: Artifact) -> Dataset:
    if artifact.type == "dataset":
        return Dataset(
            name=artifact.name,
            version=artifact.version,
            asset_path=artifact.asset_path,
            tags=artifact.tags,
            metadata=artifact.metadata,
            data=artifact.data,
        )


artifact_datasets = [artifact_to_dataset(artifact) for artifact in datasets]
selected_dataset = st.selectbox(
    "Select a dataset for modeling",
    artifact_datasets, format_func=lambda x: x.name
)

if selected_dataset:
    st.write(f"Your selected dataset: {selected_dataset.name}")

    dataset_df = selected_dataset.read()

    st.write("Dataset Preview", dataset_df)

    features = detect_feature_types(selected_dataset)

    formatted_feature_names = [
        f"{feature.name} ({feature.type})" for feature in features
    ]
    feature_map = {
        f"{feature.name} ({feature.type})": feature for feature in features}

    selected_input_feature_names = st.multiselect(
        "Select input features (multiple)", options=formatted_feature_names
    )

    selected_target_feature_name = st.selectbox(
        "Select target feature", options=formatted_feature_names
    )

    selected_input_features = [
        feature_map[name] for name in selected_input_feature_names
    ]
    selected_target_feature = feature_map[selected_target_feature_name]

    detected_feature_type_str = selected_target_feature.type

    st.write("This is the detected task type:", detected_feature_type_str)
    if detected_feature_type_str == "categorical":
        model_name = CLASSIFICATION_MODELS
        metric_name = CLASSIFICATION_METRICS
    elif detected_feature_type_str == "numerical":
        model_name = REGRESSION_MODELS
        metric_name = REGRESSION_METRICS

    selected_model_name = st.selectbox("Select your model!", model_name)
    selected_model = get_model(selected_model_name)

    split_ratio = st.slider("Select Train-Test Split Ratio", 0.1, 0.9, 0.8)

    selected_metric_name = st.multiselect("Select your metrics!", metric_name)
    selected_metrics = []
    for metric in selected_metric_name:
        selected_metrics.append(get_metric(metric))

    summary_data = {
        "Description": [
            "Dataset",
            "Input Features",
            "Target Feature",
            "Model",
            "Train-Test Split",
            "Metrics",
        ],
        "Details": [
            selected_dataset.name,
            ", ".join([feature.name for feature in selected_input_features]),
            selected_target_feature.name,
            selected_model_name,
            f"{split_ratio * 100:.0f}% Train / \
                {100 - split_ratio * 100:.0f}% Test",
            ", ".join(selected_metric_name),
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

st.title("Train the Model")
if st.button("Train!"):
    pipeline = Pipeline(
        dataset=selected_dataset,
        input_features=selected_input_features,
        target_feature=selected_target_feature,
        model=selected_model,
        split=split_ratio,
        metrics=selected_metrics,
    )

    results = pipeline.execute()
    train_df = pd.DataFrame(
        results["train_metrics"], columns=["Metric", "Training Score"]
    )
    test_df = pd.DataFrame(results["test_metrics"],
                           columns=["Metric", "Testing Score"])

    st.subheader("Training Results")
    st.table(train_df)
    st.subheader("Testing Results")
    st.table(test_df)

    st.session_state.training_complete = True
    st.session_state.pipeline = pipeline

if st.session_state.training_complete:
    pipeline_name = st.text_input("Enter Pipeline Name", value="NewPipeline")
    pipeline_version = st.text_input("Enter Pipeline Version", value="1.0")
    if st.button("Save Pipeline"):
        pipeline_data = pickle.dumps(
            {"pipeline": st.session_state.pipeline, "summary_df": summary_df}
        )

        pipeline_hash = hashlib.md5(pipeline_data).hexdigest()[:8]

        asset_path = f"pipelines/{pipeline_name}_{pipeline_hash}.pkl"

        pipeline_artifact = Artifact(
            name=pipeline_name,
            version=pipeline_version,
            asset_path=asset_path,
            data=pipeline_data,
            type="pipeline",
        )

        registered_data = automl.registry.register(pipeline_artifact)
        st.success(
            f"Pipeline '{pipeline_name}' version {pipeline_version} \
                saved successfully."
        )
