import streamlit as st
import pickle
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

automl = AutoMLSystem.get_instance()

st.title("Load and Predict with Pipelines")

pipelines = automl.registry.list(type="pipeline")

cache = automl.registry.list()
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Select Pipeline")

with col2:
    if st.button("Clear all cache", key="delete_cache"):
        for items in cache:
            automl.registry.delete(items.id)

        st.warning("Cache has been cleared, refresh the page.")

if pipelines:
    selected_pipeline = st.selectbox(
        "Select a pipeline to load",
        pipelines,
        format_func=lambda x: f"{x.name} (v{x.version})",
    )

    if selected_pipeline:
        pipeline_data = pickle.loads(selected_pipeline.data)
        loaded_pipeline = pipeline_data["pipeline"]
        loaded_summary_df = pipeline_data["summary_df"]

        st.subheader(
            f"Pipeline Summary: \
            {selected_pipeline.name} (v{selected_pipeline.version})"
        )
        st.table(loaded_summary_df)

        st.subheader("Make Predictions")
        uploaded_csv = st.file_uploader(
            "Upload CSV for prediction", type=["csv"])

        if uploaded_csv:
            input_data = pd.read_csv(uploaded_csv)
            st.write("Uploaded Data Preview", input_data)

            converted_dataset = Dataset.from_dataframe(
                input_data,
                name="temp_dataset_name",
                version="1.0",
                asset_path="temp_dataset_asset_path"
            )

            features = detect_feature_types(converted_dataset)

            formatted_feature_names = [
                f"{feature.name} ({feature.type})" for feature in features
            ]
            feature_map = {
                f"{feature.name} ({feature.type})":
                feature for feature in features}

            selected_input_feature_names = st.multiselect(
                "Select input features (multiple)",
                options=formatted_feature_names
            )

            selected_target_feature_name = st.selectbox(
                "Select target feature", options=formatted_feature_names
            )

            selected_input_features = [
                feature_map[name] for name in selected_input_feature_names
            ]
            selected_target_feature = feature_map[selected_target_feature_name]

            detected_feature_type_str = selected_target_feature.type
            if st.button("Predict!"):
                try:
                    loaded_pipeline.set_dataset_and_features(
                        new_dataset=converted_dataset,
                        new_input_features=selected_input_features,
                        new_target_feature=selected_target_feature
                    )

                    if len(selected_input_features) != len(
                            loaded_pipeline._input_features):
                        raise ValueError(
                            "Make sure the number of input features \
                                matches what the pipeline was trained on."
                        )
                    if detected_feature_type_str !=  \
                            loaded_pipeline._target_feature.type:
                        raise ValueError(
                            "Make sure the target feature type matches \
                                what the pipeline was trained on."
                        )

                    results = loaded_pipeline.execute()

                    test_df = pd.DataFrame(
                        results["test_metrics"],
                        columns=["Metric", "Testing Score"]
                    )

                    st.subheader("Testing Results")
                    st.table(test_df)

                except ValueError as e:
                    st.error(str(e))


else:
    st.warning("No pipelines available. Please save a pipeline first.")
