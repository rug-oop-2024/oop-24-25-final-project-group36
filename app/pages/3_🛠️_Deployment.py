import streamlit as st
import pickle
import pandas as pd
from app.core.system import AutoMLSystem

automl = AutoMLSystem.get_instance()

st.title("Load and Predict with Pipelines")

pipelines = automl.registry.list(type="pipeline")

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
            st.write("Uploaded Data Preview", input_data.head())

            feature_names = list(input_data.columns)
            selected_input_features = st.multiselect(
                "Select input features", feature_names
            )
            selected_target_feature = st.selectbox(
                "Select target feature", feature_names
            )


else:
    st.warning("No pipelines available. Please save a pipeline first.")
