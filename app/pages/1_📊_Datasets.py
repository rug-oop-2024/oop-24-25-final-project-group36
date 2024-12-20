import os
import pandas as pd
import hashlib
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import streamlit as st
from typing import Any
automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
st.title("Dataset")


def generate_file_hash(file: Any) -> str:
    """
    Generate an MD5 hash for a file-like object.

    Args:
        file:
            A file-like object opened in binary mode.
            The file pointer will be reset to the beginning after hashing.

    Returns:
        str: The MD5 hash of the file's contents as a hexadecimal string.
    """
    hasher = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b""):
        hasher.update(chunk)
    file.seek(0)
    return hasher.hexdigest()


existing_asset_paths = {dataset.asset_path for dataset in datasets}

uploaded_csv_file = st.file_uploader(
    "Upload your CSV file here:)", type=["csv"])
if uploaded_csv_file:
    file_name = os.path.splitext(uploaded_csv_file.name)[0]
    file_hash = generate_file_hash(uploaded_csv_file)

    asset_path = f"datasets/{file_name}_{file_hash}.csv"
    dataframe = pd.read_csv(uploaded_csv_file)
    st.write("Dataset Preview", dataframe)

    if asset_path not in existing_asset_paths:
        converted_dataset = Dataset.from_dataframe(
            dataframe, name=file_name, version="1.0", asset_path=asset_path
        )

        automl.registry.register(converted_dataset)
        st.success(f"'{converted_dataset.name}' \
                   successfully uploaded and registered!")
        st.rerun()

cache = automl.registry.list()
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Registered Datasets")

with col2:
    if st.button("Clear all cache", key="delete_cache"):
        for items in cache:
            automl.registry.delete(items.id)

        st.warning("Cache has been cleared, refresh the page.")

for dataset in datasets:
    with st.container(border=True):
        st.write(f"{dataset.name}")
