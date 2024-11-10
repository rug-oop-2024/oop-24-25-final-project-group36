import io

import pandas as pd
from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """
    A specialized Artifact that represents a dataset.
    Provides methods for converting data between
    pandas DataFrames and byte-encoded CSV format for storage.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize Dataset artifact with the specified arguments,
        enforcing the artifact type as "dataset".

        Args:
            *args:
                Positional arguments for initializing the base Artifact class.
            **kwargs:
                Keyword arguments for initializing the base Artifact class.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> "Dataset":
        """
        Create Dataset instance from pandas DataFrame.

        Args:
            data (pd.DataFrame): The data to store in the dataset.
            name (str): The name of the dataset.
            asset_path (str): The path where the dataset asset will be stored.
            version (str, optional):
                The version of the dataset. The defualt is "1.0.0".

        Returns:
            Dataset: A new Dataset instance containing the provided DataFrame.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Read and decode the dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset loaded into a DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Converts pandas DataFrame to byte-encoded CSV
        and save it using Artifact's save method.

        Args:
            data (pd.DataFrame): The DataFrame to save.

        Returns:
            bytes: The byte-encoded CSV data.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
