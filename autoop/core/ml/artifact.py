import base64
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """
    A base class of an artifact, which stores information
    about a file or data asset.

    Attributes:
    - name: Name of the artifact.
    - version: Version of the artifact.
    - asset_path: Path to the artifact asset.
    - data: Raw data as bytes.
    - type: Type of the artifact (e.g., "artifact").
    - tags: List of tags associated with the artifact.
    - metadata: Dictionary of additional metadata.
    """

    name: str
    version: str
    asset_path: str
    data: bytes
    type: str
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)

    def read(self) -> Union[bytes, pd.DataFrame]:
        """
        Reads the artifact data. Returns bytes for non-dataset
        types or DataFrame for dataset types.
        """
        return self.data

    def save(self, data: bytes) -> None:
        """
        Saves the artifact data by updating the data attribute with new bytes.
        """
        self.data = data

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the artifact based
        on the asset path and version.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"
