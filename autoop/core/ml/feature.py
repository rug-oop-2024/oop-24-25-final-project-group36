from typing import Literal

from pydantic import BaseModel


class Feature(BaseModel):
    """
    A model representing a feature with a name and type.

    Attributes:
        name (str): The name of the feature.
        type (Literal["categorical", "numerical"]):
            The type of the feature; either "categorical" or "numerical".
    """
    name: str
    type: Literal["categorical", "numerical"]

    def __str__(self) -> str:
        return f"Feature(name={self.name}, type={self.type})"
