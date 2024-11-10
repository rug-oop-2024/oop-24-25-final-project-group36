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
        """
        Returns string representation of the Feature instance.

        Generates string that includes the name and type of the feature
        in the format: "Feature(name={name}, type={type})".

        Returns:
            str: A string describing the Feature object.
        """
        return f"Feature(name={self.name}, type={self.type})"
