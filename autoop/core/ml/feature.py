from typing import Literal

from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    type: Literal["categorical", "numerical"]

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"
