"""Bounding box model for UML diagram elements."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    """Axis-aligned bounding box with integer pixel coordinates.

    Attributes:
        x_min: Left edge coordinate (>= 0).
        y_min: Top edge coordinate (>= 0).
        x_max: Right edge coordinate (strictly > x_min).
        y_max: Bottom edge coordinate (strictly > y_min).
    """

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(gt=0)
    y_max: int = Field(gt=0)

    @model_validator(mode="after")
    def check_ordering(self) -> BBox:
        """Ensure min coordinates are strictly less than max coordinates."""
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be < x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be < y_max")
        return self

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """Box area in square pixels."""
        return self.width * self.height
