"""Bounding box model for detected UML elements."""

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    """Axis-aligned bounding box with validation rules.

    Attributes:
        x_min: Left edge coordinate (>= 0).
        y_min: Top edge coordinate (>= 0).
        x_max: Right edge coordinate (> 0).
        y_max: Bottom edge coordinate (> 0).
    """

    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)

    @model_validator(mode="after")
    def check_ordering(self) -> "BBox":
        """Ensure x_min < x_max and y_min < y_max.

        Returns:
            The validated BBox instance.

        Raises:
            ValueError: If coordinates are not strictly ordered.
        """
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be strictly less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be strictly less than y_max")
        return self

    @property
    def width(self) -> float:
        """Return the box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return the box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Return the box area."""
        return self.width * self.height
