"""Visibility enum and UML attribute model."""

from enum import Enum

from pydantic import BaseModel, field_validator


class Visibility(str, Enum):
    """UML visibility markers."""

    PUBLIC = "+"
    PRIVATE = "-"
    PROTECTED = "#"
    PACKAGE = "~"


class Attribute(BaseModel):
    """A single UML class attribute.

    Attributes:
        name: Attribute name (non-empty).
        type: Optional attribute type.
        visibility: UML visibility marker, defaults to PUBLIC.
    """

    name: str
    type: str | None = None
    visibility: Visibility = Visibility.PUBLIC

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty attribute names.

        Args:
            value: Candidate name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Attribute name must not be empty")
        return value
