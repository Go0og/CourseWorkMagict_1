"""UML class attribute model."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator


class Visibility(str, Enum):
    """UML visibility modifiers."""

    PUBLIC = "+"
    PRIVATE = "-"
    PROTECTED = "#"
    PACKAGE = "~"


class Attribute(BaseModel):
    """A single class attribute (field).

    Attributes:
        name: Attribute name (non-empty).
        type_str: Declared type as string (OCR output), optional.
        visibility: UML visibility modifier.
    """

    name: str
    type_str: str | None = None
    visibility: Visibility = Visibility.PUBLIC

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only names."""
        if not value.strip():
            raise ValueError("Attribute name must not be empty")
        return value
