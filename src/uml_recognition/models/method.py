"""UML class method (operation) model."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from uml_recognition.models.attribute import Visibility


class Parameter(BaseModel):
    """A single method parameter.

    Attributes:
        name: Parameter name (non-empty).
        type_str: Parameter type as string, optional.
    """

    name: str
    type_str: str | None = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only names."""
        if not value.strip():
            raise ValueError("Parameter name must not be empty")
        return value


class Method(BaseModel):
    """A single class method (operation).

    Attributes:
        name: Method name (non-empty).
        parameters: Ordered list of method parameters.
        return_type: Return type as string, optional.
        visibility: UML visibility modifier.
    """

    name: str
    parameters: list[Parameter] = []
    return_type: str | None = None
    visibility: Visibility = Visibility.PUBLIC

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only names."""
        if not value.strip():
            raise ValueError("Method name must not be empty")
        return value
