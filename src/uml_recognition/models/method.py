"""UML method and parameter models."""


from pydantic import BaseModel, Field, field_validator

from uml_recognition.models.attribute import Visibility


class Parameter(BaseModel):
    """A single method parameter.

    Attributes:
        name: Parameter name.
        type: Optional parameter type.
    """

    name: str
    type: str | None = None


class Method(BaseModel):
    """A single UML class method.

    Attributes:
        name: Method name (non-empty).
        return_type: Optional returned type.
        parameters: Ordered list of method parameters.
        visibility: UML visibility marker, defaults to PUBLIC.
    """

    name: str
    return_type: str | None = None
    parameters: list[Parameter] = Field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty method names.

        Args:
            value: Candidate name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Method name must not be empty")
        return value
