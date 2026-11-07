"""UML class model and root diagram models."""


from pydantic import BaseModel, Field, field_validator

from uml_recognition.models.attribute import Attribute
from uml_recognition.models.bbox import BBox
from uml_recognition.models.method import Method


class UmlClass(BaseModel):
    """A single UML class box detected on the diagram.

    Attributes:
        name: Class name (non-empty).
        bbox: Bounding box of the class on the image.
        stereotype: Optional UML stereotype (e.g. «interface»).
        attributes: List of class attributes.
        methods: List of class methods.
    """

    name: str
    bbox: BBox
    stereotype: str | None = None
    attributes: list[Attribute] = Field(default_factory=list)
    methods: list[Method] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty class names.

        Args:
            value: Candidate name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("UmlClass name must not be empty")
        return value


class Metadata(BaseModel):
    """Processing metadata attached to the exported diagram.

    Attributes:
        source_image: Path to the source image.
        processing_time_ms: Pipeline runtime in milliseconds.
        schema_version: Output JSON schema version.
    """

    source_image: str
    processing_time_ms: float
    schema_version: str = "1.0"


class UmlDiagram(BaseModel):
    """Root model of the module output JSON.

    Attributes:
        classes: Detected UML classes.
        relations: Detected relations between classes.
        metadata: Processing metadata.
    """

    classes: list[UmlClass] = Field(default_factory=list)
    relations: list["Relation"] = Field(default_factory=list)
    metadata: Metadata


from uml_recognition.models.relation import Relation

__all__ = ["Metadata", "Relation", "UmlClass", "UmlDiagram"]
