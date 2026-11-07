"""UML class, metadata and root diagram models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from uml_recognition.models.attribute import Attribute
from uml_recognition.models.bbox import BBox
from uml_recognition.models.method import Method
from uml_recognition.models.relation import Relation


class UmlClass(BaseModel):
    """A single UML class (node) detected in the diagram.

    Attributes:
        name: Class name (non-empty).
        attributes: List of class attributes.
        methods: List of class methods.
        bbox: Bounding box of the class rectangle.
        confidence: Detection confidence in [0, 1].
    """

    name: str
    attributes: list[Attribute] = []
    methods: list[Method] = []
    bbox: BBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only names."""
        if not value.strip():
            raise ValueError("Class name must not be empty")
        return value


class Metadata(BaseModel):
    """Processing metadata attached to the output diagram.

    Attributes:
        source_image: Path to the input image.
        processing_time_ms: Pipeline run duration in milliseconds.
        schema_version: Version of the JSON schema used.
    """

    source_image: str
    processing_time_ms: int = Field(ge=0)
    schema_version: str = "1.0"


class UmlDiagram(BaseModel):
    """Root model: full extracted UML class diagram.

    Attributes:
        classes: Detected UML classes.
        relations: Detected relations between classes.
        metadata: Processing metadata.
    """

    classes: list[UmlClass] = []
    relations: list[Relation] = []
    metadata: Metadata
