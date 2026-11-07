"""UML relation (edge) model."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from uml_recognition.models.bbox import BBox


class RelationType(str, Enum):
    """Supported UML relation types."""

    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    AGGREGATION = "aggregation"
    DEPENDENCY = "dependency"
    REALIZATION = "realization"


class Relation(BaseModel):
    """A directed relation between two classes identified by their names.

    Note:
        Referential integrity (source/target must exist among diagram
        classes) is enforced at the Pipeline level, not by this model.

    Attributes:
        source: Name of the source class.
        target: Name of the target class.
        type: Relation type.
        label: Optional relation label.
        bbox: Optional bounding box of the relation line/label.
        confidence: Detection confidence in [0, 1].
    """

    source: str
    target: str
    type: RelationType
    label: str | None = None
    bbox: BBox | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("source", "target")
    @classmethod
    def endpoint_not_empty(cls, value: str) -> str:
        """Reject empty class references."""
        if not value.strip():
            raise ValueError("Relation endpoint must reference a non-empty class name")
        return value
