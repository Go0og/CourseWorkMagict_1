"""Relation model and relation type enum."""

from enum import Enum

from pydantic import BaseModel


class RelationType(str, Enum):
    """Supported UML relation types."""

    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    DEPENDENCY = "dependency"
    AGGREGATION = "aggregation"
    COMPOSITION = "composition"


class Relation(BaseModel):
    """A directed relation between two UML classes.

    Attributes:
        source_class_id: Name of the source class.
        target_class_id: Name of the target class.
        relation_type: Type of the UML relation.
        multiplicity_source: Optional multiplicity at the source end.
        multiplicity_target: Optional multiplicity at the target end.
    """

    source_class_id: str
    target_class_id: str
    relation_type: RelationType
    multiplicity_source: str | None = None
    multiplicity_target: str | None = None
