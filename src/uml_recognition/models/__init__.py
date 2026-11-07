"""Pydantic v2 data models for the UML recognition module.

Mirrors the output JSON schema defined in docs/json-schema.md.
"""

from uml_recognition.models.attribute import Attribute, Visibility
from uml_recognition.models.bbox import BBox
from uml_recognition.models.method import Method, Parameter
from uml_recognition.models.relation import Relation, RelationType
from uml_recognition.models.uml_class import Metadata, UmlClass, UmlDiagram

__all__ = [
    "Attribute",
    "BBox",
    "Metadata",
    "Method",
    "Parameter",
    "Relation",
    "RelationType",
    "UmlClass",
    "UmlDiagram",
    "Visibility",
]
