"""Tests for Pydantic v2 data models (docs/json-schema.md)."""

import json

import pytest
from pydantic import ValidationError

from uml_recognition.models import (
    Attribute,
    BBox,
    Metadata,
    Method,
    Parameter,
    Relation,
    RelationType,
    UmlClass,
    UmlDiagram,
    Visibility,
)
from uml_recognition.pipeline import Pipeline


def make_fake_diagram() -> UmlDiagram:
    """Build a UmlDiagram with fake sample data.

    Returns:
        A populated, valid UmlDiagram instance.
    """
    return UmlDiagram(
        classes=[
            UmlClass(
                name="User",
                bbox=BBox(x_min=100.0, y_min=50.0, x_max=300.0, y_max=250.0),
                stereotype=None,
                attributes=[Attribute(name="id", type="int", visibility=Visibility.PRIVATE)],
                methods=[
                    Method(
                        name="login",
                        return_type="bool",
                        parameters=[Parameter(name="password", type="str")],
                        visibility=Visibility.PUBLIC,
                    )
                ],
            )
        ],
        relations=[
            Relation(
                source_class_id="User",
                target_class_id="Order",
                relation_type=RelationType.ASSOCIATION,
                multiplicity_source="1",
                multiplicity_target="*",
            )
        ],
        metadata=Metadata(source_image="data/raw/example.png", processing_time_ms=1523.45),
    )


class TestSerialization:
    """model_dump_json round-trip tests."""

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json must not raise and must yield parseable JSON."""
        diagram = make_fake_diagram()
        raw = diagram.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["classes"][0]["name"] == "User"
        assert parsed["relations"][0]["relation_type"] == "association"
        assert parsed["metadata"]["schema_version"] == "1.0"

    def test_roundtrip_validate(self) -> None:
        """model_validate_json must reconstruct an equal model."""
        diagram = make_fake_diagram()
        restored = UmlDiagram.model_validate_json(diagram.model_dump_json())
        assert restored == diagram


class TestValidationRules:
    """Schema validation rules from docs/json-schema.md section 3."""

    def test_bbox_negative_coord_rejected(self) -> None:
        """BBox coordinates must be >= 0."""
        with pytest.raises(ValidationError):
            BBox(x_min=-1.0, y_min=0.0, x_max=10.0, y_max=10.0)

    def test_bbox_ordering_enforced(self) -> None:
        """x_min < x_max and y_min < y_max are strictly required."""
        with pytest.raises(ValidationError):
            BBox(x_min=10.0, y_min=0.0, x_max=5.0, y_max=10.0)
        with pytest.raises(ValidationError):
            BBox(x_min=0.0, y_min=10.0, x_max=5.0, y_max=5.0)

    def test_bbox_properties(self) -> None:
        """width/height/area properties compute correctly."""
        box = BBox(x_min=100.0, y_min=50.0, x_max=300.0, y_max=250.0)
        assert box.width == 200.0
        assert box.height == 200.0
        assert box.area == 40000.0

    def test_empty_name_rejected(self) -> None:
        """Empty names are invalid for UmlClass/Attribute/Method."""
        box = BBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        with pytest.raises(ValidationError):
            UmlClass(name="", bbox=box)
        with pytest.raises(ValidationError):
            Attribute(name="")
        with pytest.raises(ValidationError):
            Method(name="")

    def test_visibility_enum_values(self) -> None:
        """Visibility serializes as the UML symbols +/-/#/~."""
        assert [v.value for v in Visibility] == ["+", "-", "#", "~"]


class TestPipelineStub:
    """Iteration-1 Pipeline stub behaviour."""

    def test_run_returns_empty_valid_diagram(self, capsys: pytest.CaptureFixture[str]) -> None:
        """run() logs the path and returns a valid empty UmlDiagram."""
        diagram = Pipeline().run("data/raw/example.png")
        captured = capsys.readouterr()
        assert "data/raw/example.png" in captured.out
        assert diagram.classes == []
        assert diagram.relations == []
        assert diagram.metadata.source_image == "data/raw/example.png"
        assert diagram.metadata.processing_time_ms >= 0.0
        # Must serialize without errors.
        json.loads(diagram.model_dump_json())
