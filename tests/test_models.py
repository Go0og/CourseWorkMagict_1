"""Tests for UML data models: serialization, validation and pipeline stub."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
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


def make_diagram() -> UmlDiagram:
    """Build a UmlDiagram with fake but valid data.

    Returns:
        A fully populated UmlDiagram instance.
    """
    return UmlDiagram(
        classes=[
            UmlClass(
                name="User",
                attributes=[
                    Attribute(name="id", type_str="int", visibility=Visibility.PRIVATE),
                    Attribute(name="name", type_str="str", visibility=Visibility.PROTECTED),
                ],
                methods=[
                    Method(
                        name="login",
                        parameters=[Parameter(name="password", type_str="str")],
                        return_type="bool",
                        visibility=Visibility.PUBLIC,
                    )
                ],
                bbox=BBox(x_min=10, y_min=20, x_max=110, y_max=120),
                confidence=0.95,
            ),
            UmlClass(
                name="Order",
                bbox=BBox(x_min=200, y_min=50, x_max=320, y_max=160),
            ),
        ],
        relations=[
            Relation(
                source="User",
                target="Order",
                type=RelationType.ASSOCIATION,
                label="places",
                confidence=0.87,
            )
        ],
        metadata=Metadata(source_image="test.png", processing_time_ms=42),
    )


class TestSerialization:
    """JSON serialization round-trip."""

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json must not raise and must produce parseable JSON."""
        diagram = make_diagram()
        raw = diagram.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["classes"][0]["name"] == "User"
        assert parsed["relations"][0]["type"] == "association"
        assert parsed["metadata"]["source_image"] == "test.png"

    def test_roundtrip_validate(self) -> None:
        """Serialized JSON must re-validate into an equal model."""
        diagram = make_diagram()
        restored = UmlDiagram.model_validate_json(diagram.model_dump_json())
        assert restored == diagram


class TestValidationRules:
    """Schema validation rules from docs/json-schema.md."""

    def test_bbox_negative_coord_rejected(self) -> None:
        """Negative coordinates must fail validation."""
        with pytest.raises(ValidationError):
            BBox(x_min=-1, y_min=0, x_max=10, y_max=10)

    def test_bbox_ordering_enforced(self) -> None:
        """x_min >= x_max and y_min >= y_max must fail validation."""
        with pytest.raises(ValidationError):
            BBox(x_min=100, y_min=0, x_max=50, y_max=10)
        with pytest.raises(ValidationError):
            BBox(x_min=0, y_min=100, x_max=50, y_max=50)

    def test_bbox_properties(self) -> None:
        """width/height/area computed correctly."""
        box = BBox(x_min=10, y_min=20, x_max=110, y_max=70)
        assert box.width == 100
        assert box.height == 50
        assert box.area == 5000

    def test_empty_name_rejected(self) -> None:
        """Empty names are rejected on all named entities."""
        fake_box = BBox(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(ValidationError):
            UmlClass(name="   ", bbox=fake_box)
        with pytest.raises(ValidationError):
            Attribute(name="")
        with pytest.raises(ValidationError):
            Method(name=" ")

    def test_visibility_enum_values(self) -> None:
        """Visibility uses UML symbols; invalid values rejected."""
        assert Attribute(name="x", visibility="+").visibility is Visibility.PUBLIC
        with pytest.raises(ValidationError):
            Attribute(name="x", visibility="!")

    def test_relation_confidence_bounds(self) -> None:
        """Confidence must lie in [0, 1]."""
        with pytest.raises(ValidationError):
            Relation(source="A", target="B", type=RelationType.INHERITANCE, confidence=1.5)


class TestPipelineRun:
    """Pipeline end-to-end run on a real (tiny) image."""

    def test_run_returns_valid_diagram_with_stub_data(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """run() logs the path and returns a dict with fake stub data."""
        image_path = tmp_path / "img.png"
        cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
        result = Pipeline().run(str(image_path))
        captured = capsys.readouterr()
        assert str(image_path) in captured.out
        assert isinstance(result, dict)
        assert len(result["classes"]) >= 1
        assert len(result["relations"]) >= 1
        assert result["metadata"]["source_image"] == str(image_path)
        assert UmlDiagram.model_validate(result).model_dump_json()
