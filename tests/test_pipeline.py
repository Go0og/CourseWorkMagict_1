"""End-to-end tests for the v0.1 pipeline (iteration 3).

Real PaddleOCR is mocked at the module boundary: tests exercise the full
pipeline flow (loader -> detector -> parser -> resolver) without network
or model downloads.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# --- Mock paddleocr BEFORE importing uml_recognition modules ---------------
_MOCK_OCR_LINES = ["User", "- id: int", "+ login(): bool"]


class _FakePaddleOCR:
    """Stand-in for paddleocr.PaddleOCR returning canned lines."""

    def __init__(self, *args, **kwargs) -> None:
        """Accept and ignore any constructor kwargs."""

    def ocr(self, img, cls=True):  # noqa: ANN001, ANN201 - mimic PaddleOCR API
        """Return PaddleOCR-shaped results with fixed text lines."""
        entries = [([[[0, 0], [10, 0], [10, 10], [0, 10]], (line, 0.99)] for line in _MOCK_OCR_LINES)]
        return [[entries]]


_fake_module = types.ModuleType("paddleocr")
_fake_module.PaddleOCR = _FakePaddleOCR
sys.modules.setdefault("paddleocr", _fake_module)

from uml_recognition.pipeline import DEFAULT_IMAGE_PATH, Pipeline  # noqa: E402
from uml_recognition.services.image_loader import ImageLoader  # noqa: E402


@pytest.fixture()
def test_image(tmp_path: Path) -> str:
    """Draw a 200x200 image with one big white rectangle + text; save to PNG.

    Returns:
        Path string of the generated test image.
    """
    image_path = tmp_path / "uml.png"
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.putText(canvas, "User", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    assert cv2.imwrite(str(image_path), canvas)
    return str(image_path)


class TestImageLoader:
    """Tests for the ImageLoader stage."""

    def test_load_returns_array_and_meta(self, test_image: str) -> None:
        """Loaded image is an ndarray with correct size metadata."""
        image, meta = ImageLoader().load(test_image)
        assert isinstance(image, np.ndarray)
        assert image.shape == (200, 200, 3)
        assert meta == {"original_width": 200, "original_height": 200}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ImageLoader().load(str(tmp_path / "nope.png"))

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """Zero-byte file raises ValueError."""
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            ImageLoader().load(str(empty))


class TestPipelineRun:
    """End-to-end Pipeline.run tests on a drawn UML-like image."""

    def test_run_returns_dict_with_expected_keys(self, test_image: str) -> None:
        """Pipeline returns a JSON-ready dict with classes/relations/metadata."""
        result = Pipeline().run(test_image)
        assert isinstance(result, dict)
        assert {"classes", "relations", "metadata"} <= set(result)
        assert result["metadata"]["processing_time_ms"] > 0
        assert result["metadata"]["source_image"] == test_image

    def test_detects_class_with_non_empty_name(self, test_image: str) -> None:
        """At least one class is detected and OCR-parsed with a name."""
        result = Pipeline().run(test_image)
        assert len(result["classes"]) >= 1
        names = [cls["name"] for cls in result["classes"]]
        assert all(name for name in names)
        assert "User" in names

    def test_parsed_attributes_and_methods(self, test_image: str) -> None:
        """Mocked OCR lines are parsed into attribute and method models."""
        result = Pipeline().run(test_image)
        user = next(cls for cls in result["classes"] if cls["name"] == "User")
        assert any(attr["name"] == "id" and attr["visibility"] == "-" for attr in user["attributes"])
        assert any(meth["name"] == "login" and meth["visibility"] == "+" for meth in user["methods"])

    def test_relations_stub_is_empty(self, test_image: str) -> None:
        """v0.1 RelationResolver returns no relations."""
        result = Pipeline().run(test_image)
        assert result["relations"] == []

    def test_bboxes_within_image_bounds(self, test_image: str) -> None:
        """Detected boxes fit inside the loaded image dimensions."""
        result = Pipeline().run(test_image)
        for cls in result["classes"]:
            bbox = cls["bbox"]
            assert 0 <= bbox["x_min"] < bbox["x_max"] <= 200
            assert 0 <= bbox["y_min"] < bbox["y_max"] <= 200

    def test_default_image_path_constant(self) -> None:
        """v0.1 default input path matches the spec."""
        assert DEFAULT_IMAGE_PATH == "data/raw/test.png"
