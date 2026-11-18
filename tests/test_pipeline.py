"""End-to-end tests for the pipeline (iteration 4: pytesseract OCR + relations)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from uml_recognition.pipeline import DEFAULT_IMAGE_PATH, Pipeline
from uml_recognition.services.image_loader import ImageLoader


@pytest.fixture(scope="module")
def uml_test_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a realistic 300x300 UML-like PNG and return its path.

    Draws a white rectangle with black border at (50, 50)-(250, 150)
    containing the class name "User" and attribute "+id: int".
    """
    image_path = tmp_path_factory.mktemp("uml") / "user_class.png"
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 150), (0, 0, 0), 2)
    cv2.fillPoly(image, [np.array([[50, 50], [250, 50], [250, 150], [50, 150]])], (255, 255, 255))
    cv2.rectangle(image, (50, 50), (250, 150), (0, 0, 0), 2)
    cv2.putText(image, "User", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "+id: int", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    assert cv2.imwrite(str(image_path), image)
    return str(image_path)


@pytest.fixture(scope="module")
def pipeline_result(uml_test_image: str) -> dict:
    """Run the full pipeline once on the synthetic UML image."""
    return Pipeline().run(uml_test_image)


@pytest.fixture(scope="module")
def two_class_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a 700x500 diagram with two class boxes and a connector line."""
    image_path = tmp_path_factory.mktemp("uml") / "two_classes.png"
    image = np.full((500, 700, 3), 255, dtype=np.uint8)
    for (x1, y1, x2, y2), name in [
        ((50, 60, 250, 180), "User"),
        ((450, 260, 650, 380), "Order"),
    ]:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(image, name, (x1 + 20, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # Connector line between the boxes.
    cv2.line(image, (250, 120), (450, 320), (0, 0, 0), 2)
    assert cv2.imwrite(str(image_path), image)
    return str(image_path)


class TestImageLoader:
    """Tests for the ImageLoader stage."""

    def test_load_returns_array_and_meta(self, uml_test_image: str) -> None:
        """Loaded image is an ndarray with correct size metadata."""
        image, meta = ImageLoader().load(uml_test_image)
        assert isinstance(image, np.ndarray)
        assert image.shape == (300, 300, 3)
        assert meta == {"original_width": 300, "original_height": 300}

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
    """End-to-end tests: real OpenCV detection + PaddleOCR parsing."""

    def test_run_returns_dict_with_expected_keys(self, pipeline_result: dict) -> None:
        """Pipeline returns a JSON-ready dict with classes/relations/metadata."""
        assert isinstance(pipeline_result, dict)
        assert {"classes", "relations", "metadata"} <= set(pipeline_result)
        assert pipeline_result["metadata"]["processing_time_ms"] > 0

    def test_at_least_one_class_detected(self, pipeline_result: dict) -> None:
        """The rectangle in the test image is detected as a class."""
        assert len(pipeline_result["classes"]) >= 1

    def test_class_name_recognized(self, pipeline_result: dict) -> None:
        """OCR recognizes the class name; it is non-empty and not fake data."""
        names = [cls["name"] for cls in pipeline_result["classes"]]
        assert all(name for name in names)
        assert "FakeUser" not in names  # stub data must be gone
        assert any("user" in name.lower() for name in names)

    def test_bboxes_within_image_bounds(self, pipeline_result: dict) -> None:
        """Detected boxes fit inside the 300x300 test image."""
        for cls in pipeline_result["classes"]:
            bbox = cls["bbox"]
            assert 0 <= bbox["x_min"] < bbox["x_max"] <= 300
            assert 0 <= bbox["y_min"] < bbox["y_max"] <= 300

    def test_single_class_no_relations(self, pipeline_result: dict) -> None:
        """One class cannot be connected to anything -> empty relations."""
        assert len(pipeline_result["classes"]) == 1
        assert pipeline_result["relations"] == []

    def test_relation_detection(self, two_class_image: str) -> None:
        """Two boxes joined by a line produce at least one relation."""
        result = Pipeline().run(two_class_image)
        assert len(result["classes"]) >= 2
        assert len(result["relations"]) >= 1
        relation = result["relations"][0]
        assert relation["source"] != relation["target"]
        assert relation["type"] == "association"

    def test_ocr_fallback_unknown_name(self, monkeypatch: pytest.MonkeyPatch,
                                       uml_test_image: str) -> None:
        """With no OCR engine available, classes get name='Unknown' (no crash)."""
        import uml_recognition.services.text_parser as tp_module

        monkeypatch.setenv("TESSERACT_CMD", "/nonexistent/tesseract.exe")

        original_init = tp_module.TextParser.__init__

        def init_without_paddle(self: tp_module.TextParser) -> None:
            original_init(self)
            self.engine = None  # simulate both engines unavailable

        monkeypatch.setattr(tp_module.TextParser, "__init__", init_without_paddle)
        result = Pipeline().run(uml_test_image)
        assert len(result["classes"]) >= 1
        assert all(cls["name"] == "Unknown" for cls in result["classes"])

    def test_default_image_path_constant(self) -> None:
        """v0.1 default input path matches the spec."""
        assert DEFAULT_IMAGE_PATH == "data/raw/test.png"
