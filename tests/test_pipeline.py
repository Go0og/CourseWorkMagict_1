"""End-to-end tests for the MVP pipeline (iteration 2)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from uml_recognition.pipeline import DEFAULT_IMAGE_PATH, Pipeline
from uml_recognition.services.image_loader import ImageLoader


@pytest.fixture()
def test_image(tmp_path: Path) -> str:
    """Create a 100x100 black PNG image in a temp dir and return its path."""
    image_path = tmp_path / "test.png"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    return str(image_path)


class TestImageLoader:
    """Tests for the ImageLoader stage."""

    def test_load_returns_array_and_meta(self, test_image: str) -> None:
        """Loaded image is an ndarray with correct size metadata."""
        image, meta = ImageLoader().load(test_image)
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert meta == {"original_width": 100, "original_height": 100}

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
    """Tests for the end-to-end Pipeline.run on a real image."""

    def test_run_returns_dict_with_expected_keys(self, test_image: str) -> None:
        """Pipeline returns a JSON-ready dict with classes/relations/metadata."""
        result = Pipeline().run(test_image)
        assert isinstance(result, dict)
        assert {"classes", "relations", "metadata"} <= set(result)
        assert result["metadata"]["processing_time_ms"] > 0
        assert len(result["classes"]) == 2  # stub: FakeUser + relation target
        assert len(result["relations"]) == 1
        assert result["metadata"]["source_image"] == test_image

    def test_stub_bboxes_within_image_bounds(self, test_image: str) -> None:
        """Fake detection boxes fit inside the loaded image dimensions."""
        result = Pipeline().run(test_image)
        for cls in result["classes"]:
            bbox = cls["bbox"]
            assert 0 <= bbox["x_min"] < bbox["x_max"] <= 100
            assert 0 <= bbox["y_min"] < bbox["y_max"] <= 100

    def test_default_image_path_constant(self) -> None:
        """v0.1 default input path matches the spec."""
        assert DEFAULT_IMAGE_PATH == "data/raw/test.png"
