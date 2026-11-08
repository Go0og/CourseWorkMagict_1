"""End-to-end UML recognition pipeline (MVP, iteration 2)."""

from __future__ import annotations

import time

from uml_recognition.models import Metadata, UmlDiagram
from uml_recognition.services.element_detector import ElementDetector
from uml_recognition.services.image_loader import ImageLoader

DEFAULT_IMAGE_PATH = "data/raw/test.png"


class Pipeline:
    """Extracts a UML class diagram from an image.

    Stage order: ImageLoader -> ElementDetector (YOLO stub for now).
    Returns the diagram as a JSON-ready dict.
    """

    def run(self, image_path: str = DEFAULT_IMAGE_PATH) -> dict:
        """Run the pipeline on an image and return the parsed diagram.

        Args:
            image_path: Path to the input image file.

        Returns:
            JSON-serializable dict (Pydantic model_dump with mode="json")
            of the extracted UmlDiagram with fresh processing metadata.
        """
        print(f"[Pipeline] Processing image: {image_path}")
        start = time.time()

        image, meta = ImageLoader().load(image_path)
        result: UmlDiagram = ElementDetector().detect(image, meta)

        processing_time_ms = max(int((time.time() - start) * 1000), 1)
        result.metadata = Metadata(
            source_image=image_path,
            processing_time_ms=processing_time_ms,
            schema_version="1.0",
        )
        return result.model_dump(mode="json")
