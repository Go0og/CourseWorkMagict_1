"""End-to-end UML recognition pipeline (v0.1 MVP, iteration 3)."""

from __future__ import annotations

import time

import numpy as np

from uml_recognition.models import BBox, Metadata, UmlDiagram
from uml_recognition.services.element_detector import ElementDetector
from uml_recognition.services.image_loader import ImageLoader
from uml_recognition.services.relation_resolver import RelationResolver
from uml_recognition.services.text_parser import TextParser

DEFAULT_IMAGE_PATH = "data/raw/test.png"


class Pipeline:
    """Extracts a UML class diagram from an image.

    Stage order: ImageLoader -> ElementDetector (OpenCV contours) ->
    TextParser (PaddleOCR per crop) -> RelationResolver (stub).
    Returns the diagram as a JSON-ready dict.
    """

    def run(self, image_path: str = DEFAULT_IMAGE_PATH) -> dict:
        """Run the full pipeline on an image and return the parsed diagram.

        Args:
            image_path: Path to the input image file.

        Returns:
            JSON-serializable dict (Pydantic model_dump with mode="json")
            of the extracted UmlDiagram with fresh processing metadata.
        """
        print(f"[Pipeline] Processing image: {image_path}")
        start = time.time()

        image, _meta = ImageLoader().load(image_path)
        boxes: list[BBox] = ElementDetector().detect(image)

        parser = TextParser()
        classes = [parser.parse(self._crop(image, box), box) for box in boxes]

        relations = RelationResolver().resolve(classes)

        processing_time_ms = max(int((time.time() - start) * 1000), 1)
        result = UmlDiagram(
            classes=classes,
            relations=relations,
            metadata=Metadata(
                source_image=image_path,
                processing_time_ms=processing_time_ms,
                schema_version="1.0",
            ),
        )
        return result.model_dump(mode="json")

    @staticmethod
    def _crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Cut out the region of interest described by a bounding box.

        Args:
            image: Full loaded BGR image.
            bbox: Region to extract.

        Returns:
            The cropped BGR image for OCR parsing.
        """
        return image[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
