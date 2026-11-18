"""End-to-end UML recognition pipeline (MVP)."""
from __future__ import annotations

import time

import numpy as np

from uml_recognition.models import BBox, Metadata, UmlClass, UmlDiagram
from uml_recognition.services.element_detector import ElementDetector
from uml_recognition.services.image_loader import ImageLoader
from uml_recognition.services.relation_resolver import RelationResolver
from uml_recognition.services.text_parser import TextParser

DEFAULT_IMAGE_PATH = "data/raw/test.png"


class Pipeline:
    """Extracts a UML class diagram from an image."""

    def __init__(self) -> None:
        self.image_loader = ImageLoader()
        self.element_detector = ElementDetector()
        self.text_parser = TextParser()
        self.relation_resolver = RelationResolver()

    @staticmethod
    def _crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Cut out the rectangle region described by bbox from the image."""
        height, width = image.shape[:2]
        x_min = max(bbox.x_min, 0)
        y_min = max(bbox.y_min, 0)
        x_max = min(bbox.x_max, width)
        y_max = min(bbox.y_max, height)
        return image[y_min:y_max, x_min:x_max]

    def run(self, image_path: str = DEFAULT_IMAGE_PATH) -> dict:
        """Run the pipeline on an image and return the parsed diagram."""
        print(f"[Pipeline] Processing image: {image_path}")
        start = time.time()

        # Stage 1: load image.
        image, _meta = self.image_loader.load(image_path)
        original_image = image.copy()  # Сохраняем оригинал для OCR

        # Stage 2: find candidate class rectangles (classic OpenCV).
        bboxes = self.element_detector.detect_bboxes(image)

        # Разбиваем боксы, охватывающие >35% площади (склеенные классы)
        h_img, w_img = image.shape[:2]
        img_area = h_img * w_img
        expanded_bboxes = []
        for bbox in bboxes:
            bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
            if bbox_area > img_area * 0.35:
                mid_y = (bbox.y_min + bbox.y_max) // 2
                expanded_bboxes.append(BBox(x_min=bbox.x_min, y_min=bbox.y_min,
                                            x_max=bbox.x_max, y_max=mid_y))
                expanded_bboxes.append(BBox(x_min=bbox.x_min, y_min=mid_y,
                                            x_max=bbox.x_max, y_max=bbox.y_max))
                print(f"[Pipeline] Split large bbox at y={mid_y}")
            else:
                expanded_bboxes.append(bbox)
        bboxes = expanded_bboxes

        # Stage 3: OCR + parse each rectangle into a UmlClass.
        classes: list[UmlClass] = []
        for bbox in bboxes:
            crop = self._crop(original_image, bbox)  # Используем оригинал для OCR
            if crop.size == 0:
                print(f"[Pipeline] Empty crop for bbox {bbox}, skipping")
                continue
            classes.append(self.text_parser.parse(crop, bbox))

        # Stage 4: resolve relations via Hough line detection.
        relations = self.relation_resolver.resolve(image, classes)

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
        print(f"[Pipeline] Done: {len(classes)} classes, {len(relations)} relations "
              f"in {processing_time_ms} ms")
        return result.model_dump(mode="json")
