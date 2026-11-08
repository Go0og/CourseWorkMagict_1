"""Element detection stage (YOLO stub for iteration 2)."""

from __future__ import annotations

import numpy as np

from uml_recognition.models import (
    Attribute,
    BBox,
    Metadata,
    Method,
    Relation,
    RelationType,
    UmlClass,
    UmlDiagram,
    Visibility,
)


class ElementDetector:
    """Detects UML elements on an image.

    STUB: returns one fake class and one fake relation so the end-to-end
    pipeline can be exercised before real YOLO/OCR integration.
    """

    def detect(self, image: np.ndarray, meta: dict) -> UmlDiagram:
        """Run detection on a loaded image.

        Args:
            image: Loaded image as numpy array (unused in the stub).
            meta: Image metadata with "original_width"/"original_height".

        Returns:
            A valid UmlDiagram containing fake test data scaled to the
            image dimensions.
        """
        print("[ElementDetector] Stub detection (fake data)")
        width = int(meta["original_width"])
        height = int(meta["original_height"])

        x_min = max(width // 10, 0)
        y_min = max(height // 10, 0)
        x_max = max(int(x_min + width * 0.3), x_min + 1)
        y_max = max(int(y_min + height * 0.3), y_min + 1)

        fake_class = UmlClass(
            name="FakeUser",
            attributes=[Attribute(name="login", type_str="str", visibility=Visibility.PRIVATE)],
            methods=[Method(name="login", return_type="bool", visibility=Visibility.PUBLIC)],
            bbox=BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            confidence=0.95,
        )
        fake_target = UmlClass(
            name="FakeSession",
            attributes=[Attribute(name="token", type_str="str", visibility=Visibility.PRIVATE)],
            methods=[],
            bbox=BBox(
                x_min=min(int(x_max + width * 0.2), max(width - 2, 1)),
                y_min=y_min,
                x_max=max(width - 1, 2),
                y_max=y_max,
            ),
            confidence=0.9,
        )
        relation = Relation(
            source=fake_class.name,
            target=fake_target.name,
            type=RelationType.ASSOCIATION,
            label="has",
            confidence=0.85,
        )
        return UmlDiagram(
            classes=[fake_class, fake_target],
            relations=[relation],
            metadata=Metadata(source_image="stub", processing_time_ms=1),
        )
