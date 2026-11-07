"""UML recognition pipeline (MVP stub for iteration 1)."""

from __future__ import annotations

import time

from uml_recognition.models import Metadata, UmlDiagram


class Pipeline:
    """Extracts a UML class diagram from an image.

    Currently a stub: logs the input path and returns a valid but empty
    UmlDiagram. Real CV/OCR stages arrive in later iterations.
    """

    def run(self, image_path: str) -> UmlDiagram:
        """Run the pipeline on an image and return the parsed diagram.

        Args:
            image_path: Path to the input image file.

        Returns:
            A valid UmlDiagram object (empty lists with fake metadata in MVP).
        """
        print(f"[Pipeline] Processing image: {image_path}")
        start = time.perf_counter()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        metadata = Metadata(
            source_image=image_path,
            processing_time_ms=max(elapsed_ms, 1),
            schema_version="1.0",
        )
        return UmlDiagram(classes=[], relations=[], metadata=metadata)
