"""Pipeline orchestrator stub for the UML extraction pipeline.

Iteration 1 (MVP): the pipeline only logs the input path and returns an
empty but schema-valid UmlDiagram. Real services (ImageLoader,
ElementDetector, TextParser, RelationResolver, JsonExporter) will be
plugged in during later iterations.
"""

import time

from uml_recognition.models import Metadata, UmlDiagram


class Pipeline:
    """Orchestrates the image-to-UML-JSON processing flow.

    Attributes:
        schema_version: Version tag written into output metadata.
    """

    def __init__(self, schema_version: str = "1.0") -> None:
        """Initialize the pipeline.

        Args:
            schema_version: Schema version for output metadata.
        """
        self.schema_version = schema_version

    def run(self, image_path: str) -> UmlDiagram:
        """Process an image and return the extracted UML diagram.

        MVP stub: logs the path and returns an empty valid diagram
        with fake metadata.

        Args:
            image_path: Path to the source PNG image.

        Returns:
            A schema-valid UmlDiagram instance.
        """
        print(f"[Pipeline] Processing image: {image_path}")
        started = time.perf_counter()
        # TODO(iteration 2): ImageLoader -> ElementDetector -> TextParser
        # -> RelationResolver -> JsonExporter
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return UmlDiagram(
            classes=[],
            relations=[],
            metadata=Metadata(
                source_image=image_path,
                processing_time_ms=elapsed_ms,
                schema_version=self.schema_version,
            ),
        )
