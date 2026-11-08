"""Image loading stage of the UML recognition pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    """Loads an input image and extracts basic size metadata.

    Stage 1 of the pipeline: reads a file from disk via OpenCV and
    validates that it exists, is non-empty and decodable.
    """

    def load(self, image_path: str) -> tuple[np.ndarray, dict]:
        """Read an image file and return it with size metadata.

        Args:
            image_path: Path to the input image file.

        Returns:
            A tuple of (BGR image as numpy array, metadata dict with keys
            "original_width" and "original_height").

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is empty or cannot be decoded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is empty: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        height, width = image.shape[:2]
        meta = {"original_width": int(width), "original_height": int(height)}
        print(f"[ImageLoader] Loaded {image_path} ({width}x{height})")
        return image, meta
