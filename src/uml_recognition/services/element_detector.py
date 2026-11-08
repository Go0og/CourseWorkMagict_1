"""Element detection stage: OpenCV contour heuristic (iteration 3).

Real YOLO is intentionally NOT used in v0.1 MVP (no dataset). Instead we
find rectangle-like contours with Canny + findContours and treat them as
candidate UML class boxes.
"""

from __future__ import annotations

import cv2
import numpy as np

from uml_recognition.models import BBox


class ElementDetector:
    """Detects candidate UML class rectangles on an image.

    Heuristic pipeline: grayscale -> GaussianBlur -> Canny -> findContours,
    then filter by area and aspect ratio to keep only rectangle-like blobs.
    """

    def detect(self, image: np.ndarray) -> list[BBox]:
        """Find bounding boxes of potential UML classes.

        Args:
            image: Loaded BGR image as numpy array.

        Returns:
            List of BBox objects for contours with area > 5000 px and
            aspect ratio within [0.5, 2.0].
        """
        print("[ElementDetector] Contour-based detection (OpenCV heuristic)")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # Close small gaps so rectangle borders form a single contour.
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        height, width = image.shape[:2]
        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            # Rectangle-likeness: bounding-box area vs filled contour area.
            fill_ratio = cv2.contourArea(contour) / float(w * h)
            if fill_ratio < 0.9:
                continue
            area = w * h
            if area <= 5000:
                continue
            aspect_ratio = w / h
            if not 0.5 <= aspect_ratio <= 2.0:
                continue
            x_min, y_min = max(int(x), 0), max(int(y), 0)
            x_max, y_max = min(int(x + w), width), min(int(y + h), height)
            if x_min >= x_max or y_min >= y_max:
                continue
            candidates.append((x_min, y_min, x_max, y_max))

        # Drop boxes fully contained inside larger boxes (nested contours).
        boxes: list[BBox] = []
        for box in candidates:
            if any(
                other != box
                and other[0] <= box[0]
                and other[1] <= box[1]
                and box[2] <= other[2]
                and box[3] <= other[3]
                and (box[2] - box[0]) * (box[3] - box[1])
                < (other[2] - other[0]) * (other[3] - other[1])
                for other in candidates
            ):
                continue
            boxes.append(BBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]))

        print(f"[ElementDetector] Found {len(boxes)} candidate class box(es)")
        return boxes
