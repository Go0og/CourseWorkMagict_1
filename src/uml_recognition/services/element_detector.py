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

    Pipeline within ``detect_bboxes``:
        1. Convert to grayscale (cv2.cvtColor).
        2. Reduce noise (cv2.GaussianBlur).
        3. Find edges (cv2.Canny).
        4. Extract contours (cv2.findContours).
        5. Keep near-rectangular contours with an adaptive minimum area
           (``max(1000, 1% of the image)``) and an aspect ratio between
           0.3 and 3.0 so small PlantUML boxes are not filtered out.
    """

    MIN_ASPECT_RATIO: float = 0.3
    MAX_ASPECT_RATIO: float = 3.0
    AREA_FRACTION: float = 0.01
    MIN_AREA_FLOOR: int = 1000

    def detect_bboxes(self, image: np.ndarray) -> list[BBox]:
        """Find bounding boxes of potential UML classes in the image.

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

        # Adaptive area threshold: 1% of the image, but never below 1000 px.
        h_img, w_img = image.shape[:2]
        min_area = max(self.MIN_AREA_FLOOR, (w_img * h_img) * self.AREA_FRACTION)
        print(f"[ElementDetector] Image {w_img}x{h_img}, adaptive min_area={min_area:.0f}")

        bboxes: list[BBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_area:
                continue
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

        bboxes = self._drop_nested(bboxes)
        bboxes.sort(key=lambda b: (b.y_min, b.x_min))
        print(f"[ElementDetector] Detected {len(bboxes)} candidate class rectangles")
        return bboxes

    @staticmethod
    def _drop_nested(bboxes: list[BBox], tolerance: int = 6) -> list[BBox]:
        """Remove boxes that are nested inside other boxes with ~equal area.

        Canny detects both outer and inner edges of a thick rectangle
        border; keeping only the outer box avoids duplicate classes.

        Args:
            bboxes: Raw detected bounding boxes.
            tolerance: Max coordinate difference treated as "same box".

        Returns:
            Filtered list without nested duplicates (smallest kept first
            is not assumed; order-independent).
        """
        result: list[BBox] = []
        for box in bboxes:
            dominated = False
            keep: list[BBox] = []
            for other in result:
                if _contains(other, box, tolerance):
                    dominated = True
                    break
                if _contains(box, other, tolerance):
                    continue  # drop the previously kept inner box
                keep.append(other)
            if dominated:
                continue
            keep.append(box)
            result = keep
        return result


def _contains(outer: BBox, inner: BBox, tolerance: int) -> bool:
    """True if ``inner`` lies within ``outer`` expanded by ``tolerance``."""
    return (outer.x_min - tolerance <= inner.x_min
            and outer.y_min - tolerance <= inner.y_min
            and outer.x_max + tolerance >= inner.x_max
            and outer.y_max + tolerance >= inner.y_max)
