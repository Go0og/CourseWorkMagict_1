"""Element detection: classic OpenCV contour detection."""
from __future__ import annotations

import cv2
import numpy as np

from uml_recognition.models import BBox


class ElementDetector:
    """Detects candidate UML class rectangles via OpenCV contours."""

    MIN_ASPECT_RATIO: float = 0.3
    MAX_ASPECT_RATIO: float = 3.0
    AREA_FRACTION: float = 0.01
    MIN_AREA_FLOOR: int = 1000

    def detect_bboxes(self, image: np.ndarray) -> list[BBox]:
        """Find bounding boxes of potential UML classes in the image."""
        # Автоинверсия для светлого фона (PlantUML и т.п.)
        mean_brightness = float(cv2.mean(image)[0])
        if mean_brightness > 127:
            image = cv2.bitwise_not(image)
            print(f"[ElementDetector] Inverted image (mean brightness={mean_brightness:.0f})")

        print("[ElementDetector] Running classic OpenCV contour detection")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # ВАЖНО: RETR_EXTERNAL берёт только внешние контуры (целые классы)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            aspect_ratio = w / h
            if not self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                continue
            # PlantUML-рамка с иконкой может дать 4-14 углов
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) < 4 or len(approx) > 14:
                continue
            try:
                bboxes.append(
                    BBox(x_min=int(x), y_min=int(y), x_max=int(x + w), y_max=int(y + h))
                )
            except ValueError:
                print(f"[ElementDetector] Skipping invalid bbox at ({x}, {y}, {w}, {h})")

        bboxes = self._drop_nested(bboxes)
        bboxes.sort(key=lambda b: (b.y_min, b.x_min))
        print(f"[ElementDetector] Detected {len(bboxes)} candidate class rectangles")
        return bboxes

    @staticmethod
    def _drop_nested(bboxes: list[BBox], tolerance: int = 6) -> list[BBox]:
        """Remove boxes that are nested inside other boxes."""
        result: list[BBox] = []
        for box in bboxes:
            dominated = False
            keep: list[BBox] = []
            for other in result:
                if _contains(other, box, tolerance):
                    dominated = True
                    break
                if _contains(box, other, tolerance):
                    continue
                keep.append(other)
            if dominated:
                continue
            keep.append(box)
            result = keep
        return result


def _contains(outer: BBox, inner: BBox, tolerance: int) -> bool:
    """True if `inner` lies within `outer` expanded by `tolerance`."""
    return (outer.x_min - tolerance <= inner.x_min
            and outer.y_min - tolerance <= inner.y_min
            and outer.x_max + tolerance >= inner.x_max
            and outer.y_max + tolerance >= inner.y_max)
