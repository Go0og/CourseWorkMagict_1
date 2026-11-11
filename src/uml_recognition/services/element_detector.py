"""Element detection stage: classic OpenCV contour detection (iteration 3).

Finds rectangles that are likely UML class boxes on a diagram image.
YOLOv8-based detection is planned for iteration 4; this MVP uses only
classical computer vision (grayscale -> blur -> Canny -> contours).
"""

from __future__ import annotations

import cv2
import numpy as np

from uml_recognition.models import BBox


class ElementDetector:
    """Detects candidate UML class rectangles via OpenCV contours.

    Pipeline within ``detect_bboxes``:
        1. Convert to grayscale (cv2.cvtColor).
        2. Reduce noise (cv2.GaussianBlur).
        3. Find edges (cv2.Canny).
        4. Extract contours (cv2.findContours).
        5. Keep near-rectangular contours with area > 5000 px and an
           aspect ratio between 0.5 and 2.0.
    """

    MIN_AREA: int = 5000
    MIN_ASPECT_RATIO: float = 0.5
    MAX_ASPECT_RATIO: float = 2.0

    def detect_bboxes(self, image: np.ndarray) -> list[BBox]:
        """Find bounding boxes of potential UML classes in the image.

        Args:
            image: Loaded image as a numpy array (BGR or grayscale).

        Returns:
            List of BBox objects for detected rectangle candidates,
            ordered top-to-bottom, left-to-right.
        """
        print("[ElementDetector] Running classic OpenCV contour detection")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        # Close small gaps so rectangle borders form complete contours.
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        bboxes: list[BBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= self.MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = w / h
            if not self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                continue
            # The contour must look like a rectangle (4 corners).
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            try:
                bboxes.append(
                    BBox(x_min=int(x), y_min=int(y), x_max=int(x + w), y_max=int(y + h))
                )
            except ValueError:
                print(f"[ElementDetector] Skipping invalid bbox at ({x}, {y}, {w}, {h})")

        bboxes.sort(key=lambda b: (b.y_min, b.x_min))
        print(f"[ElementDetector] Detected {len(bboxes)} candidate class rectangles")
        return bboxes
