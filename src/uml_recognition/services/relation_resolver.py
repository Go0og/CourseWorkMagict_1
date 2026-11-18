"""Relation resolution stage (iteration 4): Hough-line based MVP.

Finds straight connector lines between detected class rectangles using
the probabilistic Hough transform and maps each line endpoint to the
nearest class bounding box center to build association relations.
"""

from __future__ import annotations

import cv2
import numpy as np

from uml_recognition.models import Relation, UmlClass


class RelationResolver:
    """Resolves relations between detected UML classes via line detection.

    Algorithm: Canny edges -> HoughLinesP -> for every segment, find the
    closest class bbox centers to both endpoints; if they differ, emit an
    ``association`` relation. Duplicate (source, target) pairs are removed.
    """

    def resolve(self, image: np.ndarray, classes: list[UmlClass]) -> list[Relation]:
        """Find relations between the given classes by detecting connector lines.

        Args:
            image: Full loaded image (BGR numpy array).
            classes: Detected UML classes with populated bboxes.

        Returns:
            List of Relation objects (type="association"); empty when
            fewer than two classes exist or no connector lines were found.
        """
        if len(classes) < 2:
            print(f"[RelationResolver] {len(classes)} classes -> nothing to connect")
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        if lines is None:
            print("[RelationResolver] No lines detected -> 0 relations")
            return []

        boxes = [self._bbox_tuple(cls) for cls in classes]
        centers = [self._center(cls) for cls in classes]

        relations: list[Relation] = []
        seen: set[tuple[str, str]] = set()
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in line)
            # Skip segments that are just the borders of a class rectangle.
            if self._inside_any_box(x1, y1, x2, y2, boxes):
                continue
            source_idx = self._nearest_class(x1, y1, centers)
            target_idx = self._nearest_class(x2, y2, centers)
            if source_idx is None or target_idx is None or source_idx == target_idx:
                continue
            source, target = classes[source_idx].name, classes[target_idx].name
            key = (min(source, target), max(source, target))
            if key in seen:
                continue
            seen.add(key)
            relations.append(Relation(source=source, target=target,
                                      type="association", confidence=0.7))

        print(f"[RelationResolver] {len(lines)} raw lines -> {len(relations)} relations")
        return relations

    @staticmethod
    def _bbox_tuple(uml_class: UmlClass) -> tuple[int, int, int, int] | None:
        """Return (x_min, y_min, x_max, y_max) of a class or None."""
        if uml_class.bbox is None:
            return None
        return (uml_class.bbox.x_min, uml_class.bbox.y_min,
                uml_class.bbox.x_max, uml_class.bbox.y_max)

    @staticmethod
    def _center(uml_class: UmlClass) -> tuple[float, float] | None:
        """Return the center point of a class bbox or None."""
        if uml_class.bbox is None:
            return None
        return ((uml_class.bbox.x_min + uml_class.bbox.x_max) / 2.0,
                (uml_class.bbox.y_min + uml_class.bbox.y_max) / 2.0)

    @staticmethod
    def _nearest_class(x: int, y: int,
                       centers: list[tuple[float, float] | None],
                       max_distance: float = 250.0) -> int | None:
        """Index of the class whose bbox center is nearest to (x, y).

        Args:
            x: Point x coordinate.
            y: Point y coordinate.
            centers: Precomputed bbox centers (None entries skipped).
            max_distance: Maximum allowed distance in pixels.

        Returns:
            Class index, or None if every center is farther than
            ``max_distance``.
        """
        best_idx: int | None = None
        best_dist = max_distance
        for idx, center in enumerate(centers):
            if center is None:
                continue
            dist = float(np.hypot(center[0] - x, center[1] - y))
            if dist < best_dist:
                best_dist, best_idx = dist, idx
        return best_idx

    @staticmethod
    def _inside_any_box(x1: int, y1: int, x2: int, y2: int,
                        boxes: list[tuple[int, int, int, int] | None],
                        margin: int = 5) -> bool:
        """True if the segment lies (almost) along some class rectangle edge.

        Long horizontal/vertical segments matching a box border are the
        rectangle sides themselves, not connectors between classes.
        """
        for box in boxes:
            if box is None:
                continue
            bx1, by1, bx2, by2 = box
            in_x = bx1 - margin <= x1 <= bx2 + margin and bx1 - margin <= x2 <= bx2 + margin
            in_y = by1 - margin <= y1 <= by2 + margin and by1 - margin <= y2 <= by2 + margin
            horizontal = abs(y1 - y2) <= margin
            vertical = abs(x1 - x2) <= margin
            if (in_x and horizontal) or (in_y and vertical):
                return True
        return False
