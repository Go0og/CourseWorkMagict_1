"""Text parsing stage: PaddleOCR + regex heuristics (iteration 3).

Reads the text inside a detected class rectangle and turns it into a
UmlClass model: first plain line -> class name, lines starting with a
visibility symbol are split into attributes (contain ':') and methods
(contain '(').
"""

from __future__ import annotations

import re

import numpy as np

from uml_recognition.models import Attribute, BBox, Method, Parameter, UmlClass, Visibility

_VISIBILITY_BY_SYMBOL: dict[str, Visibility] = {
    "+": Visibility.PUBLIC,
    "-": Visibility.PRIVATE,
    "#": Visibility.PROTECTED,
    "~": Visibility.PACKAGE,
}


class TextParser:
    """Parses OCR text of a class crop into a UmlClass model.

    Attributes:
        ocr: Underlying PaddleOCR engine (lazy-initialized on first use).
    """

    def __init__(self) -> None:
        """Initialize the parser; the heavy OCR engine loads lazily."""
        self._ocr: object | None = None

    @property
    def ocr(self) -> object:
        """Return the PaddleOCR instance, creating it on first access."""
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=False,
            )
        return self._ocr

    def parse(self, image_crop: np.ndarray, bbox: BBox) -> UmlClass:
        """Recognize text in a crop and build a UmlClass.

        Args:
            image_crop: BGR crop of one detected class box.
            bbox: Bounding box the crop was taken from.

        Returns:
            A filled UmlClass. If OCR finds nothing, the class is returned
            with name="Unknown".
        """
        print(f"[TextParser] OCR for bbox {bbox.model_dump()}")
        lines = self._recognize_lines(image_crop)
        return self._lines_to_class(lines, bbox)

    def _recognize_lines(self, image_crop: np.ndarray) -> list[str]:
        """Run PaddleOCR on the crop and return recognized text lines."""
        try:
            raw = self.ocr.ocr(np.asarray(image_crop), cls=True)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - MVP: never break the pipeline
            print(f"[TextParser] OCR failed: {exc}")
            return []
        lines: list[str] = []
        for page in self._as_sequence(raw):
            for entry in self._as_sequence(page):
                lines.extend(self._collect_texts(entry))
        return [line for line in lines if line]

    @staticmethod
    def _as_sequence(node: object) -> list:
        """Return an iterable node (list/generator/tuple) as a plain list."""
        if isinstance(node, (list, tuple)):
            return list(node)
        if hasattr(node, "__iter__") and not isinstance(node, (str, bytes, dict)):
            try:
                return list(node)  # type: ignore[call-overload]
            except TypeError:
                return []
        return []

    @classmethod
    def _collect_texts(cls, node: object) -> list[str]:
        """Recursively collect recognized strings from one OCR result node.

        Handles both the classic PaddleOCR format ``[box, (text, conf)]``
        and the newer nested layouts (lists/dicts with a "text" key).

        Args:
            node: Any element of the raw OCR response.

        Returns:
            Recognized text strings found under this node, top to bottom.
        """
        if isinstance(node, tuple):
            head = str(node[0]).strip() if node else ""
            return [head] if head else []
        if isinstance(node, dict):
            text = str(node.get("text", "")).strip()
            return [text] if text else []
        if isinstance(node, np.ndarray):
            node = node.tolist()
        results: list[str] = []
        for child in cls._as_sequence(node):
            results.extend(cls._collect_texts(child))
        return results

    def _lines_to_class(self, lines: list[str], bbox: BBox) -> UmlClass:
        """Convert raw OCR lines into a UmlClass via simple heuristics.

        Args:
            lines: Recognized text lines (top to bottom).
            bbox: Class bounding box to attach to the model.

        Returns:
            UmlClass with parsed name, attributes and methods.
        """
        name = "Unknown"
        attributes: list[Attribute] = []
        methods: list[Method] = []

        for line in lines:
            visibility = _VISIBILITY_BY_SYMBOL.get(line[0])
            body = line[1:].strip() if visibility else ""
            if visibility and body:
                if "(" in body:
                    methods.append(self._parse_method(body, visibility))
                elif ":" in body:
                    attr = self._parse_attribute(body, visibility)
                    if attr is not None:
                        attributes.append(attr)
                continue
            if name == "Unknown" and re.fullmatch(r"[A-Za-z_][\w]*", line):
                name = line

        return UmlClass(name=name, attributes=attributes, methods=methods, bbox=bbox)

    @staticmethod
    def _parse_attribute(body: str, visibility: Visibility) -> Attribute | None:
        """Parse 'name: type' into an Attribute (None if name invalid)."""
        attr_name, _, type_str = body.partition(":")
        attr_name = attr_name.strip()
        if not re.fullmatch(r"[A-Za-z_][\w]*", attr_name):
            return None
        return Attribute(
            name=attr_name,
            type_str=type_str.strip() or None,
            visibility=visibility,
        )

    @staticmethod
    def _parse_method(body: str, visibility: Visibility) -> Method:
        """Parse 'name(params): ret' into a Method."""
        head, sep, tail = body.partition("(")
        head = head.strip()
        method_name = head if re.fullmatch(r"[A-Za-z_][\w]*", head) else head or "run"
        params_part = tail.split(")")[0] if sep else ""
        parameters: list[tuple[str, str | None]] = []
        for chunk in params_part.split(","):
            param_name, _, param_type = chunk.partition(":")
            param_name = param_name.strip()
            if re.fullmatch(r"[A-Za-z_][\w]*", param_name):
                parameters.append((param_name, param_type.strip() or None))
        return_type = tail.split(")", 1)[-1].lstrip(":").strip() or None
        return Method(
            name=method_name,
            parameters=[Parameter(name=p_name, type_str=p_type) for p_name, p_type in parameters],
            return_type=return_type,
            visibility=visibility,
        )
