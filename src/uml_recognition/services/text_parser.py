"""Text recognition and parsing stage: PaddleOCR + heuristic UML parser.

Reads the text inside a detected class rectangle and splits it into
class name, attributes ("+id: int") and methods ("+login(): bool").
"""

from __future__ import annotations

import os
import re

import numpy as np

from uml_recognition.models import Attribute, BBox, Method, Parameter, UmlClass, Visibility

VISIBILITY_CHARS = {"+": Visibility.PUBLIC, "-": Visibility.PRIVATE,
                    "#": Visibility.PROTECTED, "~": Visibility.PACKAGE}


class TextParser:
    """Recognizes text in an image crop and builds a UmlClass model.

    Uses PaddleOCR for recognition and regex heuristics to split the
    recognized lines into a class name, attributes and methods.
    """

    def __init__(self) -> None:
        """Initialize the PaddleOCR engine (lazy, tolerant to API versions).

        Tries the classic PaddleOCR 2.x API first
        (``use_angle_cls=True, lang='en', show_log=False``); if the
        installed version is 3.x (which dropped those arguments), falls
        back to the new constructor signature.
        """
        # Avoid oneDNN runtime crashes on some CPU builds of PaddlePaddle.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR  # heavy import kept local on purpose

        print("[TextParser] Initializing PaddleOCR...")
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._api_v2 = True
        except (ValueError, TypeError):
            # PaddleOCR >= 3.x: unknown legacy kwargs, use defaults.
            self.ocr = PaddleOCR(lang="en", enable_mkldnn=False)
            self._api_v2 = False
        print("[TextParser] PaddleOCR ready")

    def _recognize_lines(self, image_crop: np.ndarray) -> list[str]:
        """Run OCR on a crop and return recognized text lines top-to-bottom.

        Args:
            image_crop: BGR crop of a single class rectangle.

        Returns:
            Recognized strings sorted by vertical position; empty list
            if nothing was recognized.
        """
        try:
            if self._api_v2:
                raw = self.ocr.ocr(image_crop, cls=True)
            else:
                raw = self.ocr.predict(image_crop)
        except Exception as exc:  # noqa: BLE001 - OCR backend raises various errors
            print(f"[TextParser] OCR call failed: {exc}")
            return []

        lines: list[tuple[float, str]] = []
        pages = raw if isinstance(raw, list) else [raw]
        for page in pages or []:
            # PaddleOCR 2.x format: [[ [box, (text, score)], ... ]]
            if isinstance(page, list):
                for item in page:
                    try:
                        box, (text, _score) = item[0], item[1]
                        lines.append((float(box[0][1]), str(text)))
                    except (IndexError, TypeError, ValueError):
                        continue
            # PaddleOCR 3.x format: dict with rec_texts / rec_polys
            elif isinstance(page, dict) and "rec_texts" in page:
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                for i, text in enumerate(page["rec_texts"]):
                    y = float(polys[i][0][1]) if i < len(polys) else float(i)
                    lines.append((y, str(text)))

        lines.sort(key=lambda t: t[0])
        return [text for _, text in lines]

    @staticmethod
    def _parse_attribute(line: str, visibility: Visibility) -> Attribute | None:
        """Parse an attribute line like ``+id: int`` into an Attribute."""
        body = line[1:].strip()
        name, _, type_str = body.partition(":")
        name = name.strip()
        if not name:
            return None
        return Attribute(name=name, type_str=type_str.strip() or None, visibility=visibility)

    @staticmethod
    def _parse_method(line: str, visibility: Visibility) -> Method | None:
        """Parse a method line like ``+login(user: str): bool`` into a Method."""
        match = re.match(r"^[+\-#~]\s*(\w+)\s*\(([^)]*)\)\s*:?\s*(.*)$", line)
        if not match:
            return None
        name, params_str, ret = match.groups()
        parameters: list[Parameter] = []
        for chunk in params_str.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            pname, _, ptype = chunk.partition(":")
            parameters.append(Parameter(name=pname.strip(), type_str=ptype.strip() or None))
        return Method(name=name, parameters=parameters, return_type=ret.strip() or None,
                      visibility=visibility)

    def parse(self, image_crop: np.ndarray, bbox: BBox) -> UmlClass:
        """Recognize text in a crop and build a UmlClass.

        Heuristics:
            * First plain line (no visibility prefix) -> class name.
            * Lines starting with +/-/#/~ containing '(' -> Method.
            * Lines starting with +/-/#/~ containing ':' -> Attribute.

        Args:
            image_crop: BGR crop of the class rectangle.
            bbox: Bounding box of the crop in full-image coordinates.

        Returns:
            A populated UmlClass; if no text was recognized, returns
            UmlClass(name="Unknown") with the given bbox.
        """
        lines = self._recognize_lines(image_crop)
        print(f"[TextParser] OCR lines: {lines}")

        name: str | None = None
        attributes: list[Attribute] = []
        methods: list[Method] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            vis = VISIBILITY_CHARS.get(line[0])
            if vis is not None:
                if "(" in line:
                    method = self._parse_method(line, vis)
                    if method:
                        methods.append(method)
                elif ":" in line:
                    attr = self._parse_attribute(line, vis)
                    if attr:
                        attributes.append(attr)
            elif name is None:
                name = line

        if name is None:
            print("[TextParser] No text recognized -> UmlClass(name='Unknown')")
            return UmlClass(name="Unknown", bbox=bbox, confidence=0.0)

        return UmlClass(name=name, attributes=attributes, methods=methods, bbox=bbox)
