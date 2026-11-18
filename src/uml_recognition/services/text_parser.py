"""Text recognition and parsing stage: pytesseract (primary) + PaddleOCR fallback.

Reads the text inside a detected class rectangle and splits it into
class name, attributes ("+id: int") and methods ("+login(): bool").

Engine selection order:
    1. pytesseract (Tesseract binary from TESSERACT_CMD env var or PATH).
    2. PaddleOCR (optional dependency: ``pip install uml-recognition[paddle]``).
    3. None -> parse() returns UmlClass(name="Unknown").
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

from uml_recognition.models import Attribute, BBox, Method, Parameter, UmlClass, Visibility

VISIBILITY_CHARS = {"+": Visibility.PUBLIC, "-": Visibility.PRIVATE,
                    "#": Visibility.PROTECTED, "~": Visibility.PACKAGE}


class TextParser:
    """Recognizes text in an image crop and builds a UmlClass model.

    Uses pytesseract when a Tesseract binary is available, otherwise
    falls back to PaddleOCR; if neither engine works, all crops are
    parsed as UmlClass(name="Unknown"). Regex heuristics split the
    recognized lines into a class name, attributes and methods.
    """

    def __init__(self) -> None:
        """Initialize the best available OCR engine.

        Tries pytesseract first (checks the TESSERACT_CMD environment
        variable, then PATH); on failure tries PaddleOCR; on second
        failure sets ``engine`` to None so the pipeline never crashes.
        """
        self.engine: str | None = None
        self.ocr = None  # PaddleOCR instance when engine == "paddleocr"
        self._api_v2 = False

        # --- Attempt 1: pytesseract -------------------------------------
        try:
            import pytesseract

            tesseract_cmd = os.environ.get("TESSERACT_CMD")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # Raises TesseractNotFoundError / subprocess errors if missing.
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self.engine = "pytesseract"
            print("[TextParser] Engine: pytesseract")
            return
        except Exception as exc:  # noqa: BLE001 - any import/binary failure
            print(f"[TextParser] pytesseract unavailable: {exc}")

        # --- Attempt 2: PaddleOCR ---------------------------------------
        try:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            from paddleocr import PaddleOCR

            print("[TextParser] Trying PaddleOCR...")
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self._api_v2 = True
            except (ValueError, TypeError):
                # PaddleOCR >= 3.x: unknown legacy kwargs, use defaults.
                self.ocr = PaddleOCR(lang="en", enable_mkldnn=False)
                self._api_v2 = False
            self.engine = "paddleocr"
            print("[TextParser] Engine: paddleocr")
            return
        except Exception as exc:  # noqa: BLE001 - any import/init failure
            print(f"[TextParser] PaddleOCR unavailable: {exc}")

        # --- No engine available ----------------------------------------
        self.engine = None
        print("[TextParser] Engine: none (OCR disabled, names will be 'Unknown')")

    # ------------------------------------------------------------------
    # OCR backends
    # ------------------------------------------------------------------
    def _recognize_lines(self, image_crop: np.ndarray) -> list[str]:
        """Run OCR on a crop and return recognized text lines top-to-bottom.

        Args:
            image_crop: BGR crop of a single class rectangle.

        Returns:
            Recognized strings sorted by vertical position; empty list
            if nothing was recognized or no engine is available.
        """
        if self.engine == "pytesseract":
            return self._recognize_pytesseract(image_crop)
        if self.engine == "paddleocr":
            return self._recognize_paddleocr(image_crop)
        return []

    def _recognize_pytesseract(self, image_crop: np.ndarray) -> list[str]:
        """Recognize text with pytesseract (numpy BGR -> PIL -> string).

        Small crops are upscaled 4x and binarized (Otsu) first, since
        Tesseract performs poorly on low-resolution text.
        """
        try:
            from PIL import Image

            rgb = image_crop
            if image_crop.ndim == 3 and image_crop.shape[2] == 3:
                rgb = image_crop[:, :, ::-1]  # BGR -> RGB
            gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
            if max(gray.shape) < 400:  # upscale small crops for better OCR
                gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_image = Image.fromarray(binary)
            text = self._pytesseract.image_to_string(pil_image, lang="eng",
                                                     config="--psm 6")
        except Exception as exc:  # noqa: BLE001 - tesseract raises various errors
            print(f"[TextParser] pytesseract call failed: {exc}")
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _recognize_paddleocr(self, image_crop: np.ndarray) -> list[str]:
        """Recognize text with PaddleOCR (supports 2.x and 3.x API formats)."""
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

    # ------------------------------------------------------------------
    # Heuristic UML line parsing
    # ------------------------------------------------------------------
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
            A populated UmlClass; if no engine is available or no text
            was recognized, returns UmlClass(name="Unknown") with the
            given bbox.
        """
        if self.engine is None:
            print("[TextParser] No OCR engine -> UmlClass(name='Unknown')")
            return UmlClass(name="Unknown", bbox=bbox, confidence=0.0)

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
