"""Text recognition and parsing stage: pytesseract (primary) + PaddleOCR fallback.

Reads the text inside a detected class rectangle and splits it into
class name, attributes ("+id: int") and methods ("+login(): bool").

Engine selection order:
    1. pytesseract (Tesseract binary from TESSERACT_CMD env var or PATH).
    2. PaddleOCR (optional dependency: ``pip install uml-recognition[paddle]``).
    3. None -> parse() returns UmlClass(name="Unknown").
"""

from __future__ import annotations

import re

import cv2
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

        Args:
            image_crop: BGR crop of one detected class box.
            bbox: Bounding box the crop was taken from.

        Returns:
            A populated UmlClass; if no engine is available or no text
            was recognized, returns UmlClass(name="Unknown") with the
            given bbox.
        """
        if self.engine is None:
            print("[TextParser] No OCR engine -> UmlClass(name='Unknown')")
            return UmlClass(name="Unknown", bbox=bbox, confidence=0.0)

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
