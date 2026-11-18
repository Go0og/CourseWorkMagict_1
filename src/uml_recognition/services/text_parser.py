"""Text parsing stage: pytesseract OCR with preprocessing."""
from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from uml_recognition.models import Attribute, BBox, Method, Parameter, UmlClass, Visibility

_VISIBILITY_BY_SYMBOL: dict[str, Visibility] = {
    "+": Visibility.PUBLIC,
    "-": Visibility.PRIVATE,
    "#": Visibility.PROTECTED,
    "~": Visibility.PACKAGE,
}

# Подписи связей, которые НЕ должны становиться именами классов
_RELATION_LABELS = {
    "places", "contains", "uses", "extends", "implements",
    "depends", "includes", "orders", "orderstatus",
}


class TextParser:
    """Parses OCR text of a class crop into a UmlClass model."""

    def __init__(self) -> None:
        print("[TextParser] Initializing pytesseract...")
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        print("[TextParser] pytesseract ready")

    def parse(self, image_crop: np.ndarray, bbox: BBox) -> UmlClass:
        """Recognize text in a crop and build a UmlClass."""
        print(f"[TextParser] Parsing crop shape={image_crop.shape}")
        lines = self._recognize_lines(image_crop)
        return self._lines_to_class(lines, bbox)

    def _recognize_lines(self, image_crop: np.ndarray) -> list[str]:
        """Run OCR with preprocessing and return recognized text lines."""
        crop = np.asarray(image_crop)
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # 1. Otsu-бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Upscale 4x для мелкого текста
        binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # 3. Обрезка пустот
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            binary = binary[y:y + h, x:x + w]

        try:
            pil_image = Image.fromarray(binary)
            # PSM 6 = блок текста (работает лучше для многострочных кропов)
            raw_text = pytesseract.image_to_string(
                pil_image, lang="eng+rus", config="--psm 6"
            )
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            print(f"[TextParser] OCR found {len(lines)} lines: {lines[:5]}")
            return lines
        except Exception as exc:
            print(f"[TextParser] OCR failed: {exc}")
            return []

    def _lines_to_class(self, lines: list[str], bbox: BBox) -> UmlClass:
        """Convert raw OCR lines into a UmlClass via simple heuristics."""
        name = "Unknown"
        attributes: list[Attribute] = []
        methods: list[Method] = []

        for line in lines:
            visibility = _VISIBILITY_BY_SYMBOL.get(line[0]) if line else None
            body = line[1:].strip() if visibility else ""

            if visibility and body:
                if "(" in body:
                    methods.append(self._parse_method(body, visibility))
                elif ":" in body:
                    attr = self._parse_attribute(body, visibility)
                    if attr is not None:
                        attributes.append(attr)
                continue

            # Строка без модификатора доступа — кандидат на имя класса
            if name == "Unknown" and line.strip():
                candidate = self._extract_class_name(line)
                if candidate:
                    name = candidate
                    print(f"[TextParser] Detected class name: '{name}'")

        # FALLBACK 1: если имя = "C" (иконка PlantUML), берём следующую строку
        if name == "C" and len(lines) > 1:
            fallback = self._extract_class_name(lines[1])
            if fallback:
                name = fallback
                print(f"[TextParser] Fallback from 'C' to second line: '{name}'")
        
        # FALLBACK 2: если имя всё ещё Unknown или "C", но есть атрибуты/методы —
        # берём ПЕРВУЮ строку и чистим от мусора
        if name in ("Unknown", "C") and (attributes or methods) and lines:
            fallback = self._extract_class_name(lines[0])
            if fallback and fallback != "C":
                name = fallback
                print(f"[TextParser] Fallback class name from first line: '{name}'")

        return UmlClass(name=name, attributes=attributes, methods=methods, bbox=bbox)

    @staticmethod
    def _extract_class_name(raw_line: str) -> str | None:
        """Extract a class name from a raw OCR line.

        Handles PlantUML artifacts like 'C User', '(c) User', '© User', etc.
        Returns None if the line looks like a relation label or pure garbage.
        """
        if not raw_line or not raw_line.strip():
            return None

        # Убираем всё, что не буква/цифра/подчёркивание/кириллица в начале
        cleaned = re.sub(r"^[^A-Za-zА-Яа-яЁё0-9]*", "", raw_line).strip()

        # Берём первое "слово" (до пробела или конца строки)
        first_word = cleaned.split()[0] if cleaned else ""

        # Убираем trailing punctuation
        first_word = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_]+$", "", first_word)

        if not first_word:
            return None

        # Отсекаем подписи связей (places, contains, etc.)
        if first_word.lower() in _RELATION_LABELS:
            return None

        # Отсекаем чистые цифры и мусор
        if re.fullmatch(r"\d+", first_word):
            return None

        # Имя класса НЕ должно содержать ":" или "()" — это атрибут/метод
        if ":" in first_word or "(" in first_word or ")" in first_word:
            return None

        # Имя класса должно быть >= 2 символов (отсекаем "C", "S", "id")
        if len(first_word) < 2:
            return None

        return first_word

    @staticmethod
    def _parse_attribute(body: str, visibility: Visibility) -> Attribute | None:
        """Parse 'name: type' into an Attribute."""
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
            parameters=[
                Parameter(name=p_name, type_str=p_type)
                for p_name, p_type in parameters
            ],
            return_type=return_type,
            visibility=visibility,
        )
