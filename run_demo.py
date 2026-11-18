"""End-to-end demo: run the UML recognition pipeline on data/raw images.

Usage:
    python run_demo.py

Searches for *.png / *.jpg in data/raw/. If none are found, generates a
synthetic test diagram (two class boxes User/Order connected by a line).
Prints the resulting JSON to the terminal and saves it to
data/raw/result.json together with a short summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from uml_recognition.pipeline import Pipeline

RAW_DIR = Path("data/raw")


def find_images() -> list[Path]:
    """Return all PNG/JPG images in data/raw/, sorted by name."""
    if not RAW_DIR.is_dir():
        return []
    return sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in RAW_DIR.glob(ext)])


def create_test_image(path: Path) -> Path:
    """Generate a synthetic 700x500 UML-like diagram with two classes.

    Draws white rectangles with black borders containing "User" /
    "+id: int" and "Order" / "+total: float", plus a connector line
    between them.

    Args:
        path: Destination PNG path.

    Returns:
        The written image path.
    """
    image = np.full((500, 700, 3), 255, dtype=np.uint8)
    # Class boxes (no internal compartment line so each box is one rectangle).
    for (x1, y1, x2, y2), name, attr in [
        ((50, 60, 250, 180), "User", "+id: int"),
        ((450, 260, 650, 380), "Order", "+total: float"),
    ]:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(image, name, (x1 + 20, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, attr, (x1 + 20, y1 + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Connector line between the boxes (kept short so it does not touch
    # the rectangles and merge their contours).
    cv2.line(image, (270, 140), (430, 300), (0, 0, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image), f"Failed to write {path}"
    print(f"[Demo] Generated test image: {path}")
    return path


def main() -> None:
    """Run the pipeline over every image in data/raw and save the JSON."""
    images = find_images()
    if not images:
        print("[Demo] No images in data/raw/ -> generating a test diagram")
        images = [create_test_image(RAW_DIR / "demo_diagram.png")]

    pipeline = Pipeline()
    all_results: list[dict] = []
    for image_path in images:
        print(f"\n[Demo] Processing: {image_path}")
        result = pipeline.run(str(image_path))
        all_results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    out_path = RAW_DIR / "result.json"
    payload = all_results[0] if len(all_results) == 1 else all_results
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total_classes = sum(len(r["classes"]) for r in all_results)
    total_relations = sum(len(r["relations"]) for r in all_results)
    total_ms = sum(r["metadata"]["processing_time_ms"] for r in all_results)
    print(f"\n[Demo] Summary: {len(images)} image(s), {total_classes} classes, "
          f"{total_relations} relations, {total_ms} ms total")
    print(f"[Demo] JSON saved to {out_path}")


if __name__ == "__main__":
    main()
