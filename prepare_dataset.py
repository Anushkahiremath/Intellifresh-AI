"""Audit image corpora without inventing labels or counting duplicate copies.

Produces a deduplicated annotation inventory, not a fabricated training dataset.
Existing split folders are preserved. Review source/batch groups before training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

FRUITS = "apple banana cherry grape kiwi mango orange peach pear plum strawberry watermelon".split()
CLASSES = [f"{fruit}_{condition}" for fruit in FRUITS for condition in ("fresh", "rotten")]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def folder_label(path):
    # Never infer fruit type from image filenames or a generic fresh/rotten folder.
    for part in reversed(path.parent.parts):
        normalized = "".join(c for c in part.lower() if c.isalpha())
        for fruit in FRUITS:
            for condition in ("fresh", "rotten"):
                if normalized in {fruit + condition, condition + fruit}:
                    return f"{fruit}_{condition}"
    return ""


def audit(roots, output, target=45000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    unique, rejected, duplicates, seen_paths = {}, [], [], set()
    files = 0
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            files += 1
            try:
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.load()
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                rejected.append({"image_path": str(path), "reason": str(error)})
                continue
            label = folder_label(path)
            split = next((part.lower() for part in reversed(path.parent.parts) if part.lower() in {"train", "valid", "test"}), "")
            if digest in unique:
                previous = unique[digest]
                duplicates.append({"image_path": str(path), "duplicate_of": previous["image_path"], "split": split, "retained_split": previous["split"], "label_conflict": bool(label and previous["class_name"] and label != previous["class_name"])})
                continue
            unique[digest] = {"image_path": str(path), "split": split, "class_name": label,
                              "temperature_c": "", "humidity_pct": "", "shelf_life_days": "",
                              "batch_id": "", "source_image_id": "", "sha256": digest,
                              "source": str(root)}
    columns = ["image_path", "split", "class_name", "temperature_c", "humidity_pct", "shelf_life_days", "batch_id", "source_image_id", "sha256", "source"]
    with (output / "annotation_inventory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(unique.values())
    counts = Counter(row["class_name"] or "unlabeled" for row in unique.values())
    report = {"files_scanned": files, "unique_images": len(unique), "duplicate_copies": len(duplicates),
              "invalid_images": len(rejected), "target_unique_images": target,
              "additional_unique_images_needed": max(0, target - len(unique)),
              "class_counts": dict(counts), "missing_classes": [name for name in CLASSES if not counts[name]],
              "multimodal_ready": False,
              "note": "Inventory needs verified class labels, measured environmental/shelf-life targets and reviewed split groups. Pixel hashes detect exact decoded duplicates, not near duplicates."}
    for name, value in (("dataset_audit.json", report), ("duplicates.json", duplicates), ("invalid_images.json", rejected)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="data/audit")
    parser.add_argument("--target", type=int, default=45000)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("target must be positive")
    print(json.dumps(audit(args.roots, args.output, args.target), indent=2))
