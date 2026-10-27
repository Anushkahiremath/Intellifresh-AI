"""Build a reproducible, deduplicated 40k-50k binary freshness manifest.

Uses folder labels from Freshness44. Preserves identifiable capture/augmentation
families in one split. Unknown provenance still requires a near-duplicate review.
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

VARIANT_PREFIX = re.compile(r"^(?:(?:rotated_by_\d+|saltandpepper|translation|vertical_flip|horizontal_flip)_)+", re.I)


def image_digest(path):
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(), None
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return None, str(error)


def source_group(path, category):
    stem = path.stem.lower().split(".rf.")[0]
    stem = VARIANT_PREFIX.sub("", stem)
    stem = re.sub(r"(?:[_-](?:aug|flip|rotate|rot|copy|zoom|shift)\w*)+$", "", stem)
    # Timestamped pictures from the same day/category are one capture session.
    match = re.match(r"(?:img)?(20\d{6})[_-]?\d{6}", stem)
    return category.rsplit("_", 1)[0] + ":" + (match.group(1) if match else stem)


def prepare(root, output, target=45000, seed=42):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, rejected, seen = [], [], {}
    conflicts, conflicting_hashes = [], set()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    print(f"Verifying {len(paths)} files with four image-decoding workers", flush=True)
    pool = ThreadPoolExecutor(max_workers=4)
    digests = pool.map(image_digest, paths)
    for index, (path, (digest, error)) in enumerate(zip(paths, digests)):
        if (index + 1) % 1000 == 0:
            print(f"Audited {index + 1}/{len(paths)} images", flush=True)
        category = path.parent.name.lower()
        if category.endswith("_fresh"):
            label = 0
        elif category.endswith(("_rotten", "_stale")):
            label = 1
        else:
            rejected.append({"path": str(path), "reason": "Unrecognized class directory"})
            continue
        if error:
            rejected.append({"path": str(path), "reason": str(error)})
            continue
        if digest in seen:
            if seen[digest]["label"] != label:
                conflicting_hashes.add(digest)
                conflicts.append({"sha256": digest, "image_path": str(path), "conflicts_with": seen[digest]["image_path"]})
            continue
        row = {"image_path": str(path), "label": label, "category": category, "sha256": digest,
               "known_source_augmentation": bool(VARIANT_PREFIX.match(path.stem)),
               "source_group": source_group(path, category)}
        seen[digest] = row
        rows.append(row)
    pool.shutdown()
    # Exclude every copy of ambiguous images; never guess which annotation is correct.
    rows = [row for row in rows if row["sha256"] not in conflicting_hashes]
    (output / "label_conflicts.json").write_text(json.dumps(conflicts, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No labeled images found")
    # Ten group-stratified folds give approximately 80/10/10; large groups affect ratios.
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    table["split"] = "train"
    for fold, (_, held) in enumerate(splitter.split(table, table.category, table.source_group)):
        if fold == 0:
            table.loc[held, "split"] = "test"
        elif fold == 1:
            table.loc[held, "split"] = "valid"
    # Known pre-generated augmentations belong only in training, never evaluation.
    excluded_variants = (table.known_source_augmentation & (table.split != "train")).sum()
    table = table[~(table.known_source_augmentation & (table.split != "train"))].reset_index(drop=True)
    # Sample proportionally by source category and split, without replacement.
    if len(table) > target:
        rng = np.random.default_rng(seed)
        strata = table.groupby(["category", "split"]).size()
        allocations = strata * target / len(table)
        quotas = allocations.astype(int)
        for key in (allocations - quotas).sort_values(ascending=False).index[:target - quotas.sum()]:
            quotas[key] += 1
        table = pd.concat([part.iloc[rng.choice(len(part), int(quotas[name]), replace=False)] for name, part in table.groupby(["category", "split"])]).reset_index(drop=True)
    if set(table.label) != {0, 1} or any(set(part.label) != {0, 1} for _, part in table.groupby("split")):
        raise ValueError("Both freshness classes must be present in each split")
    table.to_csv(output / "freshness.csv", index=False)
    report = {"source": "https://www.kaggle.com/datasets/siavash93/freshness44", "seed": seed,
              "files_scanned": len(paths), "unique_decoded_images": len(seen), "unambiguous_unique_images": len(rows), "duplicates_removed": len(paths) - len(rejected) - len(seen),
              "conflicting_unique_images_quarantined": len(conflicting_hashes), "invalid_or_unrecognized_images": len(rejected),
              "selected_images": len(table), "target_images": target, "target_met": 40000 <= len(table) <= 50000,
              "known_source_augmentations_in_training": int(table.known_source_augmentation.sum()),
              "known_augmentations_excluded_from_evaluation": int(excluded_variants),
              "selected_source_groups": int(table.source_group.nunique()),
              "split_counts": table.split.value_counts().to_dict(), "class_counts": table.category.value_counts().to_dict(),
              "labels": {"0": "Fresh", "1": "Stale"},
              "limitations": "Count is decoded-distinct labeled files, not independent fruit specimens. Source includes augmentations; recognizable variants are training-only. Exact duplicates removed and known filename families grouped. Unknown near duplicates/capture identities need review. Sources may overlap pretrained model data. No sensor/shelf-life annotations."}
    (output / "freshness_dataset.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "rejected_images.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="raw_data/freshness44")
    parser.add_argument("--output", default="data/freshness")
    parser.add_argument("--target", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 40000 <= args.target <= 50000:
        parser.error("target must be between 40000 and 50000 unique images")
    print(json.dumps(prepare(args.root, args.output, args.target, args.seed), indent=2))
