"""Dataset and CSV validation for aligned image/environment/target records."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from multimodal_model import DEFAULT_CLASSES

REQUIRED_COLUMNS = {"image_path", "split", "class_name", "temperature_c", "humidity_pct", "shelf_life_days"}


def transforms_for(split: str):
    common = [v2.ToImage(), v2.Resize((256, 256))]
    if split == "train":
        common += [v2.RandomResizedCrop((224, 224), scale=(0.75, 1.0)), v2.RandomHorizontalFlip(), v2.RandomRotation(15), v2.ColorJitter(.2, .2, .1, .03)]
    else:
        common += [v2.CenterCrop((224, 224))]
    return v2.Compose(common + [v2.ToDtype(torch.float32, scale=True), v2.Normalize([.485, .456, .406], [.229, .224, .225])])


class FruitMultimodalDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, image_root: str | Path | None = None, stats: dict | None = None):
        self.csv_path = Path(csv_path)
        table = pd.read_csv(self.csv_path)
        missing = REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        table["split"] = table["split"].astype(str).str.strip().str.lower()
        if not set(table["split"]) <= {"train", "valid", "test"}:
            raise ValueError("split must be train, valid or test")
        root = Path(image_root) if image_root else self.csv_path.parent
        resolved = table.image_path.map(lambda value: str((root / str(value)).resolve()).casefold())
        if resolved.duplicated().any():
            raise ValueError("Duplicate image paths in CSV; each image must occur once")
        for column in ("sha256", "source_image_id"):
            if column in table and (table.dropna(subset=[column]).groupby(column)["split"].nunique() > 1).any():
                raise ValueError(f"Data leakage: {column} appears in multiple splits")
        if "batch_id" in table.columns:
            leaked = table.groupby("batch_id")["split"].nunique()
            if (leaked > 1).any():
                raise ValueError(f"Data leakage: batch_id values occur in multiple splits: {leaked[leaked > 1].index.tolist()}")
        self.rows = table.loc[table["split"].str.lower() == split.lower()].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows for split '{split}'")
        unknown = sorted(set(self.rows.class_name) - set(DEFAULT_CLASSES))
        if unknown:
            raise ValueError(f"Unknown class_name values: {unknown}. Expected the configured 24 classes.")
        numeric = self.rows[["temperature_c", "humidity_pct", "shelf_life_days"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("temperature_c, humidity_pct and shelf_life_days must be measured numeric values; missing targets are not imputed.")
        if not numeric.humidity_pct.between(0, 100).all() or not numeric.shelf_life_days.ge(0).all():
            raise ValueError("humidity_pct must be 0..100 and shelf_life_days must be non-negative.")
        self.rows[["temperature_c", "humidity_pct", "shelf_life_days"]] = numeric
        self.image_root = Path(image_root) if image_root else self.csv_path.parent
        self.transform = transforms_for(split)
        self.class_to_idx = {name: i for i, name in enumerate(DEFAULT_CLASSES)}
        if stats is None:
            if split.lower() != "train":
                raise ValueError("Validation/test require normalization statistics from training")
            env = self.rows[["temperature_c", "humidity_pct"]]
            mean, std = env.mean(), env.std().replace(0, 1).fillna(1)
            stats = {"mean": mean.tolist(), "std": std.tolist()}
        self.stats = stats
        if np.shape(stats.get("mean")) != (2,) or np.shape(stats.get("std")) != (2,) or not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std"]).all() or not (np.asarray(stats["std"]) > 0).all():
            raise ValueError("Environment statistics must contain two finite means and positive standard deviations")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = self.image_root / str(row.image_path)
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        raw_env = torch.tensor([row.temperature_c, row.humidity_pct], dtype=torch.float32)
        env = (raw_env - torch.tensor(self.stats["mean"])) / torch.tensor(self.stats["std"])
        return image_tensor, env, torch.tensor(self.class_to_idx[row.class_name]), torch.tensor(row.shelf_life_days, dtype=torch.float32), str(path)


def save_dataset_metadata(path: str | Path, stats: dict) -> None:
    Path(path).write_text(json.dumps({"classes": DEFAULT_CLASSES, "environment_stats": stats}, indent=2), encoding="utf-8")
