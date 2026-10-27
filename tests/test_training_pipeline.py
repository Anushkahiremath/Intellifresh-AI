import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from PIL import Image

from dataset import FruitMultimodalDataset
from prepare_dataset import audit, folder_label
from prepare_freshness import source_group, prepare
from train import snapshot
from train_freshness import run


class TinyMobileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(), torch.nn.Identity(), torch.nn.Conv2d(4, 4, 1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Identity())
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(4, 2))

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


class TrainingPipelineTests(unittest.TestCase):
    def test_preparation_quarantines_conflicting_labels_and_prevents_group_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ("Fresh", "Rotten"):
                images = root / "source" / f"Apple_{label}"
                images.mkdir(parents=True)
                for index in range(60):
                    Image.new("RGB", (8, 8), (index, 80 if label == "Fresh" else 160, 20)).save(images / f"sample_{index}.png")
            for label in ("Fresh", "Rotten"):
                Image.new("RGB", (8, 8), "blue").save(root / "source" / f"Apple_{label}" / "conflict.png")
            with patch("builtins.print"):
                report = prepare(root / "source", root / "output", target=100)
            manifest = pd.read_csv(root / "output/freshness.csv")
            self.assertEqual(report["conflicting_unique_images_quarantined"], 1)
            self.assertEqual(len(manifest), 100)
            self.assertFalse(manifest.image_path.str.endswith("conflict.png").any())
            self.assertTrue((manifest.groupby("source_group").split.nunique() == 1).all())
            self.assertEqual(set(manifest.split), {"train", "valid", "test"})
            self.assertFalse(manifest.sha256.duplicated().any())

    def test_freshness_training_exports_compatible_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for split in ("train", "valid", "test"):
                for label in (0, 1):
                    path = root / f"{split}_{label}.png"
                    Image.new("RGB", (8, 8), "red" if label else "green").save(path)
                    # Test loader/optimization wiring; these synthetic fixtures are not accuracy evidence.
                    rows.append(dict(image_path=str(path), label=label, split=split, sha256=path.stem, source_group=path.stem))
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            initial = root / "initial.pth"
            torch.save(TinyMobileNet().state_dict(), initial)
            args = Namespace(csv=str(manifest), initial_weights=str(initial), output=str(root / "output"), seed=42, cpu=True, cpu_threads=2, batch_size=2, workers=0, lr=1e-4, epochs=1, patience=1)
            with patch("train_freshness.models.mobilenet_v2", return_value=TinyMobileNet()), patch("builtins.print"):
                run(args)
            candidate = root / "output/freshness_candidate.pth"
            TinyMobileNet().load_state_dict(torch.load(candidate, weights_only=True))
            import json
            report = json.loads((root / "output/freshness_training.json").read_text())
            self.assertEqual(report["labels"], ["Fresh", "Stale"])
            self.assertEqual(len(report["candidate_test"]["confusion_matrix"]), 2)
            self.assertEqual(report["split_counts"], {"train": 2, "valid": 2, "test": 2})

    def test_best_checkpoint_is_independent_on_cpu(self):
        model = torch.nn.Linear(2, 1)
        state = snapshot(model)
        before = state["weight"].clone()
        with torch.no_grad():
            model.weight.add_(10)
        self.assertTrue(torch.equal(state["weight"], before))
        self.assertFalse(torch.equal(state["weight"], model.weight))

    def test_audit_deduplicates_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images" / "apple_fresh"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(images / "a.png")
            Image.new("RGB", (8, 8), "red").save(images / "b.bmp")
            (images / "invalid.jpg").write_text("invalid")
            report = audit([root / "images"], root / "out")
            self.assertEqual(report["unique_images"], 1)
            self.assertEqual(report["duplicate_copies"], 1)
            self.assertEqual(report["invalid_images"], 1)
            self.assertEqual(report["class_counts"], {"apple_fresh": 1})

    def test_no_fruit_label_guessed_from_filename(self):
        self.assertEqual(folder_label(Path("train/rotten/apple_photo.jpg")), "")

    def test_capture_sessions_stay_together_across_conditions(self):
        self.assertEqual(source_group(Path("20190809_115439.jpg"), "apple_fresh"), source_group(Path("20190809_125439.jpg"), "apple_rotten"))
        self.assertEqual(source_group(Path("rotated_by_15_Screen Shot 2018.jpg"), "apple_fresh"), source_group(Path("Screen Shot 2018.jpg"), "apple_fresh"))

    def test_invalid_and_leaking_multimodal_records_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            rows = pd.DataFrame([
                dict(image_path="a.png", split="train", class_name="apple_fresh", temperature_c=20, humidity_pct=65, shelf_life_days=5, batch_id="same"),
                dict(image_path="b.png", split="valid", class_name="apple_fresh", temperature_c=20, humidity_pct=65, shelf_life_days=5, batch_id="same"),
            ])
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                FruitMultimodalDataset(path, "train")
            rows.loc[1, "batch_id"] = "different"
            rows["temperature_c"] = rows["temperature_c"].astype(float)
            rows.loc[0, "temperature_c"] = float("inf")
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "numeric"):
                FruitMultimodalDataset(path, "train")
            rows.loc[0, "temperature_c"] = 20
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "statistics from training"):
                FruitMultimodalDataset(path, "valid")


if __name__ == "__main__":
    unittest.main()
