"""Fine-tune the app's binary MobileNetV2 on the prepared image-only corpus.

Writes a candidate checkpoint; never replaces the deployed model automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train import seed_everything, snapshot


def image_transform(training=False):
    operations = [transforms.Resize((224, 224))]
    if training:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), transforms.ColorJitter(.15, .15, .1, .02)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class FreshnessImages(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows.reset_index(drop=True)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(row.label)


def evaluate(model, loader, device, callback=None):
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(loader):
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
            if callback and batch % 25 == 0:
                callback(batch + 1, len(loader))
    return {"accuracy": accuracy_score(truth, predicted), "macro_f1": f1_score(truth, predicted, average="macro"),
            "per_class": classification_report(truth, predicted, labels=[0, 1], target_names=["Fresh", "Stale"], output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    def progress(stage, **details):
        from datetime import datetime, timezone
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps({"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details}, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")
    table = pd.read_csv(args.csv)
    required = {"image_path", "label", "split", "sha256", "source_group"}
    if not required <= set(table):
        raise ValueError(f"Manifest missing {sorted(required - set(table))}")
    if table[list(required)].isna().any().any() or set(table.label) != {0, 1} or set(table.split) != {"train", "valid", "test"}:
        raise ValueError("Manifest needs complete labels 0/1 and train/valid/test partitions")
    if table.image_path.duplicated().any() or table.sha256.duplicated().any():
        raise ValueError("Manifest contains duplicate paths or images")
    if (table.groupby("source_group").split.nunique() > 1).any():
        raise ValueError("Capture/augmentation groups leak across splits")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = {}
    for split in ("train", "valid", "test"):
        rows = table[table.split == split]
        if set(rows.label) != {0, 1}:
            raise ValueError(f"Both labels required in {split}")
        loaders[split] = DataLoader(FreshnessImages(rows, split == "train"), batch_size=args.batch_size,
                                    shuffle=split == "train", num_workers=args.workers, pin_memory=device.type == "cuda")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(model.classifier[1].in_features, 2))
    model.load_state_dict(torch.load(args.initial_weights, map_location="cpu", weights_only=True))
    model.to(device)
    # Validation baseline is available for monitoring; test is reserved until selection.
    progress("baseline_validation", device=str(device), split_counts=table.split.value_counts().to_dict())
    baseline_validation = evaluate(model, loaders["valid"], device, lambda batch, batches: progress("baseline_validation", batch=batch, batches=batches, device=str(device)))
    (output / "baseline_validation.json").write_text(json.dumps(baseline_validation, indent=2), encoding="utf-8")
    for parameter in model.features[:-4].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.features[-4:].parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 10}], weight_decay=1e-4)
    counts = table[table.split == "train"].label.value_counts().sort_index()
    weights = torch.tensor((counts.sum() / (2 * counts)).to_numpy(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    best_score, best_state, stale, history, best_epoch = baseline_validation["macro_f1"], snapshot(model), 0, [], 0
    for epoch in range(args.epochs):
        progress("training", epoch=epoch + 1, max_epochs=args.epochs, device=str(device))
        model.train()
        model.features[:-4].eval()
        loss_sum = 0.0
        for batch, (images, labels) in enumerate(loaders["train"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item() * len(images)
            if batch % 100 == 0:
                progress("training", epoch=epoch + 1, batch=batch + 1, batches=len(loaders["train"]), device=str(device))
                print(f"Epoch {epoch + 1}, batch {batch + 1}/{len(loaders['train'])}", flush=True)
        validation = evaluate(model, loaders["valid"], device, lambda batch, batches: progress("validation", epoch=epoch + 1, batch=batch, batches=batches, device=str(device)))
        score = validation["macro_f1"]
        scheduler.step(score)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / len(loaders["train"].dataset), "validation_accuracy": validation["accuracy"], "validation_macro_f1": score}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score, best_state, stale, best_epoch = score, snapshot(model), 0, epoch + 1
            torch.save(best_state, output / "best_validation.pth")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.patience:
            break
    progress("test_evaluation", best_epoch=best_epoch, device=str(device))
    model.load_state_dict(best_state)
    candidate = evaluate(model, loaders["test"], device, lambda batch, batches: progress("candidate_test", batch=batch, batches=batches, device=str(device)))
    model.load_state_dict(torch.load(args.initial_weights, map_location=device, weights_only=True))
    baseline = evaluate(model, loaders["test"], device, lambda batch, batches: progress("baseline_test", batch=batch, batches=batches, device=str(device)))
    torch.save(best_state, output / "freshness_candidate.pth")
    report = {"baseline_test": baseline, "candidate_test": candidate, "best_epoch": best_epoch,
              "test_accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
              "test_macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
              "history": history, "arguments": vars(args), "labels": ["Fresh", "Stale"],
              "csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              "split_counts": table.split.value_counts().to_dict(),
              "limitation": "Public data may overlap the initial checkpoint's training corpus. Confirm gains on independently collected images before claiming generalization."}
    (output / "freshness_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress("complete", best_epoch=best_epoch, candidate_accuracy=candidate["accuracy"], baseline_accuracy=baseline["accuracy"])
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/freshness/freshness.csv")
    parser.add_argument("--initial-weights", default="models/freshness_mobilenetv2_state_dict.pth")
    parser.add_argument("--output", default="artifacts/freshness")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.cpu_threads) < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("Invalid training parameters")
    try:
        run(args)
    except Exception as error:
        from datetime import datetime, timezone
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "progress.json").write_text(json.dumps({"stage": "failed", "error": str(error), "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        raise
