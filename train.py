"""Train/evaluate multimodal and image-only ResNet50 models.

Example: python train.py --csv data/environment.csv --image-root data --epochs 25 --compare
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from dataset import FruitMultimodalDataset
from multimodal_model import DEFAULT_CLASSES, ImageOnlyResNet50, MultimodalResNet50


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(model, loader, device, detailed=False):
    model.eval(); truth, predicted, shelf_truth, shelf_pred = [], [], [], []
    with torch.no_grad():
        for images, env, labels, shelf, _ in loader:
            output = model(images.to(device), env.to(device))
            truth.extend(labels.numpy()); predicted.extend(output["logits"].argmax(1).cpu().numpy())
            shelf_truth.extend(shelf.numpy()); shelf_pred.extend(output["shelf_life"].cpu().numpy())
    report = {
        "accuracy": accuracy_score(truth, predicted),
        "macro_precision": precision_score(truth, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(truth, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "mae_days": mean_absolute_error(shelf_truth, shelf_pred),
        "rmse_days": mean_squared_error(shelf_truth, shelf_pred) ** .5,
    }
    if detailed:
        report["per_class"] = classification_report(truth, predicted, labels=list(range(len(DEFAULT_CLASSES))), target_names=DEFAULT_CLASSES, output_dict=True, zero_division=0)
        report["confusion_matrix"] = confusion_matrix(truth, predicted, labels=list(range(len(DEFAULT_CLASSES)))).tolist()
    return report


def snapshot(model):
    # CPU tensors must be cloned; otherwise optimizer updates mutate the saved best state.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one(args, image_only: bool):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_set = FruitMultimodalDataset(args.csv, "train", args.image_root)
    valid_set = FruitMultimodalDataset(args.csv, "valid", args.image_root, train_set.stats)
    test_set = FruitMultimodalDataset(args.csv, "test", args.image_root, train_set.stats)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    model = (ImageOnlyResNet50 if image_only else MultimodalResNet50)(len(DEFAULT_CLASSES), pretrained=not args.no_pretrained).to(device)
    if not image_only:
        for parameter in model.image_encoder[:-2].parameters(): parameter.requires_grad = False
        head_parameters = list(model.environment_encoder.parameters()) + list(model.classifier.parameters()) + list(model.shelf_life_regressor.parameters())
    else:
        for parameter in model.image_encoder[:-2].parameters(): parameter.requires_grad = False
        head_parameters = list(model.classifier.parameters()) + list(model.shelf_life_regressor.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder[-2:].parameters(), "lr": args.lr},
        {"params": head_parameters, "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    counts = train_set.rows.class_name.value_counts().reindex(DEFAULT_CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"Training data missing classes: {counts[counts == 0].index.tolist()}")
    weights = torch.tensor((len(train_set) / (len(DEFAULT_CLASSES) * counts)).to_numpy(), dtype=torch.float32, device=device)
    classification_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    regression_loss = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
    history, stale_epochs, best_epoch = [], 0, 0
    best_score, best_state = -float("inf"), None
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        # Frozen BatchNorm statistics must not drift with augmented training batches.
        model.image_encoder[:-2].eval()
        for images, env, labels, shelf, _ in train_loader:
            images, env, labels, shelf = images.to(device), env.to(device), labels.to(device), shelf.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, env)
            loss = classification_loss(output["logits"], labels) + args.regression_weight * regression_loss(output["shelf_life"], shelf)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check targets and learning rate")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); running += loss.item() * images.size(0)
        report = metrics(model, valid_loader, device)
        score = report["macro_f1"] - .01 * report["mae_days"]
        epoch_report = {"epoch": epoch + 1, "train_loss": running / len(train_set), **report}
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        scheduler.step(score)
        if score > best_score:
            best_score = score
            best_state = snapshot(model)
            best_epoch, stale_epochs = epoch + 1, 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    test_report = metrics(model, test_loader, device, detailed=True)
    name = "image_only" if image_only else "multimodal"
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    provenance = {"arguments": vars(args), "csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(), "split_sizes": {"train": len(train_set), "valid": len(valid_set), "test": len(test_set)}, "best_epoch": best_epoch}
    torch.save({"model_state": best_state, "classes": DEFAULT_CLASSES, "environment_stats": train_set.stats, "model_type": name, "metrics": test_report, "training": provenance}, output_dir / f"{name}.pt")
    (output_dir / f"{name}_training.json").write_text(json.dumps({**provenance, "history": history}, indent=2), encoding="utf-8")
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    return {key: value for key, value in test_report.items() if key not in {"per_class", "confusion_matrix"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True); parser.add_argument("--image-root")
    parser.add_argument("--output", default="artifacts"); parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--regression-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true"); parser.add_argument("--compare", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args(); seed_everything(args.seed)
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.workers < 0 or args.regression_weight < 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        parser.error("Invalid training parameters: counts/lr must be positive, weights/workers non-negative, smoothing in [0, 1)")
    results = {"multimodal": train_one(args, False)}
    if args.compare: results["image_only"] = train_one(args, True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
