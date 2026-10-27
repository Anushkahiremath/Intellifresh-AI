# IntelliFresh dataset and model training

## What changed

The original local corpus contains **359 unique images** and a second identical copy. The example environmental CSV is a schema example, not a measured training dataset. A new image-only pipeline downloads Freshness44, removes exact decoded-image duplicates, and selects up to **45,000 decoded-distinct labeled image files** without replacement. The completed October 14, 2026 preparation selected **45,000 files**: **38,587 train / 3,105 validation / 3,308 test**. Of these, 6,177 are recognizable source augmentations, restricted to training. Two decoded images with conflicting labels were excluded in all copies. Actual counts are recorded in `data/freshness/freshness_dataset.json`.

Source: [Freshness44, version 1](https://www.kaggle.com/datasets/siavash93/freshness44). Keep the source attribution. Kaggle reports its license as unknown; verify redistribution rights before packaging the images for publication. The images have freshness/category labels, not measured environmental or shelf-life labels.

## Build the larger dataset

From the project directory with requirements installed:

```powershell
python download_dataset.py
python prepare_freshness.py --target 45000
```

The downloaded version contains 53,616 labeled image files before auditing and filtering. The download is approximately 7.2 GB compressed and extraction needs additional disk space. Existing original images are preserved. Preparation reads the actual class folders, rejects unreadable images, removes identical decoded images, and quarantines every copy of images with conflicting freshness labels and writes `label_conflicts.json`. It selects proportionally by source category and writes absolute image paths to `data/freshness/freshness.csv`. Regenerate the CSV after moving the dataset to another machine.

Known augmentation filenames and timestamped capture sessions are grouped before an approximately 80/10/10 train/validation/test split. Actual ratios are in the report because group sizes differ. These checks do not establish independence for undocumented capture sessions or all near duplicates. Review visually similar images and record acquisition IDs for a publication-quality test set. The source already contains augmentations. Recognizable rotated/flipped/noisy variants are grouped with their originals and retained only in training; their count is reported separately. The 45,000-image target means decoded-distinct labeled files, not 45,000 independent fruit specimens. Copied files are removed, and no extra images are generated to inflate the count.

Freshness44 includes vegetables and fruit types outside the research model's 12-fruit taxonomy. The 45k target applies to the **binary visual Fresh/Stale task**, not 45k measured records across the research model's 24 classes.

## Train the model used by the app

```powershell
python train_freshness.py --epochs 20 --batch-size 32
```

1. Load the app's existing MobileNetV2 weights. Label 0 means Fresh and label 1 means Stale.
2. Resize RGB images to 224 by 224 and normalize with ImageNet statistics, matching deployed preprocessing. Apply modest flips, rotation and colour variation only to training images.
3. Freeze early image features and their batch-normalization statistics. Train the last four feature blocks and classifier using mini-batch backpropagation.
4. Minimize class-weighted cross-entropy with 0.05 label smoothing. AdamW uses learning rates of 0.00001 for feature blocks and 0.0001 for the classifier, weight decay of 0.0001, and gradient clipping.
5. Measure macro-F1 on validation images each epoch. Reduce learning rate after a plateau and stop after five epochs without improvement. Keep an independent copy of the best weights. The initial model remains the selected candidate if fine-tuning never improves validation macro-F1.
6. After checkpoint selection, evaluate both the initial model and selected candidate on the same held-out test images. Export accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, learning history and dataset fingerprint.

Outputs are `artifacts/freshness/freshness_candidate.pth` and `artifacts/freshness/freshness_training.json`. CPU training works but can take many hours for this corpus; a CUDA GPU is recommended. The script automatically uses CUDA when available.

The downloaded corpus may overlap data used to train the initial third-party checkpoint. A comparison on this public split is useful for development but cannot prove improvement on unseen real-world produce. Collect a separate camera/session test set for that claim. Prediction confidence is not accuracy.

After reviewing a completed evaluation, activate a candidate explicitly and restart either app:

```powershell
$env:FRESHNESS_MODEL_PATH = "artifacts/freshness/freshness_candidate.pth"
streamlit run streamlit_app.py
# or: python app.py
```

No downloaded model is silently replaced. Existing coverage indicators remain conservative until each produce category is validated.

## Train the separate multimodal research model

The ResNet50 model needs verified `class_name`, measured `temperature_c`, `humidity_pct`, and observed `shelf_life_days` for every image. Shelf life must be established from follow-up observations using a consistent endpoint. Do not randomly generate those values and report the resulting regression as real-world performance.

```powershell
python prepare_dataset.py --roots "Quality Dataset" raw_data/corpus_b --output data/audit
# Complete verified annotations in a separate data/environment.csv.
python train.py --csv data/environment.csv --image-root data --epochs 25 --compare
```

Use `data/environment_template.csv` for the CSV schema only. Preserve whole fruit/batch/capture groups across splits; require all 24 classes in training. Include the audited pixel `sha256` and acquisition `source_image_id` where available. Validation/test reuse training environmental normalization statistics.

ResNet50 creates 2,048 image features; an MLP turns two environmental inputs into 32 features. Concatenation produces 2,080 features. A classifier predicts 24 fruit-condition classes, and a regression head estimates non-negative remaining days. Training minimizes class-weighted smoothed cross-entropy plus 0.5 times shelf-life mean-squared error. AdamW, learning-rate reduction, early stopping, gradient clipping and fixed frozen-layer statistics improve training control. Checkpoints are selected using validation macro-F1 minus 0.01 times shelf-life MAE. Both experimental runs reset the same seed and use the same partitions.

`artifacts/multimodal.pt` is loaded by the Streamlit research workflow on restart. Its class order and normalization statistics are stored with the weights. Reports contain held-out metrics, per-class results, confusion matrices, training history, best epoch, parameters and the CSV SHA-256 fingerprint.

## Verify the implementation

```powershell
python -m unittest discover -s tests -v
```

Tests check independent CPU checkpoint snapshots, decoded duplicate detection, corrupt-image handling, label inference boundaries, capture grouping, leakage rejection, conflicting-label quarantine, manifest generation, a complete small training run and invalid environmental values. Passing these tests verifies behavior, not model accuracy.

Training progress is recorded in `artifacts/freshness/progress.json` and shown in the application. A progress file is a last-recorded status, not proof that the process is still alive. The final measured results appear only in `freshness_training.json`.
