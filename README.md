# IntelliFresh AI

IntelliFresh AI is a Streamlit app for produce recognition, freshness review, fruit counting, batch checks, and model evaluation. Run it from the project root:

```powershell
streamlit run streamlit_app.py
```

## What the app does

- **Inspect produce:** upload or capture a photo, review the available freshness estimate, handling guidance, and downloadable report.
- **Choose the inspection model:** use MobileNetV2 for freshness-only output, the dual MobileNetV2 + EfficientNet-B0 workflow, or the IntelliFresh multimodal checkpoint when installed. The freshness-only mode does not infer produce identity or shelf life.
- **Recognize produce:** identify one item from the 76-label produce classifier.
- **Detect and count fruit:** find visible apples, bananas, and oranges in one image, show boxes and per-item confidence, summarize counts, and download the detection list.
- **Review batches:** analyze multiple uploaded images and export the results.
- **Review model performance:** display saved validation metrics and training history when the evaluation JSON files are present.
- **Capture with a camera:** use a connected device camera from a supported browser.

The fruit detector is a separate, pretrained COCO object-detection model. Its supported fruit labels are apple, banana, and orange; it does not count every produce type. Occlusion, overlap, image quality, and the confidence threshold affect results. A model confidence score is not a measure of model accuracy.

Freshness and shelf-life outputs are estimates, not food-safety certification. Always inspect produce and follow established food-handling guidance. When a trained model cannot be loaded, the app clearly labels its visual-only diagnostic and does not present it as a trained freshness classification.

## Run locally

Python 3.11 is recommended. From PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The repository includes the MobileNetV2 freshness weights, EfficientNet-B0 produce weights, and the TorchVision fruit detector weights. If either classifier checkpoint is missing, the app downloads it from its public Hugging Face model repository; if the detector checkpoint is missing, TorchVision downloads it when fruit counting is first used. Internet access is therefore only needed for a missing model or optional online service.

The upload limit is 10 MB per image, as configured in `.streamlit/config.toml`. The app uses CPU inference by default when CUDA is unavailable; CPU startup and analysis can take longer.

### Optional Gemini guidance

Gemini is optional; the application has a local guidance fallback. For local use, add your key to `.env` (which is excluded from Git):

```dotenv
GEMINI_API_KEY=your_key_here
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_key_here"
```

Never commit `.env` or `.streamlit/secrets.toml`. The app does not display the key.

## Deploy to Streamlit Community Cloud

The GitHub repository is public and the app entry point is `streamlit_app.py`. To create the hosted app:

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that owns or administers [this repository](https://github.com/Anushkahiremath/Intellifresh-AI).
2. Select **Create app** → **Yup, I have an app**.
3. Choose repository `Anushkahiremath/Intellifresh-AI`, branch `main`, and main file path `streamlit_app.py`. Optionally choose an available custom `streamlit.app` subdomain.
4. Open **Advanced settings**. Select **Python 3.11** and, only if you want Gemini-powered guidance, enter the optional secret:

   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

   Leave Secrets empty to use the local guidance fallback. Never enter a key in a source file, GitHub issue, or committed `.env`.
5. Select **Save** → **Deploy**. The first build installs the requirements and downloads the repository, including its model files. Allow several minutes for the initial build and model initialization.
6. When the app reports that it is running, verify image upload, produce recognition, fruit counting, batch review, and report downloads using the generated `streamlit.app` URL. Use the Cloud app's **Manage app** logs if build or startup fails.

The committed model weights mean normal startup does not need to fetch classifier or detector checkpoints from Hugging Face or PyTorch. The repository is about 100 MB; the 74 MB detector weight exceeds GitHub's 50 MB recommendation (GitHub warns, but accepts it below the 100 MB per-file limit), so a fresh clone/build takes longer. CPU inference is supported. If the host runs out of memory or terminates the app, use a host with more memory or move inference to a separately hosted model service.

I cannot create the Cloud app on your behalf from this workspace: Streamlit Community Cloud requires an interactive sign-in, an owner/admin-authorized GitHub connection, and acceptance of its deployment settings. The steps above are the remaining one-time account actions.

## Model and evaluation files

- `models/` includes the MobileNetV2 freshness, EfficientNet-B0 produce-recognition, and Faster R-CNN MobileNetV3 fruit-detection weights, plus the produce label configuration. If present, `FRESHNESS_MODEL_PATH` can point to an alternate freshness checkpoint.
- The custom IntelliFresh multimodal option requires its trained checkpoint. Set `MODEL_CHECKPOINT` to the checkpoint path in the deployment environment; the option is shown as unavailable when the checkpoint is not installed.
- `artifacts/freshness/baseline_validation.json` and `artifacts/freshness/history.json` provide the saved validation metrics and training history used in the performance view. Keep these small JSON files in the deployment repository if you want the evaluation cards and charts to appear. Training checkpoints, logs, databases, uploads, datasets, and other local artifacts are excluded by `.gitignore`.
- `download_model.py` can also be used to download the two Hugging Face image-classification models before deployment.
- `TRAINING_GUIDE.md` documents dataset preparation, training, and evaluation workflows. Do not describe a validation score as test-set performance.

### Pretrained model sources

- [Freshness MobileNetV2](https://huggingface.co/Piyu12/fruit-veg-freshness-mobilenetv2) — model card lists Apache-2.0.
- [Produce EfficientNet-B0](https://huggingface.co/bhumong/fruit-classifier-efficientnet-b0) — model card lists MIT.
- [TorchVision Faster R-CNN MobileNetV3 Large FPN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html) — official COCO detector checkpoint.

Review the upstream model and dataset terms before redistributing or using the weights beyond this project.

## Data, privacy, and limitations

- Uploaded images are processed for the requested analysis and held in the app session; they are not added to a persistent image database by this Streamlit interface.
- Gemini assessment is optional. When configured, assessment inputs are sent to Google's Gemini service. Do not upload sensitive or personally identifying images.
- Streamlit Community Cloud is suitable for demonstrations and small workloads, not a guaranteed private or high-availability production service. Review the host's data retention, access control, and resource policies before deployment.
- Image classifiers analyze an image as a whole; they do not locate separate objects. The dedicated detector locates only its supported COCO classes and may miss partially hidden or unusual fruit.
- AI output can be wrong and must not be used as the sole food-safety decision.

## Verify

Run the project tests:

```powershell
python -m unittest discover -s tests -v
```

Check that the Streamlit app starts:

```powershell
streamlit run streamlit_app.py
```

The Flask app in `app.py` is a separate interface and is not required to run the Streamlit application.
