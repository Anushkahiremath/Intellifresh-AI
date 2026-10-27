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

1. Push the project to a GitHub repository. The three inference model weights are included so the deployed app does not need to fetch them at startup. Do not commit `.env`, `.streamlit/secrets.toml`, databases, user uploads, raw datasets, training checkpoints, or virtual environments.
2. In Streamlit Community Cloud, choose **Create app**, select the repository and branch, and set the main file to `streamlit_app.py`.
3. Use Python 3.11 in the app's advanced settings when available. Ensure the deployment has enough memory for PyTorch and both image models; a 2 GB or larger memory instance is recommended, especially when using fruit detection. Smaller instances may be unable to load the models.
4. Add optional API credentials in the app's **Settings → Secrets** using TOML:

   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

5. Deploy and wait for the first model downloads to finish. First startup requires outbound access to Hugging Face and PyTorch's model host. Try **Reboot app** if initial downloads were interrupted.
6. Test image upload, produce recognition, fruit counting, batch review, and report downloads on the deployed URL.

The detector checkpoint is a large GitHub file, so a fresh clone and Streamlit Cloud build take longer than a source-only app. The bundled weights avoid repeat downloads after deployment. If the deployment host has insufficient memory, use a host with adequate RAM or configure the model services to load from your own approved artifact store.

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
