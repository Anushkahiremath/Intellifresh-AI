"""IntelliFresh AI produce quality inspection app."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from quality_engine import _visual_fallback, analyze_fruit, model_status
from recommendation import recommendation_for
from services.gemini_service import generate_gemini_assessment
from services.gradcam import create_gradcam as create_hf_gradcam
from services.pdf_generator import generate_pdf_report
from services.predictor import FreshnessPredictor, ProducePredictor

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IntelliFresh AI | Produce quality",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application styling
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 4% 4%, rgba(255, 104, 76, 0.34), transparent 34rem),
        radial-gradient(ellipse at 96% 12%, rgba(144, 91, 255, 0.30), transparent 38rem),
        radial-gradient(ellipse at 72% 96%, rgba(21, 189, 190, 0.19), transparent 36rem),
        linear-gradient(135deg, #100e24 0%, #1b1534 46%, #111f37 100%);
    background-attachment: fixed;
}

.stApp {
    background: transparent;
    color: #f4f0ff;
}

[data-testid="stMain"] {
    background: transparent;
}

[data-testid="stMain"] .block-container {
    color: #f4f0ff;
}

.block-container {
    padding-top: 1.65rem;
    padding-bottom: 3.5rem;
    max-width: 1500px;
}

[data-testid="stHeader"] {
    background: rgba(16, 14, 36, 0.58);
    backdrop-filter: blur(18px);
}
[data-testid="stToolbar"] {
    right: 1.2rem;
}

/* Hero Header */
.hero-banner {
    background:
        radial-gradient(circle at 84% 8%, rgba(255, 173, 86, 0.58), transparent 18rem),
        radial-gradient(circle at 98% 95%, rgba(27, 214, 205, 0.30), transparent 22rem),
        linear-gradient(118deg, #7a274e 0%, #68378d 46%, #35488e 100%);
    border-radius: 28px;
    padding: 2.9rem 3.1rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 28px 75px -28px rgba(8, 5, 24, 0.92), 0 0 42px rgba(151, 92, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.24);
    position: relative;
    overflow: hidden;
}
.hero-banner:after {
    content: "";
    position: absolute;
    width: 18rem;
    height: 18rem;
    right: -5rem;
    bottom: -11rem;
    border: 1px solid rgba(255, 255, 255, 0.23);
    border-radius: 50%;
    box-shadow: 0 0 0 2rem rgba(255,255,255,0.035), 0 0 0 4rem rgba(255,255,255,0.025);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 239, 213, 0.48);
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #fff0d6;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 850;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    color: #ffffff;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #fff2f8;
    max-width: 980px;
    line-height: 1.55;
    margin: 0;
    font-weight: 400;
}
.overview-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem;
    margin: -0.25rem 0 1.4rem;
}
.overview-card {
    min-height: 92px;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(226, 211, 255, 0.22);
    border-radius: 17px;
    background:
        radial-gradient(circle at 100% 0%, rgba(255, 132, 100, 0.16), transparent 8rem),
        linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.055));
    box-shadow: 0 14px 30px -23px rgba(0,0,0,0.85);
}
.overview-label {
    color: #d7cbe9;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
    font-weight: 750;
}
.overview-value {
    color: #fff5eb;
    font-size: 1.2rem;
    line-height: 1.35;
    font-weight: 800;
    margin-top: 0.3rem;
}
.overview-note {
    color: #d7cce3;
    font-size: 0.76rem;
    margin-top: 0.08rem;
}

/* Cards & Containers */
.glass-card {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #e8def1;
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 12px 30px -16px rgba(15, 23, 42, 0.28);
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover {
    border-color: #c4a9ee;
    transform: translateY(-2px);
    box-shadow: 0 18px 34px -16px rgba(15, 23, 42, 0.34);
}

/* Status Badges */
.badge-fresh {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #14532d;
    border: 1px solid #86efac;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.badge-stale {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #7f1d1d;
    border: 1px solid #fca5a5;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

/* Custom Metrics */
.custom-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.85rem;
    margin: 1rem 0;
}
.custom-metric-item {
    background: linear-gradient(145deg, #ffffff 0%, #f8f2ff 58%, #fff4eb 100%);
    border: 1px solid rgba(238, 222, 255, 0.95);
    border-radius: 18px;
    padding: 1rem 0.9rem;
    text-align: center;
    box-shadow: 0 14px 28px -20px rgba(0, 0, 0, 0.75);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.custom-metric-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 34px -19px rgba(0, 0, 0, 0.88);
}
.custom-metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #70677f;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.custom-metric-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #29233d;
    overflow-wrap: anywhere;
}
.performance-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 1rem;
    margin: 1.15rem 0 0.6rem;
}
.performance-card {
    position: relative;
    overflow: hidden;
    min-height: 132px;
    padding: 1.2rem 1.35rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.82);
    background:
        radial-gradient(circle at 100% 0%, rgba(255, 160, 105, 0.25), transparent 9rem),
        linear-gradient(145deg, #ffffff 0%, #f5efff 62%, #fff4ea 100%);
    box-shadow: 0 18px 35px -24px rgba(0,0,0,0.78);
}
.performance-card-label {
    color: #665777;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.74rem;
    font-weight: 750;
}
.performance-card-value {
    color: #241a3b;
    font-size: clamp(1.85rem, 3vw, 2.45rem);
    font-weight: 850;
    line-height: 1.2;
    margin-top: 0.7rem;
    font-variant-numeric: tabular-nums;
}
.performance-card-note {
    color: #766b84;
    font-size: 0.82rem;
    margin-top: 0.25rem;
}

/* Callout Alert Boxes */
.hazard-box {
    background: linear-gradient(120deg, #fff1ef 0%, #fff8f4 100%);
    border: 1px solid #ffd1c9;
    border-left: 5px solid #ef5c55;
    border-radius: 15px;
    padding: 1.25rem;
    margin: 1rem 0;
    color: #71323a;
}
.fresh-box {
    background: linear-gradient(120deg, #effaf2 0%, #f7fff5 100%);
    border: 1px solid #c6ebce;
    border-left: 5px solid #36a269;
    border-radius: 15px;
    padding: 1.25rem;
    margin: 1rem 0;
    color: #14532d;
}
.prototype-box {
    background: linear-gradient(120deg, #f3efff 0%, #fff8f1 100%);
    border: 1px solid #ded2f6;
    border-left: 5px solid #7957c8;
    border-radius: 15px;
    padding: 1.25rem;
    margin: 1rem 0;
    color: #34284d;
}
.prototype-box h3 {
    margin: 0.65rem 0 0.35rem;
    color: #34284d;
}
.prototype-box p {
    margin: 0;
    color: #5e526e;
}
.prototype-badge {
    display: inline-block;
    background: #e8defc;
    border: 1px solid #cbb8f0;
    color: #563d94;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(255, 120, 94, 0.25), transparent 24rem),
        radial-gradient(ellipse at 100% 70%, rgba(109, 87, 206, 0.17), transparent 20rem),
        linear-gradient(180deg, #171327 0%, #211a39 52%, #15273b 100%) !important;
    border-right: 1px solid rgba(220, 204, 255, 0.22) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}
.sidebar-brand {
    padding: 1rem 1rem 1.05rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.18);
    background:
        radial-gradient(circle at 100% 0%, rgba(255, 174, 85, 0.28), transparent 8rem),
        linear-gradient(135deg, rgba(244, 106, 95, 0.22), rgba(113, 76, 174, 0.27));
    box-shadow: 0 16px 35px -26px rgba(0,0,0,0.9);
}
.sidebar-brand-kicker {
    color: #ffce9c;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.sidebar-brand-title {
    color: #fff5ef;
    font-size: 1.15rem;
    font-weight: 850;
    margin-top: 0.25rem;
}
.sidebar-brand-copy {
    color: #e4dcef;
    font-size: 0.78rem;
    line-height: 1.4;
    margin-top: 0.28rem;
}
.sidebar-panel {
    padding: 0.85rem 0.9rem;
    border: 1px solid rgba(223, 208, 255, 0.17);
    border-radius: 15px;
    background: rgba(255,255,255,0.065);
}
.sidebar-panel-title {
    color: #ffddbf;
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.4rem;
}
.sidebar-panel-copy {
    color: #e5deef;
    font-size: 0.8rem;
    line-height: 1.45;
}
.sidebar-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.55rem;
}
.sidebar-stat {
    padding: 0.7rem;
    border-radius: 13px;
    border: 1px solid rgba(225, 209, 255, 0.14);
    background: rgba(255,255,255,0.07);
}
.sidebar-stat-value {
    color: #fff4e8;
    font-size: 1.15rem;
    line-height: 1.2;
    font-weight: 850;
    font-variant-numeric: tabular-nums;
}
.sidebar-stat-label {
    color: #d9cfe6;
    font-size: 0.68rem;
    line-height: 1.3;
    margin-top: 0.2rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.35rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.35rem 0.45rem;
    border-radius: 10px;
    transition: background 0.18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
    color: #fff4e8 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #e4dcef !important;
}

/* Custom Buttons */
.stButton>button {
    background: linear-gradient(118deg, #fb765e 0%, #c651a4 48%, #7557de 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 0.72rem 1.5rem !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
    box-shadow: 0 10px 26px -13px rgba(240, 94, 141, 0.85) !important;
}
.stButton>button:hover {
    transform: translateY(-2px);
    filter: brightness(1.09);
    box-shadow: 0 16px 31px -12px rgba(182, 91, 227, 0.8) !important;
}
.stDownloadButton>button {
    background: linear-gradient(118deg, #ffae49 0%, #f07850 52%, #dc4c76 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.6rem !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 22px -5px rgba(239, 104, 92, 0.5) !important;
}
.stDownloadButton>button:hover {
    box-shadow: 0 15px 28px -4px rgba(239, 104, 92, 0.67) !important;
    transform: translateY(-2px);
}

[data-testid="stTabs"] button[role="tab"] {
    border-radius: 13px 13px 0 0;
    font-weight: 700;
    color: #e5dcf5;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #ffd3a0;
    border-bottom: 3px solid #ff8b67;
}

[data-testid="stFileUploader"] section {
    border: 1.5px dashed #bca3ea;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(250,242,255,0.94));
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] p {
    color: #352b45 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #fffefd 0%, #f5efff 56%, #fff3e9 100%) !important;
    border: 1.5px dashed #9878c8 !important;
    border-radius: 18px !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(115deg, #f2765d, #9c53bb) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] span {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #62566f !important;
    opacity: 1 !important;
}

[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4 {
    color: #fff0e8;
}

[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] label {
    color: #e9e2f5;
}

[data-testid="stMain"] [data-testid="stAlert"] {
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.17);
    backdrop-filter: blur(10px);
}

[data-testid="stMain"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(227, 210, 255, 0.2);
    border-radius: 15px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #f3ebff 70%, #fff2e8);
    border: 1px solid rgba(238, 222, 255, 0.95);
    border-radius: 17px;
    padding: 1rem 1.1rem;
    box-shadow: 0 14px 28px -21px rgba(0, 0, 0, 0.85);
}
[data-testid="stMetric"] label {
    color: #6a5a7f !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #32214e !important;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #ffb34f 0%, #ff7461 42%, #b45bd6 72%, #36c4c0 100%);
}

[data-testid="stDataFrame"] {
    border: 1px solid #d8c8ed;
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stCameraInput"] {
    border: 1px solid rgba(215, 187, 255, 0.4);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 18px 36px -24px rgba(0,0,0,0.9);
}

@media (max-width: 700px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .hero-banner { padding: 1.7rem 1.4rem; border-radius: 20px; }
    .hero-subtitle { font-size: 0.95rem; }
    .overview-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; }
    .overview-card { min-height: 82px; padding: 0.8rem; }
}
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">IntelliFresh AI™</div>
    <div class="hero-subtitle">
        Inspect one item or a full batch. Recognize produce, count supported fruit, and review storage conditions—with clear model results and exportable reports.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Model Initialization -----------------
try:
    status = model_status()
    model_status_error = False
except Exception:
    status = {"loaded": False}
    model_status_error = True

@st.cache_resource
def load_vision_models():
    models_dir = Path(__file__).resolve().parent / "models"
    try:
        from huggingface_hub import hf_hub_download

        freshness_path = Path(os.getenv("FRESHNESS_MODEL_PATH") or models_dir / "freshness_mobilenetv2_state_dict.pth")
        if not freshness_path.is_file():
            freshness_path = Path(hf_hub_download(
                repo_id="Piyu12/fruit-veg-freshness-mobilenetv2",
                filename="freshness_mobilenetv2_state_dict.pth",
            ))
        produce_path = models_dir / "produce_efficientnet_b0.pth"
        if not produce_path.is_file():
            produce_path = Path(hf_hub_download(
                repo_id="bhumong/fruit-classifier-efficientnet-b0",
                filename="fruit_classifier_efficientnet_b0_best.pth",
            ))
        config_path = models_dir / "produce_config.json"
        if not config_path.is_file():
            config_path = Path(hf_hub_download(
                repo_id="bhumong/fruit-classifier-efficientnet-b0",
                filename="config.json",
            ))
        download_error = None
    except OSError as exc:
        download_error = type(exc).__name__
        freshness_path = Path(os.getenv("FRESHNESS_MODEL_PATH") or models_dir / "freshness_mobilenetv2_state_dict.pth")
        produce_path = models_dir / "produce_efficientnet_b0.pth"
        config_path = models_dir / "produce_config.json"

    freshness = FreshnessPredictor(freshness_path)
    produce = ProducePredictor(produce_path, config_path)
    return freshness, produce, download_error


@st.cache_resource(show_spinner=False)
def load_fruit_detector():
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    checkpoint_path = Path(__file__).resolve().parent / "models" / "fruit_detector_mobilenet_v3_large_fpn.pth"
    if checkpoint_path.is_file():
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), weights.transforms(), weights.meta["categories"], device


def detect_and_annotate_fruits(image, confidence_threshold):
    import torch
    from PIL import ImageDraw

    model, transforms, categories, device = load_fruit_detector()
    display_image = image.convert("RGB")
    display_image.thumbnail((1280, 1280))
    tensor = transforms(display_image).to(device)

    with torch.inference_mode():
        prediction = model([tensor])[0]

    annotated = display_image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []
    supported_labels = {"apple": "Apple", "banana": "Banana", "orange": "Orange"}
    for box, label_index, score in zip(
        prediction["boxes"].cpu().tolist(),
        prediction["labels"].cpu().tolist(),
        prediction["scores"].cpu().tolist(),
    ):
        label = categories[label_index].lower()
        if label not in supported_labels or score < confidence_threshold:
            continue
        name = supported_labels[label]
        x1, y1, x2, y2 = (int(value) for value in box)
        draw.rectangle((x1, y1, x2, y2), outline="#ff684c", width=4)
        draw.text((x1 + 4, max(0, y1 - 20)), f"{name} {score:.0%}", fill="#ff684c", stroke_width=2, stroke_fill="white")
        detections.append({
            "Fruit": name,
            "Confidence (%)": round(score * 100, 1),
            "Left": x1,
            "Top": y1,
            "Right": x2,
            "Bottom": y2,
        })
    return annotated, detections


def get_gemini_api_key():
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        return None


STORAGE_PRESET_VALUES = {
    "Room ambient": (20.0, 65.0),
    "Cold chain": (4.0, 90.0),
    "Warm & humid": (32.0, 85.0),
    "Custom": (22.0, 60.0),
}


def apply_storage_preset():
    temperature, humidity = STORAGE_PRESET_VALUES[st.session_state["storage_preset"]]
    st.session_state["storage_temperature"] = temperature
    st.session_state["storage_humidity"] = humidity


with st.spinner("Loading vision models. The first run may download model weights..."):
    hf_freshness, hf_produce, model_download_error = load_vision_models()
hf_ready = hf_freshness.available and hf_produce.available
MODEL_DUAL = "Dual model · MobileNetV2 + EfficientNet-B0"
MODEL_MOBILENET = "MobileNetV2 · freshness only"
MODEL_OURS = "Our model · IntelliFresh multimodal"
MODEL_OURS_UNAVAILABLE = "Our model · checkpoint not installed"
MODEL_VISUAL = "Visual diagnostic only · no trained model"

model_choices = []
model_choice_modes = {}
if hf_ready:
    model_choices.append(MODEL_DUAL)
    model_choice_modes[MODEL_DUAL] = "dual"
if hf_freshness.available:
    model_choices.append(MODEL_MOBILENET)
    model_choice_modes[MODEL_MOBILENET] = "mobilenet"
if status.get("loaded"):
    model_choices.append(MODEL_OURS)
    model_choice_modes[MODEL_OURS] = "multimodal"
else:
    model_choices.append(MODEL_OURS_UNAVAILABLE)
    model_choice_modes[MODEL_OURS_UNAVAILABLE] = "unavailable_multimodal"
if not model_choices or all(mode.startswith("unavailable") for mode in model_choice_modes.values()):
    model_choices.append(MODEL_VISUAL)
    model_choice_modes[MODEL_VISUAL] = "visual_fallback"

default_model_choice = (
    MODEL_DUAL if hf_ready
    else MODEL_OURS if status.get("loaded")
    else MODEL_MOBILENET if hf_freshness.available
    else MODEL_VISUAL
)
if "model_selection" not in st.session_state or st.session_state["model_selection"] not in model_choices:
    st.session_state["model_selection"] = default_model_choice
selected_model_label = st.session_state["model_selection"]
selected_inference_mode = model_choice_modes[selected_model_label]

mode_summaries = {
    "dual": ("Dual vision models", "Produce ID + freshness classification"),
    "mobilenet": ("MobileNetV2 freshness", "Fresh / stale classification only"),
    "multimodal": ("IntelliFresh multimodal", "Produce condition + storage context"),
    "unavailable_multimodal": ("Our model is not installed", "Add MODEL_CHECKPOINT to enable it"),
    "visual_fallback": ("Visual diagnostic only", "No trained model is available"),
}
mode_summary = mode_summaries[selected_inference_mode]
produce_label_count = len(hf_produce.labels) if hf_produce.available else "—"
st.markdown(
    f"""
    <div class="overview-strip">
        <div class="overview-card">
            <div class="overview-label">Inspection</div>
            <div class="overview-value">{escape(mode_summary[0])}</div>
            <div class="overview-note">{escape(mode_summary[1])}</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Produce labels</div>
            <div class="overview-value">{produce_label_count}</div>
            <div class="overview-note">Whole-image recognition classes</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Fruit counting</div>
            <div class="overview-value">3 supported types</div>
            <div class="overview-note">Apple · banana · orange</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Batch review</div>
            <div class="overview-value">Up to 30 images</div>
            <div class="overview-note">Review and export results</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

SHELF_LIFE_BASELINES = {
    "apple": 21.0, "banana": 7.0, "cherry": 5.0, "grape": 10.0,
    "kiwi": 14.0, "mango": 8.0, "orange": 18.0, "peach": 6.0,
    "pear": 10.0, "plum": 8.0, "strawberry": 4.0, "watermelon": 10.0,
    "tomato": 9.0, "capsicum": 12.0, "bitter gourd": 5.0,
}

def estimate_shelf_life(produce_name, fresh_probability, temperature, humidity):
    """Estimate remaining days using physiological baseline and storage condition penalties."""
    name = str(produce_name).lower().replace("_", " ")
    baseline = next((days for fruit, days in SHELF_LIFE_BASELINES.items() if fruit in name), 7.0)
    freshness_factor = max(0.03, min(1.0, float(fresh_probability) / 100.0)) ** 1.35
    temperature_factor = math.exp(-0.055 * (float(temperature) - 20.0))
    humidity_factor = math.exp(-0.012 * abs(float(humidity) - 65.0))
    return round(max(0.1, min(60.0, baseline * freshness_factor * temperature_factor * humidity_factor)), 1)

# ----------------- Sidebar Controls -----------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-kicker">FreshVision workspace</div>
            <div class="sidebar-brand-title">Control center</div>
            <div class="sidebar-brand-copy">Set storage conditions and check the tools available for this session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_inference_mode in {"dual", "mobilenet", "multimodal"}:
        st.success(f"Selected model ready · {mode_summary[0]}")
    else:
        st.warning(mode_summary[0])
        st.caption(
            "Select an available model below. Diagnostic mode does not produce a trained classification."
        )
    if model_download_error:
        st.caption(
            "Model weights could not be downloaded "
            f"({model_download_error}). Check deployment internet access and try again."
        )
    if model_status_error:
        st.caption("The multimodal checkpoint could not be loaded in this runtime.")

    st.markdown("### Analysis model")
    st.selectbox(
        "Choose the model for inspection and batch review",
        options=model_choices,
        key="model_selection",
        help=(
            "Dual combines MobileNetV2 freshness classification with EfficientNet produce recognition. "
            "MobileNetV2 returns freshness only. Our model requires its trained multimodal checkpoint."
        ),
    )
    if selected_inference_mode == "mobilenet":
        st.caption("Freshness only: produce identity and shelf-life estimate are not generated.")
    elif selected_inference_mode == "unavailable_multimodal":
        st.caption("To enable this option, provide a trained checkpoint with the MODEL_CHECKPOINT environment variable.")
    elif selected_inference_mode == "visual_fallback":
        st.caption("No trained model is currently available; results are visual diagnostics only.")

    if "storage_temperature" not in st.session_state:
        st.session_state["storage_temperature"] = STORAGE_PRESET_VALUES["Room ambient"][0]
    if "storage_humidity" not in st.session_state:
        st.session_state["storage_humidity"] = STORAGE_PRESET_VALUES["Room ambient"][1]

    st.markdown("---")
    st.markdown("### Storage profile")
    st.radio(
        "Choose a starting profile",
        ["Room ambient", "Cold chain", "Warm & humid", "Custom"],
        index=0,
        key="storage_preset",
        on_change=apply_storage_preset,
    )

    temperature = st.slider(
        "Storage temperature (°C)",
        min_value=-5.0,
        max_value=45.0,
        step=0.5,
        key="storage_temperature",
    )
    humidity = st.slider(
        "Relative humidity (%)",
        min_value=15.0,
        max_value=95.0,
        step=1.0,
        key="storage_humidity",
    )

    if temperature <= 6.0 and humidity >= 85.0:
        climate_status = "Cold-chain profile"
        climate_detail = "Cool and humid storage conditions selected."
    elif temperature >= 28.0 or humidity >= 88.0:
        climate_status = "Warm / humid profile"
        climate_detail = "Higher temperature or humidity may shorten quality windows."
    else:
        climate_status = "Ambient profile"
        climate_detail = "Standard room conditions selected."
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="sidebar-panel-title">Current conditions</div>
            <div class="sidebar-stat-grid">
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value">{temperature:g}°C</div>
                    <div class="sidebar-stat-label">temperature</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value">{humidity:g}%</div>
                    <div class="sidebar-stat-label">relative humidity</div>
                </div>
            </div>
            <div class="sidebar-panel-copy" style="margin-top:0.6rem">
                <strong>{climate_status}</strong><br>{climate_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_history = st.session_state.get("inspection_history", [])
    session_batch = st.session_state.get("last_batch")
    session_batch_count = len(session_batch) if isinstance(session_batch, pd.DataFrame) else 0
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="sidebar-panel-title">Workspace snapshot</div>
            <div class="sidebar-stat-grid">
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value">{len(session_history)}</div>
                    <div class="sidebar-stat-label">inspections this session</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value">{session_batch_count}</div>
                    <div class="sidebar-stat-label">latest batch items</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="sidebar-panel-title">Analysis tools</div>
            <div class="sidebar-panel-copy">
                {'✓' if hf_produce.available else '○'} Produce recognition · {produce_label_count} labels<br>
                {'✓' if hf_freshness.available else '○'} Freshness classifier · Fresh / stale<br>
                On demand · object count (apple / banana / orange)<br>
                ✓ Camera capture · browser supported
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Images stay in the current app session unless you choose to export a result.")

# ----------------- Helper: Run Inference on Upload -----------------
def run_uploaded(image_source, temperature, humidity, inference_mode):
    """Processes uploaded image bytes or file through the inference and explainability pipeline."""
    with tempfile.TemporaryDirectory() as folder:
        suffix = ".jpg"
        image_path = Path(folder) / ("input" + suffix)

        if hasattr(image_source, "getvalue"):
            image_bytes = image_source.getvalue()
        elif isinstance(image_source, bytes):
            image_bytes = image_source
        else:
            image_bytes = Path(image_source).read_bytes()

        image_path.write_bytes(image_bytes)

        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The provided file is not a valid or readable image.") from exc

        cam_path = Path(folder) / "gradcam.png"

        if inference_mode == "multimodal":
            result = analyze_fruit(image_path, "auto", temperature, humidity, cam_path)
            recommendation, source = recommendation_for(result, temperature, humidity)
            result["recommendation"] = recommendation
            result["recommendation_source"] = source
        elif inference_mode in {"dual", "mobilenet"}:
            freshness = hf_freshness.predict(image_path)
            tensor = freshness.pop("tensor")
            class_index = freshness.pop("class_index")
            create_hf_gradcam(hf_freshness.model, tensor, class_index, image_path, cam_path)

            if inference_mode == "dual":
                identification = hf_produce.predict(image_path)
                shelf_life = estimate_shelf_life(
                    identification["produce_name"],
                    freshness["fresh_probability"],
                    temperature,
                    humidity,
                )
                result = {
                    **freshness,
                    **identification,
                    "model_mode": "Dual Vision AI (MobileNetV2 + EfficientNet-B0)",
                    "class_name": freshness["prediction"].lower(),
                    "fruit": identification["produce_name"],
                    "status": freshness["prediction"],
                    "freshness_score": freshness["fresh_probability"],
                    "shelf_life_days": shelf_life,
                    "metrics": {
                        "Fresh probability": freshness["fresh_probability"],
                        "Rotten probability": freshness["stale_probability"],
                        "Model confidence": freshness["confidence"],
                    },
                    "forecast": [
                        {
                            "day": day,
                            "freshness": round(
                                max(
                                    0,
                                    freshness["fresh_probability"]
                                    * (1 - day / max(shelf_life, 1)),
                                ),
                                1,
                            ),
                        }
                        for day in range(8)
                    ],
                    "disclaimer": "AI-assisted produce estimates. Do not use as the sole food-safety decision.",
                },
            else:
                result = {
                    **freshness,
                    "model_mode": "MobileNetV2 freshness-only classifier",
                    "class_name": freshness["prediction"].lower(),
                    "fruit": "Unidentified produce",
                    "status": freshness["prediction"],
                    "freshness_score": freshness["fresh_probability"],
                    "shelf_life_days": None,
                    "metrics": {
                        "Fresh probability": freshness["fresh_probability"],
                        "Stale probability": freshness["stale_probability"],
                        "Model confidence": freshness["confidence"],
                    },
                    "forecast": [],
                    "disclaimer": (
                        "MobileNetV2 classifies freshness only. Produce identity and shelf life "
                        "were not assessed."
                    ),
                }
            recommendation, source = recommendation_for(result, temperature, humidity)
            result["recommendation"] = recommendation
            result["recommendation_source"] = source
        elif inference_mode == "visual_fallback":
            result = _visual_fallback(image_path, temperature, humidity)
            result["recommendation"] = (
                "A trained produce and freshness model is unavailable. "
                "Use visual inspection and established food-safety practices."
            )
            result["recommendation_source"] = "Visual diagnostic only"
        else:
            raise RuntimeError("The selected model is unavailable. Choose an available model in the sidebar.")

        result["image_bytes"] = image_path.read_bytes()
        result["gradcam_bytes"] = cam_path.read_bytes() if cam_path.exists() else None
        return result

# ----------------- Main Tabs -----------------
tab_inspect, tab_recognize, tab_batch, tab_metrics = st.tabs([
    "Inspect produce",
    "Recognize & count",
    "Batch review",
    "Model performance",
])

# ==============================================================================
# TAB 1: Produce Inspection & Audit
# ==============================================================================
with tab_inspect:
    st.markdown("#### Inspect a produce image")
    st.caption("Upload a photo or take one, then review the quality estimate and storage guidance.")

    col_input, col_env = st.columns([1.1, 1], gap="medium")

    with col_input:
        input_method = st.segmented_control(
            "Choose how to add a produce image",
            ["Upload image", "Capture with camera"],
            default="Upload image",
            selection_mode="single",
            key="inspection_input_method",
        )
        uploaded_file = None
        if input_method == "Upload image":
            uploaded_file = st.file_uploader(
                "Choose an image (JPG, PNG, WebP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="produce_uploader"
            )
        else:
            uploaded_file = st.camera_input("Capture a fresh image now")
        if uploaded_file:
            with Image.open(io.BytesIO(uploaded_file.getvalue())) as preview:
                st.image(
                    preview,
                    caption=f"Inspection image · {preview.width} × {preview.height}",
                    alt="Image selected for produce inspection",
                    width="stretch",
                )

    with col_env:
        st.markdown("**Storage conditions in use**")
        st.caption("Adjust the shared temperature and humidity in the left control panel.")
        temp_col, hum_col = st.columns(2)
        temp_col.metric("Temperature", f"{temperature:.1f}°C")
        hum_col.metric("Relative humidity", f"{humidity:.0f}%")
        if temperature <= 6.0 and humidity >= 85.0:
            st.info("Cold-chain conditions selected for this inspection.")
        elif temperature >= 28.0 or humidity >= 88.0:
            st.warning("Warm or humid conditions may shorten quality windows.")
        else:
            st.success("Ambient storage conditions selected.")

        st.markdown("---")
        st.markdown("**Inspection workflow**")
        st.markdown(
            "1. Add a clear image or capture a photo.  \n"
            "2. Run the available image models.  \n"
            "3. Review the result and export if needed."
        )
        st.caption(
            "Use **Recognize & count** for supported fruit counts, or **Batch review** "
            "to process multiple images."
        )

    st.markdown("---")

    # Inspection Action Button
    if selected_inference_mode == "unavailable_multimodal":
        inspection_action = "Selected model unavailable · choose another"
    elif selected_inference_mode == "visual_fallback":
        inspection_action = "Run visual diagnostic"
    else:
        inspection_action = "Inspect with selected model"
    if selected_inference_mode == "unavailable_multimodal":
        st.warning(
            "The IntelliFresh multimodal checkpoint is not installed. "
            "Set MODEL_CHECKPOINT to its trained checkpoint or choose another model."
        )
    analyze_btn = st.button(
        inspection_action,
        type="primary",
        width="stretch",
        disabled=selected_inference_mode == "unavailable_multimodal",
    )

    if analyze_btn:
        if not uploaded_file:
            st.warning("⚠️ Please provide an image file or capture a photo first.")
        else:
            with st.spinner("Analyzing the image and preparing results..."):
                try:
                    result = run_uploaded(
                        uploaded_file,
                        temperature,
                        humidity,
                        selected_inference_mode,
                    )

                    if result.get("model_mode") == "visual_prototype":
                        gemini_report = {
                            "verdict": "Visual diagnostic only — no trained model result",
                            "usable_days_summary": "No shelf-life estimate is available.",
                            "safe_to_use": "Not assessed",
                            "culinary_action": "Do not use this visual diagnostic to make food-safety decisions.",
                            "biological_reactions_if_consumed": "A trained model was unavailable; biological risk was not assessed.",
                            "spoilage_pathogens": "Not assessed",
                            "storage_protocol": "Follow established produce storage guidance.",
                            "risk_severity": "Not assessed",
                            "source": "Visual diagnostic only",
                        }
                    elif result.get("model_mode") == "MobileNetV2 freshness-only classifier":
                        gemini_report = {
                            "verdict": f"MobileNetV2 classification: {result['status']}",
                            "usable_days_summary": "No shelf-life estimate is generated in freshness-only mode.",
                            "safe_to_use": "Not assessed",
                            "culinary_action": (
                                "Confirm produce identity and inspect it directly. "
                                "Do not use this freshness-only classification as a food-safety decision."
                            ),
                            "biological_reactions_if_consumed": (
                                "Food-safety risk was not assessed by the selected freshness-only model."
                            ),
                            "spoilage_pathogens": "Not assessed",
                            "storage_protocol": "Select an identified produce type for tailored storage guidance.",
                            "risk_severity": "Not assessed",
                            "source": "MobileNetV2 freshness classifier only",
                        }
                    else:
                        gemini_report = generate_gemini_assessment(
                            produce_name=result.get("fruit", result.get("produce_name", "Fruit")),
                            status=result.get("status", "Unknown"),
                            confidence=result.get("confidence", 90.0),
                            freshness_score=result.get("freshness_score", 50.0),
                            shelf_life_days=result.get("shelf_life_days"),
                            temperature=temperature,
                            humidity=humidity,
                            gemini_api_key=get_gemini_api_key(),
                        )

                    limited_scope = result.get("model_mode") in {
                        "visual_prototype",
                        "MobileNetV2 freshness-only classifier",
                    }
                    pdf_bytes = (
                        None if limited_scope
                        else generate_pdf_report(result, gemini_report, temperature, humidity)
                    )

                    st.session_state["last_inspection"] = {
                        "result": result,
                        "gemini": gemini_report,
                        "pdf": pdf_bytes,
                        "temp": temperature,
                        "hum": humidity,
                    }
                    history_entry = {
                        "Inspected at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                        "Produce": str(result.get("fruit", result.get("produce_name", "Unknown"))),
                        "Status": str(result.get("status", "Unknown")),
                        "Quality score (%)": float(result.get("freshness_score", 0.0)),
                        "Confidence (%)": float(result.get("confidence", 0.0)),
                        "Shelf life (days)": result.get("shelf_life_days"),
                        "Temperature (°C)": float(temperature),
                        "Humidity (%)": float(humidity),
                    }
                    history = st.session_state.setdefault("inspection_history", [])
                    history.insert(0, history_entry)
                    del history[20:]
                except Exception as exc:
                    st.error(
                        f"Inspection failed ({type(exc).__name__}). "
                        "Sensitive details are hidden; please retry or contact the app maintainer."
                    )

    # Render Results if available in session
    if "last_inspection" in st.session_state:
        insp = st.session_state["last_inspection"]
        res = insp["result"]
        gemini = insp["gemini"]
        pdf_bytes = insp["pdf"]

        result_status = str(res["status"]).lower()
        is_prototype = res.get("model_mode") == "visual_prototype"
        is_mobilenet_only = res.get("model_mode") == "MobileNetV2 freshness-only classifier"
        limited_scope = is_prototype or is_mobilenet_only
        is_fresh = result_status == "fresh"

        result_heading_col, clear_result_col = st.columns([5, 1])
        with result_heading_col:
            st.markdown("### 📋 Inspection results")
        with clear_result_col:
            if st.button("✕ Clear result", key="clear_latest_inspection"):
                st.session_state.pop("last_inspection", None)
                st.rerun()

        st.markdown("---")
        if is_prototype:
            st.warning(
                "Visual diagnostic only: a trained model did not run. "
                "The displayed image features are not a freshness classification or food-safety result."
            )
        elif is_mobilenet_only:
            st.info(
                "Freshness-only mode: this model does not identify the produce or assess shelf life "
                "or food safety."
            )
        else:
            st.progress(
                int(max(0, min(100, round(float(res.get("freshness_score", 0)))))),
                text=f"Freshness score · {float(res.get('freshness_score', 0)):.1f}%",
            )

        # 1. Executive Verdict Banner
        if is_mobilenet_only:
            st.markdown(f"""
            <div class="prototype-box">
                <span class="prototype-badge">MobileNetV2 · freshness only</span>
                <h3>Model classification: {escape(str(res['status']))}</h3>
                <p>Produce identity, shelf life, and food-safety risk were not assessed by this model.</p>
            </div>
            """, unsafe_allow_html=True)
        elif is_fresh and not is_prototype:
            verdict = escape(str(gemini.get("verdict", "Freshness assessment complete")))
            usable_days_summary = escape(str(gemini.get("usable_days_summary", "")))
            safe_to_use = escape(str(gemini.get("safe_to_use", "Review guidance")))
            fruit_name = escape(str(res.get("fruit", "Produce")))
            st.markdown(f"""
            <div class="fresh-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="badge-fresh">✓ {escape(str(res['status']).upper())}</span>
                        <h3 style="margin:0.4rem 0 0.2rem 0; color:#14532d;">{fruit_name} — {verdict}</h3>
                        <p style="margin:0; font-size:0.95rem;">{usable_days_summary}</p>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.8rem; text-transform:uppercase; color:#166534; font-weight:700;">Safe to Use</span><br/>
                        <strong style="font-size:1.4rem; color:#14532d;">{safe_to_use}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif is_prototype:
            st.markdown("""
            <div class="prototype-box">
                <span class="prototype-badge">Visual diagnostic</span>
                <h3>Trained model unavailable</h3>
                <p>No produce label, freshness verdict, shelf-life estimate, or safety result was generated.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            verdict = escape(str(gemini.get("verdict", "Freshness assessment complete")))
            usable_days_summary = escape(str(gemini.get("usable_days_summary", "")))
            safe_to_use = escape(str(gemini.get("safe_to_use", "Review guidance")))
            fruit_name = escape(str(res.get("fruit", "Produce")))
            st.markdown(f"""
            <div class="hazard-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="badge-stale">⚠️ {escape(str(res['status']).upper())} / HAZARDOUS</span>
                        <h3 style="margin:0.4rem 0 0.2rem 0; color:#7f1d1d;">{fruit_name} — {verdict}</h3>
                        <p style="margin:0; font-size:0.95rem;">{usable_days_summary}</p>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.8rem; text-transform:uppercase; color:#991b1b; font-weight:700;">Safe to Use</span><br/>
                        <strong style="font-size:1.4rem; color:#7f1d1d;">{safe_to_use}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. PDF Download Action Card
        download_col1, download_col2, download_col3 = st.columns([2, 1, 1])
        with download_col1:
            if limited_scope:
                st.write("📄 **Limited model scope:** Download the result data. No food-safety audit PDF is generated for this mode.")
            else:
                st.write("📄 **Official audit report:** Includes visual evidence, Grad-CAM attention heatmap, and storage guidance.")
        with download_col2:
            if pdf_bytes:
                st.download_button(
                    label="📥 Download Official Audit PDF",
                    data=pdf_bytes,
                    file_name=f"IntelliFresh_Audit_{res['fruit']}_{res['status']}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
        with download_col3:
            export_result = {
                key: value for key, value in res.items()
                if key.lower() not in {
                    "image_bytes", "gradcam_bytes", "api_key", "gemini_api_key",
                    "google_api_key", "token", "authorization",
                }
            }
            st.download_button(
                label="Download result data",
                data=json.dumps(export_result, indent=2, ensure_ascii=False, default=str),
                file_name="IntelliFresh_Inspection.json",
                mime="application/json",
                help="Download prediction and environment details without image data or credentials.",
            )

        # 3. High-Level Telemetry Cards
        st.markdown(f"""
        <div class="custom-metric-grid">
            <div class="custom-metric-item">
                <div class="custom-metric-label">{'Produce identity' if is_mobilenet_only else 'Identified Produce'}</div>
                <div class="custom-metric-val">{'Not assessed' if is_mobilenet_only else escape(str(res['fruit']))}</div>
            </div>
            <div class="custom-metric-item">
                <div class="custom-metric-label">{'Visual signal' if is_prototype else 'Freshness score'}</div>
                <div class="custom-metric-val" style="color:{'#6244a0' if limited_scope else '#16a34a' if is_fresh else '#dc2626'}">{res['freshness_score']:.1f}%</div>
            </div>
            <div class="custom-metric-item">
                <div class="custom-metric-label">Model Confidence</div>
                <div class="custom-metric-val">{'Not available' if is_prototype else f"{float(res['confidence']):.1f}%"}</div>
            </div>
            <div class="custom-metric-item">
                <div class="custom-metric-label">Est. Usable Days</div>
                <div class="custom-metric-val">{'Unavailable' if res.get('shelf_life_days') is None else f"{float(res['shelf_life_days']):.1f} d"}</div>
            </div>
            <div class="custom-metric-item">
                <div class="custom-metric-label">Risk Severity</div>
                <div class="custom-metric-val" style="color:{'#6244a0' if limited_scope else '#16a34a' if is_fresh else '#dc2626'}">{escape(str(gemini.get('risk_severity', 'LOW')))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        result_metrics = res.get("metrics", {})
        if result_metrics:
            st.markdown("#### Model result matrix")
            result_matrix = pd.DataFrame(
                [{"Signal": str(name), "Score (%)": float(value)} for name, value in result_metrics.items()]
            )
            st.dataframe(
                result_matrix,
                column_config={
                    "Score (%)": st.column_config.ProgressColumn(
                        "Score (%)", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
                hide_index=True,
            )
            st.bar_chart(result_matrix.set_index("Signal"), y="Score (%)")

        # 4. Visual Evidence & Grad-CAM Explainable AI
        st.markdown("#### 🔍 Visual Evidence & Explainable AI (Grad-CAM)")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(res["image_bytes"], caption="Uploaded Produce Specimen", alt="Uploaded produce specimen", width="stretch")
        with col_img2:
            if res["gradcam_bytes"]:
                st.image(res["gradcam_bytes"], caption="Grad-CAM Neural Attention Heatmap (Discoloration/Pathology)", alt="Grad-CAM heatmap for the produce classification", width="stretch")
            else:
                st.info("Grad-CAM visualization rendered in audit report.")

        with st.expander("Additional handling and safety guidance"):
            st.markdown("**Recommended action**")
            st.write(gemini.get("culinary_action", "Follow standard produce handling guidance."))
            st.markdown("**Storage**")
            st.write(gemini.get("storage_protocol", "Use clean, appropriately cool storage."))
            st.markdown("**Spoilage indicators**")
            st.write(gemini.get("spoilage_pathogens", "Not available"))
            st.markdown("**Food-safety notes**")
            st.write(gemini.get("biological_reactions_if_consumed", "No assessment is available."))
            st.caption(f"Guidance source: {escape(str(gemini.get('source', 'Expert rules')))}")

        # 6. Interactive Degradation Chart
        if res.get("forecast"):
            st.markdown("#### 📈 Projected Freshness Retention Curve (7 Days)")
            forecast_df = pd.DataFrame(res["forecast"]).set_index("day")
            st.line_chart(forecast_df, y="freshness")

    if st.session_state.get("inspection_history"):
        with st.expander("Recent inspections in this session", expanded=False):
            history_df = pd.DataFrame(st.session_state["inspection_history"])
            st.dataframe(history_df, hide_index=True)
            history_col1, history_col2 = st.columns(2)
            with history_col1:
                st.download_button(
                    "Download session history",
                    data=history_df.to_csv(index=False),
                    file_name="IntelliFresh_Session_History.csv",
                    mime="text/csv",
                )
            with history_col2:
                if st.button("Clear session history"):
                    st.session_state["inspection_history"] = []
                    st.rerun()


with tab_recognize:
    st.markdown("#### Recognize produce or count fruit")
    st.caption(
        "Identify one produce item with the 76-label classifier, or detect and count "
        "visible apples, bananas, and oranges in one image."
    )
    recognition_mode = st.segmented_control(
        "Choose image analysis",
        ["Recognize produce", "Detect & count fruit"],
        default="Recognize produce",
        key="recognition_mode",
    )
    recognition_input_method = st.segmented_control(
        "Choose image source",
        ["Upload image", "Capture with camera"],
        default="Upload image",
        key="recognition_input_method",
    )
    if recognition_input_method == "Upload image":
        recognition_upload = st.file_uploader(
            "Choose a produce image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="recognition_uploader",
        )
    else:
        recognition_upload = st.camera_input(
            "Capture produce for recognition",
            key="recognition_camera",
        )

    confidence_threshold = 0.4
    if recognition_mode == "Detect & count fruit":
        confidence_threshold = st.slider(
            "Minimum detection confidence",
            min_value=0.2,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Raise this to hide uncertain boxes; partially hidden fruit may be missed.",
            key="fruit_detection_threshold",
        )

    if recognition_upload:
        try:
            recognition_image_bytes = recognition_upload.getvalue()
            recognition_image_digest = hashlib.sha256(recognition_image_bytes).hexdigest()
            with Image.open(io.BytesIO(recognition_image_bytes)) as recognition_preview:
                st.image(
                    recognition_preview,
                    caption=f"Selected image · {recognition_preview.width} × {recognition_preview.height}",
                    alt="Image selected for produce recognition",
                    width="stretch",
                )
        except (UnidentifiedImageError, OSError):
            st.error("This file is not a valid or readable image.")
            recognition_upload = None
            recognition_image_digest = None
    else:
        recognition_image_digest = None

    action_label = (
        "Detect and count fruit"
        if recognition_mode == "Detect & count fruit"
        else "Recognize produce"
    )
    if st.button(action_label, type="primary", width="stretch", key="run_recognition"):
        if not recognition_upload:
            st.warning("Choose or capture an image first.")
        elif recognition_mode == "Recognize produce" and not hf_produce.available:
            st.error("The produce recognition model is unavailable. Check model downloads and restart the app.")
        else:
            try:
                with st.spinner(
                    "Analyzing the image. First-time fruit counting may download detector weights..."
                ):
                    with Image.open(io.BytesIO(recognition_upload.getvalue())) as source_image:
                        source_image = source_image.convert("RGB")
                        if recognition_mode == "Recognize produce":
                            recognition_result = hf_produce.predict(source_image)
                        else:
                            annotated_image, detections = detect_and_annotate_fruits(
                                source_image,
                                confidence_threshold,
                            )

                if recognition_mode == "Recognize produce":
                    st.session_state["recognition_result"] = {
                        "mode": recognition_mode,
                        "image_digest": recognition_image_digest,
                        "result": recognition_result,
                    }
                else:
                    st.session_state["recognition_result"] = {
                        "mode": recognition_mode,
                        "image_digest": recognition_image_digest,
                        "confidence_threshold": confidence_threshold,
                        "image": annotated_image,
                        "detections": detections,
                    }
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(
                    f"Recognition could not be completed ({type(exc).__name__}). "
                    "Check model availability and try a clear, well-lit image."
                )

    saved_recognition = st.session_state.get("recognition_result")
    if saved_recognition and (
        saved_recognition.get("mode") != recognition_mode
        or saved_recognition.get("image_digest") != recognition_image_digest
        or (
            recognition_mode == "Detect & count fruit"
            and saved_recognition.get("confidence_threshold") != confidence_threshold
        )
    ):
        saved_recognition = None
    if saved_recognition:
        st.markdown("---")
        if saved_recognition["mode"] == "Recognize produce":
            recognized = saved_recognition["result"]
            result_col, confidence_col, scope_col = st.columns(3)
            result_col.metric("Recognized produce", recognized["produce_name"])
            confidence_col.metric("Model confidence", f'{recognized["produce_confidence"]:.1f}%')
            scope_col.metric("Freshness coverage", recognized["freshness_coverage"])
            st.caption(
                "This image classifier labels the overall image; it does not locate or count "
                "multiple separate items. Confidence is not prediction accuracy."
            )
        else:
            fruit_detections = saved_recognition["detections"]
            annotated_image = saved_recognition["image"]
            total_detected = len(fruit_detections)
            st.metric("Visible supported fruits detected", total_detected)
            st.image(
                annotated_image,
                caption="Detected fruit locations",
                alt="Fruit detection boxes and confidence scores",
                width="stretch",
            )
            if fruit_detections:
                detections_df = pd.DataFrame(fruit_detections)
                counts_df = (
                    detections_df.groupby("Fruit", as_index=False)
                    .size()
                    .rename(columns={"size": "Count"})
                )
                count_col, detail_col = st.columns([1, 1.5])
                with count_col:
                    st.markdown("**Count by fruit**")
                    st.dataframe(
                        counts_df,
                        hide_index=True,
                        alt="Detected count grouped by fruit type",
                    )
                with detail_col:
                    st.markdown("**Detected items**")
                    st.dataframe(
                        detections_df[["Fruit", "Confidence (%)"]],
                        hide_index=True,
                        alt="Detected fruit labels and confidence values",
                    )
                st.download_button(
                    "Download detection results",
                    data=detections_df.to_csv(index=False),
                    file_name="IntelliFresh_Fruit_Detections.csv",
                    mime="text/csv",
                    key="download_fruit_detections",
                )
            else:
                st.info(
                    "No supported fruit was detected above this confidence threshold. "
                    "This does not prove that the image contains no fruit."
                )
            st.caption(
                "The detector supports COCO apple, banana, and orange labels only. "
                "Occlusion, crowding, image quality, and the confidence threshold can affect counts."
            )

# ==============================================================================
# TAB 2: Batch Crate QC
# ==============================================================================
with tab_batch:
    st.markdown("#### Batch review")
    st.caption("Review and export results for up to 30 images. Batch conditions start from the sidebar profile.")

    batch_col1, batch_col2 = st.columns(2)
    batch_temp = batch_col1.number_input("Crate Storage Temperature (°C)", -5.0, 45.0, float(temperature), step=0.5, key="b_t")
    batch_hum = batch_col2.number_input("Crate Relative Humidity (%)", 15.0, 95.0, float(humidity), step=1.0, key="b_h")

    batch_files = st.file_uploader(
        "Upload crate produce images (Max 30 items per batch)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    include_batch_filenames = st.checkbox(
        "Include original filenames in batch results",
        value=False,
        help="Off by default so personal or identifying filenames are not repeated in the results or export.",
    )

    if batch_files:
        st.caption(f"📁 {len(batch_files)} items queued for bulk inspection.")
        if len(batch_files) > 30:
            st.warning("A batch can contain at most 30 images. Remove extra files to continue.")

    if st.button(
        "Analyze crate batch",
        type="primary",
        disabled=(
            not batch_files
            or len(batch_files) > 30
            or selected_inference_mode == "unavailable_multimodal"
        ),
        width="stretch",
    ):
        progress_bar = st.progress(0)
        batch_rows = []

        for idx, file in enumerate(batch_files):
            try:
                out = run_uploaded(file, batch_temp, batch_hum, selected_inference_mode)
                batch_rows.append({
                    "Filename": file.name if include_batch_filenames else f"Item {idx + 1:02d}",
                    "Produce": out["fruit"],
                    "Status": out["status"],
                    "Quality Score (%)": out["freshness_score"],
                    "Confidence (%)": out["confidence"],
                    "Shelf Life (Days)": out["shelf_life_days"],
                    "Action": (
                        "Review manually"
                        if out.get("model_mode") in {
                            "visual_prototype",
                            "MobileNetV2 freshness-only classifier",
                        }
                        else
                        "Accept / Export" if out["status"].lower() == "fresh"
                        else "Reject / Discard"
                    )
                })
            except Exception as e:
                batch_rows.append({
                    "Filename": file.name,
                    "Produce": "Error",
                    "Status": "Failed",
                    "Quality Score (%)": None,
                    "Confidence (%)": None,
                    "Shelf Life (Days)": None,
                    "Action": f"Analysis failed ({type(e).__name__})"
                })
            progress_bar.progress((idx + 1) / len(batch_files))

        st.session_state["last_batch"] = pd.DataFrame(batch_rows)

    if "last_batch" in st.session_state:
        batch_df = st.session_state["last_batch"]
        st.markdown("### 📊 Batch QC Inspection Summary")
        total_items = len(batch_df)
        fresh_items = int((batch_df["Status"].str.lower() == "fresh").sum())
        failed_items = int((batch_df["Status"].str.lower() == "failed").sum())
        other_items = total_items - fresh_items - failed_items
        summary_cols = st.columns(4)
        summary_cols[0].metric("Images analyzed", total_items)
        summary_cols[1].metric("Fresh", fresh_items)
        summary_cols[2].metric("Needs review", other_items)
        summary_cols[3].metric("Failed", failed_items)

        status_filter = st.selectbox(
            "Filter batch results",
            ["All results", "Fresh", "Not fresh", "Failed"],
            key="batch_status_filter",
        )
        if status_filter == "Fresh":
            visible_batch = batch_df[batch_df["Status"].str.lower() == "fresh"]
        elif status_filter == "Not fresh":
            visible_batch = batch_df[
                ~batch_df["Status"].str.lower().isin(["fresh", "failed"])
            ]
        elif status_filter == "Failed":
            visible_batch = batch_df[batch_df["Status"].str.lower() == "failed"]
        else:
            visible_batch = batch_df

        st.dataframe(visible_batch, hide_index=True)
        if failed_items:
            st.warning(f"{failed_items} image(s) could not be analyzed. Sensitive error details are hidden.")

        batch_chart_button_col, batch_export_col = st.columns([1, 2])
        with batch_chart_button_col:
            if st.button(
                "📊 Show batch quality chart",
                key="toggle_batch_chart",
                width="stretch",
            ):
                st.session_state["show_batch_chart"] = not st.session_state.get("show_batch_chart", False)
        if st.session_state.get("show_batch_chart"):
            chart_data = batch_df.copy()
            chart_data["Quality Score (%)"] = pd.to_numeric(
                chart_data["Quality Score (%)"], errors="coerce"
            )
            chart_data = chart_data.dropna(subset=["Quality Score (%)"])
            if chart_data.empty:
                st.info("No quality scores are available to chart.")
            else:
                st.bar_chart(
                    chart_data.set_index("Filename")[["Quality Score (%)"]],
                    y="Quality Score (%)",
                )

        # Batch Download CSV
        with batch_export_col:
            st.download_button(
                "📥 Download Batch Inspection CSV",
                data=batch_df.to_csv(index=False),
                file_name="IntelliFresh_Batch_QC_Report.csv",
                mime="text/csv",
                width="stretch",
            )

# Model performance
with tab_metrics:
    st.markdown("#### Model performance")
    st.caption("Evaluation snapshots and training progress from saved model artifacts.")
    artifacts_dir = Path(__file__).resolve().parent / "artifacts" / "freshness"
    metrics_path = artifacts_dir / "baseline_validation.json"
    history_path = artifacts_dir / "history.json"
    if metrics_path.exists():
        try:
            with metrics_path.open(encoding="utf-8") as metrics_file:
                validation = json.load(metrics_file)
            per_class = validation.get("per_class", {})
            matrix = validation.get("confusion_matrix")
            sample_count = int(per_class.get("macro avg", {}).get("support", 0))
            st.markdown(
                f"""
                <div class="performance-card-grid">
                    <div class="performance-card">
                        <div class="performance-card-label">Validation accuracy</div>
                        <div class="performance-card-value">{float(validation['accuracy']) * 100:.2f}%</div>
                        <div class="performance-card-note">Correct predictions in the saved validation run</div>
                    </div>
                    <div class="performance-card">
                        <div class="performance-card-label">Macro F1</div>
                        <div class="performance-card-value">{float(validation['macro_f1']) * 100:.2f}%</div>
                        <div class="performance-card-note">Balanced Fresh and Stale performance</div>
                    </div>
                    <div class="performance-card">
                        <div class="performance-card-label">Validation images</div>
                        <div class="performance-card-value">{sample_count:,}</div>
                        <div class="performance-card-note">Images included in this evaluation</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Saved MobileNetV2 baseline validation results · Not a test-set score.")

            class_rows = []
            for class_name in ("Fresh", "Stale"):
                class_metrics = per_class.get(class_name)
                if class_metrics:
                    class_rows.append({
                        "Class": class_name,
                        "Precision (%)": round(float(class_metrics["precision"]) * 100, 2),
                        "Recall (%)": round(float(class_metrics["recall"]) * 100, 2),
                        "F1 (%)": round(float(class_metrics["f1-score"]) * 100, 2),
                        "Validation samples": int(class_metrics["support"]),
                    })
            st.markdown("##### Explore results")
            accuracy_button_col, class_button_col, matrix_button_col = st.columns(3)
            accuracy_visible = st.session_state.get("show_accuracy_graph", False)
            if accuracy_button_col.button(
                "Hide accuracy graph" if accuracy_visible else "📈 Show accuracy graph",
                key="toggle_accuracy_graph",
                disabled=not history_path.exists(),
                width="stretch",
            ):
                st.session_state["show_accuracy_graph"] = not accuracy_visible
            if class_button_col.button(
                "Hide class metrics" if st.session_state.get("show_class_metrics") else "🎯 Class metrics",
                key="toggle_class_metrics",
                disabled=not class_rows,
                width="stretch",
            ):
                st.session_state["show_class_metrics"] = not st.session_state.get("show_class_metrics", False)
            if matrix_button_col.button(
                "Hide confusion matrix" if st.session_state.get("show_confusion_matrix") else "🧩 Confusion matrix",
                key="toggle_confusion_matrix",
                disabled=not matrix or len(matrix) != 2,
                width="stretch",
            ):
                st.session_state["show_confusion_matrix"] = not st.session_state.get("show_confusion_matrix", False)

            if class_rows and st.session_state.get("show_class_metrics"):
                st.markdown("##### Class performance")
                class_df = pd.DataFrame(class_rows)
                st.dataframe(class_df, hide_index=True)
                class_chart_data = class_df.melt(
                    id_vars="Class",
                    value_vars=["Precision (%)", "Recall (%)", "F1 (%)"],
                    var_name="Metric",
                    value_name="Score (%)",
                )
                class_chart = (
                    alt.Chart(class_chart_data)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("Score (%):Q", title="Score", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("Class:N", title=None, sort=["Fresh", "Stale"]),
                        color=alt.Color(
                            "Metric:N",
                            scale=alt.Scale(
                                domain=["Precision (%)", "Recall (%)", "F1 (%)"],
                                range=["#f07850", "#7655d5", "#21a6a2"],
                            ),
                            legend=alt.Legend(orient="top"),
                        ),
                        xOffset="Metric:N",
                        tooltip=["Class:N", "Metric:N", alt.Tooltip("Score (%):Q", format=".2f")],
                    )
                    .properties(height=220, background="#ffffff")
                    .configure_axis(
                        labelColor="#51475c",
                        titleColor="#40334e",
                        gridColor="#eee8f5",
                    )
                    .configure_legend(labelColor="#51475c", title=None)
                    .configure_view(stroke=None)
                )
                st.altair_chart(
                    class_chart,
                    alt="Precision, recall, and F1 scores for Fresh and Stale produce.",
                )

            if matrix and len(matrix) == 2 and st.session_state.get("show_confusion_matrix"):
                matrix_col, explanation_col = st.columns([1.2, 1])
                with matrix_col:
                    st.markdown("##### Confusion matrix")
                    matrix_df = pd.DataFrame(
                        matrix,
                        index=["Actual fresh", "Actual stale"],
                        columns=["Predicted fresh", "Predicted stale"],
                    )
                    st.dataframe(matrix_df)
                with explanation_col:
                    st.markdown("##### How to read it")
                    st.write("Rows are the actual class; columns are the model prediction.")
                    st.caption(
                        f"Correct fresh: {matrix[0][0]:,} · Fresh classified stale: {matrix[0][1]:,}"
                    )
                    st.caption(
                        f"Stale classified fresh: {matrix[1][0]:,} · Correct stale: {matrix[1][1]:,}"
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(
                f"Saved validation metrics could not be displayed ({type(exc).__name__}). "
                "Check the validation JSON artifact."
            )
    else:
        st.info("No saved validation metrics are available yet. Train or evaluate a model to populate this panel.")

    if history_path.exists():
        try:
            with history_path.open(encoding="utf-8") as history_file:
                training_history = pd.DataFrame(json.load(history_file))
            if not training_history.empty and {"epoch", "validation_accuracy", "validation_macro_f1"}.issubset(training_history.columns):
                if st.session_state.get("show_accuracy_graph"):
                    st.markdown("##### Validation accuracy by epoch")
                    history_chart = (
                        training_history.set_index("epoch")[
                            ["validation_accuracy", "validation_macro_f1"]
                        ].rename(columns={
                            "validation_accuracy": "Validation accuracy",
                            "validation_macro_f1": "Validation macro F1",
                        }) * 100
                    )
                    accuracy_chart_data = (
                        history_chart.rename_axis("Epoch")
                        .reset_index()
                        .melt(id_vars="Epoch", var_name="Metric", value_name="Score (%)")
                    )
                    low_score = max(0, math.floor(accuracy_chart_data["Score (%)"].min() - 1))
                    high_score = min(100, math.ceil(accuracy_chart_data["Score (%)"].max() + 1))
                    accuracy_chart = (
                        alt.Chart(accuracy_chart_data)
                        .mark_line(point=True, strokeWidth=3)
                        .encode(
                            x=alt.X(
                                "Epoch:Q",
                                title="Training epoch",
                                axis=alt.Axis(format="d", tickMinStep=1),
                            ),
                            y=alt.Y(
                                "Score (%):Q",
                                title="Validation score",
                                scale=alt.Scale(domain=[low_score, high_score]),
                                axis=alt.Axis(labelExpr="datum.value + '%'"),
                            ),
                            color=alt.Color(
                                "Metric:N",
                                scale=alt.Scale(
                                    domain=["Validation accuracy", "Validation macro F1"],
                                    range=["#ed684f", "#7052c4"],
                                ),
                                legend=alt.Legend(orient="top"),
                            ),
                            tooltip=[
                                alt.Tooltip("Epoch:Q", format=".0f"),
                                "Metric:N",
                                alt.Tooltip("Score (%):Q", format=".2f"),
                            ],
                        )
                        .properties(height=340, background="#ffffff")
                        .configure_axis(
                            labelColor="#51475c",
                            titleColor="#40334e",
                            gridColor="#eee8f5",
                        )
                        .configure_legend(labelColor="#51475c", title=None)
                        .configure_view(stroke=None)
                    )
                    st.altair_chart(
                        accuracy_chart,
                        alt="Validation accuracy and macro F1 by training epoch.",
                    )
                if st.download_button(
                    "⬇️ Download training history",
                    data=training_history.to_csv(index=False),
                    file_name="IntelliFresh_Validation_History.csv",
                    mime="text/csv",
                    key="download_training_history",
                ):
                    st.toast("Training history download is ready.")
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"Saved training history could not be displayed ({type(exc).__name__}).")
