"""Inference facade for the trained multimodal checkpoint.

Without a checkpoint the app reports visual-only diagnostics and never substitutes
a heuristic value for trained shelf-life regression.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "artifacts" / "multimodal.pt"
_runtime = None

def _load_runtime():
    global _runtime
    if _runtime is not None: return _runtime
    checkpoint_path = Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint_path.exists(): _runtime = False; return _runtime
    import torch
    from dataset import transforms_for
    from multimodal_model import MultimodalResNet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultimodalResNet50(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    _runtime = {"model": model, "device": device, "classes": checkpoint["classes"], "stats": checkpoint["environment_stats"], "transform": transforms_for("test"), "path": checkpoint_path}
    return _runtime

def model_status() -> dict:
    runtime = _load_runtime()
    return {"loaded": bool(runtime), "checkpoint": str(runtime["path"]) if runtime else None, "mode": "multimodal_resnet50" if runtime else "visual_prototype"}

def analyze_fruit(path: str | Path, requested_type: str = "auto", temperature_c: float = 20.0, humidity_pct: float = 65.0, gradcam_path: str | Path | None = None) -> dict:
    temperature_c, humidity_pct = float(temperature_c), float(humidity_pct)
    if not -30 <= temperature_c <= 60: raise ValueError("temperature_c must be between -30 and 60")
    if not 0 <= humidity_pct <= 100: raise ValueError("humidity_pct must be between 0 and 100")
    runtime = _load_runtime()
    if not runtime: return _visual_fallback(path, temperature_c, humidity_pct)
    import torch
    with Image.open(path) as source: tensor = runtime["transform"](source.convert("RGB")).unsqueeze(0).to(runtime["device"])
    mean = torch.tensor(runtime["stats"]["mean"], device=runtime["device"])
    std = torch.tensor(runtime["stats"]["std"], device=runtime["device"])
    environment = (torch.tensor([[temperature_c, humidity_pct]], device=runtime["device"]) - mean) / std
    with torch.no_grad():
        output = runtime["model"](tensor, environment); probability = output["logits"].softmax(1); confidence, index = probability.max(1)
    class_name = runtime["classes"][index.item()]; shelf_days = max(0.0, output["shelf_life"].item())
    if gradcam_path:
        from gradcam import generate_gradcam
        generate_gradcam(runtime["model"], tensor, environment, gradcam_path, index.item())
    rotten_indices = [i for i, name in enumerate(runtime["classes"]) if name.endswith("_rotten")]
    freshness = float(np.clip(100 * (1 - probability[0, rotten_indices].sum().item()), 0, 100))
    grade = "A" if freshness >= 80 else "B" if freshness >= 65 else "C" if freshness >= 45 else "D"
    return {"class_name": class_name, "fruit": class_name.rsplit("_",1)[0].title(), "status": class_name.rsplit("_",1)[1].title(), "grade": grade, "freshness_score": round(freshness,1), "confidence": round(confidence.item()*100,1), "shelf_life_days": round(shelf_days,1), "shelf_life_source": "trained_regression", "model_mode": "multimodal_resnet50", "environment": {"temperature_c": temperature_c, "humidity_pct": humidity_pct}, "metrics": {"Fresh probability": round(freshness,1), "Rotten probability": round(100-freshness,1), "Model confidence": round(confidence.item()*100,1)}, "forecast": [{"day": d, "freshness": round(max(0, freshness*(1-d/max(shelf_days,1))),1)} for d in range(8)], "disclaimer": "Model estimate only. Do not use as the sole food-safety decision."}

def _visual_fallback(path: str | Path, temperature: float, humidity: float) -> dict:
    with Image.open(path) as source: image=source.convert("RGB"); image.thumbnail((640,640))
    rgb=np.asarray(image,dtype=np.float32); gray=np.asarray(image.convert("L"),dtype=np.float32); hsv=np.asarray(image.convert("HSV"),dtype=np.float32)
    saturation,brightness=hsv[...,1].mean()/2.55,hsv[...,2].mean()/2.55; texture=min(100,np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES)).std()*2.15)
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; browning=min(100,(((r>g*1.08)&(g>b*1.12)&(r<185)&(g<145)).mean()*170)+(gray<55).mean()*35)
    freshness=float(np.clip(72+saturation*.2-abs(brightness-58)*.3-browning*.8,0,100))
    return {"class_name": None, "fruit": "Untrained visual estimate", "status": "Prototype", "grade": "N/A", "freshness_score": round(freshness,1), "confidence": 0.0, "shelf_life_days": None, "shelf_life_source": "unavailable_no_checkpoint", "model_mode": "visual_prototype", "environment": {"temperature_c": temperature, "humidity_pct": humidity}, "metrics": {"Color vitality": round(float(saturation),1), "Brightness": round(float(brightness),1), "Browning risk": round(float(browning),1), "Texture detail": round(float(texture),1)}, "forecast": [], "disclaimer": "No trained checkpoint is installed. Shelf life and 24-class predictions are unavailable; this is only a visual diagnostic."}
