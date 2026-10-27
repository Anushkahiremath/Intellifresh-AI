"""Singleton MobileNetV2 inference service."""
from __future__ import annotations
from pathlib import Path
import os
from PIL import Image, ImageOps

class ModelUnavailableError(RuntimeError): pass

class FreshnessPredictor:
    labels = ["Fresh", "Stale"]
    def __init__(self, model_path):
        self.model_path = Path(os.getenv("FRESHNESS_MODEL_PATH") or model_path); self.model = None; self.device = "unavailable"; self.load_error = None
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            self.load_error = f"Model weights not found at {self.model_path}. Run python download_model.py."
            return
        try:
            import torch
            from torchvision import models
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.mobilenet_v2(weights=None)
            model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(model.classifier[1].in_features, 2))
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state); model.to(self.device); model.eval(); self.model = model
            print(f"FreshVision MobileNetV2 loaded on {self.device}")
        except Exception as exc:
            self.load_error = f"Could not load MobileNetV2 weights: {exc}"; self.model = None

    @property
    def available(self): return self.model is not None

    def preprocess_image(self, image_or_path):
        from torchvision import transforms
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        if isinstance(image_or_path, (str, Path)):
            with Image.open(image_or_path) as image: return transform(ImageOps.exif_transpose(image).convert("RGB"))
        return transform(ImageOps.exif_transpose(image_or_path).convert("RGB"))

    def predict(self, image_or_path):
        if not self.available: raise ModelUnavailableError(self.load_error or "Model is unavailable.")
        import torch
        tensor = self.preprocess_image(image_or_path).unsqueeze(0).to(self.device)
        with torch.no_grad(): probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax().item())
        return {"prediction": self.labels[index], "confidence": round(float(probabilities[index])*100,2), "fresh_probability": round(float(probabilities[0])*100,2), "stale_probability": round(float(probabilities[1])*100,2), "tensor": tensor, "class_index": index}

class ProducePredictor:
    """EfficientNet-B0 produce identifier with labels supplied by its repository."""
    supported_freshness = {"apple", "banana", "bitter gourd", "capsicum", "orange", "tomato"}
    def __init__(self, model_path, config_path):
        self.model_path=Path(model_path); self.config_path=Path(config_path); self.model=None; self.device="unavailable"; self.load_error=None; self.labels={}
        self.load_model()
    def load_model(self):
        if not self.model_path.exists() or not self.config_path.exists():
            self.load_error="Produce identification weights/config are missing. Run python download_model.py."; return
        try:
            import json, torch
            from torchvision.models import efficientnet_b0
            config=json.loads(self.config_path.read_text(encoding="utf-8")); self.labels={int(k):v for k,v in config["id2label"].items()}
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model=efficientnet_b0(weights=None); model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,len(self.labels))
            state=torch.load(self.model_path,map_location=self.device,weights_only=True)
            if isinstance(state,dict) and "model_state_dict" in state: state=state["model_state_dict"]
            elif isinstance(state,dict) and "state_dict" in state: state=state["state_dict"]
            model.load_state_dict(state); model.to(self.device).eval(); self.model=model
            print(f"FreshVision EfficientNet-B0 ({len(self.labels)} labels) loaded on {self.device}")
        except Exception as exc: self.load_error=f"Could not load produce identifier: {exc}"; self.model=None
    @property
    def available(self): return self.model is not None
    def predict(self,image_or_path):
        if not self.available: return {"produce_name":"Unavailable","produce_confidence":0.0,"freshness_coverage":"Unknown"}
        import torch
        from torchvision import transforms
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        if isinstance(image_or_path,(str,Path)):
            with Image.open(image_or_path) as image: tensor=transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        else: tensor=transform(image_or_path.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(): probabilities=torch.softmax(self.model(tensor),1)[0]
        confidence,index=probabilities.max(0); name=self.labels[int(index.item())]
        normalized=name.lower().replace("_"," ").strip()
        coverage="Supported" if any(item in normalized for item in self.supported_freshness) else "Experimental"
        return {"produce_name":name,"produce_confidence":round(float(confidence)*100,2),"freshness_coverage":coverage}
