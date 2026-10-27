"""Dependency-free MobileNetV2 Grad-CAM; failures are non-fatal."""
from __future__ import annotations
import numpy as np
from PIL import Image

def create_gradcam(model, tensor, class_index: int, original_path, output_path):
    import torch
    activations, gradients = [], []
    def capture(_module, _inputs, output):
        activations.append(output); output.register_hook(lambda grad: gradients.append(grad))
    handle = model.features[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True); logits = model(tensor); logits[0,class_index].backward()
        weights = gradients[0].mean((2,3), keepdim=True)
        cam = torch.relu((weights*activations[0]).sum(1))[0]
        cam = (cam-cam.min())/(cam.max()-cam.min()).clamp_min(1e-8)
        with Image.open(original_path) as source: original=source.convert("RGB"); size=original.size
        heat = np.asarray(Image.fromarray(np.uint8(cam.detach().cpu().numpy()*255)).resize(size,Image.Resampling.BILINEAR),dtype=np.float32)/255
        rgb=np.asarray(original,dtype=np.float32)/255
        color=np.stack([heat, np.maximum(0,1-abs(heat-.5)*2)*.55, 1-heat],axis=-1)
        Image.fromarray(np.uint8(np.clip(rgb*.58+color*.42,0,1)*255)).save(output_path)
        return True
    except Exception as exc:
        print(f"Grad-CAM skipped: {exc}"); return False
    finally: handle.remove()
