"""Grad-CAM generation for the ResNet image branch."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def generate_gradcam(model, image_tensor: torch.Tensor, environment: torch.Tensor, output_path, class_index: int | None = None):
    activations, gradients = [], []
    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))
    forward_handle = model.gradcam_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        output = model(image_tensor, environment)
        index = int(output["logits"].argmax(1).item()) if class_index is None else class_index
        output["logits"][0, index].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1))[0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        heatmap_array = np.asarray(Image.fromarray(np.uint8(heatmap.detach().cpu().numpy() * 255)).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        mean = np.array([.485, .456, .406]); std = np.array([.229, .224, .225])
        original = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * std + mean
        original = np.clip(original, 0, 1)
        colored = np.stack((heatmap_array, np.maximum(0, 1-abs(heatmap_array-.5)*2)*.55, 1-heatmap_array), axis=-1)
        overlay = np.uint8(np.clip(original*.55 + colored*.45, 0, 1)*255)
        Image.fromarray(overlay).save(output_path)
        return output_path
    finally:
        forward_handle.remove()
