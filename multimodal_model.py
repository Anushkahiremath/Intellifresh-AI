"""PyTorch models for 24-class fruit quality and shelf-life prediction."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


DEFAULT_CLASSES = [
    f"{fruit}_{condition}"
    for fruit in (
        "apple", "banana", "cherry", "grape", "kiwi", "mango",
        "orange", "peach", "pear", "plum", "strawberry", "watermelon",
    )
    for condition in ("fresh", "rotten")
]


class MultimodalResNet50(nn.Module):
    """2048 image features + 32 environmental features = 2080 fused features."""

    image_feature_dim = 2048
    environment_feature_dim = 32
    fusion_feature_dim = 2080

    def __init__(self, num_classes: int = 24, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.environment_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )
        self.shelf_life_regressor = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Softplus(),
        )

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.image_encoder[-2][-1].conv3

    def forward(self, images: torch.Tensor, environment: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = torch.flatten(self.image_encoder(images), 1)
        environment_features = self.environment_encoder(environment)
        fused = torch.cat((image_features, environment_features), dim=1)
        return {
            "logits": self.classifier(fused),
            "shelf_life": self.shelf_life_regressor(fused).squeeze(1),
            "features": fused,
        }


class ImageOnlyResNet50(nn.Module):
    """Ablation baseline with identical output heads and no environmental input."""

    def __init__(self, num_classes: int = 24, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)
        self.shelf_life_regressor = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1), nn.Softplus())

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.image_encoder[-2][-1].conv3

    def forward(self, images: torch.Tensor, environment: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = torch.flatten(self.image_encoder(images), 1)
        return {
            "logits": self.classifier(features),
            "shelf_life": self.shelf_life_regressor(features).squeeze(1),
            "features": features,
        }
