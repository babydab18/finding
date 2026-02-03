import torch
from torch import nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def get_device(device_preference: str) -> torch.device:
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
