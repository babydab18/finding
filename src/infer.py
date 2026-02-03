from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from model import build_model, get_device


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image_path: Path, config_path: Path, class_names: list[str]) -> tuple[str, float]:
    config = load_config(config_path)
    device = get_device(config["training"]["device"])
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(torch.load(config["output"]["best_model_path"], map_location=device))
    model.to(device)
    model.eval()

    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, dim=1)
    label = class_names[pred.item()]
    return label, confidence.item()


if __name__ == "__main__":
    classes = ["elephant", "giraffe", "zebra", "lion", "buffalo"]
    prediction, score = predict(Path("data/sample.jpg"), Path("src/config.yaml"), classes)
    print(f"Prediction: {prediction} ({score:.2%})")
