import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import AerialSpeciesDataset, DatasetConfig
from model import build_model, get_device


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return accuracy_score(targets, predictions)


def train(config_path: Path) -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    device = get_device(config["training"]["device"])

    train_transform, val_transform = build_transforms()

    train_dataset = AerialSpeciesDataset(
        DatasetConfig(
            csv_path=Path(config["data"]["train_csv"]),
            image_root=Path(config["data"]["image_root"]),
        ),
        transform=train_transform,
    )
    val_dataset = AerialSpeciesDataset(
        DatasetConfig(
            csv_path=Path(config["data"]["val_csv"]),
            image_root=Path(config["data"]["image_root"]),
        ),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    metrics = {"val_accuracy": []}

    checkpoints_dir = Path(config["output"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=sum(epoch_losses) / len(epoch_losses))

        val_accuracy = evaluate(model, val_loader, device)
        metrics["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), config["output"]["best_model_path"])

    metrics_path = Path(config["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    train(Path("src/config.yaml"))
