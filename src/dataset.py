from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    csv_path: Path
    image_root: Path


class AerialSpeciesDataset(Dataset):
    def __init__(self, config: DatasetConfig, transform=None):
        self.data = pd.read_csv(config.csv_path)
        self.image_root = Path(config.image_root)
        self.transform = transform

        required_columns = {"image_path", "label"}
        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
