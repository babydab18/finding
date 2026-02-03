# Aerial Animal Species Classifier

This project provides a starter pipeline for training a machine-learning model that can identify animal species from aerial imagery. It includes data layout guidance, a training script, and an inference helper so you can iterate on model choices and datasets quickly.

## 1) Data preparation

Organize your dataset and labels with a CSV format. The default configuration expects:

```
data/
  images/
    image_001.jpg
    image_002.jpg
  train.csv
  val.csv
```

Each CSV should contain two columns:

| image_path | label |
| --- | --- |
| image_001.jpg | 0 |
| image_002.jpg | 1 |

Labels should be integer-encoded class IDs (0..N-1). Keep a separate mapping of IDs to species names for inference.

## 2) Configuration

Update `src/config.yaml` to match your dataset:

- `data.train_csv` / `data.val_csv` to the CSV files.
- `data.image_root` for your image folder.
- `model.num_classes` to match your species count.
- `training.device` to `cuda` (if available) or `cpu`.

## 3) Install dependencies

```
pip install -r requirements.txt
```

## 4) Train the model

```
python src/train.py
```

This writes checkpoints to `outputs/checkpoints` and saves the best model to `outputs/checkpoints/best_model.pt`.

## 5) Run inference

Update the `classes` list in `src/infer.py` to match your species mapping, then run:

```
python src/infer.py
```

## Next steps

- Add data augmentation tailored to your aerial imagery (e.g., color jitter, random crops based on altitude).
- Expand to more architectures in `src/model.py`.
- Add experiment tracking (Weights & Biases or MLflow).
- Collect more diverse aerial imagery to improve generalization.
