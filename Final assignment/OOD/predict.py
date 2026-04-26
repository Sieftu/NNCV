"""
OOD inference entrypoint. For each input image, writes a single-channel
segmentation mask to /output and appends an (image_name, include) row to
/output/predictions.csv where include is False for images flagged as OOD.
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    pred_max = torch.argmax(pred, dim=1, keepdim=True).float()
    pred_max = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    return pred_max.cpu().detach().numpy().squeeze()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))
    print(f"Found {len(image_files)} images to process.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                seg_pred, include_decision = model(img_tensor)

            seg_pred = postprocess(seg_pred, original_shape)

            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

            predictions.append({
                'image_name': str(relative_path).replace('\\', '/'),
                'include': bool(include_decision),
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")


if __name__ == "__main__":
    main()
