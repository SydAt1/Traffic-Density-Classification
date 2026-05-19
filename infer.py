#!/usr/bin/env python3
"""
EfficientNet-B0 traffic density inference script.

Usage:
    python infer.py image1.jpg image2.jpg ...
    python infer.py path/to/folder/
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

WEIGHTS_PATH = Path(__file__).parent / "efficientnet_b0_weights.pth"
CLASS_NAMES = ["heavy", "incident", "light", "moderate"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        l_eq = self.clahe.apply(l)
        img_lab_eq = cv2.merge((l_eq, a, b))
        img_rgb = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, len(CLASS_NAMES)),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image_path: Path, device: torch.device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    confidence, idx = torch.max(probs, 0)
    return {
        "path": str(image_path),
        "label": CLASS_NAMES[idx.item()],
        "confidence": confidence.item(),
        "probabilities": {cls: probs[i].item() for i, cls in enumerate(CLASS_NAMES)},
    }


def collect_images(paths: list[str]) -> list[Path]:
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            images.extend(f for f in sorted(path.iterdir()) if f.suffix.lower() in IMG_EXTENSIONS)
        elif path.suffix.lower() in IMG_EXTENSIONS:
            images.append(path)
        else:
            print(f"Skipping {p} (not a recognised image type)")
    return images


def main():
    parser = argparse.ArgumentParser(description="Traffic density classifier (EfficientNet-B0)")
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument("--weights", default=str(WEIGHTS_PATH), help="Path to .pth weights file")
    parser.add_argument("--no-probs", action="store_true", help="Hide per-class probabilities")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = load_model(Path(args.weights), device)

    image_paths = collect_images(args.images)
    if not image_paths:
        print("No images found.")
        sys.exit(1)

    for img_path in image_paths:
        try:
            result = predict(model, img_path, device)
            label = result["label"].upper()
            conf = result["confidence"] * 100
            print(f"\n{img_path.name}")
            print(f"  Prediction : {label}  ({conf:.1f}%)")
            if not args.no_probs:
                for cls, prob in result["probabilities"].items():
                    bar = "█" * int(prob * 20)
                    print(f"  {cls:<10} {prob*100:5.1f}%  {bar}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
