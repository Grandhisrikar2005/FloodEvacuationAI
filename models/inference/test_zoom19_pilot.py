
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOSAIC_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "zoom19_pilot"
    / "vijayawada_zoom19_pilot.tif"
)

# Corrected model location
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "best_model.pth"
)

# Must match the class order used during training.
CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

PATCH_SIZE = 224

GRID_POSITIONS = [
    ("Top Left", 0.17, 0.17),
    ("Top Centre", 0.50, 0.17),
    ("Top Right", 0.83, 0.17),
    ("Middle Left", 0.17, 0.50),
    ("Centre", 0.50, 0.50),
    ("Middle Right", 0.83, 0.50),
    ("Bottom Left", 0.17, 0.83),
    ("Bottom Centre", 0.50, 0.83),
    ("Bottom Right", 0.83, 0.83),
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# MODEL LOADING
# ==================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASS_NAMES),
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ==================================================
# PILOT PATCH TEST
# ==================================================

def main():
    if not MOSAIC_PATH.exists():
        raise FileNotFoundError(
            f"Pilot mosaic not found:\n{MOSAIC_PATH}"
        )

    print("=" * 65)
    print("MOBILENETV2 — ZOOM-19 PILOT TEST")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")
    print(f"Mosaic : {MOSAIC_PATH}")

    model = load_model()

    print("\nModel loaded successfully.")
    print("\nPREDICTIONS")
    print("-" * 65)

    with rasterio.open(MOSAIC_PATH) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 RGB bands; found {src.count}."
            )

        width = src.width
        height = src.height

        print(f"Image dimensions: {width} x {height}\n")

        with torch.inference_mode():
            for label, fx, fy in GRID_POSITIONS:
                center_x = int(fx * width)
                center_y = int(fy * height)

                x0 = max(
                    0,
                    min(
                        center_x - PATCH_SIZE // 2,
                        width - PATCH_SIZE,
                    ),
                )
                y0 = max(
                    0,
                    min(
                        center_y - PATCH_SIZE // 2,
                        height - PATCH_SIZE,
                    ),
                )

                patch = src.read(
                    indexes=[1, 2, 3],
                    window=(
                        (y0, y0 + PATCH_SIZE),
                        (x0, x0 + PATCH_SIZE),
                    ),
                )

                patch = np.moveaxis(patch, 0, -1)
                patch_image = Image.fromarray(
                    patch.astype(np.uint8)
                )

                input_tensor = PREPROCESS(
                    patch_image
                ).unsqueeze(0).to(DEVICE)

                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

                confidence, predicted_index = torch.max(
                    probabilities, dim=0
                )

                predicted_class = CLASS_NAMES[
                    predicted_index.item()
                ]
                confidence_pct = confidence.item() * 100

                print(
                    f"{label:16s} | "
                    f"{predicted_class:12s} | "
                    f"{confidence_pct:6.2f}%"
                )

                if confidence_pct < 60:
                    print(
                        " " * 19
                        + "Warning: low-confidence prediction"
                    )

    print("\n" + "=" * 65)
    print("PILOT TEST COMPLETED")
    print("=" * 65)
    print(
        "Note: confidence is not proof of correctness. "
        "These nine patches are a diagnostic test, "
        "not an accuracy evaluation."
    )


if __name__ == "__main__":
    main()
