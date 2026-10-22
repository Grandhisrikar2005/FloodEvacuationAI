from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import rasterio
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "best_model.pth"
)

INPUT_RASTER = (
    PROJECT_ROOT
    / "dataset"
    / "georeferenced_landcover_utm44.tif"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water"
]


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 60)
print("TESTING LAND-COVER MODEL ON GEOREFERENCED MOSAIC")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not INPUT_RASTER.exists():

    raise FileNotFoundError(
        f"\nLand-cover raster not found:\n{INPUT_RASTER}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading MobileNetV2...")

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# OPEN GEOREFERENCED RASTER
# ============================================================

with rasterio.open(INPUT_RASTER) as src:

    print()
    print("Raster information:")
    print("Width :", src.width)
    print("Height:", src.height)
    print("CRS   :", src.crs)
    print("Bands :", src.count)

    image_width = src.width
    image_height = src.height

    # --------------------------------------------------------
    # Test patch locations
    # --------------------------------------------------------

    patch_locations = [

        ("Top Left", 0, 0),

        (
            "Top Centre",
            0,
            max(0, (image_width - IMAGE_SIZE) // 2)
        ),

        (
            "Top Right",
            0,
            max(0, image_width - IMAGE_SIZE)
        ),

        (
            "Centre",
            max(0, (image_height - IMAGE_SIZE) // 2),
            max(0, (image_width - IMAGE_SIZE) // 2)
        ),

        (
            "Bottom Left",
            max(0, image_height - IMAGE_SIZE),
            0
        ),

        (
            "Bottom Centre",
            max(0, image_height - IMAGE_SIZE),
            max(0, (image_width - IMAGE_SIZE) // 2)
        ),

        (
            "Bottom Right",
            max(0, image_height - IMAGE_SIZE),
            max(0, image_width - IMAGE_SIZE)
        )
    ]

    print()
    print("=" * 60)
    print("PATCH PREDICTIONS")
    print("=" * 60)

    # --------------------------------------------------------
    # Process patches
    # --------------------------------------------------------

    for name, row, col in patch_locations:

        print()
        print("-" * 60)
        print(f"Patch: {name}")
        print(f"Raster position: row={row}, col={col}")

        # Read RGB patch
        patch = src.read(
            [1, 2, 3],
            window=rasterio.windows.Window(
                col,
                row,
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        # ----------------------------------------------------
        # Convert CHW -> HWC
        # ----------------------------------------------------

        patch = np.transpose(
            patch,
            (1, 2, 0)
        )

        # ----------------------------------------------------
        # Convert to uint8
        # ----------------------------------------------------

        patch = np.clip(
            patch,
            0,
            255
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Convert NumPy -> PIL
        # ----------------------------------------------------

        pil_image = Image.fromarray(
            patch,
            mode="RGB"
        )

        # ----------------------------------------------------
        # Apply exact training preprocessing
        # ----------------------------------------------------

        tensor = transform(
            pil_image
        )

        tensor = tensor.unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        class_index = prediction.item()
        confidence_value = confidence.item() * 100

        predicted_class = CLASS_NAMES[
            class_index
        ]

        print(
            f"Prediction : {predicted_class}"
        )

        print(
            f"Confidence : {confidence_value:.2f}%"
        )

        print(
            "All class probabilities:"
        )

        for i, class_name in enumerate(CLASS_NAMES):

            probability = (
                probabilities[0, i].item()
                * 100
            )

            print(
                f"  {class_name:<12}: "
                f"{probability:6.2f}%"
            )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)