
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================
ROOT = Path(__file__).resolve().parents[2]

INPUT_RASTER = (
    ROOT / "dataset" / "zoom19_pilot"
    / "vijayawada_zoom19_pilot.tif"
)

# Use the experimental model, not the original checkpoint.
MODEL_PATH = (
    ROOT / "models" / "saved_models"
    / "expanded_best_model.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "landcover"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TIF = OUTPUT_DIR / "vijayawada_landcover_expanded.tif"
OUTPUT_PNG = OUTPUT_DIR / "vijayawada_landcover_expanded_preview.png"

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

# Class colours for visualization: RGB
PALETTE = np.array([
    [230, 190, 70],   # Agriculture
    [170, 140, 110],  # Barren
    [220, 50, 50],    # Roads
    [160, 70, 180],   # Urban
    [40, 160, 70],    # Vegetation
    [30, 100, 220],   # Water
], dtype=np.uint8)

IMAGE_SIZE = 224
PATCH_SIZE = 224
STRIDE = 112
BATCH_SIZE = 16
NUM_CLASSES = len(CLASS_NAMES)
NODATA_VALUE = 255

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# 2. CHECK INPUT FILES
# =====================================================
if not INPUT_RASTER.exists():
    raise FileNotFoundError(
        f"Satellite mosaic not found:\n{INPUT_RASTER}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Experimental model not found:\n{MODEL_PATH}"
    )

print("=" * 65)
print("EXPERIMENTAL LAND-COVER MAP GENERATION")
print("=" * 65)
print(f"Device: {device}")
print(f"Input raster: {INPUT_RASTER}")
print(f"Model: {MODEL_PATH}")


# =====================================================
# 3. LOAD EXPERIMENTAL MODEL
# =====================================================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.last_channel,
    NUM_CLASSES,
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Experimental model loaded successfully.")


# =====================================================
# 4. READ SATELLITE MOSAIC
# =====================================================
with rasterio.open(INPUT_RASTER) as src:
    if src.count < 3:
        raise ValueError(
            "Input raster must have at least three RGB bands."
        )

    raster = src.read([1, 2, 3])
    profile = src.profile.copy()
    height, width = src.height, src.width

print(f"Raster dimensions: {width} x {height}")
print(f"Raster CRS: {profile.get('crs')}")
print(f"Raster transform: {profile.get('transform')}")


# =====================================================
# 5. PREDICT OVERLAPPING PATCHES
# =====================================================
# Each patch votes for its predicted class over its footprint.
# Overlapping votes are combined using majority voting.
vote_counts = np.zeros(
    (NUM_CLASSES, height, width),
    dtype=np.uint16,
)

coverage = np.zeros(
    (height, width),
    dtype=np.uint16,
)

patch_tensors = []
patch_positions = []


def predict_batch():
    """Predict the queued patches and accumulate their votes."""
    if not patch_tensors:
        return

    batch = torch.stack(patch_tensors).to(device)

    with torch.inference_mode():
        logits = model(batch)
        predictions = logits.argmax(dim=1).cpu().numpy()

    for prediction, (row, col, patch_height, patch_width) in zip(
        predictions, patch_positions
    ):
        vote_counts[
            int(prediction),
            row:row + patch_height,
            col:col + patch_width,
        ] += 1

        coverage[
            row:row + patch_height,
            col:col + patch_width,
        ] += 1

    patch_tensors.clear()
    patch_positions.clear()


# Include the final edge positions so the whole raster is covered.
def make_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    final_position = length - patch_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


row_positions = make_positions(height, PATCH_SIZE, STRIDE)
col_positions = make_positions(width, PATCH_SIZE, STRIDE)

total_patches = len(row_positions) * len(col_positions)
processed_patches = 0

print(f"Patches to process: {total_patches}")
print("Generating predictions...")

for row in row_positions:
    for col in col_positions:
        patch_height = min(PATCH_SIZE, height - row)
        patch_width = min(PATCH_SIZE, width - col)

        crop = raster[
            :,
            row:row + patch_height,
            col:col + patch_width,
        ]

        # Pad small edge crops to 224 x 224.
        padded = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

        rgb_crop = np.transpose(crop, (1, 2, 0))
        padded[:patch_height, :patch_width, :] = rgb_crop

        patch_image = Image.fromarray(padded, mode="RGB")
        patch_tensors.append(preprocess(patch_image))
        patch_positions.append(
            (row, col, patch_height, patch_width)
        )

        if len(patch_tensors) >= BATCH_SIZE:
            predict_batch()

        processed_patches += 1

    print(
        f"Processed rows through {row + PATCH_SIZE} "
        f"({processed_patches}/{total_patches} patches)"
    )

predict_batch()


# =====================================================
# 6. BUILD THE CLASS MAP
# =====================================================
landcover_map = np.full(
    (height, width),
    NODATA_VALUE,
    dtype=np.uint8,
)

valid_pixels = coverage > 0

landcover_map[valid_pixels] = np.argmax(
    vote_counts[:, valid_pixels],
    axis=0,
).astype(np.uint8)


# =====================================================
# 7. REPORT CLASS PROPORTIONS
# =====================================================
print("\nPREDICTED CLASS PROPORTIONS")
print("-" * 40)

total_valid_pixels = int(valid_pixels.sum())

for class_index, class_name in enumerate(CLASS_NAMES):
    count = int(
        np.count_nonzero(
            landcover_map[valid_pixels] == class_index
        )
    )

    percentage = (
        100.0 * count / total_valid_pixels
        if total_valid_pixels
        else 0.0
    )

    print(
        f"{class_name:12s}: "
        f"{percentage:6.2f}% ({count:,} pixels)"
    )

print("-" * 40)
print(f"Covered pixels: {total_valid_pixels:,}")
print(
    "Note: these are model predictions, "
    "not validated land-cover measurements."
)


# =====================================================
# 8. SAVE GEOREFERENCED GEOTIFF
# =====================================================
output_profile = profile.copy()
output_profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=NODATA_VALUE,
    compress="lzw",
)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    **output_profile,
) as destination:
    destination.write(landcover_map, 1)

print(f"\nGeoTIFF saved:\n{OUTPUT_TIF}")


# =====================================================
# 9. SAVE COLOUR PREVIEW
# =====================================================
preview = np.zeros(
    (height, width, 3),
    dtype=np.uint8,
)

preview[valid_pixels] = PALETTE[
    landcover_map[valid_pixels]
]

Image.fromarray(preview, mode="RGB").save(
    OUTPUT_PNG,
)

print(f"Preview saved:\n{OUTPUT_PNG}")

print("\nMAP GENERATION COMPLETE")
print("This map is experimental and must be validated before use")
print("in flood-risk calculations.")
