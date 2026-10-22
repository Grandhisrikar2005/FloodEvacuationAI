
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================
ROOT = Path(__file__).resolve().parents[2]

RASTER_PATH = (
    ROOT / "dataset" / "zoom19_pilot"
    / "vijayawada_zoom19_pilot.tif"
)

ORIGINAL_MODEL_PATH = (
    ROOT / "models" / "saved_models" / "best_model.pth"
)

EXPERIMENTAL_MODEL_PATH = (
    ROOT / "models" / "saved_models" / "expanded_best_model.pth"
)

PATCH_DIR = (
    ROOT / "dataset" / "zoom19_pilot"
    / "diagnostic_patches"
)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = PATCH_DIR / "model_comparison.jpg"

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

PATCH_SIZE = 224

PATCHES = [
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# 2. LOAD BOTH MODELS
# =====================================================
def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel, len(CLASS_NAMES)
    )

    state = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


print("=" * 72)
print("ORIGINAL VS EXPERIMENTAL MODEL — ZOOM-19 COMPARISON")
print("=" * 72)
print(f"Device: {device}")

if not RASTER_PATH.exists():
    raise FileNotFoundError(f"Satellite raster not found: {RASTER_PATH}")

original_model = load_model(ORIGINAL_MODEL_PATH)
experimental_model = load_model(EXPERIMENTAL_MODEL_PATH)

print("Both models loaded successfully.")


# =====================================================
# 3. READ SATELLITE MOSAIC
# =====================================================
with rasterio.open(RASTER_PATH) as src:
    if src.count < 3:
        raise ValueError("Satellite raster must have at least 3 RGB bands.")

    raster = src.read([1, 2, 3])
    height, width = src.height, src.width

print(f"Satellite dimensions: {width} x {height}")


# =====================================================
# 4. PREDICT IDENTICAL PATCHES
# =====================================================
results = []

for name, x_fraction, y_fraction in PATCHES:
    col = int(x_fraction * width)
    row = int(y_fraction * height)

    col = min(max(col - PATCH_SIZE // 2, 0), width - PATCH_SIZE)
    row = min(max(row - PATCH_SIZE // 2, 0), height - PATCH_SIZE)

    crop = raster[
        :,
        row:row + PATCH_SIZE,
        col:col + PATCH_SIZE,
    ]

    # Convert raster bands to an RGB image.
    rgb = np.transpose(crop, (1, 2, 0)).astype(np.uint8)
    patch_image = Image.fromarray(rgb, mode="RGB")

    patch_path = PATCH_DIR / f"{name.lower().replace(' ', '_')}.jpg"
    patch_image.save(patch_path, quality=95)

    tensor = preprocess(patch_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        original_probs = torch.softmax(
            original_model(tensor), dim=1
        )[0].cpu().numpy()

        experimental_probs = torch.softmax(
            experimental_model(tensor), dim=1
        )[0].cpu().numpy()

    original_index = int(np.argmax(original_probs))
    experimental_index = int(np.argmax(experimental_probs))

    results.append({
        "name": name,
        "image": patch_image.copy(),
        "original_class": CLASS_NAMES[original_index],
        "original_conf": float(original_probs[original_index]),
        "experimental_class": CLASS_NAMES[experimental_index],
        "experimental_conf": float(experimental_probs[experimental_index]),
    })

    print(f"\n{name.upper()}")
    print(
        f"  Original:     {CLASS_NAMES[original_index]:12s} "
        f"({original_probs[original_index] * 100:.2f}%)"
    )
    print(
        f"  Experimental: {CLASS_NAMES[experimental_index]:12s} "
        f"({experimental_probs[experimental_index] * 100:.2f}%)"
    )


# =====================================================
# 5. SUMMARY
# =====================================================
original_water = sum(
    r["original_class"] == "Water" for r in results
)
experimental_water = sum(
    r["experimental_class"] == "Water" for r in results
)

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"Original model Water predictions:     {original_water}/9")
print(f"Experimental model Water predictions: {experimental_water}/9")
print("These are diagnostic patches, not a labelled accuracy test.")


# =====================================================
# 6. SAVE VISUAL COMPARISON SHEET
# =====================================================
cell_width = 330
image_height = 220
text_height = 72
gap = 12
columns = 3
rows = 3

sheet = Image.new(
    "RGB",
    (
        gap + columns * (cell_width + gap),
        gap + rows * (image_height + text_height + gap),
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)

for index, result in enumerate(results):
    row = index // columns
    col = index % columns

    x = gap + col * (cell_width + gap)
    y = gap + row * (image_height + text_height + gap)

    image = result["image"].resize((cell_width, image_height))
    sheet.paste(image, (x, y))

    original_text = (
        f"Original: {result['original_class']} "
        f"{result['original_conf'] * 100:.1f}%"
    )
    experimental_text = (
        f"New: {result['experimental_class']} "
        f"{result['experimental_conf'] * 100:.1f}%"
    )

    draw.text((x + 3, y + image_height + 5),
              result["name"], fill="black")
    draw.text((x + 3, y + image_height + 25),
              original_text, fill="black")
    draw.text((x + 3, y + image_height + 45),
              experimental_text, fill="black")

sheet.save(OUTPUT_IMAGE, quality=95)

print(f"\nComparison image saved to:\n{OUTPUT_IMAGE}")
print("\nComparison completed. Neither model checkpoint was modified.")
