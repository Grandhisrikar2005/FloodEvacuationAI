
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGE = ROOT / "dataset" / "georeferenced_landcover_utm44.tif"
MODEL_PATH = ROOT / "models" / "saved_models" / "expanded_best_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "landcover"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "vijayawada_landcover_expanded.tif"

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

IMAGE_SIZE = 224
TILE_SIZE = 224
BATCH_SIZE = 16
NODATA = 255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(INPUT_IMAGE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_PATH}"
        )

    model = load_model()
    print("Loaded model:", MODEL_PATH)
    print("Device:", device)

    with rasterio.open(INPUT_IMAGE) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=NODATA,
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )

        tile_jobs = []

        for row in range(0, src.height, TILE_SIZE):
            for col in range(0, src.width, TILE_SIZE):
                width = min(TILE_SIZE, src.width - col)
                height = min(TILE_SIZE, src.height - row)

                # Avoid partial edge tiles that would distort scale.
                if width != TILE_SIZE or height != TILE_SIZE:
                    continue

                tile_jobs.append((row, col))

        total_tiles = len(tile_jobs)
        print("Image dimensions:", src.width, "x", src.height)
        print("Tiles to classify:", total_tiles)

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            for start in range(0, total_tiles, BATCH_SIZE):
                batch_jobs = tile_jobs[start:start + BATCH_SIZE]
                tensors = []
                windows = []

                for row, col in batch_jobs:
                    window = Window(col, row, TILE_SIZE, TILE_SIZE)
                    rgb = src.read([1, 2, 3], window=window)
                    rgb = np.moveaxis(rgb, 0, -1)

                    image = Image.fromarray(rgb.astype(np.uint8), "RGB")
                    tensors.append(transform(image))
                    windows.append(window)

                inputs = torch.stack(tensors).to(device)

                with torch.inference_mode():
                    logits = model(inputs)
                    predictions = logits.argmax(dim=1).cpu().numpy()

                for window, class_id in zip(windows, predictions):
                    dst.write(
                        np.full(
                            (TILE_SIZE, TILE_SIZE),
                            int(class_id),
                            dtype=np.uint8,
                        ),
                        1,
                        window=window,
                    )

                print(
                    f"Processed {min(start + len(batch_jobs), total_tiles)}"
                    f"/{total_tiles} tiles"
                )

    print("\nLand-cover prediction completed.")
    print("Output:", OUTPUT_PATH)
    print("Class encoding:")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"  {class_id} = {name}")
    print(f"  {NODATA} = NoData")


if __name__ == "__main__":
    main()
