
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = os.path.join(
    ROOT, "dataset", "zoom19_pilot",
    "vijayawada_zoom19_pilot.tif"
)

MASK_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_aligned.tif"
)

OUTPUT_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_overlay.png"
)


def main():
    with rasterio.open(IMAGE_PATH) as src:
        image = src.read()
        if image.shape[0] >= 3:
            image = np.moveaxis(image[:3], 0, -1)
        else:
            image = image[0]

    with rasterio.open(MASK_PATH) as src:
        mask = src.read(1)
        nodata = src.nodata

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Grid mismatch: image {image.shape[:2]}, mask {mask.shape}"
        )

    candidates = mask == 1
    valid = np.ones(mask.shape, dtype=bool)
    if nodata is not None:
        valid = mask != nodata

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[candidates] = [1.0, 0.0, 0.0, 0.65]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.imshow(overlay)
    ax.set_title("Vijayawada SAR Flood-Change Candidates")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Overlay saved:", OUTPUT_PATH)
    print("Image dimensions:", image.shape[:2])
    print("Candidate pixels:", int(candidates.sum()))
    print("NoData pixels:", int((~valid).sum()))
    print("Note: red pixels are SAR-change candidates, not confirmed floods.")


if __name__ == "__main__":
    main()
