
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_TIF = (
    ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "reference"
    / "NRSC_APSAC_Vijayawada_reference_georef.tif"
)

OUTPUT_TIF = (
    ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "reference"
    / "reference_flood_mask_v3_2024-09-01.tif"
)

OUTPUT_PREVIEW = (
    ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "reference"
    / "reference_flood_mask_v3_preview.png"
)


# ============================================================
# SETTINGS
# ============================================================

# Keep the existing cyan color rule from V2.
MIN_COMPONENT_PIXELS = 3

# 8-connected neighbourhood.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("CREATING REFERENCE FLOOD MASK V3")
    print("=" * 65)

    if not INPUT_TIF.exists():
        raise FileNotFoundError(f"Reference image not found: {INPUT_TIF}")

    with rasterio.open(INPUT_TIF) as src:
        if src.count < 3:
            raise ValueError("Expected an RGB image with at least 3 bands.")

        rgb = src.read([1, 2, 3])
        profile = src.profile.copy()

        print("Input:", INPUT_TIF)
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Input NoData:", src.nodata)

    red = rgb[0].astype(np.int16)
    green = rgb[1].astype(np.int16)
    blue = rgb[2].astype(np.int16)

    # Official map legend: cyan = Flood Inundation.
    cyan = (
        (green > 180)
        & (blue > 180)
        & (red < 100)
        & (green > red * 1.5)
        & (blue > red * 1.5)
    )

    initial_count = int(cyan.sum())

    # Remove only extremely tiny connected components.
    # This may remove genuine small patches too, so compare V2 and V3.
    labels, count = ndimage.label(cyan, structure=STRUCTURE)
    sizes = np.bincount(labels.ravel())

    keep = sizes >= MIN_COMPONENT_PIXELS
    keep[0] = False
    cleaned = keep[labels]

    removed = int(initial_count - cleaned.sum())

    # 0 = not classified as cyan flood inundation
    # 1 = cyan flood-inundation candidate
    # NoData remains unset because the RGB image has no defined
    # NoData footprint.
    mask = cleaned.astype(np.uint8)

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=None,
        compress="deflate"
    )

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(mask, 1)

    # Save a simple visual preview for inspection.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(np.moveaxis(rgb, 0, -1))
    axes[0].set_title("Original reference image")

    axes[1].imshow(cyan, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("V2 cyan extraction")

    axes[2].imshow(cleaned, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("V3 after tiny-component filtering")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PREVIEW, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nRESULTS")
    print("-" * 45)
    print("Initial cyan pixels:", f"{initial_count:,}")
    print("Connected components:", f"{count:,}")
    print("Pixels removed:", f"{removed:,}")
    print("Final candidate pixels:", f"{int(cleaned.sum()):,}")
    print("Output mask:", OUTPUT_TIF)
    print("Preview:", OUTPUT_PREVIEW)
    print("\nEncoding: 0 = not cyan, 1 = cyan candidate, NoData = unset")
    print("This is map-color extraction, not independent ground truth.")


if __name__ == "__main__":
    main()
