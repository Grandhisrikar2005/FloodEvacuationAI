from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FLOOD_MASK = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "best_flood_mask_2024-09-01.tif"
)

OUTPUT_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "best_flood_mask_visualization.png"
)


# Maximum display size
MAX_SIZE = 1200


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISUALIZING BEST FLOOD MASK")
    print("=" * 60)

    print(f"Input:")
    print(FLOOD_MASK)

    if not FLOOD_MASK.exists():
        print("ERROR: Flood mask not found.")
        return

    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    with rasterio.open(FLOOD_MASK) as src:

        mask = src.read(1)

        height, width = mask.shape

        print()
        print(f"Dimensions: {width} x {height}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")

    # --------------------------------------------------------
    # Downsample for visualization
    # --------------------------------------------------------

    scale = max(
        height / MAX_SIZE,
        width / MAX_SIZE
    )

    step = max(
        1,
        int(np.ceil(scale))
    )

    display_mask = mask[
        ::step,
        ::step
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    flood_pixels = int(
        np.sum(mask == 1)
    )

    total_pixels = mask.size

    percentage = (
        flood_pixels /
        total_pixels *
        100
    )

    print()
    print(f"Flood pixels: {flood_pixels:,}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Flood percentage: {percentage:.2f}%")

    print()
    print(
        f"Visualization dimensions: "
        f"{display_mask.shape[1]} x "
        f"{display_mask.shape[0]}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(12, 9))

    plt.imshow(
        display_mask,
        cmap="Blues",
        interpolation="nearest"
    )

    plt.title(
        "Filtered Flood Mask - "
        "-2 dB + 1000 Pixel Minimum Component"
    )

    plt.axis("off")

    plt.tight_layout()

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print("Visualization saved to:")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()