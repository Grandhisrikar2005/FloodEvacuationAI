import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MASK_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_analysis",
    "cleaned_flood_mask_2024-09-01.tif"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_analysis",
    "flood_mask_visualization.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLOOD MASK VISUALIZATION")
    print("=" * 70)

    with rasterio.open(MASK_PATH) as src:

        print("\nCRS:", src.crs)
        print("Original size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)

        # ----------------------------------------------------
        # Statistics from full-resolution raster
        # ----------------------------------------------------

        flood_pixels = 0
        total_pixels = src.width * src.height

        # Count flood pixels block-by-block
        for _, window in src.block_windows(1):

            data = src.read(
                1,
                window=window
            )

            flood_pixels += np.count_nonzero(
                data == 1
            )

        flood_percentage = (
            flood_pixels /
            total_pixels *
            100
        )

        pixel_area_m2 = (
            abs(src.transform.a) *
            abs(src.transform.e)
        )

        flood_area_km2 = (
            flood_pixels *
            pixel_area_m2 /
            1_000_000
        )

        print("\nFlood pixels:", flood_pixels)
        print("Total pixels:", total_pixels)

        print(
            f"Flood percentage: "
            f"{flood_percentage:.2f}%"
        )

        print(
            f"Flood area: "
            f"{flood_area_km2:.2f} km²"
        )

        # ----------------------------------------------------
        # Downsample for visualization
        # ----------------------------------------------------

        MAX_SIZE = 1200

        scale = min(
            MAX_SIZE / src.width,
            MAX_SIZE / src.height
        )

        out_width = max(
            1,
            int(src.width * scale)
        )

        out_height = max(
            1,
            int(src.height * scale)
        )

        print("\nVisualization size:",
              out_width,
              "x",
              out_height)

        # Read only a reduced version
        small_mask = src.read(
            1,
            out_shape=(
                out_height,
                out_width
            ),
            resampling=rasterio.enums.Resampling.nearest
        )

    # ========================================================
    # CREATE IMAGE
    # ========================================================

    print("\nCreating visualization...")

    plt.figure(
        figsize=(12, 9)
    )

    plt.imshow(
        small_mask,
        cmap="gray",
        interpolation="nearest"
    )

    plt.title(
        "Sentinel-1 Preliminary Flood Inundation Mask\n"
        "1 September 2024"
    )

    plt.xlabel("Image Column")
    plt.ylabel("Image Row")

    plt.colorbar(
        label="0 = No Flood | 1 = Flood"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nVisualization saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()