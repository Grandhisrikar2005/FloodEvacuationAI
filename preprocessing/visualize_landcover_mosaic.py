from pathlib import Path

import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOSAIC = PROJECT_ROOT / "dataset" / "georeferenced_landcover_base.tif"


def main():
    print("=" * 60)
    print("VISUALIZING GEOREFERENCED LAND-COVER MOSAIC")
    print("=" * 60)

    with rasterio.open(MOSAIC) as src:
        image = src.read()

        transformer = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True
        )

        left_lon, bottom_lat = transformer.transform(
            src.bounds.left,
            src.bounds.bottom
        )

        right_lon, top_lat = transformer.transform(
            src.bounds.right,
            src.bounds.top
        )

        print(f"Longitude: {left_lon:.6f} -> {right_lon:.6f}")
        print(f"Latitude:  {bottom_lat:.6f} -> {top_lat:.6f}")

    # Convert from bands,rows,cols to rows,cols,bands
    rgb = image.transpose(1, 2, 0)

    plt.figure(figsize=(10, 10))

    plt.imshow(
        rgb,
        extent=[
            left_lon,
            right_lon,
            bottom_lat,
            top_lat
        ]
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Georeferenced Land-Cover Mosaic")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = PROJECT_ROOT / "dataset" / "georeferenced_landcover_preview.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Preview saved to:")
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()