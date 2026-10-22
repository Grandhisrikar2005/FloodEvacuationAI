
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOSAIC_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "zoom19_pilot"
    / "vijayawada_zoom19_pilot.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "zoom19_pilot"
    / "vijayawada_zoom19_preview.png"
)


def main():
    if not MOSAIC_PATH.exists():
        raise FileNotFoundError(
            f"Mosaic not found: {MOSAIC_PATH}"
        )

    with rasterio.open(MOSAIC_PATH) as src:
        image = reshape_as_image(src.read([1, 2, 3]))

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(image)
    ax.set_title(
        "Vijayawada Zoom-19 Satellite Imagery Pilot",
        fontsize=14,
    )
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"\nPreview saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
