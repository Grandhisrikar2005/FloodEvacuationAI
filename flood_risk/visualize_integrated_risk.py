
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "outputs" / "flood_risk" /
    "experimental_integrated_risk.tif"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "flood_risk" /
    "experimental_integrated_risk_preview.png"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with rasterio.open(INPUT_PATH) as src:
        risk = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = risk != nodata

    if not valid.any():
        raise ValueError("Risk raster contains no valid pixels.")

    display = np.ma.masked_where(~valid, risk)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        display,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Experimental heuristic score (0–100)",
        shrink=0.8,
    )

    ax.set_title(
        "Experimental Flood-Risk Screening Map\n"
        "SAR Change Candidates + Land-Cover Heuristic"
    )
    ax.set_xlabel("Raster column")
    ax.set_ylabel("Raster row")

    fig.text(
        0.5,
        0.015,
        f"CRS: {crs} | Bounds: {bounds}\n"
        "Not a calibrated flood probability or a validated safety map.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Risk preview saved:", OUTPUT_PATH)
    print("Valid pixels:", int(valid.sum()))
    print("Minimum score:", int(risk[valid].min()))
    print("Maximum score:", int(risk[valid].max()))


if __name__ == "__main__":
    main()
