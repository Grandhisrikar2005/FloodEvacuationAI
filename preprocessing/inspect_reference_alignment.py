
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "dataset" / "temporal" / "krishna_flood_2024"

SAR_PATH = (
    BASE / "flood_analysis"
    / "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)
REFERENCE_PATH = (
    BASE / "reference"
    / "reference_flood_mask_v3_2024-09-01.tif"
)
OUTPUT_PATH = (
    BASE / "flood_analysis"
    / "reference_alignment_inspection.png"
)

THRESHOLD_DB = -2.0
MIN_COMPONENT_PIXELS = 1000


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(SAR_PATH) as sar, \
            rasterio.open(REFERENCE_PATH) as ref:

        if sar.crs is None or ref.crs is None:
            raise ValueError("Both rasters must have a CRS.")

        bounds = transform_bounds(
            ref.crs, sar.crs, *ref.bounds, densify_pts=21
        )

        win_raw = from_bounds(
            *bounds, transform=sar.transform
        ).round_offsets().round_lengths()

        c0 = max(0, int(win_raw.col_off))
        r0 = max(0, int(win_raw.row_off))
        c1 = min(sar.width, int(win_raw.col_off + win_raw.width))
        r1 = min(sar.height, int(win_raw.row_off + win_raw.height))

        if c1 <= c0 or r1 <= r0:
            raise ValueError("No raster overlap.")

        window = Window(c0, r0, c1 - c0, r1 - r0)
        transform = sar.window_transform(window)

        change = sar.read(1, window=window)
        sar_valid = sar.read_masks(1, window=window) > 0

        sar_valid &= np.isfinite(change)
        if sar.nodata is not None:
            sar_valid &= change != sar.nodata

        raw = sar_valid & (change <= THRESHOLD_DB)

        # Filter connected components within the reference crop.
        # This is only a visual diagnostic; crop-edge components
        # may be truncated.
        from scipy import ndimage

        labels, count = ndimage.label(
            raw, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.ravel())
        keep = sizes >= MIN_COMPONENT_PIXELS
        keep[0] = False
        filtered = keep[labels]

        reference = np.zeros(change.shape, dtype=np.uint8)
        coverage = np.zeros(change.shape, dtype=np.uint8)

        reproject(
            source=rasterio.band(ref, 1),
            destination=reference,
            src_transform=ref.transform,
            src_crs=ref.crs,
            src_nodata=ref.nodata,
            dst_transform=transform,
            dst_crs=sar.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        footprint = np.ones((ref.height, ref.width), dtype=np.uint8)

        reproject(
            source=footprint,
            destination=coverage,
            src_transform=ref.transform,
            src_crs=ref.crs,
            src_nodata=None,
            dst_transform=transform,
            dst_crs=sar.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        valid = (
            sar_valid
            & (coverage == 1)
            & np.isin(reference, [0, 1])
        )

    # Downsample for display only.
    stride = max(1, int(np.ceil(max(change.shape) / 1600)))

    ref_img = reference[::stride, ::stride]
    raw_img = raw[::stride, ::stride]
    filtered_img = filtered[::stride, ::stride]
    valid_img = valid[::stride, ::stride]

    # Agreement map:
    # Green = both positive; red = SAR-only; blue = reference-only.
    agreement = np.full(ref_img.shape, 4, dtype=np.uint8)
    agreement[valid_img & (raw_img == 0) & (ref_img == 0)] = 0
    agreement[valid_img & (raw_img == 1) & (ref_img == 1)] = 1
    agreement[valid_img & (raw_img == 1) & (ref_img == 0)] = 2
    agreement[valid_img & (raw_img == 0) & (ref_img == 1)] = 3

    from matplotlib.colors import ListedColormap

    binary_cmap = ListedColormap(["#f2f2f2", "#00cfe8", "#808080"])
    agreement_cmap = ListedColormap([
        "#f2f2f2", "#00a650", "#ff3030", "#2864dc", "#808080"
    ])

    ref_display = np.full(ref_img.shape, 2, dtype=np.uint8)
    raw_display = np.full(raw_img.shape, 2, dtype=np.uint8)
    filtered_display = np.full(filtered_img.shape, 2, dtype=np.uint8)

    ref_display[valid_img] = ref_img[valid_img]
    raw_display[valid_img] = raw_img[valid_img]
    filtered_display[valid_img] = filtered_img[valid_img]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(ref_display, cmap=binary_cmap, vmin=0, vmax=2)
    axes[0].set_title("APSAC reference V3")

    axes[1].imshow(raw_display, cmap=binary_cmap, vmin=0, vmax=2)
    axes[1].set_title("Raw SAR candidates (-2 dB)")

    axes[2].imshow(filtered_display, cmap=binary_cmap, vmin=0, vmax=2)
    axes[2].set_title("Filtered SAR candidates")

    axes[3].imshow(agreement, cmap=agreement_cmap, vmin=0, vmax=4)
    axes[3].set_title("Reference vs raw SAR\nGreen=TP, Red=FP, Blue=FN")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Reference Alignment Inspection\n"
        "Same projected grid and overlapping footprint; visual alignment "
        "still requires inspection"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("Alignment inspection image saved:")
    print(OUTPUT_PATH)
    print("\nConnected components in reference crop:", f"{count:,}")
    print("Valid overlap pixels:", f"{int(valid.sum()):,}")
    print(
        "\nThis plot helps inspect spatial agreement; it cannot alone "
        "prove that the reference labels are correct."
    )


if __name__ == "__main__":
    main()
