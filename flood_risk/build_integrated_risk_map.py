
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]

LANDCOVER_PATH = (
    ROOT / "outputs" / "landcover" /
    "vijayawada_landcover_expanded.tif"
)

FLOOD_PATH = (
    ROOT / "dataset" / "temporal" / "krishna_flood_2024" /
    "flood_analysis" / "best_flood_mask_v3_2024-09-01.tif"
)

OUTPUT_DIR = ROOT / "outputs" / "flood_risk"
OUTPUT_PATH = OUTPUT_DIR / "experimental_integrated_risk.tif"

# Existing class encoding:
# 0 Agriculture, 1 Barren, 2 Roads, 3 Urban,
# 4 Vegetation, 5 Water, 255 NoData.
#
# This is a provisional heuristic, NOT a trained flood-risk model.
LANDCOVER_WEIGHTS = {
    0: 0.45,  # Agriculture
    1: 0.55,  # Barren
    2: 0.65,  # Roads
    3: 0.70,  # Urban
    4: 0.35,  # Vegetation
    5: 0.90,  # Water
}

FLOOD_CANDIDATE_SCORE = 0.90
NODATA = 255
WINDOW_SIZE = 1024


def main():
    for path in (LANDCOVER_PATH, FLOOD_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Rename or remove it before rerunning."
        )

    with rasterio.open(LANDCOVER_PATH) as lc, \
            rasterio.open(FLOOD_PATH) as flood:

        # Restrict processing to the land-cover extent transformed
        # into the flood raster's CRS.
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        bounds = transform_bounds(
            lc.crs, flood.crs, *lc.bounds, densify_pts=21
        )

        left = max(bounds[0], flood.bounds.left)
        bottom = max(bounds[1], flood.bounds.bottom)
        right = min(bounds[2], flood.bounds.right)
        top = min(bounds[3], flood.bounds.top)

        if left >= right or bottom >= top:
            raise ValueError("No overlap between land cover and flood mask.")

        window = from_bounds(
            left, bottom, right, top,
            transform=flood.transform
        ).round_offsets().round_lengths()

        col = max(0, int(window.col_off))
        row = max(0, int(window.row_off))
        end_col = min(flood.width, int(window.col_off + window.width))
        end_row = min(flood.height, int(window.row_off + window.height))

        window = Window(col, row, end_col - col, end_row - row)
        transform = flood.window_transform(window)

        height = int(window.height)
        width = int(window.width)

        print("Study-area dimensions:", width, "x", height)
        print("Output CRS:", flood.crs)

        profile = flood.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform,
            dtype="uint8",
            count=1,
            nodata=NODATA,
            compress="lzw",
        )

        # Reproject categorical land-cover classes onto the SAR grid.
        landcover = np.full((height, width), 255, dtype=np.uint8)

        reproject(
            source=rasterio.band(lc, 1),
            destination=landcover,
            src_transform=lc.transform,
            src_crs=lc.crs,
            src_nodata=lc.nodata,
            dst_transform=transform,
            dst_crs=flood.crs,
            dst_nodata=255,
            resampling=Resampling.nearest,
        )

        # Output encoding:
        # 0-100 = provisional heuristic risk score
        # 255 = NoData
        #
        # Candidate flood pixels receive a high score.
        # Other pixels receive a class-dependent baseline score.
        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            for r in range(0, height, WINDOW_SIZE):
                for c in range(0, width, WINDOW_SIZE):
                    h = min(WINDOW_SIZE, height - r)
                    w = min(WINDOW_SIZE, width - c)

                    local_window = Window(c, r, w, h)

                    flood_values = flood.read(
                        1,
                        window=Window(
                            col + c, row + r, w, h
                        ),
                    )
                    local_lc = landcover[r:r + h, c:c + w]

                    valid = (
                        (flood_values != flood.nodata)
                        & np.isin(local_lc, list(LANDCOVER_WEIGHTS))
                    )

                    scores = np.full((h, w), NODATA, dtype=np.uint8)

                    for class_id, weight in LANDCOVER_WEIGHTS.items():
                        scores[valid & (local_lc == class_id)] = round(
                            weight * 100
                        )

                    candidate = valid & (flood_values == 1)
                    scores[candidate] = round(
                        FLOOD_CANDIDATE_SCORE * 100
                    )

                    dst.write(scores, 1, window=local_window)

                print(f"Processed risk rows {min(r + WINDOW_SIZE, height)}/{height}")

    print("\nExperimental risk raster created:")
    print(OUTPUT_PATH)
    print("Encoding: 0-100 = heuristic score; 255 = NoData")
    print("This is a prototype score, not a calibrated flood probability.")


if __name__ == "__main__":
    main()
