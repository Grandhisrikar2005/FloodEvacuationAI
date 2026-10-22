
import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLOOD_DIR = os.path.join(
    ROOT, "dataset", "temporal", "krishna_flood_2024",
    "flood_analysis"
)

FLOOD_PATH = os.path.join(
    FLOOD_DIR, "cleaned_flood_mask_v2_2024-09-01.tif"
)

PILOT_PATH = os.path.join(
    ROOT, "dataset", "zoom19_pilot",
    "vijayawada_zoom19_pilot.tif"
)

OUTPUT_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_v2.tif"
)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with rasterio.open(PILOT_PATH) as pilot:
        pilot_bounds = pilot.bounds
        pilot_crs = pilot.crs

    with rasterio.open(FLOOD_PATH) as src:
        # Transform pilot bounds into the flood raster's CRS.
        bounds = transform_bounds(
            pilot_crs, src.crs,
            *pilot_bounds,
            densify_pts=21
        )

        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()

        # Clamp the crop to the source raster's extent.
        col_start = max(0, int(window.col_off))
        row_start = max(0, int(window.row_off))
        col_end = min(src.width, int(window.col_off + window.width))
        row_end = min(src.height, int(window.row_off + window.height))

        if col_end <= col_start or row_end <= row_start:
            raise ValueError(
                "Pilot area does not overlap the flood raster."
            )

        from rasterio.windows import Window
        crop_window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start
        )

        data = src.read(1, window=crop_window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=src.window_transform(crop_window),
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            dst.write(data, 1)

        valid = data != 255
        candidates = valid & (data == 1)

        print("\nVIJAYAWADA FLOOD CANDIDATE CROP")
        print("Output:", OUTPUT_PATH)
        print("CRS:", src.crs)
        print("Dimensions:", data.shape[1], "x", data.shape[0])
        print("Valid pixels:", int(valid.sum()))
        print("Candidate pixels:", int(candidates.sum()))
        print("NoData pixels:", int((~valid).sum()))
        print("Candidate percentage of valid pixels:",
              round(100 * candidates.sum() / max(valid.sum(), 1), 2), "%")


if __name__ == "__main__":
    main()
