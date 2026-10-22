
import os
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "Before-event SAR": os.path.join(
        ROOT, "dataset", "temporal", "krishna_flood_2024",
        "before_20240820", "sentinel1_2024-08-20.tif"
    ),
    "After-event SAR": os.path.join(
        ROOT, "dataset", "temporal", "krishna_flood_2024",
        "flood_20240901", "sentinel1_2024-09-01.tif"
    ),
    "APSAC reference mask": os.path.join(
        ROOT, "dataset", "temporal", "krishna_flood_2024",
        "reference", "reference_flood_mask_2024-09-01.tif"
    ),
    "Georeferenced reference": os.path.join(
        ROOT, "dataset", "temporal", "krishna_flood_2024",
        "reference", "NRSC_APSAC_Vijayawada_reference_georef.tif"
    ),
    "Vijayawada pilot": os.path.join(
        ROOT, "dataset", "zoom19_pilot",
        "vijayawada_zoom19_pilot.tif"
    ),
    "Aligned candidate mask": os.path.join(
        ROOT, "outputs", "landcover",
        "vijayawada_flood_candidates_aligned.tif"
    ),
}


def main():
    for name, path in FILES.items():
        print(f"\n{'=' * 60}\n{name}")

        if not os.path.exists(path):
            print("NOT FOUND:", path)
            continue

        with rasterio.open(path) as src:
            print("File:", os.path.basename(path))
            print("Dimensions:", src.width, "x", src.height)
            print("Bands:", src.count)
            print("CRS:", src.crs)
            print("Resolution:", src.res)
            print("Bounds:", src.bounds)
            print("NoData:", src.nodata)
            print("Data type:", src.dtypes[0])

            if src.count == 1:
                # Sample pixels from a small section, not the whole raster.
                sample = src.read(
                    1,
                    window=rasterio.windows.Window(
                        0, 0, min(512, src.width), min(512, src.height)
                    )
                )
                values, counts = np.unique(sample, return_counts=True)
                print("Sample values/counts:")
                for value, count in zip(values[:20], counts[:20]):
                    print(f"  {value}: {count}")

                if len(values) > 20:
                    print("More values exist; only first 20 shown.")


if __name__ == "__main__":
    main()
