
from pathlib import Path

import osmnx as ox
import numpy as np
import rasterio
import geopandas as gpd

from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = ROOT / "outputs" / "evacuation" / "vijayawada_road_network.graphml"

MASK_PATH = (
    ROOT / "dataset" / "temporal" / "krishna_flood_2024"
    / "flood_analysis" / "best_flood_mask_v3_2024-09-01.tif"
)

OUTPUT_PATH = ROOT / "outputs" / "evacuation" / "flood_candidate_roads.geojson"


def main():
    graph = ox.load_graphml(GRAPH_PATH)

    edges = ox.graph_to_gdfs(
        graph, nodes=False, edges=True, fill_edge_geometry=True
    )

    with rasterio.open(MASK_PATH) as raster:
        edges = edges.to_crs(raster.crs)
        fractions = []
        counts = []

        for _, row in edges.iterrows():
            line = row.geometry

            if line is None or line.is_empty:
                fractions.append(0.0)
                counts.append(0)
                continue

            # Sample every 10 m, including the endpoint.
            distances = list(np.arange(0, line.length, 10.0))
            distances.append(line.length)

            points = [line.interpolate(d) for d in distances]
            coords = [(p.x, p.y) for p in points]

            values = np.array(
                [s[0] for s in raster.sample(coords)],
                dtype=np.float64,
            )

            valid = np.isfinite(values) & np.isin(values, [0, 1])
            candidate_count = int(np.sum(valid & (values == 1)))

            fraction = (
                candidate_count / int(np.sum(valid))
                if np.any(valid) else 0.0
            )

            fractions.append(fraction)
            counts.append(candidate_count)

    edges["candidate_sample_count"] = counts
    edges["candidate_fraction"] = fractions

    affected = edges[edges["candidate_sample_count"] > 0].copy()

    # Export in WGS84 for easy viewing in GIS applications.
    affected = affected.to_crs("EPSG:4326")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    affected.to_file(OUTPUT_PATH, driver="GeoJSON")

    print("Total road edges:", len(edges))
    print("Road edges with candidate samples:", len(affected))
    print("Output:", OUTPUT_PATH)

    if not affected.empty:
        print("\nTop 10 affected road edges:")
        print(
            affected[
                ["candidate_sample_count", "candidate_fraction", "geometry"]
            ]
            .sort_values("candidate_fraction", ascending=False)
            .head(10)
            .drop(columns="geometry")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
