
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = ROOT / "outputs" / "evacuation" / "vijayawada_road_network.graphml"
FLOOD_ROADS_PATH = ROOT / "outputs" / "evacuation" / "flood_candidate_roads.geojson"
OUTPUT_PATH = ROOT / "outputs" / "evacuation" / "flood_candidate_roads_map.png"


def main():
    print("Loading road network...")
    graph = ox.load_graphml(GRAPH_PATH)

    print("Loading candidate-affected roads...")
    affected = gpd.read_file(FLOOD_ROADS_PATH)

    # Plot in a projected CRS so the map uses metre-based coordinates.
    roads = ox.graph_to_gdfs(
        graph, nodes=False, edges=True, fill_edge_geometry=True
    ).to_crs("EPSG:32644")

    affected = affected.to_crs("EPSG:32644")

    fig, ax = plt.subplots(figsize=(12, 10))

    roads.plot(
        ax=ax,
        color="lightgray",
        linewidth=0.6,
        label="Road network",
    )

    affected.plot(
        ax=ax,
        color="red",
        linewidth=2.5,
        label="Roads with flood-candidate samples",
    )

    ax.set_title(
        "Vijayawada Road Network — Flood Candidate Segments",
        fontsize=14,
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    ax.set_aspect("equal")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMap saved to:\n{OUTPUT_PATH}")
    print(f"Candidate-affected road edges: {len(affected)}")


if __name__ == "__main__":
    main()
