
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "evacuation"
OUTPUT.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = OUTPUT / "vijayawada_road_network.graphml"
MASK_PATH = (
    ROOT / "dataset" / "temporal" / "krishna_flood_2024"
    / "flood_analysis" / "best_flood_mask_v3_2024-09-01.tif"
)

MAP_PATH = OUTPUT / "route_demo_comparison.png"
REPORT_PATH = OUTPUT / "route_demo_report.txt"

FLOOD_PENALTY = 12.0
SAMPLE_SPACING_M = 10.0


def sample_line(line, spacing):
    if line is None or line.is_empty:
        return []

    distances = list(np.arange(0, line.length, spacing))
    distances.append(line.length)
    return [line.interpolate(d) for d in distances]


def get_edge_geometry(graph, u, v, data):
    geometry = data.get("geometry")
    if geometry is not None and not geometry.is_empty:
        return geometry

    return LineString([
        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
    ])


def annotate_flood_exposure(graph):
    edges = ox.graph_to_gdfs(
        graph, nodes=False, edges=True, fill_edge_geometry=True
    )

    with rasterio.open(MASK_PATH) as raster:
        projected = edges.to_crs(raster.crs)

        for (u, v, key), row in projected.iterrows():
            line = row.geometry
            data = graph[u][v][key]

            points = sample_line(line, SAMPLE_SPACING_M)
            if not points:
                data["flood_fraction"] = 0.0
                data["unknown_fraction"] = 1.0
                continue

            values = np.asarray([
                item[0]
                for item in raster.sample(
                    [(point.x, point.y) for point in points]
                )
            ], dtype=float)

            unknown = ~np.isfinite(values)
            if raster.nodata is not None:
                unknown |= values == raster.nodata
            unknown |= ~np.isin(values, [0, 1])

            valid = ~unknown
            flood_fraction = (
                float(np.mean(values[valid] == 1))
                if np.any(valid) else 0.0
            )
            unknown_fraction = float(np.mean(unknown))

            length = max(float(data.get("length", 1.0)), 1.0)
            data["flood_fraction"] = flood_fraction
            data["unknown_fraction"] = unknown_fraction

            data["flood_cost"] = length * (
                1 + FLOOD_PENALTY * flood_fraction
                + unknown_fraction
            )

    return graph


def choose_edge(options, weight):
    return min(
        options.values(),
        key=lambda data: float(data.get(weight, float("inf"))),
    )


def route_metrics(graph, route):
    distance = 0.0
    candidate_distance = 0.0
    unknown_distance = 0.0

    for u, v in zip(route[:-1], route[1:]):
        options = graph.get_edge_data(u, v)
        if not options:
            continue

        data = choose_edge(options, "length")
        length = float(data.get("length", 0.0))

        distance += length
        candidate_distance += length * float(
            data.get("flood_fraction", 0.0)
        )
        unknown_distance += length * float(
            data.get("unknown_fraction", 1.0)
        )

    return {
        "distance_km": distance / 1000,
        "candidate_exposure_equivalent_km": candidate_distance / 1000,
        "unknown_equivalent_km": unknown_distance / 1000,
    }


def find_candidate_endpoints(graph):
    # Candidate-marked roads are already identified from the raster.
    affected_nodes = set()

    for u, v, data in graph.edges(data=True):
        if float(data.get("flood_fraction", 0.0)) > 0:
            affected_nodes.update([u, v])

    if len(affected_nodes) < 2:
        raise RuntimeError(
            "Not enough candidate-exposed road nodes were found."
        )

    # Search for pairs that are connected and far enough apart
    # to make route comparisons meaningful.
    candidates = sorted(affected_nodes)
    best = None
    best_score = -1

    for origin in candidates:
        lengths = nx.single_source_dijkstra_path_length(
            graph, origin, weight="length", cutoff=1800
        )

        for destination in candidates:
            if destination == origin or destination not in lengths:
                continue

            distance = lengths[destination]
            if distance < 500:
                continue

            # Prefer pairs with at least one candidate-exposed edge
            # on the shortest route.
            route = nx.shortest_path(
                graph, origin, destination, weight="length"
            )

            exposed_length = 0.0
            for u, v in zip(route[:-1], route[1:]):
                data = choose_edge(graph[u][v], "length")
                exposed_length += (
                    float(data.get("length", 0.0))
                    * float(data.get("flood_fraction", 0.0))
                )

            score = exposed_length
            if score > best_score:
                best_score = score
                best = (origin, destination, route)

    if best is None or best_score <= 0:
        raise RuntimeError(
            "No connected pair with candidate exposure on the "
            "shortest route was found. Do not force a result; "
            "inspect the candidate-road distribution."
        )

    return best[0], best[1]


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing road graph: {GRAPH_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Missing flood mask: {MASK_PATH}")

    print("Loading cached road network...")
    graph = ox.load_graphml(GRAPH_PATH)

    print("Sampling flood exposure on roads...")
    graph = annotate_flood_exposure(graph)

    print("Searching for a demonstration route pair...")
    origin, destination = find_candidate_endpoints(graph)

    print("Selected origin node:", origin)
    print("Selected destination node:", destination)

    baseline = nx.shortest_path(
        graph, origin, destination, weight="length"
    )
    flood_aware = nx.shortest_path(
        graph, origin, destination, weight="flood_cost"
    )

    routes = {
        "Shortest-distance": baseline,
        "Flood-penalized": flood_aware,
    }

    lines = [
        "ROUTE DEMONSTRATION REPORT",
        "Candidate mask exposure is a heuristic, not confirmed flooding.",
        f"Origin node: {origin}",
        f"Destination node: {destination}",
        "",
    ]

    for name, route in routes.items():
        stats = route_metrics(graph, route)
        lines.append(name)
        for key, value in stats.items():
            lines.append(f"{key}: {value:.4f}")
        lines.append("")

        print(f"\n{name}")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#d0d0d0",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    colors = {
        "Shortest-distance": "#1565c0",
        "Flood-penalized": "#c62828",
    }

    for name, route in routes.items():
        route_gdf = ox.routing.route_to_gdf(
            graph, route, weight="length"
        )
        route_gdf.plot(
            ax=ax,
            color=colors[name],
            linewidth=3,
            label=name,
            zorder=5,
        )

    origin_data = graph.nodes[origin]
    destination_data = graph.nodes[destination]

    ax.scatter(
        origin_data["x"], origin_data["y"],
        color="green", s=90, label="Origin", zorder=10
    )
    ax.scatter(
        destination_data["x"], destination_data["y"],
        color="black", marker="*", s=120,
        label="Destination", zorder=10
    )

    ax.set_title("Shortest vs Flood-Penalized Route")
    ax.legend()
    fig.savefig(MAP_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved map:", MAP_PATH)
    print("Saved report:", REPORT_PATH)
    print(
        "\nThe selected endpoints are algorithmically chosen "
        "demonstration points, not verified safe locations."
    )


if __name__ == "__main__":
    main()
