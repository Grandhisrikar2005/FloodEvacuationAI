
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import LineString


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "evacuation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = OUTPUT_DIR / "vijayawada_road_network.graphml"

FLOOD_MASK_PATH = (
    ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "best_flood_mask_v3_2024-09-01.tif"
)

ROUTES_PATH = OUTPUT_DIR / "flood_aware_routes.geojson"
MAP_PATH = OUTPUT_DIR / "flood_aware_routes.png"
REPORT_PATH = OUTPUT_DIR / "flood_aware_route_report.txt"

# Demonstration coordinates: latitude, longitude.
# Replace with verified locations before real-world use.
ORIGIN = (16.5062, 80.6480)
DESTINATION = (16.5150, 80.6350)

# Research heuristics, not calibrated flood probabilities.
FLOOD_PENALTY = 8.0
UNKNOWN_PENALTY = 1.0

FLOOD_VALUE = 1
DRY_VALUE = 0
NODATA_VALUE = 255

SAMPLE_SPACING_METERS = 10.0
DEFAULT_SPEED_KPH = 30.0


# ============================================================
# LOAD ROAD NETWORK
# ============================================================

def load_road_network():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Road graph not found:\n{GRAPH_PATH}\n"
            "Run the road-network downloader first."
        )

    print("Loading cached road network...")
    graph = ox.load_graphml(GRAPH_PATH)

    if len(graph.nodes) == 0 or len(graph.edges) == 0:
        raise ValueError("The cached road network is empty.")

    print(f"Road nodes: {len(graph.nodes):,}")
    print(f"Road edges: {len(graph.edges):,}")

    return graph


# ============================================================
# EDGE GEOMETRY
# ============================================================

def get_edge_geometry(graph, u, v, data):
    geometry = data.get("geometry")

    if geometry is not None and not geometry.is_empty:
        return geometry

    return LineString([
        (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
        (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
    ])


def sample_line(line, spacing):
    """Generate points along a road geometry."""

    if line is None or line.is_empty:
        return []

    if line.length <= 0:
        return [line.interpolate(0)]

    distances = list(np.arange(0, line.length, spacing))
    distances.append(line.length)

    return [line.interpolate(d) for d in distances]


# ============================================================
# FLOOD MASK SAMPLING
# ============================================================

def add_flood_exposure(graph):
    if not FLOOD_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Flood mask not found:\n{FLOOD_MASK_PATH}"
        )

    print("\nSampling flood mask along road segments...")
    print(f"Raster: {FLOOD_MASK_PATH}")

    # Build a stable list of edge references.
    edge_refs = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = get_edge_geometry(graph, u, v, data)
        edge_refs.append((u, v, key, geometry))

    with rasterio.open(FLOOD_MASK_PATH) as raster:
        print(f"Raster CRS: {raster.crs}")
        print(f"Raster bounds: {raster.bounds}")
        print(f"Raster NoData: {raster.nodata}")

        # Transform road geometry into the raster CRS.
        edge_gdf = ox.graph_to_gdfs(
            graph,
            nodes=False,
            edges=True,
            fill_edge_geometry=True,
        )

        edge_gdf = edge_gdf.to_crs(raster.crs)
        projected_geometries = {
            index: row.geometry
            for index, row in edge_gdf.iterrows()
        }

        # Reprojected geometries use (u, v, key) indexes.
        total = len(edge_refs)
        candidate_edges = 0
        unknown_edges = 0
        no_candidate_edges = 0

        for index, (u, v, key, _) in enumerate(edge_refs, start=1):
            edge_key = (u, v, key)
            line = projected_geometries.get(edge_key)

            data = graph[u][v][key]

            if line is None or line.is_empty:
                data["flood_fraction"] = 0.0
                data["unknown_fraction"] = 1.0
                data["flood_aware_cost"] = (
                    float(data.get("length", 1.0))
                    * (1.0 + UNKNOWN_PENALTY)
                )
                unknown_edges += 1
                continue

            # Sample every 10 metres to reduce the chance of missing
            # narrow candidate areas along road segments.
            points = sample_line(line, SAMPLE_SPACING_METERS)
            coordinates = [(point.x, point.y) for point in points]

            values = np.asarray(
                [item[0] for item in raster.sample(coordinates)],
                dtype=np.float64,
            )

            unknown = ~np.isfinite(values)
            unknown |= values == NODATA_VALUE

            if raster.nodata is not None:
                unknown |= values == raster.nodata

            # Values outside the known classes are also unknown.
            unknown |= ~np.isin(values, [DRY_VALUE, FLOOD_VALUE])

            valid = ~unknown

            if np.any(valid):
                flood_fraction = float(
                    np.mean(values[valid] == FLOOD_VALUE)
                )
            else:
                flood_fraction = 0.0

            unknown_fraction = float(np.mean(unknown))

            length = max(float(data.get("length", 1.0)), 1.0)

            data["flood_fraction"] = flood_fraction
            data["unknown_fraction"] = unknown_fraction

            # Penalize candidate flood exposure and uncertain coverage.
            data["flood_aware_cost"] = length * (
                1.0
                + FLOOD_PENALTY * flood_fraction
                + UNKNOWN_PENALTY * unknown_fraction
            )

            speed = data.get("speed_kph", DEFAULT_SPEED_KPH)

            try:
                if isinstance(speed, list):
                    speed = float(speed[0])
                else:
                    speed = float(speed)
            except (ValueError, TypeError, IndexError):
                speed = DEFAULT_SPEED_KPH

            speed = max(speed, 5.0)

            data["estimated_time_seconds"] = (
                length / (speed * 1000.0 / 3600.0)
            )

            if flood_fraction > 0:
                candidate_edges += 1
            else:
                no_candidate_edges += 1

            if unknown_fraction > 0:
                unknown_edges += 1

            if index % 2000 == 0:
                print(f"Processed {index:,}/{total:,} edges")

    print(f"Processed edges: {total:,}")
    print(f"Edges with candidate samples: {candidate_edges:,}")
    print(f"Edges with no candidate samples: {no_candidate_edges:,}")
    print(f"Edges with some unknown coverage: {unknown_edges:,}")

    return graph


# ============================================================
# ROUTE CALCULATION
# ============================================================

def nearest_node(graph, location):
    latitude, longitude = location

    return ox.distance.nearest_nodes(
        graph,
        X=longitude,
        Y=latitude,
    )


def calculate_routes(graph):
    origin_node = nearest_node(graph, ORIGIN)
    destination_node = nearest_node(graph, DESTINATION)

    print(f"\nOrigin node: {origin_node}")
    print(f"Destination node: {destination_node}")

    if not nx.has_path(graph, origin_node, destination_node):
        raise ValueError("No route connects the selected locations.")

    routes = {
        "Shortest-distance": nx.shortest_path(
            graph,
            origin_node,
            destination_node,
            weight="length",
        ),
        "Fastest-estimated": nx.shortest_path(
            graph,
            origin_node,
            destination_node,
            weight="estimated_time_seconds",
        ),
        "Flood-aware": nx.shortest_path(
            graph,
            origin_node,
            destination_node,
            weight="flood_aware_cost",
        ),
    }

    return routes


# ============================================================
# SELECT PARALLEL EDGE CONSISTENTLY
# ============================================================

def get_route_edge(graph, u, v, route_type):
    options = graph.get_edge_data(u, v)

    if not options:
        return None

    if route_type == "Flood-aware":
        weight = "flood_aware_cost"
    elif route_type == "Fastest-estimated":
        weight = "estimated_time_seconds"
    else:
        weight = "length"

    return min(
        options.values(),
        key=lambda item: float(item.get(weight, float("inf"))),
    )


# ============================================================
# ROUTE STATISTICS
# ============================================================

def calculate_route_statistics(graph, route, route_type):
    total_distance = 0.0
    estimated_time = 0.0
    candidate_exposure = 0.0
    unknown_distance = 0.0
    total_penalized_cost = 0.0

    for u, v in zip(route[:-1], route[1:]):
        data = get_route_edge(graph, u, v, route_type)

        if data is None:
            continue

        length = float(data.get("length", 0.0))
        flood_fraction = float(data.get("flood_fraction", 0.0))
        unknown_fraction = float(data.get("unknown_fraction", 1.0))

        total_distance += length
        candidate_exposure += length * flood_fraction
        unknown_distance += length * unknown_fraction

        estimated_time += float(
            data.get("estimated_time_seconds", 0.0)
        )

        total_penalized_cost += float(
            data.get("flood_aware_cost", length)
        )

    return {
        "distance_km": total_distance / 1000.0,
        "estimated_minutes": estimated_time / 60.0,
        "candidate_exposure_distance_km": candidate_exposure / 1000.0,
        "unknown_distance_km": unknown_distance / 1000.0,
        "candidate_exposure_percent": (
            100.0 * candidate_exposure / total_distance
            if total_distance > 0 else 0.0
        ),
        "flood_aware_cost_km_equivalent": total_penalized_cost / 1000.0,
    }


# ============================================================
# EXPORT GEOJSON AND REPORT
# ============================================================

def export_routes(graph, routes):
    route_frames = []
    report = [
        "FLOOD-AWARE EVACUATION ROUTE REPORT",
        "=" * 48,
        "Flood exposure is sampled from a candidate mask.",
        "It is not a confirmed measurement of road inundation.",
        "Unknown raster coverage is not considered safe.",
        "Origin and destination are demonstration coordinates.",
        "",
    ]

    for route_name, nodes in routes.items():
        route_gdf = ox.routing.route_to_gdf(
            graph,
            nodes,
            weight="length",
        ).copy()

        route_gdf["route_type"] = route_name
        route_gdf["route_order"] = np.arange(len(route_gdf))

        fractions = []
        unknowns = []

        for u, v in zip(nodes[:-1], nodes[1:]):
            data = get_route_edge(graph, u, v, route_name)

            if data is None:
                fractions.append(0.0)
                unknowns.append(1.0)
            else:
                fractions.append(float(data.get("flood_fraction", 0.0)))
                unknowns.append(float(data.get("unknown_fraction", 1.0)))

        if len(fractions) == len(route_gdf):
            route_gdf["flood_fraction"] = fractions
            route_gdf["unknown_fraction"] = unknowns

        route_frames.append(route_gdf)

        stats = calculate_route_statistics(graph, nodes, route_name)

        print(f"\n{route_name}")

        for name, value in stats.items():
            print(f"  {name}: {value:.3f}")

        report.append(route_name)
        report.append("-" * len(route_name))

        for name, value in stats.items():
            report.append(f"{name}: {value:.4f}")

        report.append("")

    combined = gpd.GeoDataFrame(
        pd.concat(route_frames, ignore_index=True),
        geometry="geometry",
        crs=route_frames[0].crs,
    )

    combined.to_file(ROUTES_PATH, driver="GeoJSON")
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"\nGeoJSON saved: {ROUTES_PATH}")
    print(f"Report saved: {REPORT_PATH}")


# ============================================================
# MAP VISUALIZATION
# ============================================================

def visualize_routes(graph, routes):
    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#cccccc",
        edge_linewidth=0.45,
        bgcolor="white",
    )

    colors = {
        "Shortest-distance": "#1565c0",
        "Fastest-estimated": "#ef6c00",
        "Flood-aware": "#8e24aa",
    }

    for route_name, nodes in routes.items():
        route_gdf = ox.routing.route_to_gdf(
            graph,
            nodes,
            weight="length",
        )

        route_gdf.plot(
            ax=ax,
            color=colors[route_name],
            linewidth=3,
            label=route_name,
            zorder=5,
        )

    ax.scatter(
        ORIGIN[1],
        ORIGIN[0],
        color="green",
        s=85,
        marker="o",
        label="Origin",
        zorder=10,
    )

    ax.scatter(
        DESTINATION[1],
        DESTINATION[0],
        color="red",
        s=110,
        marker="*",
        label="Destination",
        zorder=10,
    )

    ax.set_title(
        "Vijayawada Candidate Evacuation Routes\n"
        "Distance, Estimated Time and Flood Exposure"
    )

    ax.legend(loc="best")

    fig.savefig(MAP_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Route comparison map saved: {MAP_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    ox.settings.log_console = False
    ox.settings.use_cache = True

    graph = load_road_network()
    graph = add_flood_exposure(graph)

    routes = calculate_routes(graph)

    export_routes(graph, routes)
    visualize_routes(graph, routes)

    print("\n" + "=" * 52)
    print("FLOOD-AWARE ROUTING COMPLETED")
    print("=" * 52)
    print(f"Map: {MAP_PATH}")
    print(f"GeoJSON: {ROUTES_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(
        "\nReminder: These are research prototype routes, "
        "not verified safe evacuation routes."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nFlood-aware routing failed: {error}")
        raise
