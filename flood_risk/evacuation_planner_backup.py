
from pathlib import Path

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "evacuation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = OUTPUT_DIR / "vijayawada_road_network.graphml"
ROUTES_PATH = OUTPUT_DIR / "candidate_evacuation_routes.geojson"
MAP_PATH = OUTPUT_DIR / "candidate_evacuation_routes.png"

# Pilot study area around Vijayawada.
CENTER = (16.5062, 80.6480)
NETWORK_RADIUS_METERS = 2500

# Example coordinates (latitude, longitude).
# Replace with actual study locations when available.
ORIGIN = (16.5062, 80.6480)
DESTINATION = (16.5150, 80.6350)


# ============================================================
# ROAD NETWORK
# ============================================================

def get_road_network():
    if GRAPH_PATH.exists():
        print(f"Loading cached road network: {GRAPH_PATH}")
        graph = ox.load_graphml(GRAPH_PATH)
    else:
        print("Downloading road network from OpenStreetMap...")

        graph = ox.graph_from_point(
            CENTER,
            dist=NETWORK_RADIUS_METERS,
            network_type="drive",
            simplify=True,
        )

        # OSMnx 2.x calculates edge lengths during graph creation.
        ox.save_graphml(graph, GRAPH_PATH)
        print(f"Saved road network: {GRAPH_PATH}")

    if len(graph.nodes) == 0 or len(graph.edges) == 0:
        raise ValueError("The road network is empty.")

    print(f"Road nodes: {len(graph.nodes):,}")
    print(f"Road edges: {len(graph.edges):,}")

    return graph


# ============================================================
# ROUTE CALCULATION
# ============================================================

def calculate_routes(graph):
    origin_node = ox.distance.nearest_nodes(
        graph,
        X=ORIGIN[1],
        Y=ORIGIN[0],
    )

    destination_node = ox.distance.nearest_nodes(
        graph,
        X=DESTINATION[1],
        Y=DESTINATION[0],
    )

    print(f"Origin node: {origin_node}")
    print(f"Destination node: {destination_node}")

    if not nx.has_path(graph, origin_node, destination_node):
        raise ValueError(
            "No drivable route connects the selected points. "
            "Choose different coordinates inside the study area."
        )

    # Route 1: shortest distance.
    shortest_route = nx.shortest_path(
        graph,
        source=origin_node,
        target=destination_node,
        weight="length",
    )

    # Route 2: fastest route if OSM speed data is available.
    # Missing speeds receive a provisional default of 30 km/h.
    graph_with_time = graph.copy()

    for u, v, key, data in graph_with_time.edges(keys=True, data=True):
        length_m = float(data.get("length", 1.0))
        speed_kph = data.get("speed_kph", 30)

        try:
            if isinstance(speed_kph, list):
                speed_kph = float(speed_kph[0])
            else:
                speed_kph = float(speed_kph)
        except (ValueError, TypeError, IndexError):
            speed_kph = 30.0

        speed_kph = max(speed_kph, 5.0)
        data["travel_time"] = length_m / (speed_kph * 1000 / 3600)

    fastest_route = nx.shortest_path(
        graph_with_time,
        source=origin_node,
        target=destination_node,
        weight="travel_time",
    )

    return {
        "Shortest-distance": shortest_route,
        "Fastest-estimated": fastest_route,
    }


# ============================================================
# ROUTE DETAILS AND GEOJSON EXPORT
# ============================================================

def calculate_route_length(graph, nodes):
    total = 0.0

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge_options = graph.get_edge_data(u, v)

        if not edge_options:
            continue

        shortest_edge = min(
            edge_options.values(),
            key=lambda data: float(data.get("length", float("inf"))),
        )
        total += float(shortest_edge.get("length", 0))

    return total


def export_routes(graph, routes):
    route_frames = []

    for route_name, nodes in routes.items():
        route_edges = ox.routing.route_to_gdf(
            graph,
            nodes,
            weight="length",
        ).copy()

        route_edges["route_type"] = route_name
        route_edges["route_order"] = range(len(route_edges))

        route_frames.append(route_edges)

        distance_m = calculate_route_length(graph, nodes)

        print(f"\n{route_name}")
        print(f"  Distance: {distance_m:.0f} m")
        print(f"  Road segments: {len(nodes) - 1}")

    combined = gpd.GeoDataFrame(
        gpd.pd.concat(route_frames, ignore_index=True),
        geometry="geometry",
        crs=route_frames[0].crs,
    )

    combined.to_file(ROUTES_PATH, driver="GeoJSON")
    print(f"\nGeoJSON saved: {ROUTES_PATH}")


# ============================================================
# MAP VISUALIZATION
# ============================================================

def visualize_routes(graph, routes):
    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#c7c7c7",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    colors = {
        "Shortest-distance": "#1565c0",
        "Fastest-estimated": "#e65100",
    }

    for route_name, nodes in routes.items():
        route_edges = ox.routing.route_to_gdf(
            graph,
            nodes,
            weight="length",
        )

        route_edges.plot(
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
        s=80,
        marker="o",
        label="Origin",
        zorder=10,
    )

    ax.scatter(
        DESTINATION[1],
        DESTINATION[0],
        color="red",
        s=100,
        marker="*",
        label="Destination",
        zorder=10,
    )

    ax.legend(loc="best")
    ax.set_title(
        "Vijayawada Candidate Routes\n"
        "Distance vs. Estimated Travel Time"
    )

    fig.savefig(MAP_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Route map saved: {MAP_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    ox.settings.log_console = True
    ox.settings.use_cache = True

    graph = get_road_network()
    routes = calculate_routes(graph)

    export_routes(graph, routes)
    visualize_routes(graph, routes)

    print("\n" + "=" * 50)
    print("ROUTE PLANNING COMPLETED")
    print("=" * 50)
    print(f"Road network: {GRAPH_PATH}")
    print(f"Route GeoJSON: {ROUTES_PATH}")
    print(f"Route map: {MAP_PATH}")
    print(
        "\nImportant: These are baseline routes, not flood-avoiding "
        "or verified-safe evacuation routes. The origin and destination "
        "are demonstration coordinates, and travel speeds are estimates."
    )


if __name__ == "__main__":
    main()
