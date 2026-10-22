
"""
Flood-Aware Evacuation Route Planner
Prototype using a road graph and Dijkstra's algorithm.

Road nodes and edges in this demo are illustrative, not actual roads.
"""

import heapq
import math
import os

import matplotlib.pyplot as plt
import networkx as nx


def build_demo_road_network():
    """Create a small illustrative road network."""

    graph = nx.Graph()

    # Illustrative locations, not real geographic coordinates.
    nodes = {
        "Start": (0, 2),
        "A": (1, 3),
        "B": (1, 1),
        "C": (2, 2),
        "D": (3, 3),
        "E": (3, 1),
        "F": (4, 2),
        "Shelter": (5, 2),
    }

    for name, position in nodes.items():
        graph.add_node(name, pos=position)

    # (node 1, node 2, distance in metres, flood exposure 0-1)
    roads = [
        ("Start", "A", 140, 0.05),
        ("Start", "B", 130, 0.10),
        ("A", "C", 120, 0.10),
        ("B", "C", 110, 0.15),
        ("C", "D", 130, 0.10),
        ("C", "E", 120, 0.20),
        ("D", "F", 120, 0.05),
        ("E", "F", 130, 0.10),
        ("F", "Shelter", 150, 0.05),

        # A shorter road with high flood exposure
        ("A", "D", 100, 0.95),
    ]

    for u, v, distance, flood_exposure in roads:
        graph.add_edge(
            u,
            v,
            distance_m=distance,
            flood_exposure=flood_exposure,
        )

    return graph


def find_safest_route(graph, start, destination, flood_penalty=2000):
    """
    Find a minimum-cost route.

    Cost = road distance + flood_penalty * flood exposure.

    flood_penalty controls how strongly the route avoids flood exposure.
    """

    if start not in graph:
        raise ValueError(f"Unknown start node: {start}")

    if destination not in graph:
        raise ValueError(f"Unknown destination: {destination}")

    if flood_penalty < 0 or not math.isfinite(flood_penalty):
        raise ValueError("Flood penalty must be finite and non-negative.")

    for _, _, data in graph.edges(data=True):
        exposure = data["flood_exposure"]
        distance = data["distance_m"]

        if distance <= 0 or not math.isfinite(distance):
            raise ValueError("Road distances must be positive and finite.")

        if not 0 <= exposure <= 1 or not math.isfinite(exposure):
            raise ValueError("Flood exposure must be between 0 and 1.")

    def edge_cost(u, v, data):
        return (
            data["distance_m"]
            + flood_penalty * data["flood_exposure"]
        )

    try:
        route = nx.dijkstra_path(
            graph,
            start,
            destination,
            weight=edge_cost,
        )
    except nx.NetworkXNoPath:
        return None, None

    distance = sum(
        graph[u][v]["distance_m"]
        for u, v in zip(route[:-1], route[1:])
    )

    exposure_cost = sum(
        graph[u][v]["flood_exposure"]
        for u, v in zip(route[:-1], route[1:])
    )

    return route, {
        "distance_m": distance,
        "exposure_sum": exposure_cost,
        "weighted_cost": sum(
            edge_cost(u, v, graph[u][v])
            for u, v in zip(route[:-1], route[1:])
        ),
    }


def save_route_map(graph, route, output_path):
    """Save a visualization of the illustrative road graph."""

    positions = nx.get_node_attributes(graph, "pos")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="gray",
        width=2,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="lightblue",
        node_size=800,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
    )

    if route:
        route_edges = list(zip(route[:-1], route[1:]))

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=route_edges,
            edge_color="green",
            width=4,
        )

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=[route[0]],
            node_color="blue",
            node_size=900,
        )

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=[route[-1]],
            node_color="green",
            node_size=900,
        )

    plt.title("Flood-Aware Evacuation Route — Illustrative Demo")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    road_graph = build_demo_road_network()

    start = "Start"
    destination = "Shelter"

    route, metrics = find_safest_route(
        road_graph,
        start,
        destination,
    )

    print("\n=== EVACUATION ROUTE PLANNER ===")

    if route is None:
        print("No connected route is available.")
    else:
        print("Selected route:", " -> ".join(route))
        print(f"Distance: {metrics['distance_m']:.0f} metres")
        print(f"Exposure score sum: {metrics['exposure_sum']:.2f}")
        print(f"Weighted route cost: {metrics['weighted_cost']:.2f}")

        output = "outputs/evacuation/demo_route.png"
        save_route_map(road_graph, route, output)
        print("Route visualization saved to:", output)

    print(
        "\nDEMO ONLY: Nodes, road distances, shelter, and flood exposure "
        "are illustrative. This is not a real evacuation recommendation."
    )
