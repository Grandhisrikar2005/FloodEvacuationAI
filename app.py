
from pathlib import Path
from typing import Any

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# FloodEvacuationAI — interactive flood-aware route planner
# ============================================================

ROOT = Path(__file__).resolve().parent

GRAPH_PATH = (
    ROOT / "outputs" / "evacuation"
    / "vijayawada_road_network.graphml"
)

MASK_PATH = (
    ROOT / "dataset" / "temporal" / "krishna_flood_2024"
    / "flood_analysis" / "best_flood_mask_v3_2024-09-01.tif"
)

DEFAULT_CENTER = (16.5062, 80.6480)
ASSUMED_SPEED_KMH = 30.0


# -------------------------- Page setup --------------------------

st.set_page_config(
    page_title="FloodEvacuationAI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: linear-gradient(120deg, #102c46, #146b82);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0 0 .4rem 0; }
    .hero p { color: #e0f2fe; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌊 FloodEvacuationAI</h1>
        <p>Satellite-informed flood-candidate analysis and
        risk-aware evacuation route comparison</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research prototype only. The historical raster contains flood "
    "candidates, not confirmed flooded roads. Routes and endpoints "
    "are not verified for real emergency use."
)


# -------------------------- Data loading --------------------------

@st.cache_resource(show_spinner="Loading road network and sampling raster...")
def load_network():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Road graph not found:\n{GRAPH_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Flood raster not found:\n{MASK_PATH}")

    graph = ox.load_graphml(GRAPH_PATH)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("The road network is empty.")

    edges = ox.graph_to_gdfs(
        graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )

    raster_summary = {}

    with rasterio.open(MASK_PATH) as raster:
        if raster.crs is None:
            raise ValueError("The flood raster has no CRS.")

        projected_edges = edges.to_crs(raster.crs)

        # Sample every road edge approximately every 10 metres.
        for (u, v, key), row in projected_edges.iterrows():
            data = graph[u][v][key]
            line = row.geometry

            if line is None or line.is_empty or line.length <= 0:
                data["flood_fraction"] = 0.0
                data["unknown_fraction"] = 1.0
                continue

            distances = np.arange(0.0, line.length, 10.0).tolist()
            distances.append(float(line.length))
            points = [line.interpolate(d) for d in distances]

            samples = raster.sample(
                [(p.x, p.y) for p in points],
                masked=True,
            )

            values = []
            for sample in samples:
                if sample.size == 0 or np.ma.is_masked(sample[0]):
                    values.append(np.nan)
                else:
                    values.append(float(sample[0]))

            values = np.asarray(values, dtype=float)

            # Encoding used by the supplied mask:
            # 0 = unmarked, 1 = candidate, 255/NoData = unknown.
            unknown = ~np.isfinite(values)

            if raster.nodata is not None:
                unknown |= values == float(raster.nodata)

            unknown |= ~np.isin(values, [0.0, 1.0])
            valid = ~unknown

            data["flood_fraction"] = (
                float(np.mean(values[valid] == 1.0))
                if np.any(valid)
                else 0.0
            )
            data["unknown_fraction"] = float(np.mean(unknown))

        raster_summary = {
            "crs": str(raster.crs),
            "width": raster.width,
            "height": raster.height,
            "nodata": raster.nodata,
            "bounds": tuple(raster.bounds),
        }

    edge_data = [
        data
        for _, _, _, data in graph.edges(keys=True, data=True)
    ]

    summary = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "raster": raster_summary,
        "candidate_edges": sum(
            d.get("flood_fraction", 0.0) > 0 for d in edge_data
        ),
        "unknown_edges": sum(
            d.get("unknown_fraction", 1.0) > 0 for d in edge_data
        ),
        "maximum_candidate_fraction": max(
            (d.get("flood_fraction", 0.0) for d in edge_data),
            default=0.0,
        ),
    }

    return graph, summary


# -------------------------- Route calculation --------------------------

def build_weighted_graph(graph, penalty):
    """Copy the graph and assign a flood-aware heuristic edge cost."""
    weighted = graph.copy()

    for _, _, _, data in weighted.edges(keys=True, data=True):
        length = max(float(data.get("length", 1.0)), 0.01)
        flood = float(data.get("flood_fraction", 0.0))
        unknown = float(data.get("unknown_fraction", 1.0))

        data["risk_cost"] = length * (
            1.0 + penalty * flood + unknown
        )

    return weighted


def get_edge_for_objective(graph, u, v, weight):
    """Choose the appropriate parallel edge for a route objective."""
    options = graph.get_edge_data(u, v)

    if not options:
        return None

    return min(
        options.values(),
        key=lambda data: float(
            data.get(weight, data.get("length", float("inf")))
        ),
    )


def get_route_metrics(graph, route, objective="length"):
    distance = 0.0
    exposure = 0.0
    unknown_distance = 0.0
    objective_cost = 0.0

    for u, v in zip(route[:-1], route[1:]):
        data = get_edge_for_objective(graph, u, v, objective)

        if data is None:
            continue

        length = max(float(data.get("length", 0.0)), 0.0)
        flood = float(data.get("flood_fraction", 0.0))
        unknown = float(data.get("unknown_fraction", 1.0))

        distance += length
        exposure += length * flood
        unknown_distance += length * unknown
        objective_cost += float(data.get(objective, length))

    return {
        "distance_m": distance,
        "exposure_m": exposure,
        "unknown_m": unknown_distance,
        "travel_minutes": distance / 1000 / ASSUMED_SPEED_KMH * 60,
        "objective_cost": objective_cost,
    }


def nearest_node(graph, point):
    lat, lon = point
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)


def is_inside_graph_area(graph, point, margin=0.01):
    lat, lon = point

    xs = [float(d["x"]) for _, d in graph.nodes(data=True)]
    ys = [float(d["y"]) for _, d in graph.nodes(data=True)]

    return (
        min(xs) - margin <= lon <= max(xs) + margin
        and min(ys) - margin <= lat <= max(ys) + margin
    )


def route_coordinates(graph, route):
    return [
        (float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"]))
        for n in route
    ]


def draw_route(map_object, graph, route, color, label):
    folium.PolyLine(
        route_coordinates(graph, route),
        color=color,
        weight=6,
        opacity=0.9,
        tooltip=label,
    ).add_to(map_object)


# -------------------------- Report --------------------------

def build_report(results):
    shortest = results["shortest_metrics"]
    risk = results["risk_metrics"]

    if shortest["exposure_m"] > 0:
        change = (
            1 - risk["exposure_m"] / shortest["exposure_m"]
        ) * 100
        change_text = f"{change:.1f}%"
    else:
        change_text = "N/A because baseline exposure is zero"

    return f"""FloodEvacuationAI — Route Comparison Report
================================================

Origin: {results['origin'][0]:.6f}, {results['origin'][1]:.6f}
Destination: {results['destination'][0]:.6f}, {results['destination'][1]:.6f}
Flood-avoidance penalty: {results['penalty']:.1f}

SHORTEST-DISTANCE ROUTE
Distance: {shortest['distance_m']:.1f} m
Estimated time: {shortest['travel_minutes']:.1f} min
Candidate-exposure equivalent: {shortest['exposure_m']:.1f} m
Unknown-mask equivalent: {shortest['unknown_m']:.1f} m

FLOOD-PENALIZED ROUTE
Distance: {risk['distance_m']:.1f} m
Estimated time: {risk['travel_minutes']:.1f} min
Candidate-exposure equivalent: {risk['exposure_m']:.1f} m
Unknown-mask equivalent: {risk['unknown_m']:.1f} m

Candidate-exposure change: {change_text}

LIMITATIONS
- Historical flood-candidate mask, not confirmed flooding.
- Exposure is a mask-based estimate, not a flood probability.
- Travel time assumes constant 30 km/h.
- Live traffic and road closures are not integrated.
- Endpoints are not verified evacuation centres.
- Not for real emergency navigation.
"""


# -------------------------- Session state --------------------------

defaults = {
    "origin": None,
    "destination": None,
    "map_center": DEFAULT_CENTER,
    "last_clicked": None,
    "route_results": None,
    "selection_map_version": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------- Sidebar --------------------------

with st.sidebar:
    st.title("🌊 FloodEvacuationAI")

    page = st.radio(
        "Dashboard",
        [
            "Evacuation Planner",
            "Project Overview",
            "Data & Methodology",
            "Diagnostics",
            "Limitations",
        ],
    )

    st.divider()
    st.subheader("Route settings")

    selection_mode = st.radio(
        "Map click sets",
        ["Origin", "Destination"],
        horizontal=True,
    )

    penalty = st.slider(
        "Flood-avoidance penalty",
        min_value=1.0,
        max_value=30.0,
        value=12.0,
        step=1.0,
        help="Higher values penalize candidate-overlapping road edges more strongly.",
    )

    st.caption("The penalty is a routing preference, not a flood probability.")

    if st.button("Reset selections and routes", use_container_width=True):
        st.session_state.origin = None
        st.session_state.destination = None
        st.session_state.last_clicked = None
        st.session_state.route_results = None
        st.session_state.selection_map_version += 1
        st.rerun()


# -------------------------- Load project data --------------------------

try:
    graph, summary = load_network()
except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.code(
        f"Expected graph:\n{GRAPH_PATH}\n\nExpected raster:\n{MASK_PATH}",
        language="text",
    )
    st.stop()


# -------------------------- Overview page --------------------------

if page == "Project Overview":
    st.title("Project Overview")
    st.write(
        "Compare the shortest road route with a route that penalizes "
        "overlap with a historical flood-candidate raster."
    )

    a, b, c = st.columns(3)
    a.metric("Road nodes", f"{summary['nodes']:,}")
    b.metric("Road edges", f"{summary['edges']:,}")
    c.metric("Edges with candidate samples", f"{summary['candidate_edges']:,}")

    st.markdown("### Workflow")
    st.code(
        "Historical candidate raster + road network\n"
        "                 ↓\n"
        "Sample candidate and unknown pixels along roads\n"
        "                 ↓\n"
        "Calculate shortest-distance route\n"
        "                 ↓\n"
        "Calculate flood-penalized route\n"
        "                 ↓\n"
        "Compare route distance and mask-based exposure",
        language="text",
    )

elif page == "Data & Methodology":
    st.title("Data & Methodology")
    st.write(f"Raster CRS: {summary['raster']['crs']}")
    st.write(
        f"Raster dimensions: {summary['raster']['width']} × "
        f"{summary['raster']['height']}"
    )
    st.write(f"Raster NoData value: {summary['raster']['nodata']}")
    st.write("Expected raster values: 0 = unmarked, 1 = candidate, other/NoData = unknown.")

    st.markdown("### Routing cost")
    st.latex(r"C_e = L_e(1 + \lambda F_e + U_e)")
    st.write(
        "L is edge length, F is the fraction of samples marked as candidates, "
        "U is the unknown-sample fraction, and λ is the selected penalty."
    )
    st.caption("This is a heuristic, not a calibrated hydrological model.")

elif page == "Diagnostics":
    st.title("Data and route diagnostics")

    st.markdown("### Road network and raster sampling")
    st.metric("Road edges", f"{summary['edges']:,}")
    st.metric(
        "Edges with candidate samples",
        f"{summary['candidate_edges']:,}",
    )
    st.metric(
        "Edges with unknown samples",
        f"{summary['unknown_edges']:,}",
    )
    st.metric(
        "Maximum edge candidate fraction",
        f"{summary['maximum_candidate_fraction']:.4f}",
    )

    if summary["candidate_edges"] == 0:
        st.error(
            "No road edge has any candidate samples. Check raster alignment, "
            "CRS, raster encoding, and the selected raster file."
        )
    else:
        st.success("At least some road edges contain candidate samples.")

    results = st.session_state.route_results

    if results:
        st.markdown("### Current route diagnostics")
        shortest_route = results["shortest_route"]
        risk_route = results["risk_route"]
        shortest_metrics = results["shortest_metrics"]
        risk_metrics = results["risk_metrics"]

        st.write("Shortest-route node count:", len(shortest_route))
        st.write("Flood-penalized route node count:", len(risk_route))
        st.write(
            "Identical node sequence:",
            shortest_route == risk_route,
        )
        st.write(
            "Shortest route candidate exposure:",
            f"{shortest_metrics['exposure_m']:.2f} m",
        )
        st.write(
            "Flood route candidate exposure:",
            f"{risk_metrics['exposure_m']:.2f} m",
        )
        st.write(
            "Shortest route unknown-mask equivalent:",
            f"{shortest_metrics['unknown_m']:.2f} m",
        )
        st.write(
            "Flood route unknown-mask equivalent:",
            f"{risk_metrics['unknown_m']:.2f} m",
        )

        if (
            shortest_route == risk_route
            and summary["candidate_edges"] > 0
        ):
            st.info(
                "Both algorithms selected the same node sequence. This can be "
                "valid if the shortest route is also cheapest under the penalty."
            )

        if (
            shortest_metrics["exposure_m"] == 0
            and risk_metrics["exposure_m"] == 0
        ):
            st.warning(
                "Both routes have zero measured candidate exposure. This may mean "
                "the sampled roads do not overlap candidate pixels; it does not "
                "prove that the roads are flood-free."
            )
    else:
        st.info("Generate a route in Evacuation Planner to inspect route diagnostics.")

elif page == "Limitations":
    st.title("Limitations and Responsible Use")
    st.markdown(
        """
        - The raster identifies historical flood candidates, not confirmed flooded roads.
        - The application does not ingest live road closures, traffic, or emergency alerts.
        - Weather forecasts and river levels are not integrated into the route cost.
        - User-selected endpoints are not verified evacuation shelters.
        - Travel time assumes a constant speed of 30 km/h.
        - Do not use the output as real emergency navigation.
        """
    )


# -------------------------- Evacuation planner --------------------------

else:
    st.title("Interactive Evacuation Planner")
    st.write(
        "Choose Origin or Destination in the sidebar, then click the map "
        "to place that point."
    )

    selection_map = folium.Map(
        location=st.session_state.map_center,
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if st.session_state.origin:
        folium.Marker(
            st.session_state.origin,
            tooltip="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(selection_map)

    if st.session_state.destination:
        folium.Marker(
            st.session_state.destination,
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(selection_map)

    map_result = st_folium(
        selection_map,
        height=480,
        use_container_width=True,
        key=f"selection_map_{st.session_state.selection_map_version}",
        returned_objects=["last_clicked"],
    )

    clicked = map_result.get("last_clicked") if map_result else None

    if clicked and "lat" in clicked and "lng" in clicked:
        point = (
            round(float(clicked["lat"]), 7),
            round(float(clicked["lng"]), 7),
        )

        if point != st.session_state.last_clicked:
            if selection_mode == "Origin":
                st.session_state.origin = point
            else:
                st.session_state.destination = point

            st.session_state.last_clicked = point
            st.session_state.map_center = point
            st.session_state.route_results = None
            st.rerun()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Origin**")
        if st.session_state.origin:
            st.success(
                f"{st.session_state.origin[0]:.6f}, "
                f"{st.session_state.origin[1]:.6f}"
            )
        else:
            st.info("Not selected.")

    with c2:
        st.markdown("**Destination**")
        if st.session_state.destination:
            st.success(
                f"{st.session_state.destination[0]:.6f}, "
                f"{st.session_state.destination[1]:.6f}"
            )
        else:
            st.info("Not selected.")

    st.divider()

    if st.button(
        "Generate Evacuation Plan",
        type="primary",
        use_container_width=True,
    ):
        origin = st.session_state.origin
        destination = st.session_state.destination

        if origin is None or destination is None:
            st.error("Select both an origin and a destination first.")

        elif not is_inside_graph_area(graph, origin):
            st.error("Origin is outside the supported road-network area.")

        elif not is_inside_graph_area(graph, destination):
            st.error("Destination is outside the supported road-network area.")

        else:
            try:
                with st.spinner("Calculating route alternatives..."):
                    origin_node = nearest_node(graph, origin)
                    destination_node = nearest_node(graph, destination)

                    if origin_node == destination_node:
                        raise ValueError(
                            "Both points snap to the same road node. "
                            "Select points farther apart."
                        )

                    shortest_route = ox.routing.shortest_path(
                        graph,
                        origin_node,
                        destination_node,
                        weight="length",
                    )

                    weighted_graph = build_weighted_graph(graph, penalty)

                    risk_route = ox.routing.shortest_path(
                        weighted_graph,
                        origin_node,
                        destination_node,
                        weight="risk_cost",
                    )

                    if shortest_route is None or risk_route is None:
                        raise ValueError("No connected route was found.")

                    shortest_metrics = get_route_metrics(
                        graph,
                        shortest_route,
                        "length",
                    )

                    risk_metrics = get_route_metrics(
                        weighted_graph,
                        risk_route,
                        "risk_cost",
                    )

                    st.session_state.route_results = {
                        "origin": origin,
                        "destination": destination,
                        "penalty": float(penalty),
                        "shortest_route": shortest_route,
                        "risk_route": risk_route,
                        "shortest_metrics": shortest_metrics,
                        "risk_metrics": risk_metrics,
                    }

            except Exception as exc:
                st.session_state.route_results = None
                st.error(f"Route calculation failed: {exc}")

    # Results render outside the button handler so they persist on reruns.
    results = st.session_state.route_results

    if results:
        st.divider()
        st.subheader("Route map")

        route_map = folium.Map(
            location=results["origin"],
            zoom_start=15,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        draw_route(
            route_map,
            graph,
            results["shortest_route"],
            "#1565c0",
            "Shortest-distance route",
        )

        draw_route(
            route_map,
            graph,
            results["risk_route"],
            "#d62728",
            "Flood-penalized route",
        )

        folium.Marker(
            results["origin"],
            tooltip="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(route_map)

        folium.Marker(
            results["destination"],
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(route_map)

        folium.LayerControl(collapsed=True).add_to(route_map)

        st_folium(
            route_map,
            height=550,
            use_container_width=True,
            key="persistent_route_map",
            returned_objects=[],
        )

        shortest = results["shortest_metrics"]
        risk = results["risk_metrics"]

        st.subheader("Route comparison")
        left, right = st.columns(2)

        with left:
            st.markdown("### 🔵 Shortest-distance route")
            st.metric("Distance", f"{shortest['distance_m'] / 1000:.2f} km")
            st.metric("Estimated time", f"{shortest['travel_minutes']:.1f} min")
            st.metric(
                "Candidate-exposure equivalent",
                f"{shortest['exposure_m']:.2f} m",
            )
            st.metric(
                "Unknown-mask equivalent",
                f"{shortest['unknown_m']:.2f} m",
            )

        with right:
            st.markdown("### 🔴 Flood-penalized route")
            st.metric("Distance", f"{risk['distance_m'] / 1000:.2f} km")
            st.metric("Estimated time", f"{risk['travel_minutes']:.1f} min")
            st.metric(
                "Candidate-exposure equivalent",
                f"{risk['exposure_m']:.2f} m",
            )
            st.metric(
                "Unknown-mask equivalent",
                f"{risk['unknown_m']:.2f} m",
            )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Extra distance",
            f"{risk['distance_m'] - shortest['distance_m']:+.0f} m",
        )

        if shortest["exposure_m"] > 0:
            change = (
                1 - risk["exposure_m"] / shortest["exposure_m"]
            ) * 100
            m2.metric("Candidate-exposure change", f"{change:+.1f}%")
        else:
            m2.metric("Candidate-exposure change", "N/A")

        m3.metric("Penalty used", f"{results['penalty']:.0f}")

        with st.expander("Route diagnostics"):
            st.write("Shortest-route node count:", len(results["shortest_route"]))
            st.write("Flood-route node count:", len(results["risk_route"]))
            st.write(
                "Identical node sequence:",
                results["shortest_route"] == results["risk_route"],
            )
            st.write(
                "Road edges with candidate samples:",
                f"{summary['candidate_edges']:,} / {summary['edges']:,}",
            )
            st.write(
                "Maximum candidate fraction on any edge:",
                f"{summary['maximum_candidate_fraction']:.4f}",
            )

        if summary["candidate_edges"] == 0:
            st.error(
                "The raster sampling found no candidate pixels on any road edge. "
                "Check raster alignment, CRS, mask encoding, and the selected raster."
            )
        elif shortest["exposure_m"] == 0 and risk["exposure_m"] == 0:
            st.info(
                "Both routes have zero measured candidate exposure. The selected "
                "roads may not overlap candidate pixels; this does not prove they "
                "are flood-free."
            )

        st.caption(
            "Travel time assumes a constant 30 km/h. Candidate-exposure equivalent "
            "is a raster-based estimate, not a flood probability."
        )

        st.warning(
            "These are experimental routes, not verified safe evacuation instructions. "
            "Confirm current flooding, road closures, and evacuation-centre details "
            "with local authorities."
        )

        st.download_button(
            "Download route comparison report",
            data=build_report(results),
            file_name="floodevacuationai_route_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.info(
            "Select both endpoints and click Generate Evacuation Plan. "
            "The calculated results will remain visible after Streamlit reruns."
        )
