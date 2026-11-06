"""Smart City Traffic Management and Route Optimization System - Streamlit app."""

import streamlit as st
import pandas
import matplotlib.pyplot as plt

from road_network import (
    RoadNetwork, load_from_csv, load_sample_city,
    generate_random_network, to_csv_text,
)
from route_planner import shortest_route, format_path, dijkstra, get_path
from traffic_analysis import bellman_ford, apply_changes, compare_routes
from connectivity_module import bfs, dfs, reachable_report
from mst_planner import prim, kruskal
from benchmark import run_benchmarks, save_results, COMPLEXITY_TABLE
from viz import compute_layout, draw_network, draw_line_chart, ROUTE_COLOR, UPDATED_COLOR, MST_COLOR


# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Smart City Traffic System", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    """Initialise session state keys on first run."""
    if "network" not in st.session_state:
        st.session_state["network"] = load_sample_city()
        st.session_state["layout"] = compute_layout(st.session_state["network"])
        st.session_state["changes"] = []
        st.session_state["benchmark_tables"] = None


_init_state()


def _reset_after_network_change(net):
    """Update session state after the network is replaced."""
    st.session_state["network"] = net
    st.session_state["layout"] = compute_layout(net)
    st.session_state["changes"] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Smart City Traffic System")
st.sidebar.caption("ENCA351 - Design and Analysis of Algorithms Lab")

PAGE_LABELS = [
    "Road network",
    "Route planning",
    "Traffic analysis",
    "Network connectivity",
    "Infrastructure planning",
    "Comparative analysis",
]

page = st.sidebar.radio("Navigation", PAGE_LABELS)


# ---------------------------------------------------------------------------
# Helper: guard for pages that need at least two intersections
# ---------------------------------------------------------------------------

def _require_two_nodes():
    """Show info and stop if the network has fewer than two intersections."""
    if len(st.session_state["network"].nodes()) < 2:
        st.info("Add at least two intersections to use this page.")
        st.stop()


# ---------------------------------------------------------------------------
# Page: Road network
# ---------------------------------------------------------------------------

def page_network():
    """Render the Road network page."""
    st.header("Road network")
    st.caption("Module: road_network.py. Models the city as a weighted undirected graph.")

    net = st.session_state["network"]
    pos = st.session_state["layout"]

    # -- Network loader --
    load_choice = st.selectbox(
        "Load network",
        ["Sample city (10 intersections)", "Random network", "Upload CSV"],
        key="load_choice",
    )

    if load_choice == "Random network":
        col_a, col_b = st.columns(2)
        rand_n = col_a.number_input("Number of intersections", min_value=5, max_value=100,
                                    value=25, step=1, key="rand_n")
        rand_seed = col_b.number_input("Seed", value=42, step=1, key="rand_seed")

    if load_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload a road network CSV", type=["csv"], key="upload_csv")

    if st.button("Load", key="btn_load"):
        try:
            if load_choice == "Sample city (10 intersections)":
                _reset_after_network_change(load_sample_city())
                st.success("Sample city loaded.")
            elif load_choice == "Random network":
                _reset_after_network_change(generate_random_network(int(rand_n), int(rand_seed)))
                st.success(f"Random network with {int(rand_n)} intersections loaded.")
            elif load_choice == "Upload CSV":
                if uploaded is None:
                    st.error("Please select a CSV file to upload.")
                else:
                    _reset_after_network_change(load_from_csv(uploaded))
                    st.success("Network loaded from CSV.")
            net = st.session_state["network"]
            pos = st.session_state["layout"]
        except ValueError as err:
            st.error(str(err))

    # -- Metrics --
    c1, c2, c3 = st.columns(3)
    c1.metric("Intersections", len(net.nodes()))
    c2.metric("Roads", len(net.edges()))
    c3.metric("Connected", "Yes" if net.is_connected() else "No")

    # -- Editing area --
    edit_left, edit_right = st.columns(2)

    with edit_left:
        st.subheader("Add intersection")
        new_name = st.text_input("Intersection name", key="new_intersection_name")
        if st.button("Add intersection", key="btn_add_intersection"):
            try:
                net.add_intersection(new_name)
                st.session_state["layout"] = compute_layout(net)
                st.success(f"Intersection '{new_name}' added.")
            except ValueError as err:
                st.error(str(err))

    with edit_right:
        st.subheader("Add or update road")
        nodes = net.nodes()
        if len(nodes) >= 2:
            from_node = st.selectbox("From", nodes, key="road_from")
            to_node = st.selectbox("To", nodes, index=min(1, len(nodes) - 1), key="road_to")
            travel_time = st.number_input("Travel time (min)", min_value=0.1, value=5.0,
                                          step=0.5, key="road_weight")
            if st.button("Add road", key="btn_add_road"):
                try:
                    existing = net.weight(from_node, to_node)
                    net.add_road(from_node, to_node, travel_time)
                    st.session_state["layout"] = compute_layout(net)
                    st.session_state["changes"] = []
                    if existing is not None:
                        st.info("Road updated.")
                    else:
                        st.success(f"Road {from_node} - {to_node} ({travel_time:g} min) added.")
                except ValueError as err:
                    st.error(str(err))

    # -- Network drawing --
    if net.nodes():
        fig = draw_network(net, pos, title="Current road network")
        st.pyplot(fig)
        plt.close(fig)

    # -- Tabs --
    tab_list, tab_matrix, tab_roads = st.tabs(["Adjacency list", "Adjacency matrix", "Road list"])
    with tab_list:
        st.dataframe(net.adjacency_list(), width='stretch')
    with tab_matrix:
        st.caption("0 means no road.")
        st.dataframe(net.adjacency_matrix(), width='stretch')
    with tab_roads:
        road_data = [{"From": u, "To": v, "Travel time (min)": w} for u, v, w in net.edges()]
        st.dataframe(pandas.DataFrame(road_data), width='stretch')

    # -- Download --
    st.download_button(
        "Download network as CSV",
        data=to_csv_text(net),
        file_name="road_network.csv",
        mime="text/csv",
        key="btn_download",
    )


# ---------------------------------------------------------------------------
# Page: Route planning
# ---------------------------------------------------------------------------

def page_route():
    """Render the Route planning page."""
    st.header("Route planning")
    st.caption("Module: route_planner.py. Finds shortest routes using Dijkstra's algorithm.")
    _require_two_nodes()

    net = st.session_state["network"]
    pos = st.session_state["layout"]
    nodes = net.nodes()

    col1, col2 = st.columns(2)
    source = col1.selectbox("Source", nodes, key="route_source")
    dest = col2.selectbox("Destination", nodes, index=len(nodes) - 1, key="route_dest")

    result = shortest_route(net, source, dest)

    if not result["path"]:
        st.warning("No route exists between these intersections.")
        fig = draw_network(net, pos, title="Road network")
        st.pyplot(fig)
        plt.close(fig)
        return

    path = result["path"]
    cost = result["cost"]
    hops = len(path) - 1

    st.write(f"**Route:** {format_path(path)}")
    m1, m2 = st.columns(2)
    m1.metric("Total travel cost", f"{cost:g} min")
    m2.metric("Roads on route", hops)

    fig = draw_network(net, pos, highlight_path=path, highlight_color=ROUTE_COLOR,
                       title="Shortest route")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Distance updates (Dijkstra)"):
        steps = result["steps"]
        if steps:
            df_steps = pandas.DataFrame(steps)
            df_steps.columns = ["Step", "Settled", "Neighbor", "Old distance",
                                 "New distance", "Updated"]
            df_steps["Old distance"] = df_steps["Old distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            df_steps["New distance"] = df_steps["New distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            df_steps["Updated"] = df_steps["Updated"].map({True: "Yes", False: "No"})
            st.dataframe(df_steps, width='stretch')

    with st.expander("Final distances from source"):
        dist = result["dist"]
        prev = result["prev"]
        df_dist = pandas.DataFrame([
            {
                "Intersection": n,
                "Distance": "inf" if dist[n] == float("inf") else f"{dist[n]:g}",
                "Previous": prev[n] if prev[n] is not None else "-",
            }
            for n in nodes
        ])
        st.dataframe(df_dist, width='stretch')


# ---------------------------------------------------------------------------
# Page: Traffic analysis
# ---------------------------------------------------------------------------

def page_traffic():
    """Render the Traffic analysis page."""
    st.header("Traffic analysis")
    st.caption("Module: traffic_analysis.py. Models congestion using Bellman-Ford algorithm.")
    _require_two_nodes()

    net = st.session_state["network"]
    pos = st.session_state["layout"]
    nodes = net.nodes()
    edges = net.edges()

    col1, col2 = st.columns(2)
    source = col1.selectbox("Source", nodes, key="traffic_source")
    dest = col2.selectbox("Destination", nodes, index=len(nodes) - 1, key="traffic_dest")

    # -- Congestion editor --
    st.subheader("Congestion editor")
    road_labels = [f"{u} - {v} (current: {w:g} min)" for u, v, w in edges]
    road_tuples = [(u, v) for u, v, w in edges]

    cong_left, cong_right = st.columns(2)
    with cong_left:
        road_idx = st.selectbox("Road", range(len(road_labels)),
                                format_func=lambda i: road_labels[i],
                                key="cong_road_idx")
        multiplier = st.slider("Congestion multiplier", 1.0, 10.0, 2.0, 0.5, key="cong_mult")
        closed = st.checkbox("Close this road", key="cong_closed")

    with cong_right:
        if st.button("Add change", key="btn_add_change"):
            sel_u, sel_v = road_tuples[road_idx]
            st.session_state["changes"].append({
                "u": sel_u, "v": sel_v,
                "multiplier": multiplier, "closed": closed,
            })
        if st.button("Clear changes", key="btn_clear_changes"):
            st.session_state["changes"] = []

    changes = st.session_state["changes"]
    if changes:
        df_changes = pandas.DataFrame([
            {"Road": f"{c['u']} - {c['v']}", "Multiplier": c["multiplier"],
             "Closed": "Yes" if c["closed"] else "No"}
            for c in changes
        ])
        st.dataframe(df_changes, width='stretch')

    if not changes:
        st.info("Add at least one congestion change to compare routes.")
        orig_bf = bellman_ford(net, source, log=False)
        orig_path = get_path(orig_bf["prev"], source, dest)
        fig = draw_network(net, pos, highlight_path=orig_path, highlight_color=ROUTE_COLOR,
                           title="Original route")
        st.pyplot(fig)
        plt.close(fig)
        return

    updated = apply_changes(net, changes)
    comparison = compare_routes(net, updated, source, dest)

    orig_path = comparison["original_path"]
    upd_path = comparison["updated_path"]
    orig_cost = comparison["original_cost"]
    upd_cost = comparison["updated_cost"]

    # Identify congested edges (changed roads that still exist in updated network).
    congested_pairs = []
    closed_roads = []
    for c in changes:
        u, v = c["u"], c["v"]
        if c.get("closed"):
            closed_roads.append(f"{u} - {v}")
        else:
            congested_pairs.append((u, v))

    draw_col1, draw_col2 = st.columns(2)
    with draw_col1:
        st.subheader("Original route")
        if orig_path:
            st.write(format_path(orig_path))
            cost_str = f"{orig_cost:g} min" if orig_cost is not None else "Unreachable"
            st.write(f"Cost: {cost_str}")
        fig = draw_network(net, pos, highlight_path=orig_path, highlight_color=ROUTE_COLOR,
                           title="Original route")
        st.pyplot(fig)
        plt.close(fig)

    with draw_col2:
        st.subheader("Updated route")
        if upd_path:
            st.write(format_path(upd_path))
            cost_str = f"{upd_cost:g} min" if upd_cost is not None else "Unreachable"
            st.write(f"Cost: {cost_str}")
        if closed_roads:
            st.caption(f"Closed roads: {', '.join(closed_roads)}")
        fig = draw_network(updated, pos, highlight_path=upd_path, highlight_color=UPDATED_COLOR,
                           congested_edges=congested_pairs, title="Updated route")
        st.pyplot(fig)
        plt.close(fig)

    if upd_cost is None:
        st.error("Destination unreachable after these changes.")

    # -- Metrics --
    delta = comparison["delta"]
    percent = comparison["percent"]
    m1, m2, m3 = st.columns(3)
    delta_val = f"{delta:+g} min" if delta is not None else "N/A"
    m1.metric("Cost change", delta_val,
              delta=delta if delta is not None else None,
              delta_color="inverse")
    pct_str = f"{percent:.1f}%" if percent is not None else "N/A"
    m2.metric("Percent change", pct_str)
    m3.metric("Route changed", "Yes" if comparison["route_changed"] else "No")

    # -- Impact table --
    st.subheader("Impact on every intersection")
    if not comparison["impact"].empty:
        st.dataframe(comparison["impact"], width='stretch')
    else:
        st.write("No intersection costs changed.")

    # -- Relaxation expanders --
    with st.expander("Bellman-Ford relaxations (original network)"):
        st.caption(f"Passes used: {comparison['original_passes']}")
        if comparison["original_steps"]:
            df_orig = pandas.DataFrame(comparison["original_steps"])
            df_orig.columns = ["Pass", "Road", "Old distance", "New distance"]
            df_orig["Old distance"] = df_orig["Old distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            df_orig["New distance"] = df_orig["New distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            st.dataframe(df_orig, width='stretch')

    with st.expander("Bellman-Ford relaxations (updated network)"):
        st.caption(f"Passes used: {comparison['updated_passes']}")
        if comparison["updated_steps"]:
            df_upd = pandas.DataFrame(comparison["updated_steps"])
            df_upd.columns = ["Pass", "Road", "Old distance", "New distance"]
            df_upd["Old distance"] = df_upd["Old distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            df_upd["New distance"] = df_upd["New distance"].apply(
                lambda v: "inf" if v == float("inf") else f"{v:g}")
            st.dataframe(df_upd, width='stretch')


# ---------------------------------------------------------------------------
# Page: Network connectivity
# ---------------------------------------------------------------------------

def page_connectivity():
    """Render the Network connectivity page."""
    st.header("Network connectivity")
    st.caption("Module: connectivity_module.py. Explores network using BFS and DFS.")
    _require_two_nodes()

    net = st.session_state["network"]
    pos = st.session_state["layout"]
    nodes = net.nodes()

    start = st.selectbox("Starting intersection", nodes, key="conn_start")

    bfs_order = bfs(net, start)
    dfs_order = dfs(net, start)
    bfs_report = reachable_report(net, start, bfs_order)
    dfs_report = reachable_report(net, start, dfs_order)

    col_bfs, col_dfs = st.columns(2)

    with col_bfs:
        st.subheader("Breadth-first search (BFS)")
        st.write("Traversal order: " + " > ".join(bfs_order))
        st.write(f"Reachable intersections: {bfs_report['count']} of {bfs_report['total']}")
        if bfs_report["unreachable"]:
            st.write("Unreachable: " + ", ".join(bfs_report["unreachable"]))
        fig = draw_network(net, pos, highlight_path=None, visit_order=bfs_order,
                           title="BFS traversal")
        st.pyplot(fig)
        plt.close(fig)

    with col_dfs:
        st.subheader("Depth-first search (DFS)")
        st.write("Traversal order: " + " > ".join(dfs_order))
        st.write(f"Reachable intersections: {dfs_report['count']} of {dfs_report['total']}")
        if dfs_report["unreachable"]:
            st.write("Unreachable: " + ", ".join(dfs_report["unreachable"]))
        fig = draw_network(net, pos, highlight_path=None, visit_order=dfs_order,
                           highlight_color=UPDATED_COLOR, title="DFS traversal")
        st.pyplot(fig)
        plt.close(fig)

    same_set = set(bfs_order) == set(dfs_order)
    caption_text = "BFS and DFS reached the same set of intersections." if same_set else \
        "BFS and DFS reached different sets of intersections."
    st.caption(caption_text)


# ---------------------------------------------------------------------------
# Page: Infrastructure planning
# ---------------------------------------------------------------------------

def page_infrastructure():
    """Render the Infrastructure planning page."""
    st.header("Infrastructure planning")
    st.caption("Module: mst_planner.py. Finds minimum-cost spanning tree using Prim and Kruskal.")
    _require_two_nodes()

    net = st.session_state["network"]
    pos = st.session_state["layout"]
    nodes = net.nodes()

    prim_start = st.selectbox("Prim starting intersection", nodes, key="prim_start")

    prim_result = prim(net, start=prim_start)
    kruskal_result = kruskal(net)

    disconnected = prim_result["components"] > 1
    if disconnected:
        st.warning(
            f"The network is disconnected. Showing a spanning forest "
            f"with {prim_result['components']} components."
        )

    col_p, col_k = st.columns(2)

    with col_p:
        st.subheader("Prim's algorithm")
        prim_df = pandas.DataFrame([
            {"Order": i + 1, "From": u, "To": v, "Cost (min)": w}
            for i, (u, v, w) in enumerate(prim_result["edges"])
        ])
        st.dataframe(prim_df, width='stretch')
        st.metric("Total infrastructure cost", f"{prim_result['total']:g} min")
        prim_edges = [(u, v) for u, v, w in prim_result["edges"]]
        fig = draw_network(net, pos, highlight_edges=prim_edges, highlight_color=MST_COLOR,
                           title="Prim MST")
        st.pyplot(fig)
        plt.close(fig)

    with col_k:
        st.subheader("Kruskal's algorithm")
        kruskal_df = pandas.DataFrame([
            {"Order": i + 1, "From": u, "To": v, "Cost (min)": w}
            for i, (u, v, w) in enumerate(kruskal_result["edges"])
        ])
        st.dataframe(kruskal_df, width='stretch')
        st.metric("Total infrastructure cost", f"{kruskal_result['total']:g} min")
        kruskal_edges = [(u, v) for u, v, w in kruskal_result["edges"]]
        fig = draw_network(net, pos, highlight_edges=kruskal_edges, highlight_color=MST_COLOR,
                           title="Kruskal MST")
        st.pyplot(fig)
        plt.close(fig)

    if abs(prim_result["total"] - kruskal_result["total"]) < 1e-9:
        st.success("Both algorithms give the same total cost.")
    else:
        st.warning("Prim and Kruskal returned different totals. Check the network for issues.")


# ---------------------------------------------------------------------------
# Page: Comparative analysis
# ---------------------------------------------------------------------------

def page_comparison():
    """Render the Comparative analysis page."""
    st.header("Comparative analysis")
    st.caption("Module: benchmark.py. Compares algorithm performance across network sizes.")

    all_sizes = [10, 25, 50, 100]
    sizes = st.multiselect("Network sizes", all_sizes, default=all_sizes, key="bench_sizes")
    col1, col2 = st.columns(2)
    repeats = col1.number_input("Repeats", min_value=1, max_value=100, value=20, key="bench_reps")
    bench_seed = col2.number_input("Seed", value=42, key="bench_seed")

    if st.button("Run experiments", key="btn_run_bench"):
        with st.spinner("Running benchmarks..."):
            tables = run_benchmarks(tuple(sorted(sizes)), int(repeats), int(bench_seed))
            save_results(tables)
            st.session_state["benchmark_tables"] = tables
        st.success("Benchmarks complete. Results saved to results/benchmark_results.csv.")

    tables = st.session_state.get("benchmark_tables")
    if tables is None:
        st.info("Click 'Run experiments' to generate benchmark results.")
        st.subheader("Algorithm complexity reference")
        st.dataframe(COMPLEXITY_TABLE, width='stretch')
        return

    # BFS vs DFS
    st.subheader("BFS vs DFS")
    st.dataframe(tables["traversal"], width='stretch')
    fig = draw_line_chart(tables["traversal"], "Intersections",
                          ["BFS time (ms)", "DFS time (ms)"],
                          "BFS vs DFS time", "Time (ms)")
    st.pyplot(fig)
    plt.close(fig)

    # Dijkstra vs Bellman-Ford
    st.subheader("Dijkstra vs Bellman-Ford")
    st.dataframe(tables["shortest_path"], width='stretch')
    fig = draw_line_chart(tables["shortest_path"], "Intersections",
                          ["Dijkstra time (ms)", "Bellman-Ford time (ms)"],
                          "Dijkstra vs Bellman-Ford time", "Time (ms)")
    st.pyplot(fig)
    plt.close(fig)

    # Prim vs Kruskal
    st.subheader("Prim vs Kruskal")
    st.dataframe(tables["mst"], width='stretch')
    fig = draw_line_chart(tables["mst"], "Intersections",
                          ["Prim time (ms)", "Kruskal time (ms)"],
                          "Prim vs Kruskal time", "Time (ms)")
    st.pyplot(fig)
    plt.close(fig)

    # Complexity table
    st.subheader("Algorithm complexity reference")
    st.dataframe(COMPLEXITY_TABLE, width='stretch')

    # Suitability section
    st.subheader("Suitability for traffic management")
    st.write(
        "BFS finds routes with the fewest roads but ignores travel time. "
        "DFS suits reachability checks. "
        "Dijkstra is the right default for navigation because road travel times are never negative. "
        "Bellman-Ford costs more but supports negative adjustments and detects negative cycles. "
        "Prim and Kruskal both give the cheapest way to connect all intersections, "
        "and Kruskal handles disconnected networks naturally as a forest."
    )

    # Download combined results
    try:
        with open("results/benchmark_results.csv", "r", encoding="utf-8") as fh:
            csv_data = fh.read()
        st.download_button(
            "Download results CSV",
            data=csv_data,
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="btn_download_bench",
        )
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Page dispatcher
# ---------------------------------------------------------------------------

if page == "Road network":
    page_network()
elif page == "Route planning":
    page_route()
elif page == "Traffic analysis":
    page_traffic()
elif page == "Network connectivity":
    page_connectivity()
elif page == "Infrastructure planning":
    page_infrastructure()
elif page == "Comparative analysis":
    page_comparison()
