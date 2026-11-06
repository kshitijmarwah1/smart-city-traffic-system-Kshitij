"""Benchmark: timing and memory comparisons for all graph algorithms."""

import time
import tracemalloc
from pathlib import Path

import pandas

from road_network import generate_random_network
from route_planner import dijkstra
from traffic_analysis import bellman_ford
from connectivity_module import bfs, dfs
from mst_planner import prim, kruskal


# Complexity reference table shown on the comparison page.
COMPLEXITY_TABLE = pandas.DataFrame([
    {
        "Algorithm": "BFS",
        "Time complexity": "O(V + E)",
        "Best used for": "Fewest roads and level-by-level exploration.",
    },
    {
        "Algorithm": "DFS",
        "Time complexity": "O(V + E)",
        "Best used for": "Deep exploration and reachability checks.",
    },
    {
        "Algorithm": "Dijkstra (binary heap)",
        "Time complexity": "O((V + E) log V)",
        "Best used for": "Fast routing with non-negative travel costs.",
    },
    {
        "Algorithm": "Bellman-Ford",
        "Time complexity": "O(V * E)",
        "Best used for": "Networks that may contain negative adjustments, and detecting negative cycles.",
    },
    {
        "Algorithm": "Prim (binary heap)",
        "Time complexity": "O(E log V)",
        "Best used for": "Connected networks, grows one tree outward.",
    },
    {
        "Algorithm": "Kruskal",
        "Time complexity": "O(E log E)",
        "Best used for": "Sparse networks, works edge by edge and handles forests.",
    },
])


def time_ms(function, repeats):
    """Run function repeats times and return average elapsed time in milliseconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def peak_memory_kb(function):
    """Run function once under tracemalloc and return peak memory in KB."""
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def run_benchmarks(sizes=(10, 25, 50, 100), repeats=20, seed=42):
    """Run timing and memory benchmarks for all algorithms across network sizes."""
    traversal_rows = []
    sp_rows = []
    mst_rows = []

    for size in sizes:
        net = generate_random_network(size, seed)
        source = "I1"
        target = f"I{size}"
        road_count = len(net.edges())

        # Traversal benchmarks.
        bfs_time = round(time_ms(lambda: bfs(net, source), repeats), 4)
        dfs_time = round(time_ms(lambda: dfs(net, source), repeats), 4)
        bfs_visited = bfs(net, source)
        dfs_visited = dfs(net, source)
        bfs_mem = round(peak_memory_kb(lambda: bfs(net, source)), 1)
        dfs_mem = round(peak_memory_kb(lambda: dfs(net, source)), 1)
        traversal_rows.append({
            "Intersections": size,
            "Roads": road_count,
            "BFS time (ms)": bfs_time,
            "DFS time (ms)": dfs_time,
            "BFS visited": len(bfs_visited),
            "DFS visited": len(dfs_visited),
            "BFS memory (KB)": bfs_mem,
            "DFS memory (KB)": dfs_mem,
        })

        # Shortest path benchmarks.
        dijk_time = round(time_ms(lambda: dijkstra(net, source, log=False), repeats), 4)
        bf_time = round(time_ms(lambda: bellman_ford(net, source, log=False), repeats), 4)
        dijk_result = dijkstra(net, source, log=False)
        bf_result = bellman_ford(net, source, log=False)
        dijk_cost = dijk_result["dist"][target]
        bf_cost = bf_result["dist"][target]

        from route_planner import get_path
        dijk_path = get_path(dijk_result["prev"], source, target)
        route_hops = max(0, len(dijk_path) - 1)

        dijk_mem = round(peak_memory_kb(lambda: dijkstra(net, source, log=False)), 1)
        bf_mem = round(peak_memory_kb(lambda: bellman_ford(net, source, log=False)), 1)

        sp_rows.append({
            "Intersections": size,
            "Roads": road_count,
            "Dijkstra time (ms)": dijk_time,
            "Bellman-Ford time (ms)": bf_time,
            "Dijkstra cost": dijk_cost if dijk_cost != float("inf") else None,
            "Bellman-Ford cost": bf_cost if bf_cost != float("inf") else None,
            "Route hops": route_hops,
            "Bellman-Ford passes": bf_result["passes"],
            "Dijkstra memory (KB)": dijk_mem,
            "Bellman-Ford memory (KB)": bf_mem,
        })

        # MST benchmarks.
        prim_time = round(time_ms(lambda: prim(net), repeats), 4)
        kruskal_time = round(time_ms(lambda: kruskal(net), repeats), 4)
        prim_result = prim(net)
        kruskal_result = kruskal(net)
        prim_mem = round(peak_memory_kb(lambda: prim(net)), 1)
        kruskal_mem = round(peak_memory_kb(lambda: kruskal(net)), 1)

        mst_rows.append({
            "Intersections": size,
            "Roads": road_count,
            "Prim time (ms)": prim_time,
            "Kruskal time (ms)": kruskal_time,
            "Prim cost": prim_result["total"],
            "Kruskal cost": kruskal_result["total"],
            "MST roads": len(prim_result["edges"]),
            "Prim memory (KB)": prim_mem,
            "Kruskal memory (KB)": kruskal_mem,
        })

    return {
        "traversal": pandas.DataFrame(traversal_rows),
        "shortest_path": pandas.DataFrame(sp_rows),
        "mst": pandas.DataFrame(mst_rows),
    }


def save_results(tables, path="results/benchmark_results.csv"):
    """Save all benchmark tables to a single combined CSV file."""
    frames = []
    for name, df in tables.items():
        copy = df.copy()
        copy.insert(0, "Experiment", name)
        frames.append(copy)
    combined = pandas.concat(frames, ignore_index=True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
