"""Traffic analysis using Bellman-Ford algorithm and congestion modelling."""

import pandas

from road_network import RoadNetwork
from route_planner import get_path


def bellman_ford(network, source, log=True):
    """Run Bellman-Ford from source; return dist, prev, passes, and step log."""
    nodes = network.nodes()
    n = len(nodes)

    dist = {node: float("inf") for node in nodes}
    prev = {node: None for node in nodes}
    dist[source] = 0.0

    # Precompute directed edge list: each undirected road becomes two directed edges.
    directed_edges = []
    for u, v, w in network.edges():
        directed_edges.append((u, v, w))
        directed_edges.append((v, u, w))

    steps = []
    passes_done = 0

    for pass_num in range(1, n):
        changed = False
        for u, v, w in directed_edges:
            if dist[u] == float("inf"):
                continue
            candidate = dist[u] + w
            if candidate < dist[v]:
                old = dist[v]
                dist[v] = candidate
                prev[v] = u
                changed = True
                if log:
                    steps.append({
                        "pass": pass_num,
                        "road": f"{u} -> {v}",
                        "old": old,
                        "new": candidate,
                    })
        passes_done = pass_num
        if not changed:
            break

    # One extra pass to detect negative-weight cycles.
    negative_cycle = False
    for u, v, w in directed_edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            negative_cycle = True
            break

    return {
        "dist": dist,
        "prev": prev,
        "passes": passes_done,
        "negative_cycle": negative_cycle,
        "steps": steps,
    }


def apply_changes(network, changes):
    """Return a new network with congestion changes applied; never mutates original."""
    # Build a dict keyed by sorted road pair so the last change for a road wins.
    change_map = {}
    for change in changes:
        key = tuple(sorted([change["u"], change["v"]]))
        change_map[key] = change

    new_net = network.copy()
    for key, change in change_map.items():
        u, v = change["u"], change["v"]
        if change.get("closed"):
            new_net.remove_road(u, v)
        else:
            original_w = network.weight(u, v)
            if original_w is not None:
                new_net.set_weight(u, v, original_w * change["multiplier"])
    return new_net


def compare_routes(original, updated, source, target):
    """Compare routes on original and updated networks; return full comparison dict."""
    orig_result = bellman_ford(original, source, log=True)
    upd_result = bellman_ford(updated, source, log=True)

    orig_path = get_path(orig_result["prev"], source, target)
    upd_path = get_path(upd_result["prev"], source, target)

    orig_cost = orig_result["dist"][target]
    upd_cost = upd_result["dist"][target]

    orig_cost = None if orig_cost == float("inf") else orig_cost
    upd_cost = None if upd_cost == float("inf") else upd_cost

    delta = None
    percent = None
    if orig_cost is not None and upd_cost is not None:
        delta = upd_cost - orig_cost
        if orig_cost != 0:
            percent = delta / orig_cost * 100

    route_changed = orig_path != upd_path

    # Build impact DataFrame: intersections whose dist changed.
    rows = []
    for node in original.nodes():
        o_val = orig_result["dist"][node]
        u_val = upd_result["dist"][node]
        o_str = "unreachable" if o_val == float("inf") else f"{o_val:g}"
        u_str = "unreachable" if u_val == float("inf") else f"{u_val:g}"
        if o_val != u_val:
            change_val = "unreachable" if u_val == float("inf") else f"{u_val - o_val:+g}"
            rows.append({
                "Intersection": node,
                "Original cost": o_str,
                "Updated cost": u_str,
                "Change": change_val,
            })
    impact = pandas.DataFrame(rows)

    return {
        "original_path": orig_path,
        "original_cost": orig_cost,
        "updated_path": upd_path,
        "updated_cost": upd_cost,
        "delta": delta,
        "percent": percent,
        "route_changed": route_changed,
        "impact": impact,
        "original_steps": orig_result["steps"],
        "updated_steps": upd_result["steps"],
        "original_passes": orig_result["passes"],
        "updated_passes": upd_result["passes"],
    }
