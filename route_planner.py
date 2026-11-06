"""Route planner using Dijkstra's algorithm on a RoadNetwork."""

import heapq


def dijkstra(network, source, log=True):
    """Run Dijkstra from source; return dist, prev, and optional step log."""
    dist = {node: float("inf") for node in network.nodes()}
    prev = {node: None for node in network.nodes()}
    dist[source] = 0.0

    heap = [(0.0, source)]
    settled = set()
    steps = []
    step_counter = 0

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        # Stale entry check: skip if we already found a shorter path.
        if d > dist[u]:
            continue
        settled.add(u)

        for v, w in network.neighbors(u):
            if v in settled:
                continue
            candidate = dist[u] + w
            old = dist[v]
            updated = candidate < old
            if log:
                step_counter += 1
                steps.append({
                    "step": step_counter,
                    "settled": u,
                    "neighbor": v,
                    "old": old,
                    "new": candidate,
                    "updated": updated,
                })
            if updated:
                dist[v] = candidate
                prev[v] = u
                heapq.heappush(heap, (candidate, v))

    return {"dist": dist, "prev": prev, "steps": steps}


def get_path(prev, source, target):
    """Reconstruct the path from source to target using the prev dict."""
    if source == target:
        return [source]
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = prev[current]
        if current == source:
            path.append(source)
            break
    if not path or path[-1] != source:
        return []
    path.reverse()
    return path


def shortest_route(network, source, target):
    """Return path, cost, steps, dist, and prev for the shortest route."""
    result = dijkstra(network, source, log=True)
    path = get_path(result["prev"], source, target)
    cost = result["dist"][target]
    if cost == float("inf"):
        cost = None
    return {
        "path": path,
        "cost": cost,
        "steps": result["steps"],
        "dist": result["dist"],
        "prev": result["prev"],
    }


def format_path(path):
    """Return a human-readable arrow-separated path string."""
    return " -> ".join(path)
