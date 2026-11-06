"""Minimum spanning tree (MST) algorithms: Prim and Kruskal."""

import heapq


def prim(network, start=None):
    """Build MST/forest with Prim's algorithm starting from start node."""
    nodes = network.nodes()
    if not nodes:
        return {"edges": [], "total": 0.0, "components": 0}

    if start is None:
        start = nodes[0]

    visited = set()
    mst_edges = []
    total = 0.0
    components = 0

    unvisited_roots = list(nodes)

    while unvisited_roots:
        # Find the next unvisited node to start a new component.
        root = None
        for candidate in unvisited_roots:
            if candidate not in visited:
                root = candidate
                break
        if root is None:
            break

        components += 1
        heap = []
        visited.add(root)
        for nb, w in network.neighbors(root):
            heapq.heappush(heap, (w, root, nb))

        while heap:
            w, u, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            mst_edges.append((u, v, w))
            total += w
            for nb, ww in network.neighbors(v):
                if nb not in visited:
                    heapq.heappush(heap, (ww, v, nb))

        # Remove visited nodes from unvisited_roots tracking list.
        unvisited_roots = [n for n in unvisited_roots if n not in visited]

    return {"edges": mst_edges, "total": total, "components": components}


# ------------------------------------------------------------------
# Union-Find helpers for Kruskal
# ------------------------------------------------------------------

def _find(parent, node):
    """Find root of node with path compression."""
    if parent[node] != node:
        parent[node] = _find(parent, parent[node])
    return parent[node]


def _union(parent, rank, a, b):
    """Union two sets by rank; return True if they were separate."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return False
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return True


def kruskal(network):
    """Build MST/forest with Kruskal's algorithm using Union-Find."""
    nodes = network.nodes()
    if not nodes:
        return {"edges": [], "total": 0.0, "components": len(nodes)}

    parent = {n: n for n in nodes}
    rank = {n: 0 for n in nodes}

    # Stable sort by weight preserves insertion order for ties.
    sorted_edges = sorted(network.edges(), key=lambda e: e[2])

    mst_edges = []
    total = 0.0

    for u, v, w in sorted_edges:
        if _union(parent, rank, u, v):
            mst_edges.append((u, v, w))
            total += w

    components = len(nodes) - len(mst_edges)
    return {"edges": mst_edges, "total": total, "components": components}
