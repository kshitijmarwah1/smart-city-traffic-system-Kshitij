"""Connectivity analysis using BFS and DFS traversals."""

from collections import deque


def bfs(network, start):
    """Return nodes in BFS visit order from start, using a queue."""
    visited = set()
    order = []
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor, _ in network.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs(network, start):
    """Return nodes in DFS visit order using an iterative stack."""
    visited = set()
    order = []
    # Push reversed neighbors so the first neighbor is popped first.
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        nbrs = [nb for nb, _ in network.neighbors(node)]
        for neighbor in reversed(nbrs):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def reachable_report(network, start, order):
    """Return a dict summarising reachable and unreachable intersections."""
    reachable_set = set(order)
    unreachable = [n for n in network.nodes() if n not in reachable_set]
    return {
        "reachable": order,
        "unreachable": unreachable,
        "count": len(order),
        "total": len(network.nodes()),
    }
