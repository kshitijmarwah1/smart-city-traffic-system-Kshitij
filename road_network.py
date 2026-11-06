"""Road network model: weighted undirected graph stored as an adjacency dict."""

import math
import random
from pathlib import Path

import pandas


class RoadNetwork:
    """Weighted undirected graph representing a city road network."""

    def __init__(self):
        """Initialise an empty network."""
        self.adj = {}  # {node: {neighbor: weight}}

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_intersection(self, name):
        """Add a new intersection; raise ValueError if empty or duplicate."""
        name = name.strip()
        if not name:
            raise ValueError("Intersection name must not be empty.")
        if name in self.adj:
            raise ValueError(f"Intersection '{name}' already exists.")
        self.adj[name] = {}

    def add_road(self, u, v, weight):
        """Add or overwrite a road between u and v with the given weight."""
        _validate_road_args(self, u, v, weight)
        self.adj[u][v] = weight
        self.adj[v][u] = weight

    def set_weight(self, u, v, weight):
        """Update the weight of an existing road; raise ValueError if absent."""
        _validate_road_args(self, u, v, weight)
        if v not in self.adj[u]:
            raise ValueError(f"Road between '{u}' and '{v}' does not exist.")
        self.adj[u][v] = weight
        self.adj[v][u] = weight

    def remove_road(self, u, v):
        """Remove the road between u and v in both directions."""
        if u in self.adj and v in self.adj[u]:
            del self.adj[u][v]
        if v in self.adj and u in self.adj[v]:
            del self.adj[v][u]

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def nodes(self):
        """Return intersections in insertion order."""
        return list(self.adj.keys())

    def neighbors(self, u):
        """Return list of (neighbor, weight) in insertion order."""
        return list(self.adj[u].items())

    def edges(self):
        """Return each road once as (u, v, weight) where u comes before v in nodes()."""
        node_index = {n: i for i, n in enumerate(self.adj)}
        seen = []
        for u, nbrs in self.adj.items():
            for v, w in nbrs.items():
                if node_index[u] < node_index[v]:
                    seen.append((u, v, w))
        return seen

    def weight(self, u, v):
        """Return the weight of road u-v, or None if it does not exist."""
        return self.adj.get(u, {}).get(v)

    def copy(self):
        """Return a deep copy of this network."""
        new_net = RoadNetwork()
        for node in self.adj:
            new_net.adj[node] = dict(self.adj[node])
        return new_net

    # ------------------------------------------------------------------
    # Tabular representations
    # ------------------------------------------------------------------

    def adjacency_list(self):
        """Return a DataFrame with columns Intersection and Connected roads."""
        rows = []
        for node, nbrs in self.adj.items():
            parts = [f"{nb} ({w:g})" for nb, w in nbrs.items()]
            rows.append({"Intersection": node, "Connected roads": ", ".join(parts)})
        return pandas.DataFrame(rows)

    def adjacency_matrix(self):
        """Return a DataFrame weight matrix; 0 means no road, diagonal is 0."""
        nodes = self.nodes()
        data = {col: [0.0] * len(nodes) for col in nodes}
        df = pandas.DataFrame(data, index=nodes)
        for u, nbrs in self.adj.items():
            for v, w in nbrs.items():
                df.at[u, v] = w
        return df

    def is_connected(self):
        """Return True when all intersections are reachable from the first."""
        if not self.adj:
            return False
        # Local import avoids circular dependency.
        from connectivity_module import bfs
        visited = bfs(self, self.nodes()[0])
        return len(visited) == len(self.adj)


# ------------------------------------------------------------------
# Validation helper (module-level, not part of the public API)
# ------------------------------------------------------------------

def _validate_road_args(network, u, v, weight):
    """Raise ValueError when road arguments are invalid."""
    if u not in network.adj:
        raise ValueError(f"Intersection '{u}' does not exist.")
    if v not in network.adj:
        raise ValueError(f"Intersection '{v}' does not exist.")
    if u == v:
        raise ValueError("A road cannot connect an intersection to itself.")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"Weight must be a finite number greater than 0, got {weight}.")


# ------------------------------------------------------------------
# Module-level I/O functions
# ------------------------------------------------------------------

def load_from_csv(file_or_path):
    """Load a RoadNetwork from a CSV file or uploaded file object."""
    df = pandas.read_csv(file_or_path)
    required = {"source", "destination", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}.")
    network = RoadNetwork()
    for idx, row in df.iterrows():
        csv_row = idx + 2  # 1-indexed, +1 for header
        src = str(row["source"]).strip()
        dst = str(row["destination"]).strip()
        try:
            w = float(row["weight"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row {csv_row}: weight is not a number. {exc}") from exc
        if src not in network.adj:
            network.add_intersection(src)
        if dst not in network.adj:
            network.add_intersection(dst)
        try:
            network.add_road(src, dst, w)
        except ValueError as exc:
            raise ValueError(f"Row {csv_row}: {exc}") from exc
    return network


def to_csv_text(network):
    """Return CSV text for the network (intersections with no roads are omitted)."""
    # Intersections with no roads cannot appear as source or destination,
    # so they are not representable in this three-column format.
    lines = ["source,destination,weight"]
    for u, v, w in network.edges():
        lines.append(f"{u},{v},{w:g}")
    return "\n".join(lines)


def load_sample_city():
    """Load the sample city network from data/sample_city.csv."""
    csv_path = Path(__file__).parent / "data" / "sample_city.csv"
    return load_from_csv(csv_path)


def generate_random_network(n, seed=42):
    """Generate a reproducible connected random network with n intersections."""
    rng = random.Random(seed)
    network = RoadNetwork()
    names = [f"I{i}" for i in range(1, n + 1)]
    for name in names:
        network.add_intersection(name)

    # Step 1: connect I<i> to a random earlier node to guarantee connectivity.
    for i in range(1, n):
        earlier = rng.randint(0, i - 1)
        w = rng.randint(1, 20)
        network.add_road(names[i], names[earlier], w)

    # Step 2: add extra roads until we reach the target edge count.
    target = min(int(1.5 * n), n * (n - 1) // 2)
    attempts = 0
    max_attempts = n * n * 4
    while len(network.edges()) < target and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(names, 2)
        if network.weight(a, b) is None:
            w = rng.randint(1, 20)
            network.add_road(a, b, w)
    return network
