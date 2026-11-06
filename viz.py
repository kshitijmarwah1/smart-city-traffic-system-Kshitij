"""Visualization helpers using NetworkX for layout and Matplotlib for drawing."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Colour constants used consistently across all drawings.
EDGE_COLOR = "#B0B7C3"
NODE_FILL = "#FFFFFF"
NODE_BORDER = "#2B2D42"
ROUTE_COLOR = "#2E86AB"
UPDATED_COLOR = "#E4572E"
MST_COLOR = "#3A7D44"
CONGESTED_COLOR = "#C1121F"
VISITED_FILL = "#CFE6F2"


def compute_layout(network):
    """Build a spring layout from the network for reuse across all drawings."""
    graph = nx.Graph()
    for node in network.nodes():
        graph.add_node(node)
    for u, v, w in network.edges():
        graph.add_edge(u, v, weight=w)
    return nx.spring_layout(graph, seed=42)


def _path_to_edges(path):
    """Convert a list of nodes to a set of frozensets representing edges."""
    if not path or len(path) < 2:
        return set()
    return {frozenset([path[i], path[i + 1]]) for i in range(len(path) - 1)}


def draw_network(
    network,
    pos,
    highlight_path=None,
    highlight_edges=None,
    highlight_color=ROUTE_COLOR,
    congested_edges=None,
    visit_order=None,
    title=None,
):
    """Draw the road network and return a matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    nodes = network.nodes()
    n = len(nodes)
    all_edges = network.edges()

    # Build highlighted edge set.
    hl_set = set()
    if highlight_path and len(highlight_path) >= 2:
        hl_set = _path_to_edges(highlight_path)
    if highlight_edges:
        for u, v in highlight_edges:
            hl_set.add(frozenset([u, v]))

    cong_set = set()
    if congested_edges:
        for u, v in congested_edges:
            cong_set.add(frozenset([u, v]))

    # Draw regular edges first, then highlighted, then congested.
    for u, v, w in all_edges:
        edge_key = frozenset([u, v])
        if edge_key in cong_set:
            continue  # drawn later
        if edge_key in hl_set:
            continue  # drawn later
        xs = [pos[u][0], pos[v][0]]
        ys = [pos[u][1], pos[v][1]]
        ax.plot(xs, ys, color=EDGE_COLOR, linewidth=1.2, zorder=1)

    for u, v, w in all_edges:
        edge_key = frozenset([u, v])
        if edge_key in hl_set:
            xs = [pos[u][0], pos[v][0]]
            ys = [pos[u][1], pos[v][1]]
            ax.plot(xs, ys, color=highlight_color, linewidth=3, zorder=2)

    for u, v, w in all_edges:
        edge_key = frozenset([u, v])
        if edge_key in cong_set:
            xs = [pos[u][0], pos[v][0]]
            ys = [pos[u][1], pos[v][1]]
            ax.plot(xs, ys, color=CONGESTED_COLOR, linewidth=2.5, linestyle="--", zorder=3)

    # Draw edge weight labels for small networks.
    if n <= 25:
        for u, v, w in all_edges:
            if u in pos and v in pos:
                mx = (pos[u][0] + pos[v][0]) / 2
                my = (pos[u][1] + pos[v][1]) / 2
                ax.text(mx, my, f"{w:g}", fontsize=7, ha="center", va="center",
                        color="#555555", zorder=4,
                        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7})

    # Build visit-order label map.
    visit_map = {}
    if visit_order:
        for idx, node in enumerate(visit_order):
            visit_map[node] = idx + 1

    # Highlight nodes on path or visited.
    hl_nodes = set(highlight_path) if highlight_path else set()

    # Draw nodes.
    show_labels = n <= 40
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        if node in hl_nodes:
            fill = highlight_color
            text_color = "white"
        elif node in visit_map:
            fill = VISITED_FILL
            text_color = NODE_BORDER
        else:
            fill = NODE_FILL
            text_color = NODE_BORDER

        circle = plt.Circle((x, y), radius=0.06, color=NODE_BORDER, linewidth=1.5, fill=True,
                             zorder=5)
        circle.set_facecolor(fill)
        ax.add_patch(circle)

        if show_labels:
            label = f"{node} ({visit_map[node]})" if node in visit_map else node
            ax.text(x, y + 0.085, label, fontsize=7, ha="center", va="bottom",
                    color=text_color if node in hl_nodes else NODE_BORDER, zorder=6,
                    fontweight="bold" if node in hl_nodes else "normal")

    # Set axis limits with padding.
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = 0.25
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad - 0.1, max(ys) + pad + 0.1)

    if title:
        ax.set_title(title, fontsize=10, pad=6)

    fig.tight_layout()
    return fig


def draw_line_chart(table, x_column, y_columns, title, y_label):
    """Draw a line chart with markers for benchmark time comparisons."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [ROUTE_COLOR, UPDATED_COLOR, MST_COLOR, CONGESTED_COLOR]
    for i, col in enumerate(y_columns):
        if col in table.columns:
            color = colors[i % len(colors)]
            ax.plot(table[x_column], table[col], marker="o", label=col, color=color)
    ax.set_xlabel(x_column, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
