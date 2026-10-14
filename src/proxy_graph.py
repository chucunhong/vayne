"""
Proxy Path Graph G = (V, E) (Sect. 3.5)

    A ──capacity (PC_j > τ₁)──► X_j ──use (PU_j > τ₂)──► D

A formal proxy path is the two-hop pattern A → X_j → D. Feature–feature
association links (I(X_j; X_k) > τ₃) are undirected and descriptive only:
they show clusters and redundant encodings, not direction or mediation.
Node colours follow the roles of src/proxy_risk.py.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import networkx as nx

from src.proxy_risk import EPS, role

logger = logging.getLogger(__name__)

DECISION_NODE = "DECISION"


# ── Graph construction ────────────────────────────────────────────────────────

def build_proxy_graph(
    sensitive_attr: str,
    proxy_capacity: dict[str, float],
    proxy_use:      dict[str, float],
    pair_mi:        dict[tuple[str, str], float],
    tau_capacity:   float,
    tau_use:        float,
    tau_assoc:      float,
) -> nx.DiGraph:
    """
    Node types : 'sensitive', 'proxy', 'decision'
    Edge types : 'capacity' | 'use' | 'association' (stored once, u < v)
    """
    G = nx.DiGraph()
    G.add_node(sensitive_attr, node_type="sensitive")
    G.add_node(DECISION_NODE,  node_type="decision")

    for feat in proxy_use:
        pc = proxy_capacity.get(feat, 0.0)
        pu = proxy_use[feat]
        r = role(pc, pu, tau_capacity, tau_use)
        if r == "low":
            continue
        G.add_node(feat, node_type="proxy", role=r, pc=round(pc, 4), pu=round(pu, 4))
        if pc > tau_capacity + EPS:
            G.add_edge(sensitive_attr, feat, weight=round(pc, 4), edge_type="capacity")
        if pu > tau_use + EPS:
            G.add_edge(feat, DECISION_NODE, weight=round(pu, 4), edge_type="use")

    for (fi, fj), mi in pair_mi.items():
        if mi > tau_assoc and fi in G.nodes and fj in G.nodes:
            u, v = sorted((fi, fj))
            G.add_edge(u, v, weight=round(mi, 4), edge_type="association")

    logger.info("Proxy graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def graph_from_dict(gd: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in gd["nodes"]:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in gd["edges"]:
        G.add_edge(e["source"], e["target"],
                   **{k: v for k, v in e.items() if k not in ("source", "target")})
    return G


def get_proxy_paths(G: nx.DiGraph, sensitive_attr: str) -> list[list[str]]:
    """Formal proxy paths A → X_j → D (association links are not traversed)."""
    return [[sensitive_attr, v, DECISION_NODE]
            for _, v, d in G.out_edges(sensitive_attr, data=True)
            if d.get("edge_type") == "capacity" and G.has_edge(v, DECISION_NODE)]


def graph_to_dict(G: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
    }


def graph_to_ascii(G: nx.DiGraph, sensitive_attr: str) -> str:
    paths = get_proxy_paths(G, sensitive_attr)
    if not paths:
        return f"{sensitive_attr} → (no proxy paths found)"
    lines = [sensitive_attr]
    for path in paths:
        parts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            w  = G.edges[u, v].get("weight", "?")
            et = G.edges[u, v].get("edge_type", "")
            parts.append(f"─[{et}={w}]─► {v}")
        lines.append("  " + "".join(parts))
    return "\n".join(lines)


# ── Visualization ─────────────────────────────────────────────────────────────

def _detect_font() -> str:
    """Return Roboto if installed, else best available sans-serif."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Roboto", "Liberation Sans", "Noto Sans", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def plot_proxy_graph(
    G: nx.DiGraph,
    sensitive_attr: str,
    save_path: Optional[Path] = None,
    max_assoc_links: int = 8,
) -> None:
    """
    Render the Proxy Path Graph with:
      • Large, high-contrast circular nodes
      • Roboto font (fallback to Liberation Sans / DejaVu Sans)
      • Structured column layout: A | Capacity Only | Capacity & Use | Use Only | Decision
      • Colour-coded column header badges at the bottom
      • Dashed blue  : A → proxy  (capacity edge)
      • Solid dark   : proxy → Decision  (use edge)
      • Dotted gray  : proxy — proxy  (association link, undirected; the
                       max_assoc_links strongest links are drawn,
                       width proportional to MI)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — skipping plot.")
        return

    FONT    = _detect_font()
    COL_GAP = 1.9    # horizontal spacing between columns
    ROW_GAP = 1.2    # vertical spacing between nodes in same column
    NODE_S  = 5500   # uniform node size (points²)
    # node radius in pt = sqrt(NODE_S/π); +12 for border lw + clearance
    SHRINK  = int(np.sqrt(NODE_S / np.pi)) + 12   # ≈ 54 pt
    LW_CAP  = 2.0    # capacity edge line width
    LW_USE  = 2.8    # use edge line width (thicker)

    proxy_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "proxy"]

    # ── Column layout ────────────────────────────────────────────────────────
    cap_set  = {v for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "capacity" and u == sensitive_attr}
    use_set  = {u for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "use" and v == DECISION_NODE}
    both     = sorted(cap_set & use_set)
    cap_only = sorted(cap_set - use_set)
    use_only = sorted(use_set - cap_set)

    def _ys(nodes):
        n = len(nodes)
        return [(i - (n - 1) / 2) * ROW_GAP for i in range(n)]

    cols = {sensitive_attr: -2 * COL_GAP, DECISION_NODE: 2 * COL_GAP}
    for n, y in zip(cap_only, _ys(cap_only)): cols[n] = -COL_GAP
    for n, y in zip(both,     _ys(both)):     cols[n] = 0.0
    for n, y in zip(use_only, _ys(use_only)): cols[n] =  COL_GAP

    pos = {sensitive_attr: np.array([cols[sensitive_attr], 0.0]),
           DECISION_NODE:  np.array([cols[DECISION_NODE],  0.0])}
    for n, y in zip(cap_only, _ys(cap_only)): pos[n] = np.array([cols[n], y])
    for n, y in zip(both,     _ys(both)):     pos[n] = np.array([cols[n], y])
    for n, y in zip(use_only, _ys(use_only)): pos[n] = np.array([cols[n], y])

    # ── Role colour mapping (src/proxy_risk.py) ──────────────────────────────
    _ROLE_FC = {"high_risk": "#FFDBA4", "capacity_only": "#C8EDCC",
                "use_only": "#C5E3F5", "low": "#C5E3F5"}
    _ROLE_EC = {"high_risk": "#E67E22", "capacity_only": "#27AE60",
                "use_only": "#2980B9", "low": "#2980B9"}

    def _node_fc(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt == "sensitive": return "#E8D5F5"
        if nt == "decision":  return "#DDE3EA"
        return _ROLE_FC[G.nodes[n].get("role", "low")]

    def _node_ec(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt == "sensitive": return "#9B59B6"
        if nt == "decision":  return "#7F8C8D"
        return _ROLE_EC[G.nodes[n].get("role", "low")]

    def _node_lw(n):
        return 3.0 if G.nodes[n].get("node_type") == "sensitive" else 2.0

    # ── Edge classification ──────────────────────────────────────────────────
    cap_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "capacity"]
    use_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "use"]
    assoc = sorted(((u, v, d["weight"]) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "association"), key=lambda e: -e[2])
    assoc = assoc[:max_assoc_links]
    # ── Figure ───────────────────────────────────────────────────────────────
    max_col = max(len(cap_only), len(both), len(use_only), 1)
    fig_h   = max(7.0, max_col * ROW_GAP + 4.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.rcParams["font.family"] = FONT

    # ── Edges ────────────────────────────────────────────────────────────────
    # Edges that skip a column (sensitive→both, both→DECISION) get a small
    # curvature so they don't overlap straight edges to adjacent columns.
    both_set = set(both)

    def _arrow(u, v, color, ls, lw, rad=None, alpha=1.0):
        if rad is None:
            skip = (u == sensitive_attr and v in both_set) or \
                   (u in both_set and v == DECISION_NODE)
            rad = 0.22 if skip else 0.0
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                color=color, lw=lw,
                linestyle=ls,
                shrinkA=SHRINK, shrinkB=SHRINK,
                mutation_scale=18,
                alpha=alpha,
            ),
            zorder=2,
        )

    for u, v in cap_edges:
        _arrow(u, v, "#4A90D9", "dashed", LW_CAP, alpha=0.9)
    for u, v in use_edges:
        _arrow(u, v, "#1A252F", "solid",  LW_USE, alpha=0.95)

    # Association links: undirected, descriptive only. Same-column links
    # bulge outward far enough to clear the nodes between their ends; links
    # across columns are straight. Width grows with the pairwise MI.
    ASSOC_C = "#8A94A0"
    w_max = max((w for _, _, w in assoc), default=1.0)
    for u, v, w in assoc:
        (xu, yu), (xv, yv) = pos[u], pos[v]
        if abs(xu - xv) < 1e-6:
            if yu < yv:                      # draw top → bottom
                u, v, yu, yv = v, u, yv, yu
            dist = yu - yv
            outward = 1 if xu >= 0 else -1  # right of centre bulges right
            rad = -outward * min(0.9, 1.9 / dist)
        else:
            rad = 0.0
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-", connectionstyle=f"arc3,rad={rad:.3f}",
                            color=ASSOC_C, lw=1.4 + 1.8 * w / w_max, linestyle=(0, (5, 3)),
                            shrinkA=SHRINK - 10, shrinkB=SHRINK - 10, alpha=0.9,
                            capstyle="round"),
            zorder=1,
        )

    # ── Nodes ────────────────────────────────────────────────────────────────
    def _label(name: str, max_ch: int = 9) -> str:
        """Fit name into two lines of max_ch chars, truncate with … if needed."""
        words = name.replace("-", " ").split()
        if len(words) == 1:
            return name[:max_ch] + ("…" if len(name) > max_ch else "")
        # Find split that balances both lines
        best_i, best_diff = 1, float("inf")
        for i in range(1, len(words)):
            diff = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
            if diff < best_diff:
                best_diff, best_i = diff, i
        l1 = " ".join(words[:best_i])
        l2 = " ".join(words[best_i:])
        if len(l1) > max_ch: l1 = l1[:max_ch - 1] + "…"
        if len(l2) > max_ch: l2 = l2[:max_ch - 1] + "…"
        return f"{l1}\n{l2}"

    node_list = list(G.nodes)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[_node_fc(n) for n in node_list],
        node_size=[NODE_S for _ in node_list],
        node_shape="o",
        linewidths=[_node_lw(n) for n in node_list],
        edgecolors=[_node_ec(n) for n in node_list],
        alpha=1.0,
    )
    nx.draw_networkx_labels(
        G, pos, {n: _label(n) for n in G.nodes}, ax=ax,
        font_size=13, font_weight="bold",
        font_color="#1A1A2E", font_family=FONT,
    )

    # ── Column badges ─────────────────────────────────────────────────────────
    all_ys  = [p[1] for p in pos.values()]
    y_badge = min(all_ys) - ROW_GAP * 0.9
    _BADGE = {
        "A":    ("#E8D5F5", "#9B59B6", "Protected Attribute (A)"),
        "cap":  ("#F4F4F4", "#AAAAAA", "Capacity Only"),
        "both": ("#FFE8C2", "#E67E22", "Capacity & Use"),
        "use":  ("#F4F4F4", "#AAAAAA", "Use Only"),
        "dec":  ("#DDE3EA", "#7F8C8D", "Decision"),
    }
    for x_col, key, show in [
        (cols[sensitive_attr], "A",   True),
        (-COL_GAP,            "cap",  bool(cap_only)),
        (0.0,                 "both", bool(both)),
        ( COL_GAP,            "use",  bool(use_only)),
        (cols[DECISION_NODE], "dec",  True),
    ]:
        if not show:
            continue
        fc_b, ec_b, text = _BADGE[key]
        ax.text(x_col, y_badge, text, ha="center", va="center",
                fontsize=15, fontfamily=FONT, color="#333333",
                bbox=dict(boxstyle="round,pad=0.55", facecolor=fc_b,
                          edgecolor=ec_b, linewidth=1.5, alpha=0.95),
                zorder=5)

    # ── Legend — placed outside axes to the right ────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#E8D5F5", edgecolor="#9B59B6", lw=1.8,
                       label="Protected Attribute (A)"),
        mpatches.Patch(facecolor="#FFDBA4", edgecolor="#E67E22", lw=1.8,
                       label="High-risk proxy (PC>τ₁, PU>τ₂)"),
        mpatches.Patch(facecolor="#C8EDCC", edgecolor="#27AE60", lw=1.8,
                       label="Medium risk (PC>τ₁ only)"),
        mpatches.Patch(facecolor="#C5E3F5", edgecolor="#2980B9", lw=1.8,
                       label="Use-only (PU>τ₂ only)"),
        mpatches.Patch(facecolor="#DDE3EA", edgecolor="#7F8C8D", lw=1.8,
                       label="Decision"),
        Line2D([0], [0], color="#4A90D9", lw=2.0, linestyle="dashed",
               label="Capacity edge (PC)"),
        Line2D([0], [0], color="#1A252F", lw=2.5,
               label="Use edge (PU)"),
        Line2D([0], [0], color="#8A94A0", lw=2.2, linestyle=(0, (5, 3)),
               label="Association (MI>τ₃; thicker = higher MI)"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, fontsize=14, framealpha=1.0,
              edgecolor="#CCCCCC", labelspacing=0.7,
              prop={"family": FONT, "size": 14})

    ax.axis("off")

    # Tight axis limits — no excess whitespace
    all_xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(all_xs) - 0.9, max(all_xs) + 0.9)
    ax.set_ylim(y_badge - 0.35, max(all_ys) + 0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.97, right=0.78)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Proxy graph figure saved → %s", save_path)
    else:
        plt.show()
