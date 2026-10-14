"""
Paper figures, drawn only from saved result files (no LLM calls).

  fig_accuracy_vs_k.pdf              — Fig. accuracyvsk
  fig_accuracy_vs_temperature.pdf    — Fig. accuracyvstemperature
  fig_{ds}_proxy_bar_plot.png        — Figs. pubar*
  fig_{ds}_pc_pu_scatter.png         — Figs. pcpuscatter*
  fig_{ds}_proxy_graph.png           — Figs. proxygraph*
  fig_proxy_use_and_proxy_capacity.png — conceptual PC/PU quadrant (no data)

Feature roles and colours follow src/proxy_risk.py so the bar plot, scatter
plot, and graph never disagree.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AUDIT_ENCODING, AUDIT_TEMPERATURE, DATASET_CONFIGS

logger = logging.getLogger(__name__)

# Categorical order is fixed per entity (validated: CVD ΔE ≥ 9 for adjacent pairs).
MODEL_STYLE = {
    "claude-3-haiku":              ("Claude-3-Haiku",    "#2a78d6", "o"),
    "gpt-4o-mini":                 ("GPT-4o-mini",       "#eb6834", "s"),
    "deepseek-v4-flash-20260423":  ("DeepSeek-V4-Flash", "#1baf7a", "^"),
}
ENC_STYLE = {"template": ("Template", "#2a78d6", "o", "-"),
             "markdown": ("Markdown", "#eb6834", "s", "--")}
TRAD_STYLE = {"DT": "-", "NB": "--", "MLP": "-.", "kNN": (0, (1, 3))}
ROLE_COLOR = {"high_risk": "#E74C3C", "use_only": "#3498DB",
              "capacity_only": "#BDC3C7", "low": "#BDC3C7"}
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#888888",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def _model_key(model: str) -> str:
    return model.split("/")[-1]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if path.suffix == ".pdf":   # PNG copy alongside every PDF
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Figure → %s", path)


def _style_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Full-width three-panel figures are shrunk to \\textwidth (≈6.5 in) in the
# paper; these sizes print at roughly 8–9 pt after scaling.
PANEL_W, PANEL_H = 5.0, 4.4
WIDE_RC = {"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 20,
           "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 19}
LW, MS = 3.0, 11


# ── RQ1: accuracy vs k ──────────────────────────────────────────────────────

def plot_accuracy_vs_k(bench: pd.DataFrame, trad: pd.DataFrame, path: Path,
                       datasets: list[str]) -> None:
    """
    bench: benchmark rows (Dataset, Model, Encoding, K, Temp, Split, Acc, ...)
    trad : traditional rows (Dataset, Model, Acc, ...)
    Accuracy is averaged over the two encodings.
    """
    with plt.rc_context(WIDE_RC):
        _accuracy_vs_k(bench, trad, path, datasets)


def _accuracy_vs_k(bench, trad, path, datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(PANEL_W * len(datasets), PANEL_H))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = bench[bench["Dataset"] == ds]
        ks = sorted(sub["K"].unique())
        xpos = {k: i for i, k in enumerate(ks)}
        for model, (label, color, marker) in MODEL_STYLE.items():
            m = sub[sub["Model"] == model]
            if m.empty:
                continue
            acc = m.groupby("K")["Acc"].mean().reindex(ks)
            ax.plot([xpos[k] for k in ks], acc.values, color=color, marker=marker,
                    lw=LW, ms=MS, label=label)
        for _, r in trad[trad["Dataset"] == ds].iterrows():
            ax.axhline(r["Acc"], color=MUTED, lw=1.8, ls=TRAD_STYLE.get(r["Model"], ":"),
                       label=r["Model"])
        ax.set_xticks(range(len(ks)), [str(int(k)) for k in ks])
        ax.set_title(DATASET_CONFIGS[ds]["title"], fontweight="bold", color=INK)
        ax.set_xlabel("Number of examples $k$")
        _style_axes(ax)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.98), frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout()
    _save(fig, path)


# ── RQ1: accuracy vs temperature ────────────────────────────────────────────

def plot_accuracy_vs_temperature(sweep: pd.DataFrame, path: Path, datasets: list[str],
                                 star=(AUDIT_ENCODING, AUDIT_TEMPERATURE)) -> None:
    with plt.rc_context(WIDE_RC):
        _accuracy_vs_temperature(sweep, path, datasets, star)


def _accuracy_vs_temperature(sweep, path, datasets, star):
    fig, axes = plt.subplots(1, len(datasets), figsize=(PANEL_W * len(datasets), PANEL_H))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = sweep[sweep["Dataset"] == ds]
        for enc, (label, color, marker, ls) in ENC_STYLE.items():
            e = sub[sub["Encoding"] == enc].sort_values("Temp")
            if e.empty:
                continue
            ax.plot(e["Temp"], e["Acc"], color=color, marker=marker, ls=ls, lw=LW, ms=MS,
                    label=label)
            if enc == star[0]:
                pt = e[np.isclose(e["Temp"], star[1])]
                if not pt.empty:
                    ax.scatter(pt["Temp"], pt["Acc"], marker="*", s=750, color="#eda100",
                               edgecolor=INK, zorder=5, label="Audit operating point")
        ax.set_title(DATASET_CONFIGS[ds]["title"], fontweight="bold", color=INK)
        temps = sorted(sub["Temp"].unique())
        ax.set_xticks(temps, [f"{t:g}" for t in temps])
        ax.set_xlabel("Decoding temperature")
        _style_axes(ax)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.98), frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout()
    _save(fig, path)


# ── RQ2: proxy use bar plot ─────────────────────────────────────────────────

def plot_proxy_use_bar(res: dict, path: Path) -> None:
    rows = sorted(res["proxy_table"], key=lambda r: -r["PU"])
    tau2 = res["thresholds"]["tau2"]
    feats = [r["feature"] for r in rows]
    pu = np.array([r["PU"] for r in rows])
    sd = np.array([r["PU_sd"] if r.get("PU_sd") is not None else np.nan for r in rows])
    sd = None if np.isnan(sd).all() else np.nan_to_num(sd)
    colors = [ROLE_COLOR[r["role"]] for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.36 * len(rows) + 1.5)))
    y = np.arange(len(rows))
    ax.barh(y, pu, xerr=sd, color=colors, edgecolor="white", height=0.72,
            error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
    ax.axvline(tau2, color="#E67E22", ls="--", lw=1.8)
    ax.set_yticks(y, feats)
    ax.invert_yaxis()
    ax.set_xlabel("Proxy use (PU)")
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color=ROLE_COLOR["high_risk"], label="Proxy risk zone (PC>τ₁, PU>τ₂)"),
        mpatches.Patch(color=ROLE_COLOR["use_only"], label="High PU, low PC"),
        mpatches.Patch(color=ROLE_COLOR["low"], label="Low PU"),
        plt.Line2D([0], [0], color="#E67E22", lw=1.8, ls="--", label=f"τ₂ = {tau2:.3f} (median PU)"),
    ], loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, path)


# ── RQ2: PC–PU scatter ──────────────────────────────────────────────────────

def plot_pc_pu_scatter(res: dict, path: Path) -> None:
    rows = res["proxy_table"]
    tau1, tau2 = res["thresholds"]["tau1"], res["thresholds"]["tau2"]
    pc = np.array([r["PC"] for r in rows])
    pu = np.array([r["PU"] for r in rows])
    prs = np.array([r["PRS"] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x_max = max(pc.max() * 1.15, tau1 * 1.5, 1e-3)
    y_lo = max(0.0, pu.min() - 0.1 * (pu.max() - pu.min() + 1e-6))
    y_hi = pu.max() + 0.12 * (pu.max() - pu.min() + 1e-6)
    ax.fill_between([tau1, x_max], tau2, y_hi, color="#FDECEA", zorder=0)
    sc = ax.scatter(pc, pu, c=prs, cmap="YlOrRd", s=150, edgecolors="#555555",
                    linewidths=0.8, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Proxy risk score (PRS = PC × PU)")
    ax.axvline(tau1, color="#2980B9", ls="--", lw=1.6, label=f"τ₁ = {tau1:.3f} (median PC)")
    ax.axhline(tau2, color="#E67E22", ls="--", lw=1.6, label=f"τ₂ = {tau2:.3f} (median PU)")
    ax.text(0.98, 0.97, "Proxy risk zone", transform=ax.transAxes, ha="right", va="top",
            color="#C0392B", fontweight="bold")

    texts = [ax.text(x, y, r["feature"], fontsize=9.5, color=INK)
             for x, y, r in zip(pc, pu, rows)]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=pc, y=pu, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6))
    except ImportError:
        pass
    ax.set_xlim(-0.02 * x_max, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"Proxy capacity $\mathrm{PC}_j = I(X_j; A)$  [nats]")
    ax.set_ylabel("Proxy use (PU)")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, path)


# ── RQ2: proxy path graph ───────────────────────────────────────────────────

def plot_graph(res: dict, path: Path) -> None:
    from src.proxy_graph import graph_from_dict, plot_proxy_graph
    ds = res["meta"]["dataset"]
    G = graph_from_dict(res["graph"])
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_proxy_graph(G, DATASET_CONFIGS[ds]["sensitive_col"], save_path=path)


# ── Conceptual PC/PU figure ─────────────────────────────────────────────────

def plot_pc_pu_concept(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    quads = [
        (0, 0, "#F2F2F2", "Low risk\nlittle protected information,\nlittle influence"),
        (1, 0, "#E3F0FA", "Encodes A, not used\n(latent proxy)"),
        (0, 1, "#E3F0FA", "Used, does not encode A\n(e.g. repayment history)"),
        (1, 1, "#FDECEA", "High-risk proxy\nencodes A and drives\nthe decision"),
    ]
    for x, y, color, text in quads:
        ax.add_patch(mpatches.Rectangle((x, y), 1, 1, color=color, ec="white", lw=3))
        ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=10.5,
                color="#C0392B" if (x, y) == (1, 1) else INK,
                fontweight="bold" if (x, y) == (1, 1) else "normal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["low", "high"])
    ax.set_yticks([0.5, 1.5], ["low", "high"])
    ax.set_xlabel(r"Proxy capacity  $\mathrm{PC}_j = I(X_j; A)$")
    ax.set_ylabel(r"Proxy use  $\mathrm{PU}_j$")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    _save(fig, path)
