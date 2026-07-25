"""Benchmark figure generation (matplotlib, light-theme PNG).

Colors follow a validated categorical palette (assigned in fixed order, never cycled):
  slot1 blue   #2a78d6 = Assay (per-request internal verification)
  slot2 aqua   #1baf7a = black-box audit (10% interleaved with traffic)
  slot3 yellow #eda100 = black-box audit (dedicated queries)
Low-contrast colors are backed by the legend and direct labels (relief rule).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_ASSAY = "#2a78d6"
C_BB = "#1baf7a"
C_BB_DENSE = "#eda100"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_MUTED = "#898781"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_SURFACE = "#fcfcfb"
C_CRITICAL = "#d03b3b"

SCENARIO_LABELS = {
    "honest": "Honest (control)",
    "size": "Full substitution (size: M -> M')",
    "quant": "Quantization substitution (FP32 -> INT8)",
    "random": "Randomized substitution (M w.p. p=0.9)",
    "evasive": "Audit evasion (canary -> M)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
        ],
        "figure.facecolor": C_SURFACE,
        "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE,
        "axes.edgecolor": C_AXIS,
        "axes.labelcolor": C_INK2,
        "axes.titlecolor": C_INK,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def _clip_curve(ys: List[float], cap: float) -> np.ndarray:
    return np.minimum(np.asarray(ys, dtype=float), cap)


def fig_headline(results: Dict[str, dict], threshold: float, path: str) -> None:
    """Headline figure: cumulative-evidence (log e-value) traces — Assay detects in a
    few requests thanks to full coverage, while the black box lags under randomization
    and stays flat forever under audit evasion."""
    apply_style()
    names = ["size", "quant", "random", "evasive"]
    cap = threshold * 4
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=False)
    for ax, name in zip(axes.flat, names):
        r = results[name]
        ax.axhline(threshold, color=C_MUTED, lw=1.2, ls="--", zorder=1)
        ax.plot(range(1, len(r["assay_history"]) + 1), _clip_curve(r["assay_history"], cap),
                color=C_ASSAY, lw=2.2, zorder=3, solid_capstyle="round")
        if r["bb_history_x"]:
            ax.plot(r["bb_history_x"], _clip_curve(r["bb_history_y"], cap),
                    color=C_BB, lw=2.0, zorder=2, solid_capstyle="round")
        ta = r.get("assay_t_alarm")
        if ta:
            ax.plot([ta], [min(threshold, cap)], "o", color=C_ASSAY, ms=8, zorder=4)
            ax.annotate(f"detected t={ta}", (ta, threshold), textcoords="offset points",
                        xytext=(8, 10), fontsize=9, color=C_ASSAY, fontweight="bold")
        tb = r.get("bb_t_alarm_traffic")
        if tb:
            ax.plot([tb], [threshold], "s", color=C_BB, ms=7, zorder=4)
            ax.annotate(f"BB detected t={tb}", (tb, threshold), textcoords="offset points",
                        xytext=(8, -16), fontsize=9, color=C_BB)
        elif name in ("random", "evasive"):
            ax.annotate("black box: not detected", (0.5, 0.12), xycoords="axes fraction",
                        fontsize=10, color=C_BB, fontweight="bold")
        ax.set_title(SCENARIO_LABELS[name], fontsize=11)
        ax.set_xlabel("number of requests")
        ax.set_ylabel("cumulative log e-value")
        ax.set_ylim(min(-2, threshold * -0.3), cap * 1.05)
    handles = [
        plt.Line2D([], [], color=C_ASSAY, lw=2.2, label="Assay (100% coverage)"),
        plt.Line2D([], [], color=C_BB, lw=2.0, label="black-box audit (canary, 10% interleaved)"),
        plt.Line2D([], [], color=C_MUTED, lw=1.2, ls="--", label="alarm threshold log(1/alpha)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cumulative evidence for substitution: per-request internal receipts vs external sampling audit",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_roc(rocs: Dict[str, dict], path: str) -> None:
    """Per-task ROC (per-request z statistic)."""
    apply_style()
    colors = {"size": C_ASSAY, "quant": C_BB, "random": C_BB_DENSE, "evasive": "#008300"}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=C_GRID, lw=1.0, ls="--", zorder=1)
    for name, roc in rocs.items():
        ax.plot(roc["fpr"], roc["tpr"], color=colors[name], lw=2.0, zorder=3,
                label=f'{SCENARIO_LABELS[name]}  AUC={roc["auc"]:.3f}')
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC of Assay per-request detection", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cost(cost: dict, path: str) -> None:
    """Verifier cost ratio: layer-token ratio and wall-clock ratio (full recompute = 1.0)."""
    apply_style()
    labels = ["layer-token ratio\n(blocks run x tokens)", "wall-clock ratio\n(measured time)"]
    vals = [cost["layer_token_ratio"], cost["wall_ratio"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    bars = ax.bar(labels, vals, width=0.5, color=C_ASSAY, zorder=3)
    ax.axhline(1.0, color=C_CRITICAL, lw=1.4, ls="--", zorder=2)
    ax.annotate("full recompute = 1.0", (0.98, 1.0), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 5), ha="right", fontsize=9, color=C_CRITICAL)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=11, color=C_INK, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ratio to full recompute")
    ax.set_title("Verifier cost (per request, every request verified)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_invariance(inv: Dict[str, dict], path: str) -> None:
    """Invariant-feature robustness: unmoved by benign jitter, moves only under quant/swap."""
    apply_style()
    conds = ["honest", "quant", "size"]
    cond_labels = ["benign jitter\n(honest)", "quantization\n(INT8)", "substitution\n(size)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, key, title in zip(axes, ["sign", "rank"],
                              ["sign-bit agreement", "top-rank overlap (Jaccard)"]):
        data = [inv[c][key] for c in conds]
        bp = ax.boxplot(data, tick_labels=cond_labels, widths=0.45, patch_artist=True,
                        medianprops={"color": C_INK, "lw": 1.4},
                        flierprops={"marker": ".", "markersize": 4, "markeredgecolor": C_MUTED})
        for patch in bp["boxes"]:
            patch.set_facecolor(C_ASSAY)
            patch.set_alpha(0.55)
            patch.set_edgecolor(C_ASSAY)
        for m, vals in zip([1, 2, 3], data):
            ax.annotate(f"{np.median(vals):.2f}", (m, float(np.median(vals))),
                        textcoords="offset points", xytext=(26, -4), fontsize=9, color=C_INK2)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("agreement")
    fig.suptitle("Invariant features do not move under benign nondeterminism (robustness)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_queries(qtd: Dict[str, dict], n_budget: int, path: str) -> None:
    """Queries-to-detection (median, log axis). ND = not detected within budget."""
    apply_style()
    names = ["size", "quant", "random", "evasive"]
    methods = [
        ("assay", "Assay (100% coverage)", C_ASSAY),
        ("bb", "black box (canary, 10% interleaved)", C_BB),
        ("bb_dense", "black box (dedicated queries)", C_BB_DENSE),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ypos = np.arange(len(names))[::-1] * (len(methods) + 1.2)
    nd_x = n_budget * 2
    # Bars start at x0 < 1 so that a value of exactly 1 (Assay's usual result) still
    # renders a visible bar on the log axis instead of collapsing to zero width.
    x0 = 0.62
    for mi, (key, label, color) in enumerate(methods):
        for ni, name in enumerate(names):
            v = qtd[name].get(key)
            y = ypos[ni] + (len(methods) - 1 - mi)
            if v is None:
                ax.barh(y, nd_x - x0, left=x0, height=0.85, color=color, alpha=0.22, hatch="///",
                        edgecolor=color, lw=0.8, zorder=2)
                ax.annotate("not detected within budget", (nd_x * 0.97, y), va="center", ha="right",
                            fontsize=8.5, color=C_INK2)
            else:
                ax.barh(y, max(v, x0 * 1.02) - x0, left=x0, height=0.85, color=color, zorder=3)
                ax.annotate(f"{v:g}", (max(v, x0) * 1.08, y), va="center", fontsize=9.5,
                            color=C_INK, fontweight="bold")
    ax.set_yticks(ypos + (len(methods) - 1) / 2)
    ax.set_yticklabels([SCENARIO_LABELS[n] for n in names], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(x0, nd_x * 1.15)
    ax.set_xlabel("queries to detection (median, log axis)")
    ax.set_title("Queries to detection: lower is better", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in methods]
    # Legend below the axes: inside the plot it collides with the full-width
    # "not detected" bars.
    ax.legend(handles, [l for _, l, _ in methods], fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
