"""Plots Experiment 1's headline finding: how the meta-gradient norm scales
with unroll length T, per method, on a shared log axis.

Not part of the Benchmarking platform -- a one-off analysis script reading
runs/exp1_calibration/results.jsonl (per-run rows), same pattern as
plot_ood_degradation.py. This sweep is still in progress; the script uses
only whichever seeds have FULL (method, dim, unroll_length) coverage so far,
so it stays correct as more seeds land -- re-run any time to refresh.

Uses `grad_norm_final` (the terminal profiled value for that run), not
`grad_norm_mean` -- see experiment1.py's `_read_profile` docstring: DM/RNNProp
log one point per meta-training epoch while L2O-Scale logs one point per
unroll segment, so a *mean* averages over differently-sized populations per
method and isn't safe to compare across methods. The final value is one
specific point regardless of logging frequency, so it is.

Uses median + IQR rather than mean +/- std, pooled across dims and the
completed seeds -- L2O-DM's grad norm is heavily right-skewed (mean can be
~5x the median at high T; see the accompanying investigation notes), so a
mean +/- std band would be dominated by a few extreme seeds.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = os.path.join("runs", "exp1_calibration", "results.jsonl")
OUT_NAME = "experiment1_gradnorm_scaling"
# Thesis figures directory (repo root, sibling of Benchmarking/) -- the only
# place figures are written (not also alongside this script).
THESIS_FIGS_DIR = os.path.join("..", "Figs")

# Same method->color assignment as plot_gradnorm_instability.py (l2o-dm =
# orange) for a consistent look across the whole figure set; rnnprop/scale
# added from the same validated 8-slot categorical order (dataviz skill).
METHODS = [
    ("l2o-dm", "L2O-DM", "#eb6834", "o"),
    ("l2o-rnnprop", "L2O-RNNProp", "#2a78d6", "s"),
    ("l2o-scale", "L2O-Scale", "#1baf7a", "^"),
]
METRIC = "grad_norm_final"
Y_LABEL = r"Meta-gradient norm $\Vert\nabla_\phi \mathcal{L}\Vert$ (final)" + \
    "\nmedian, IQR band (log scale)"

# dataviz skill chrome: light chart surface + recessive gridlines/ink.
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def _load_complete_rows(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    full_combo_count = len({(r["method"], r["dim"], r["unroll_length"]) for r in rows})
    per_seed_count = {}
    for r in rows:
        per_seed_count[r["seed"]] = per_seed_count.get(r["seed"], 0) + 1
    complete_seeds = {s for s, n in per_seed_count.items() if n == full_combo_count}
    return [r for r in rows if r["seed"] in complete_seeds], sorted(complete_seeds)


def main():
    rows, complete_seeds = _load_complete_rows(RESULTS_PATH)
    print("using {} rows from {} complete seeds: {}".format(
        len(rows), len(complete_seeds), complete_seeds))

    # (method, unroll_length) -> [values pooled across dims/seeds]
    by_method_t = {}
    for r in rows:
        if r.get(METRIC) is None:
            continue
        by_method_t.setdefault((r["method"], r["unroll_length"]), []).append(r[METRIC])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)

    for method_key, label, color, marker in METHODS:
        ts = sorted({t for (m, t) in by_method_t if m == method_key})
        medians = np.array([np.median(by_method_t[(method_key, t)]) for t in ts])
        q1 = np.array([np.percentile(by_method_t[(method_key, t)], 25) for t in ts])
        q3 = np.array([np.percentile(by_method_t[(method_key, t)], 75) for t in ts])

        ax.plot(ts, medians, color=color, marker=marker, markersize=5,
                linewidth=2, zorder=3, label=label)
        ax.fill_between(ts, q1, q3, color=color, alpha=0.15, zorder=2, linewidth=0)

    ax.set_title("Meta-gradient norm vs. unroll length T",
                 fontsize=12, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Unroll length T", color=INK_PRIMARY)
    ax.set_ylabel(Y_LABEL, color=INK_PRIMARY, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()

    out_stem = os.path.join(THESIS_FIGS_DIR, OUT_NAME)
    fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("wrote", out_stem + ".png")
    plt.close(fig)


if __name__ == "__main__":
    main()
