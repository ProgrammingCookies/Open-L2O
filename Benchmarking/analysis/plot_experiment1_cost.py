"""Plots Experiment 1's compute-cost side: per-optimizee-step wall-clock cost
per method (left panel), and how peak allocated GPU memory scales with LASSO
dimension per method (right panel). Two panels, each its own single axis --
never a dual-axis plot for two differently-scaled measures (dataviz skill's
"one axis" rule).

Not part of the Benchmarking platform -- a one-off analysis script reading
runs/exp1_calibration/results.jsonl. This sweep is still in progress; the
script uses only whichever seeds have FULL (method, dim, unroll_length)
coverage so far, so it stays correct as more seeds land -- re-run any time
to refresh.

`avg_time_per_iteration_s` is Table 2's cross-method-comparable timing
figure (normalized by optimizee steps, not by how often each method logs --
see experiment1.py's `_read_profile` docstring) and is fairly flat across
dims/T for a given method, so the left panel collapses it to one bar per
method (median, IQR whiskers) rather than faceting further.
`peak_allocated_mb`/`peak_reserved_mb` report the same underlying TF 'peak'
figure (this repo's own documented simplification vs. Table 2's separate
PyTorch-style columns), so only one is plotted.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = os.path.join("runs", "exp1_calibration", "results.jsonl")
OUT_NAME = "experiment1_cost_scaling"
# Thesis figures directory (repo root, sibling of Benchmarking/) -- the only
# place figures are written (not also alongside this script).
THESIS_FIGS_DIR = os.path.join("..", "Figs")

METHODS = [
    ("l2o-dm", "L2O-DM", "#eb6834", "o"),
    ("l2o-rnnprop", "L2O-RNNProp", "#2a78d6", "s"),
    ("l2o-scale", "L2O-Scale", "#1baf7a", "^"),
]
DIM_ORDER = ["25x50", "50x100", "250x500", "500x1000"]

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

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax in (ax_time, ax_mem):
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)

    # --- Left panel: avg wall-clock cost per optimizee step, one bar/method.
    # Table 2 (pre-study) states this metric in ms, not s -- converted here.
    time_vals = {m: [] for m, *_ in METHODS}
    for r in rows:
        v = r.get("avg_time_per_iteration_s")
        if v is not None and r["method"] in time_vals:
            time_vals[r["method"]].append(v * 1000.0)

    xs = np.arange(len(METHODS))
    medians = [np.median(time_vals[m]) for m, *_ in METHODS]
    q1 = [np.percentile(time_vals[m], 25) for m, *_ in METHODS]
    q3 = [np.percentile(time_vals[m], 75) for m, *_ in METHODS]
    err_low = [med - lo for med, lo in zip(medians, q1)]
    err_high = [hi - med for med, hi in zip(medians, q3)]
    colors = [c for _, _, c, _ in METHODS]
    labels = [lbl for _, lbl, _, _ in METHODS]

    ax_time.bar(xs, medians, color=colors, width=0.55, zorder=3,
               yerr=[err_low, err_high], capsize=4,
               error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "zorder": 4})
    ax_time.set_xticks(xs)
    ax_time.set_xticklabels(labels, fontsize=9)
    ax_time.set_ylabel("Avg time / optimizee step (ms)\nmedian, IQR whiskers",
                       color=INK_PRIMARY, fontsize=9)
    ax_time.set_title("Per-step wall-clock cost", fontsize=11, color=INK_PRIMARY, loc="left")

    # --- Right panel: peak allocated memory vs. dim, one line/method.
    mem_index = {}
    for r in rows:
        v = r.get("peak_allocated_mb")
        if v is None:
            continue
        mem_index.setdefault((r["method"], r["dim"]), []).append(v)

    for method_key, label, color, marker in METHODS:
        dims_present = [d for d in DIM_ORDER if (method_key, d) in mem_index]
        vals = [np.median(mem_index[(method_key, d)]) for d in dims_present]
        ax_mem.plot(range(len(dims_present)), vals, color=color, marker=marker,
                   markersize=6, linewidth=2, zorder=3, label=label)
    ax_mem.set_xticks(range(len(DIM_ORDER)))
    ax_mem.set_xticklabels(DIM_ORDER, fontsize=9)
    ax_mem.set_xlabel("LASSO dimension (m x n)", color=INK_PRIMARY)
    ax_mem.set_ylabel("Peak allocated GPU memory (MB)\nmedian across T, seeds",
                      color=INK_PRIMARY, fontsize=9)
    ax_mem.set_title("Memory vs. problem dimension", fontsize=11, color=INK_PRIMARY, loc="left")
    ax_mem.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()

    out_stem = os.path.join(THESIS_FIGS_DIR, OUT_NAME)
    fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("wrote", out_stem + ".png")
    plt.close(fig)


if __name__ == "__main__":
    main()
