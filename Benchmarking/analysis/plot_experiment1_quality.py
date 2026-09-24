"""Plots Experiment 1's quality/compute tradeoff: how Modified Relative Loss
(Eq. 10) changes with unroll length T, per method, with one line per LASSO
dimension -- directly answers Table 1's question of whether the T tradeoff
shifts with optimizee dimensionality. One standalone figure per method
(experiment1_quality_scaling_<method>.png), same layout convention as
plot_ood_degradation.py's per-method panels.

Not part of the Benchmarking platform -- a one-off analysis script reading
runs/exp1_calibration/results.jsonl. This sweep is still in progress; the
script uses only whichever seeds have FULL (method, dim, unroll_length)
coverage so far, so it stays correct as more seeds land -- re-run any time
to refresh.

Uses median + IQR across the completed seeds rather than mean +/- std --
checked directly against this run's own per-seed distributions before
choosing this (l2o-dm's modified_relative_loss has seed-to-seed coefficient
of variation as high as ~3 at some T, i.e. std exceeding the mean; l2o-scale
and l2o-rnnprop are calmer but still meaningfully skewed), matching the
convention plot_ood_degradation.py already uses for the same metric.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = os.path.join("runs", "exp1_calibration", "results.jsonl")
OUT_STEM = "experiment1_quality_scaling"
# Thesis figures directory (repo root, sibling of Benchmarking/) -- the only
# place figures are written (not also alongside this script).
THESIS_FIGS_DIR = os.path.join("..", "Figs")

METHODS = [
    ("l2o-dm", "L2O-DM"),
    ("l2o-rnnprop", "L2O-RNNProp"),
    ("l2o-scale", "L2O-Scale"),
]

# dataviz skill's validated 8-slot categorical order, first 4 slots (blue,
# orange, aqua, yellow) -- passes the adjacent-pair CVD/normal-vision gates
# for a 4-line chart (validated via scripts/validate_palette.js
# "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light); aqua/yellow sit under the
# 3:1 contrast floor on this light surface, so every line also carries a
# distinct marker + a direct legend (the relief rule) rather than relying on
# hue alone.
DIMS = [
    ("25x50", "25x50", "#2a78d6", "-", "o"),
    ("50x100", "50x100", "#eb6834", "--", "s"),
    ("250x500", "250x500", "#1baf7a", "-.", "^"),
    ("500x1000", "500x1000", "#eda100", ":", "D"),
]

METRIC = "modified_relative_loss"
Y_LABEL = r"Modified Relative Loss $R_f$ (Eq. 10)" + "\n" + r"median, IQR band (log scale)"

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

    # (method, dim, unroll_length) -> [values across seeds]
    index = {}
    for r in rows:
        if r.get(METRIC) is None:
            continue
        key = (r["method"], r["dim"], r["unroll_length"])
        index.setdefault(key, []).append(r[METRIC])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    for method_key, method_label in METHODS:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
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

        for dim_key, dim_label, color, ls, marker in DIMS:
            ts = sorted({t for (m, d, t) in index if m == method_key and d == dim_key})
            if not ts:
                continue
            vals_per_t = [index[(method_key, dim_key, t)] for t in ts]
            medians = np.array([np.median(v) for v in vals_per_t])
            q1 = np.array([np.percentile(v, 25) for v in vals_per_t])
            q3 = np.array([np.percentile(v, 75) for v in vals_per_t])

            ax.plot(ts, medians, color=color, linestyle=ls, marker=marker,
                    markersize=5, linewidth=2, zorder=3, label=dim_label)
            ax.fill_between(ts, q1, q3, color=color, alpha=0.15, zorder=2, linewidth=0)

        ax.set_title(method_label, fontsize=12, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Unroll length T", color=INK_PRIMARY)
        ax.set_ylabel(Y_LABEL, color=INK_PRIMARY, fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()

        out_name = "{}_{}".format(OUT_STEM, method_key)
        out_stem = os.path.join(THESIS_FIGS_DIR, out_name)
        fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", out_stem + ".png")
        plt.close(fig)


if __name__ == "__main__":
    main()
