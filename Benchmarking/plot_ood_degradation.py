"""Plots the OOD-degradation trend: how each method's recovery quality
(Modified Relative Loss, Eq. 10) changes as test sparsity moves away from
what it was trained on. One standalone figure per method (four total:
experiment2_ood_degradation_<method>.png), one line per
training-sparsity width (narrow/medium/wide) within each.

Not part of the Benchmarking platform -- a one-off analysis script reading
runs/exp2_real/results.jsonl (per-seed rows, not the mean/std-only
summary.json), same pattern as generate_result_tables.py.

Uses median + IQR (25th/75th percentile band) rather than mean +/- std.
Checked directly against the 10-seed distributions before choosing this:
L2O-DM's per-seed modified_relative_loss is heavily right-skewed by
occasional divergent runs (mean up to 8x the median at some conditions,
std exceeding the mean at others -- e.g. narrow/p=0.10: mean=1.0e7,
std=2.3e7, median=1.3e6), L2O-RNNProp is moderately skewed (mean
~1.2-1.6x median), while LISTA/ALISTA are essentially symmetric
(mean/median ~1.00) so the choice of statistic barely changes their
panels. Median/IQR is applied uniformly across all four panels for a
consistent statistic within one figure; the results tables
(generate_result_tables.py) still report mean (std) per Table 4's spec
-- this plot's summary statistic is independent of that.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = os.path.join("runs", "exp2_real", "results.jsonl")
OUT_STEM = "experiment2_ood_degradation"
# Thesis figures directory (repo root, sibling of Benchmarking/) -- every
# figure is saved here in addition to alongside this script.
THESIS_FIGS_DIR = os.path.join("..", "Figs")

METHODS = [
    ("l2o-dm", "L2O-DM"),
    ("l2o-rnnprop", "L2O-RNNProp"),
    ("lista", "LISTA"),
    ("alista", "ALISTA"),
]

# (key, display label, color, linestyle, marker, training range for shading)
# Colors: dataviz skill's validated 3-slot categorical order (blue/orange/aqua),
# passes CVD + normal-vision separation on the light surface (validated via
# scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light).
WIDTHS = [
    ("narrow", r"Narrow ($p = 0.10$)", "#2a78d6", "-", "o", None),
    ("medium", r"Medium ($p \in [0.075, 0.20]$)", "#eb6834", "--", "s", (0.075, 0.20)),
    ("wide", r"Wide ($p \in [0.05, 0.30]$)", "#1baf7a", ":", "^", (0.05, 0.30)),
]

METRIC = "modified_relative_loss"
Y_LABEL = r"Modified Relative Loss $R_f$ (Eq. 10)" + "\n" + r"median, IQR band (log scale)"

# dataviz skill chrome: light chart surface + recessive gridlines/ink.
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def main():
    with open(RESULTS_PATH) as f:
        rows = [json.loads(line) for line in f]

    # (method, width) -> {test_sparsity: [per-seed metric values]}
    raw = {}
    for row in rows:
        key = (row["method"], row["width"])
        raw.setdefault(key, {}).setdefault(round(row["test_sparsity"], 2), []).append(
            row[METRIC])

    # Collapse to median/Q1/Q3 per (method, width, test_sparsity).
    index = {}
    for key, by_p in raw.items():
        index[key] = {}
        for p, values in by_p.items():
            values = np.array(values)
            index[key][p] = {
                "median": np.median(values),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
            }

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
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)

        # Nested background shading for each width's own training range,
        # widest first so narrower ranges layer visibly on top.
        for width_key, _, color, _, _, train_range in WIDTHS:
            if train_range is not None:
                ax.axvspan(train_range[0], train_range[1], color=color, alpha=0.07,
                          zorder=1, linewidth=0)

        for width_key, width_label, color, ls, marker, train_range in WIDTHS:
            by_p = index.get((method_key, width_key), {})
            ps = sorted(by_p.keys())
            medians = np.array([by_p[p]["median"] for p in ps])
            lower = np.array([by_p[p]["q1"] for p in ps])
            upper = np.array([by_p[p]["q3"] for p in ps])

            ax.plot(ps, medians, color=color, linestyle=ls, marker=marker,
                    markersize=5, linewidth=2, zorder=3, label=width_label)
            ax.fill_between(ps, lower, upper, color=color, alpha=0.15, zorder=2,
                            linewidth=0)
            if width_key == "narrow":
                ax.axvline(0.10, color=color, linestyle=ls, linewidth=1, alpha=0.5,
                          zorder=1)

        ax.set_title(method_label, fontsize=12, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Test sparsity $p$", color=INK_PRIMARY)
        ax.set_ylabel(Y_LABEL, color=INK_PRIMARY, fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()

        out_name = "{}_{}".format(OUT_STEM, method_key)
        for out_dir in (".", THESIS_FIGS_DIR):
            out_stem = os.path.join(out_dir, out_name)
            fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
            print("wrote", out_stem + ".png")
        plt.close(fig)


if __name__ == "__main__":
    main()
