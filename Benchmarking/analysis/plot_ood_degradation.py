"""Plots the OOD-degradation trend for one Experiment-2 run.

    python analysis/plot_ood_degradation.py --run exp2_try3_210x300

How each method's recovery quality (Modified Relative Loss, Eq. 10) changes as
test sparsity moves away from what it was trained on. One standalone figure per
method, one line per training-sparsity width, saved to
`Figs/<run>/<stem>_ood_degradation_<method>.png`.

Which methods are legitimate columns for a run, its width triad, its sparsity
rounding and the centre line drawn for `narrow` all come from `run_specs.py` --
this was two near-identical scripts (`exp2_real` and an `exp2_try2` fork) whose
only real differences were those values.

Uses median + IQR (25th/75th percentile band) rather than mean +/- std. Checked
directly against the 10-seed distributions before choosing this: L2O-DM's
per-seed `modified_relative_loss` is heavily right-skewed by occasional
divergent runs (mean up to 8x the median at some conditions, std exceeding the
mean at others -- e.g. narrow/p=0.10: mean=1.0e7, std=2.3e7, median=1.3e6),
L2O-RNNProp is moderately skewed (mean ~1.2-1.6x median), while LISTA/ALISTA are
essentially symmetric (mean/median ~1.00) so the choice of statistic barely
changes their panels. Median/IQR is applied uniformly across every panel for a
consistent statistic within one figure; the result tables still report mean (std)
per Table 4's spec -- this plot's summary statistic is independent of that.

Not part of the Benchmarking platform -- a one-off analysis script reading
`runs/<run>/results.jsonl` (per-seed rows, not the mean/std-only summary.json).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import sys

# Benchmarking/ is this file's parent, not its own directory, so it is not on
# sys.path automatically when this runs as `python analysis/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_specs
from core.seedsets import filter_to_complete_seeds

METRIC = "modified_relative_loss"
Y_LABEL = (r"Modified Relative Loss $R_f$ (Eq. 10)" + "\n"
           + r"median, IQR band (log scale)")

# dataviz chrome: light chart surface + recessive gridlines/ink.
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def build_index(spec):
    """{(method, width): {test_sparsity: {median, q1, q3}}} for this run's methods."""
    with open(spec.results) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keep = set(spec.learned_methods)
    rows = [r for r in rows if r["method"] in keep]
    # Only seeds present in EVERY (method, width) cell -- otherwise one panel's
    # median is over N seeds and the next panel's over N-1 (see core/seedsets.py).
    rows, _, _ = filter_to_complete_seeds(rows, methods=sorted(keep), label=spec.key)

    raw = {}
    for row in rows:
        key = (row["method"], row["width"])
        p = round(float(row["test_sparsity"]), spec.sparsity_round)
        raw.setdefault(key, {}).setdefault(p, []).append(row[METRIC])

    index = {}
    for key, by_p in raw.items():
        index[key] = {p: {"median": float(np.median(v)),
                          "q1": float(np.percentile(v, 25)),
                          "q3": float(np.percentile(v, 75))}
                      for p, v in ((p, np.array(vals)) for p, vals in by_p.items())}
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=run_specs.choices())
    args = ap.parse_args()
    spec = run_specs.get(args.run)

    index = build_index(spec)
    out_dir = os.path.join(run_specs.FIGS_DIR, spec.key)
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    for method_key, method_label in spec.methods:
        if method_key in run_specs.BASELINE_KEYS:
            continue        # classical references are not a training-width story
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

        # Nested background shading for each width's own training range, widest
        # first so narrower ranges layer visibly on top.
        for _, _, color, _, _, train_range, _ in spec.plot_widths:
            if train_range is not None:
                ax.axvspan(train_range[0], train_range[1], color=color, alpha=0.07,
                           zorder=1, linewidth=0)

        for width_key, width_label, color, ls, marker, _, centre in spec.plot_widths:
            by_p = index.get((method_key, width_key), {})
            ps = sorted(by_p)
            if not ps:
                continue
            ax.plot(ps, [by_p[p]["median"] for p in ps], color=color, linestyle=ls,
                    marker=marker, markersize=5, linewidth=2, zorder=3, label=width_label)
            ax.fill_between(ps, [by_p[p]["q1"] for p in ps], [by_p[p]["q3"] for p in ps],
                            color=color, alpha=0.15, zorder=2, linewidth=0)
            if centre is not None:
                # `narrow` is a single fixed sparsity, so mark it as a line
                # rather than a band. The value comes from the spec, not a
                # literal: it is 0.10 in the uncentred design and 0.175 in the
                # centred one.
                ax.axvline(centre, color=color, linestyle=ls, linewidth=1,
                           alpha=0.5, zorder=1)

        ax.set_title(method_label, fontsize=12, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Test sparsity $p$", color=INK_PRIMARY)
        ax.set_ylabel(Y_LABEL, color=INK_PRIMARY, fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()

        out_stem = os.path.join(out_dir, "{}_ood_degradation_{}".format(
            spec.tex_stem, method_key))
        fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", out_stem + ".png")
        plt.close(fig)


if __name__ == "__main__":
    main()
