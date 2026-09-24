"""
Experiment 2, claim 3: the width effect on the LASSO objective as a paired
RATIO, so the reader can see it instead of taking it from a table.

Why: `experiment2_*_modelbased_rel_loss` plots absolute R_f per width on a
log axis spanning two decades, on which a 40% width effect is ~0.15 decades --
present but easy to miss. This script plots, per test sparsity p, the paired
ratio R_f(wide) / R_f(narrow) and R_f(medium) / R_f(narrow) computed per
seed (same seed = same A, same instances, same x0 for every width), as the
geometric mean over seeds with a 95% t-interval on log R_f, on a log axis
centred on 1. Below 1 = the wider training is better on the objective.

What it shows directly:
  * inside the overlap of the three training supports the ratio sits at ~1
    (the width effect on the objective is a few percent there);
  * LISTA: a symmetric U -- the ratio drops below 1 at both ends of the
    test range, further the farther from the training centre;
  * ALISTA: asymmetric -- above 1 (wide worse) around p = 0.10-0.175, below 1
    (wide better) from p ~ 0.24 upward, and exactly 1 at p <= 0.05 where
    every width is in the exact-recovery regime (R_f ~ 0.003).

Inputs: `runs/<run>/results.jsonl` (LISTA/ALISTA rows; never summary.json).
Outputs, per run with >= MIN_SEEDS complete seeds:
  Figs/<run>/width_ratio_rf_<run>.png                combined LISTA | ALISTA
  Figs/<run>/width_ratio_rf_<run>_{lista,alista}.png standalone panels
  Figs/multifigure/width_ratio_rf_<run>_figure.tex        subfigure snippet
  stdout: the ratios with CI and the per-seed sign counts, as markdown.

Usage: python analysis/plot_experiment2_width_ratio_rf.py [--runs exp2_try2 exp2_real ...]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

# Benchmarking/ is this file's parent, not its own directory, so it is not on
# sys.path automatically when this runs as `python analysis/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_specs

from core.seedsets import filter_to_complete_seeds
from scipy import stats

THESIS_FIGS_DIR = os.path.join("..", "Figs")
MULTIFIG_DIR = os.path.join(THESIS_FIGS_DIR, "multifigure")  # all subfigure snippets live here
MIN_SEEDS = 5
METHODS = [("lista", "LISTA"), ("alista", "ALISTA")]
WIDTHS = ["narrow", "medium", "wide"]
NARROW_C, MEDIUM_C, WIDE_C = "#2a78d6", "#eb6834", "#1baf7a"
COMPARISONS = [("wide", "narrow", r"$R_f(\mathrm{wide}) / R_f(\mathrm{narrow})$", WIDE_C, "-", "^"),
               ("medium", "narrow", r"$R_f(\mathrm{medium}) / R_f(\mathrm{narrow})$", MEDIUM_C, "--", "s")]
# Run identity -- results paths, dimensions, method lists, width triads --
# comes from the shared registry, so a run is defined in exactly one place.
# The ORDER of the keys below and the subtitle wording are presentation,
# and stay local: the order fixes panel/figure order in this script's own
# output, and each script phrases the run differently.
_ORDER = ["exp2_try2", "exp2_try3_210x300",
          # Same protocol/data as exp2_try3_210x300, LISTA/ALISTA retrained on
          # the scored objective instead of the released 2*lambda one (the
          # lasso_train_objective ablation, experiments_analysis.md 2026-09-20/21).
          # The paper arm's 2*lambda bias is a width-INDEPENDENT R_f offset, so
          # it compresses this very ratio (wide/narrow 1.3x there vs 4-7x here
          # at the sparse end) -- this is the arm to cite for the ID cost of
          # width.
          "exp2_try3_210x300_scoringonly", "exp2_real"]
RUNS = {k: run_specs.RUN_SPECS[k].subtitle for k in _ORDER}
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK_PRIMARY, INK_MUTED = "#0b0b0b", "#898781"


def load(run_dir: str):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "results.jsonl")) if l.strip()]
    # Only seeds present in EVERY (method, width) cell: a sweep stopped
    # mid-seed otherwise aggregates one method over N seeds and another
    # over N-1 (see core/seedsets.py).
    rows, _, _ = filter_to_complete_seeds(rows, methods=("lista", "alista"),
                                         label=run_dir)
    rf: Dict[Tuple[str, str, int, float], float] = {}
    for r in rows:
        if r["method"] in ("lista", "alista"):
            rf[(r["method"], r["width"], int(r["seed"]), round(float(r["test_sparsity"]), 4))] = float(r["modified_relative_loss"])
    return rf


def supports(run_dir: str, seed: int):
    with open(os.path.join(run_dir, "data", "seed{}".format(seed), "metadata.json")) as f:
        spec = json.load(f)["train_sparsities"]
    out = {}
    for w, v in spec.items():
        out[w] = (float(v), float(v)) if not isinstance(v, list) else (float(v[0]), float(v[1]))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY})

    for run_key in args.runs:
        run_dir = os.path.join("runs", run_key)
        if not os.path.exists(os.path.join(run_dir, "results.jsonl")):
            continue
        rf = load(run_dir)
        # per method: seeds complete for all three widths, and the test grid
        stats_out = {}   # (method, cmp) -> {p: (gmean, lo, hi, n_below)}
        ylo, yhi = np.inf, -np.inf
        seeds_by_method = {}
        for mkey, _ in METHODS:
            seeds = sorted({s for (m, w, s, p) in rf if m == mkey})
            ps = sorted({p for (m, w, s, p) in rf if m == mkey})
            seeds = [s for s in seeds if all((mkey, w, s, p) in rf for w in WIDTHS for p in ps)]
            if len(seeds) < MIN_SEEDS:
                print("skip {} {}: {} complete seeds < {}".format(run_key, mkey, len(seeds), MIN_SEEDS))
                continue
            seeds_by_method[mkey] = (seeds, ps)
            for wa, wb, *_ in COMPARISONS:
                d = {}
                for p in ps:
                    logr = np.array([np.log(rf[(mkey, wa, s, p)]) - np.log(rf[(mkey, wb, s, p)]) for s in seeds])
                    half = stats.t.ppf(0.975, len(seeds) - 1) * stats.sem(logr)
                    d[p] = (float(np.exp(logr.mean())), float(np.exp(logr.mean() - half)),
                            float(np.exp(logr.mean() + half)), int(np.sum(logr < 0)))
                    ylo, yhi = min(ylo, d[p][1]), max(yhi, d[p][2])
                stats_out[(mkey, wa)] = d
        if not seeds_by_method:
            continue
        pad = (yhi / ylo) ** 0.08
        ylim = (min(ylo / pad, 1 / 1.15), max(yhi * pad, 1.15))
        supp = supports(run_dir, next(iter(seeds_by_method.values()))[0][0])

        def draw(ax, mkey, legend):
            style(ax)
            for w, c in (("wide", WIDE_C), ("medium", MEDIUM_C)):
                lo_, hi_ = supp[w]
                ax.axvspan(lo_, hi_, color=c, alpha=0.07, zorder=1, linewidth=0)
            ax.axvline(supp["narrow"][0], color=NARROW_C, lw=1, alpha=0.5, zorder=1)
            ax.axhline(1.0, color=INK_PRIMARY, lw=0.8, zorder=2)
            for wa, wb, label, color, ls, marker in COMPARISONS:
                d = stats_out.get((mkey, wa))
                if d is None:
                    continue
                ps = sorted(d)
                g = np.array([d[p][0] for p in ps]); lo_ = np.array([d[p][1] for p in ps]); hi_ = np.array([d[p][2] for p in ps])
                ax.fill_between(ps, lo_, hi_, color=color, alpha=0.15, lw=0, zorder=2)
                ax.plot(ps, g, color=color, lw=2, ls=ls, marker=marker, ms=5, label=label, zorder=3)
            ax.set_ylim(*ylim)
            # A log axis over a narrow range yields one or two mathtext ticks;
            # use plain ratio ticks instead so 0.7 reads as 0.7, not 7e-1.
            cand = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 2, 3, 4, 5]
            ticks = [t for t in cand if ylim[0] <= t <= ylim[1]]
            if ylim[1] / ylim[0] > 8:   # wide range: thin the ticks
                ticks = [t for t in ticks if t in (0.05, 0.1, 0.2, 0.5, 1.0, 2, 5)]
            ax.set_yticks(ticks)
            ax.set_yticklabels(["{:g}".format(t) for t in ticks])
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xlabel("Test sparsity $p$", fontsize=9)
            ax.text(0.99, 0.02, "below 1: wider training is better", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=INK_MUTED)
            if legend:
                ax.legend(frameon=False, fontsize=8, loc="upper right")

        ylabel = "paired ratio of $R_f$ (Eq. 10), geometric mean\nover seeds, 95% $t$-interval (log scale)"
        n_seeds = {m: len(v[0]) for m, v in seeds_by_method.items()}
        out_dir = os.path.join(THESIS_FIGS_DIR, run_key)
        os.makedirs(out_dir, exist_ok=True)

        # combined
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
        fig.patch.set_facecolor(SURFACE)
        for i, (ax, (mkey, mlabel)) in enumerate(zip(axes, METHODS)):
            draw(ax, mkey, legend=(i == 0))
            ax.set_title("{}  ({} seeds)".format(mlabel, n_seeds.get(mkey, 0)), fontsize=12, loc="left")
        axes[0].set_ylabel(ylabel, fontsize=9)
        fig.suptitle("Width effect on the objective, paired by seed -- " + RUNS[run_key],
                     fontsize=10, color=INK_MUTED, x=0.005, ha="left")
        fig.tight_layout()
        stem = os.path.join(out_dir, "width_ratio_rf_{}".format(run_key))
        fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", stem + ".png")
        plt.close(fig)

        # standalone panels + snippet
        entries = []
        for mkey, mlabel in METHODS:
            if mkey not in seeds_by_method:
                continue
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            fig.patch.set_facecolor(SURFACE)
            draw(ax, mkey, legend=True)
            ax.set_title("{}  ({} seeds)".format(mlabel, n_seeds[mkey]), fontsize=12, loc="left")
            ax.set_ylabel(ylabel, fontsize=9)
            fig.tight_layout()
            name = "width_ratio_rf_{}_{}".format(run_key, mkey)
            stem = os.path.join(out_dir, name)
            fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
            print("wrote", stem + ".png")
            plt.close(fig)
            entries.append((name, mlabel, "fig:width-ratio-rf-{}-{}".format(run_key.replace("_", "-"), mkey)))
        lines = ["% Multi-panel figure generated by plot_experiment2_width_ratio_rf.py --runs " + run_key + ".",
                 "% Requires \\usepackage{graphicx} and \\usepackage{subcaption}. Paths are relative",
                 "% to the repo root (Figs/...); adjust \\graphicspath or the paths below.",
                 "% Include with: \\input{Figs/multifigure/width_ratio_rf_" + run_key + "_figure}",
                 r"\begin{figure}[htbp]", r"\centering"]
        for i, (name, cap, lab) in enumerate(entries):
            lines += [r"\begin{subfigure}[t]{0.48\linewidth}", r"  \centering",
                      r"  \includegraphics[width=\linewidth]{Figs/" + run_key + "/" + name + ".png}",
                      r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}"
                      + (r"\hfill" if i % 2 == 0 and i + 1 < len(entries) else "")]
        lines += [r"\caption{Effect of the training-sparsity width on the modified relative loss $R_f$ (Eq.~10), "
                  + RUNS[run_key] + r". For each test sparsity $p$ the ratio $R_f(\text{wide}) / R_f(\text{narrow})$ "
                  r"and $R_f(\text{medium}) / R_f(\text{narrow})$ is computed per seed (same dictionary, test "
                  r"instances and $x_0$ for every width) and summarised as the geometric mean over seeds with a "
                  r"95\% $t$-interval on $\log R_f$; below 1 means the wider training distribution is better on "
                  r"the objective. Shaded bands are the medium and wide training supports, the vertical line the "
                  r"fixed narrow training point. Both panels share the y-axis.}",
                  r"\label{fig:width-ratio-rf-" + run_key.replace("_", "-") + "}", r"\end{figure}"]
        os.makedirs(MULTIFIG_DIR, exist_ok=True)
        out_tex = os.path.join(MULTIFIG_DIR, "width_ratio_rf_{}_figure.tex".format(run_key))
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", out_tex)

        # numbers
        for mkey, mlabel in METHODS:
            if mkey not in seeds_by_method:
                continue
            seeds, ps = seeds_by_method[mkey]
            print("\n{} {} (n = {}): geometric-mean ratio [95% CI], seeds with ratio < 1".format(run_key, mlabel, len(seeds)))
            print("| p | wide/narrow | seeds<1 | medium/narrow | seeds<1 |")
            print("|---|---|---|---|---|")
            for p in ps:
                w = stats_out[(mkey, "wide")][p]; m = stats_out[(mkey, "medium")][p]
                print("| {:g} | {:.2f} [{:.2f}, {:.2f}] | {}/{} | {:.2f} [{:.2f}, {:.2f}] | {}/{} |".format(
                    p, w[0], w[1], w[2], w[3], len(seeds), m[0], m[1], m[2], m[3], len(seeds)))


if __name__ == "__main__":
    main()
