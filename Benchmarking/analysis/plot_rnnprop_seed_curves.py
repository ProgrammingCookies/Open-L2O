"""
RNNProp R_f-vs-step trajectories drawn PER SEED (no averaging), one figure per
training width, one panel per test sparsity -- to show how much the curves
differ between independently trained optimizers.

Why: the compute-matched figures show the median over seeds, and the 2026-09-19
check found that networks with uncorrelated weights (different seeds) still
trace nearly the same curve for the first ~100-200 learned steps and only
spread on the plateau. This figure lets the reader see that spread directly:
one thin line per seed, colour-coded, no band, no median.

Inputs (same as plot_compute_matched_rf.py): the per-step
`L2L_eval_loss_record.pickle-lasso_dataset` of every (width, seed, p) cell
and the run's x* cache for the exact mean f*.

Outputs, in Figs/<run>/:
  rnnprop_seed_curves_<run>_<width>.png   one figure per width (11 panels)
  Figs/multifigure/rnnprop_seed_curves_<run>_figure.tex   3-subfigure snippet
  stdout: max/min spread across seeds at k = 16, 100, 300, 999 per width.

Usage: python analysis/plot_rnnprop_seed_curves.py --run exp2_try3_210x300
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_compute_matched_rf import (RUNS, SURFACE, GRID, AXIS, INK_PRIMARY, INK_MUTED,  # noqa: E402
                                     MULTIFIG_DIR, THESIS_FIGS_DIR, _eval_dirname,
                                     load_results, mean_fstar)

WIDTH_LABELS = {"narrow": "Narrow ($p = 0.175$)", "medium": r"Medium ($p \in [0.1125, 0.2375]$)",
                "wide": r"Wide ($p \in [0.05, 0.30]$)"}
# Categorical palette for seeds (up to 10), fixed order so seed 21 is always the same colour.
SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a3ffc", "#d6336c",
                "#b8860b", "#0b8a8a", "#6b6b6b", "#c04ad6", "#3f7f3f"]
CHECK_K = (16, 100, 300, 999)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
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
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--method", default="l2o-rnnprop")
    args = ap.parse_args()
    run_dir = os.path.join("runs", args.run)
    rows = [r for r in load_results(run_dir) if r["method"] == args.method]
    if not rows:
        raise SystemExit("no {} rows in {}".format(args.method, run_dir))
    seeds = sorted({int(r["seed"]) for r in rows})
    with open(os.path.join(run_dir, "data", "seed{}".format(seeds[0]), "metadata.json")) as f:
        meta = json.load(f)
    lam, m = float(meta["lam"]), int(meta["m"])
    ps = [float(p) for p in meta["test_sparsities"]]
    widths = [w for w in ("narrow", "medium", "wide") if any(r["width"] == w for r in rows)]

    # {(width, p, seed): R_f[k]}
    curves = {}
    fstar = {}
    for r in rows:
        seed, width, p = int(r["seed"]), r["width"], float(r["test_sparsity"])
        pkl = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, args.method,
                           _eval_dirname(p), "eval", "L2L_eval_loss_record.pickle-lasso_dataset")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            traj = np.asarray(pickle.load(f), dtype=np.float64)
        if (seed, p) not in fstar:
            fstar[(seed, p)] = mean_fstar(run_dir, seed, p, lam, m)
        curves[(width, p, seed)] = traj / fstar[(seed, p)] - 1.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY})
    ncols = 4
    nrows = int(np.ceil(len(ps) / ncols))
    y_lo = min(c[1:].min() for c in curves.values()) / 3
    y_hi = max(c[1:].max() for c in curves.values()) * 1.5
    out_dir = os.path.join(THESIS_FIGS_DIR, args.run)
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    spread_lines = []
    for width in widths:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), sharex=True, sharey=True)
        axes = axes.ravel()
        fig.patch.set_facecolor(SURFACE)
        handles = {}
        for ax, p in zip(axes, ps):
            style(ax)
            for i, seed in enumerate(seeds):
                c = curves.get((width, p, seed))
                if c is None:
                    continue
                kk = np.arange(1, len(c))
                (h,) = ax.plot(kk, c[1:], color=SEED_COLOURS[i % len(SEED_COLOURS)], lw=1.2, alpha=0.9, zorder=3)
                handles["seed {}".format(seed)] = h
            ax.set_title("test $p = {:g}$".format(p), fontsize=9, color=INK_PRIMARY, pad=4)
            ax.set_xlim(1, 1000)
            ax.set_ylim(y_lo, y_hi)
        for ax in axes[len(ps):]:
            ax.axis("off")
        if len(ps) < len(axes):
            axes[len(ps)].legend(list(handles.values()), list(handles), loc="center", frameon=False, fontsize=8)
        else:
            fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles),
                       frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
        for ax in axes[::ncols]:
            ax.set_ylabel("$R_f$ after $k$ learned steps", fontsize=8)
        fig.supxlabel("learned steps $k$", fontsize=8, y=0.035)
        fig.suptitle("RNNProp, {} training width -- one curve per seed ({} seeds), no averaging -- {}"
                     .format(width, len(seeds), RUNS[args.run]["subtitle"]),
                     fontsize=9, color=INK_PRIMARY, y=1.005)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        name = "rnnprop_seed_curves_{}_{}".format(args.run, width)
        fig.savefig(os.path.join(out_dir, name + ".png"), facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", os.path.join(out_dir, name + ".png"))
        plt.close(fig)
        entries.append((name, WIDTH_LABELS[width], "fig:rnnprop-seed-curves-{}-{}".format(args.run.replace("_", "-"), width)))

        # spread across seeds at fixed k, per p
        for k in CHECK_K:
            vals = {p: [curves[(width, p, s)][k] for s in seeds if (width, p, s) in curves] for p in ps}
            ratios = [max(v) / min(v) for v in vals.values() if len(v) > 1]
            spread_lines.append("| {} | {} | {:.2f} | {:.2f} |".format(width, k, min(ratios), max(ratios)))

    lines = ["% One figure per training width, one thin curve per seed (no averaging),",
             "% generated by plot_rnnprop_seed_curves.py --run " + args.run + ".",
             "% Requires \\usepackage{graphicx} and \\usepackage{subcaption}. Paths are",
             "% relative to the repo root (Figs/...); adjust \\graphicspath or the paths below.",
             "% Include with: \\input{Figs/multifigure/rnnprop_seed_curves_" + args.run + "_figure}",
             r"\begin{figure}[htbp]", r"\centering"]
    for name, cap, lab in entries:
        lines += [r"\begin{subfigure}[t]{\linewidth}", r"  \centering",
                  r"  \includegraphics[width=\linewidth]{Figs/" + args.run + "/" + name + ".png}",
                  r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}\\[6pt]"]
    lines += [r"\caption{L2O-RNNProp, modified relative loss $R_f$ (Eq.~10) against learned steps, " + RUNS[args.run]["subtitle"]
              + r". One curve per training seed (" + str(len(seeds)) + r" independently trained optimizers per width), "
              r"no averaging; one panel per test sparsity $p$. The curves of different seeds agree closely for the "
              r"first $\sim$100--200 steps and separate only on the plateau: the early trajectory is set by the "
              r"architecture's normalised inputs and bounded ($\tanh \times 0.01$) output rather than by the learned "
              r"weights.}",
              r"\label{fig:rnnprop-seed-curves-" + args.run.replace("_", "-") + "}", r"\end{figure}"]
    os.makedirs(MULTIFIG_DIR, exist_ok=True)
    out_tex = os.path.join(MULTIFIG_DIR, "rnnprop_seed_curves_{}_figure.tex".format(args.run))
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_tex)

    print("\nspread across seeds, max/min of R_f at step k (min and max over test p):")
    print("| width | k | min over p | max over p |")
    print("|---|---|---|---|")
    print("\n".join(spread_lines))


if __name__ == "__main__":
    main()
