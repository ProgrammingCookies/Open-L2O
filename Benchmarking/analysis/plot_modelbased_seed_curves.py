"""
LISTA / ALISTA R_f-vs-layer trajectories drawn PER SEED (no averaging), one
figure per (method, training width), one panel per test sparsity -- the
model-based counterpart of `plot_rnnprop_seed_curves.py`.

Why: every other Experiment-2 model-based figure aggregates over seeds (mean +
CI, or the median in the compute-matched figure), so the reader never sees how
much two independently trained unrolled networks differ. RNNProp's per-seed
figure showed that its seeds are nearly indistinguishable for the first ~100
steps because of the architecture's bounded normalised step, and separate only
on the plateau. LISTA / ALISTA have no such bound -- every layer's weights are
free -- so this figure is what tells the reader whether their seed spread is of
the same kind, and it is drawn the same way so the two can be compared.

Two metrics, `--metric rf` (default) and `--metric nmse`:
  rf    modified relative loss (Eq. 10), the objective gap -- log y, and the
        seed spread is read as a RATIO because R_f is scale-free.
  nmse  NMSE against x_true (Eq. 11) in dB -- linear y (dB is already log), and
        the spread is read as a DIFFERENCE in dB. This is recovery quality, not
        objective quality: the two answer different questions, and the LASSO
        minimiser is not the true signal, so the curves need not agree.

Inputs: each cell's `all_test_metrics_perfile.json` (per-file `lasso_layer{i}`,
i = 0..15, written by `regenerate_modelbased_trajectories.py`; the library's own
`all_test_metrics.json` values are running means across test files for any cell
evaluated before the 2026-09-17 fix), the row's
`modified_relative_loss` for k = 16, and the run's x* cache for the exact mean f*.
For `--metric nmse`, each cell's `all_test_nmse_perfile.json` instead, written by
`compute_modelbased_nmse_trajectories.py` (no sweep stores per-layer NMSE: they
only ever attached objective metrics to the LASSO task).

Outputs, in Figs/<run>/ (`_nmse` suffix on the `--metric nmse` variants):
  {lista,alista}_seed_curves[_nmse]_<run>_<width>.png   one figure per method x width
  Figs/multifigure/{lista,alista}_seed_curves[_nmse]_<run>_figure.tex  3-subfigure snippet
  stdout: spread across seeds at k = 1, 4, 8, 16 layers per (method, width).

Usage:
  python analysis/plot_modelbased_seed_curves.py --run exp2_try3_210x300_scoringonly
  python analysis/plot_modelbased_seed_curves.py --run exp2_try3_210x300_scoringonly --metric nmse
  python analysis/plot_modelbased_seed_curves.py --run exp2_try3_210x300 --method lista
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.lasso import _test_filename  # noqa: E402
from plot_compute_matched_rf import (RUNS, SURFACE, GRID, AXIS, INK_PRIMARY, INK_MUTED,  # noqa: E402
                                     MODEL_BASED_DEPTH, MULTIFIG_DIR, THESIS_FIGS_DIR,
                                     load_results, mean_fstar)

METHOD_LABELS = {"lista": "LISTA", "alista": "ALISTA"}
WIDTH_LABELS = {"narrow": "Narrow ($p = 0.175$)", "medium": r"Medium ($p \in [0.1125, 0.2375]$)",
                "wide": r"Wide ($p \in [0.05, 0.30]$)"}
# Same categorical palette and fixed order as plot_rnnprop_seed_curves.py, so a
# given seed keeps its colour across the model-free and model-based figures.
SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a3ffc", "#d6336c",
                "#b8860b", "#0b8a8a", "#6b6b6b", "#c04ad6", "#3f7f3f"]
CHECK_K = (1, 4, 8, 16)

# R_f is scale-free, so seed spread is a ratio; NMSE is already in dB, so its
# spread is a difference. Everything downstream that differs between the two
# metrics lives here.
METRICS = {
    "rf": {"ylabel": "$R_f$ after $k$ layers",
           "tex_quantity": r"modified relative loss $R_f$ (Eq.~10)",
           "spread": lambda v: max(v) / min(v), "fmt": "{:.2f}",
           "spread_name": "max/min of R_f", "suffix": ""},
    "nmse": {"ylabel": "NMSE vs $x_{true}$ after $k$ layers [dB]",
             "tex_quantity": r"NMSE against $x_{\mathrm{true}}$ (Eq.~11)",
             "spread": lambda v: max(v) - min(v), "fmt": "{:.2f}",
             "spread_name": "max-min of NMSE [dB]", "suffix": "_nmse"},
}


def style(ax, metric="rf"):
    ax.set_facecolor(SURFACE)
    # Linear in k: there are only 16 layers, and a log axis would squeeze the
    # descent (k = 1..6, where the seeds actually differ) into the left third.
    # y: log for R_f, linear for NMSE (dB is already a log scale).
    if metric == "rf":
        ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def per_seed_layer_curves(run_dir: str, rows: List[dict], lam: float, m: int, metric: str = "rf"
                          ) -> Dict[Tuple[str, str, float, int], np.ndarray]:
    """{(method, width, p, seed): y[k], k = 0..16} -- no aggregation.

    metric "rf": same source and indexing as
    `plot_compute_matched_rf.model_based_layer_curves` (which medians this over
    seeds): k = 0 is f(x0) at the shared seeded start, k = i the iterate after i
    layers, k = 16 the row's own `modified_relative_loss` (scored on the same
    final iterate).

    metric "nmse": Eq. 11 in dB from `all_test_nmse_perfile.json`, which already
    stores k = 0..16 inclusive (its last entry is recomputed from x_pred and was
    checked against the row's `nmse_vs_true_mean` when written).
    """
    curves: Dict[Tuple[str, str, float, int], np.ndarray] = {}
    fstar_cache: Dict[Tuple[int, float], float] = {}
    src = "all_test_metrics_perfile.json" if metric == "rf" else "all_test_nmse_perfile.json"
    missing = 0
    for r in rows:
        seed, width, p = int(r["seed"]), r["width"], float(r["test_sparsity"])
        pattern = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, r["method"],
                               "checkpoints", "models", "*", "replicate_1", src)
        hits = glob.glob(pattern)
        if not hits:
            missing += 1
            continue
        with open(hits[0]) as f:
            vals = json.load(f).get(_test_filename(p))
        if vals is None:
            missing += 1
            continue
        if metric == "nmse":
            curves[(r["method"], width, p, seed)] = np.array(
                [vals["nmse_layer{}".format(i)] for i in range(MODEL_BASED_DEPTH + 1)], dtype=np.float64)
            continue
        key = (seed, p)
        if key not in fstar_cache:
            fstar_cache[key] = mean_fstar(run_dir, seed, p, lam, m)
        layers = [vals["lasso_layer{}".format(i)] for i in range(MODEL_BASED_DEPTH)]
        rf = np.array(layers, dtype=np.float64) / fstar_cache[key] - 1.0
        curves[(r["method"], width, p, seed)] = np.append(rf, float(r["modified_relative_loss"]))
    if missing and metric == "nmse":
        raise SystemExit(
            "{} model-based cells lack all_test_nmse_perfile.json -- no sweep stores per-layer "
            "NMSE. The per-layer files ship precomputed under runs/; the generator "
            "that produces them lives on the unified-slice branch.".format(missing))
    if missing:
        raise SystemExit(
            "{} model-based cells lack all_test_metrics_perfile.json. The stored "
            "all_test_metrics.json values are running means across test files for any "
            "cell evaluated before the 2026-09-17 fix, so they are not usable here -- "
            "The corrected per-file values ship under runs/; the regeneration script "
            "lives on the unified-slice branch.".format(missing))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--method", action="append", choices=sorted(METHOD_LABELS),
                    help="repeatable; default: both lista and alista")
    ap.add_argument("--metric", default="rf", choices=sorted(METRICS),
                    help="rf = modified relative loss (Eq. 10, default); nmse = Eq. 11 in dB")
    args = ap.parse_args()
    methods = args.method or ["lista", "alista"]
    spec = METRICS[args.metric]

    run_dir = os.path.join("runs", args.run)
    rows = [r for r in load_results(run_dir) if r["method"] in methods]
    if not rows:
        raise SystemExit("no {} rows in {}".format("/".join(methods), run_dir))
    seeds = sorted({int(r["seed"]) for r in rows})
    with open(os.path.join(run_dir, "data", "seed{}".format(seeds[0]), "metadata.json")) as f:
        meta = json.load(f)
    lam, m = float(meta["lam"]), int(meta["m"])
    ps = [float(p) for p in meta["test_sparsities"]]
    widths = [w for w in ("narrow", "medium", "wide") if any(r["width"] == w for r in rows)]

    curves = per_seed_layer_curves(run_dir, rows, lam, m, args.metric)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY})
    ncols = 4
    nrows = int(np.ceil(len(ps) / ncols))
    # One y-range for every panel of every figure: the point of the figure is to
    # compare spreads across seeds, widths AND the two methods, which per-panel
    # autoscaling would destroy (see the 2026-09-15 metric-figure note).
    lo, hi = min(c.min() for c in curves.values()), max(c.max() for c in curves.values())
    y_lo, y_hi = (lo / 3, hi * 1.5) if args.metric == "rf" else (lo - 2.0, hi + 2.0)
    out_dir = os.path.join(THESIS_FIGS_DIR, args.run)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MULTIFIG_DIR, exist_ok=True)
    spread_lines = []

    for method in methods:
        entries = []
        for width in widths:
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                                     sharex=True, sharey=True)
            axes = axes.ravel()
            fig.patch.set_facecolor(SURFACE)
            handles = {}
            for ax, p in zip(axes, ps):
                style(ax, args.metric)
                for i, seed in enumerate(seeds):
                    c = curves.get((method, width, p, seed))
                    if c is None:
                        continue
                    (h,) = ax.plot(np.arange(len(c)), c, color=SEED_COLOURS[i % len(SEED_COLOURS)],
                                   lw=1.2, alpha=0.9, zorder=3)
                    handles["seed {}".format(seed)] = h
                ax.set_title("test $p = {:g}$".format(p), fontsize=9, color=INK_PRIMARY, pad=4)
                ax.set_xlim(0, MODEL_BASED_DEPTH)
                ax.set_xticks([0, 4, 8, 12, 16])
                ax.set_ylim(y_lo, y_hi)
            for ax in axes[len(ps):]:
                ax.axis("off")
            if len(ps) < len(axes):
                axes[len(ps)].legend(list(handles.values()), list(handles), loc="center",
                                     frameon=False, fontsize=8, ncol=2)
            else:
                fig.legend(list(handles.values()), list(handles), loc="lower center",
                           ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
            for ax in axes[::ncols]:
                ax.set_ylabel(spec["ylabel"], fontsize=8)
            fig.supxlabel("unrolled layers $k$", fontsize=8, y=0.035)
            fig.suptitle("{} {}, {} training width -- one curve per seed ({} seeds), no averaging -- {}"
                         .format(METHOD_LABELS[method],
                                 "$R_f$" if args.metric == "rf" else "NMSE vs $x_{true}$",
                                 width, len(seeds), RUNS[args.run]["subtitle"]),
                         fontsize=9, color=INK_PRIMARY, y=1.005)
            fig.tight_layout(rect=(0, 0.02, 1, 1))
            name = "{}_seed_curves{}_{}_{}".format(method, spec["suffix"], args.run, width)
            fig.savefig(os.path.join(out_dir, name + ".png"), facecolor=SURFACE,
                        bbox_inches="tight", dpi=200)
            print("wrote", os.path.join(out_dir, name + ".png"))
            plt.close(fig)
            entries.append((name, WIDTH_LABELS[width],
                            "fig:{}-seed-curves{}-{}-{}".format(
                                method, spec["suffix"].replace("_", "-"),
                                args.run.replace("_", "-"), width)))

            # spread across seeds at fixed k, per p
            for k in CHECK_K:
                vals = [[curves[(method, width, p, s)][k] for s in seeds
                         if (method, width, p, s) in curves] for p in ps]
                sp = [spec["spread"](v) for v in vals if len(v) > 1]
                spread_lines.append(("| {} | {} | {} | " + spec["fmt"] + " | " + spec["fmt"] + " |").format(
                    METHOD_LABELS[method], width, k, min(sp), max(sp)))

        lines = ["% One figure per training width, one thin curve per seed (no averaging),",
                 "% generated by plot_modelbased_seed_curves.py --run " + args.run
                 + " --method " + method + " --metric " + args.metric + ".",
                 "% Requires \\usepackage{graphicx} and \\usepackage{subcaption}. Paths are",
                 "% relative to the repo root (Figs/...); adjust \\graphicspath or the paths below.",
                 "% Include with: \\input{Figs/multifigure/" + method + "_seed_curves"
                 + spec["suffix"] + "_" + args.run + "_figure}",
                 r"\begin{figure}[htbp]", r"\centering"]
        for name, cap, lab in entries:
            lines += [r"\begin{subfigure}[t]{\linewidth}", r"  \centering",
                      r"  \includegraphics[width=\linewidth]{Figs/" + args.run + "/" + name + ".png}",
                      r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}\\[6pt]"]
        lines += [r"\caption{" + METHOD_LABELS[method] + ", " + spec["tex_quantity"] + r" against "
                  r"unrolled layer $k$, " + RUNS[args.run]["subtitle"]
                  + r". One curve per training seed (" + str(len(seeds))
                  + r" independently trained networks per width), no averaging; one panel per test "
                  r"sparsity $p$. $k = 0$ is the shared seeded start $x_0$, $k = 16$ the network's "
                  r"output. Unlike the model-free optimizer, every layer's weights are free, so the "
                  r"seeds may separate from the first layer on.}",
                  r"\label{fig:" + method + "-seed-curves" + spec["suffix"].replace("_", "-") + "-"
                  + args.run.replace("_", "-") + "}",
                  r"\end{figure}"]
        out_tex = os.path.join(MULTIFIG_DIR, "{}_seed_curves{}_{}_figure.tex".format(
            method, spec["suffix"], args.run))
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", out_tex)

    print("\nspread across seeds, {} at layer k (min and max over test p):".format(spec["spread_name"]))
    print("| method | width | k | min over p | max over p |")
    print("|---|---|---|---|---|")
    print("\n".join(spread_lines))


if __name__ == "__main__":
    main()
