"""
Compute-matched comparison for Experiment 2: modified relative loss R_f as a
function of the iteration budget, for the model-free optimizer (L2O-RNNProp),
the classical solvers (ISTA / FISTA) and the fixed-depth unrolled networks
(LISTA / ALISTA at their 16 layers).

Why: the pre-study's S7.2 protocol evaluates each family "to its own standard,
fixed budget" -- 1000 learned steps for the model-free optimizer, 16 layers
for LISTA/ALISTA. That is the right per-protocol comparison, but it cannot by
itself answer "do the families differ in the width tradeoff": a 1000-step
run on a convex problem lands near x* whatever it was trained on, so the
model-free width-insensitivity in results.jsonl may be convergence, not
robustness. This script adds the matched-budget view.

No new GPU work. Inputs already on disk:
  * `checkpoints/s<seed>/<width>/l2o-rnnprop/eval_p<p>/eval/
     L2L_eval_loss_record.pickle-lasso_dataset` -- the batch-mean objective
     f(x_k) at every k = 0..999 (x_k = iterate after k learned steps; k = 0 is
     the shared seeded x0), written by evaluate_rnnprop.py for every finished
     model-free triple;
  * the run's `xstar_cache/` -- x* for every test file, content-addressed on
    (A, b, lam, 50_000, 1e-10), the same entries results.jsonl's R_f used, so
    mean f* is exact rather than back-solved from final_loss;
  * `classical_trajectories/` + `classical_baselines.jsonl` from
    classical_baselines.py (run that first);
  * results.jsonl for LISTA / ALISTA's 16-layer R_f (all seeds).

Outputs:
  Figs/<run>/compute_matched_rf_<run>.png   R_f vs. iteration, one panel
      per test sparsity, shared log axes; RNNProp per width (median over
      seeds), ISTA / FISTA, LISTA / ALISTA per layer k = 1..16 (one thin
      curve per training width, from the regenerated per-file metrics) ending
      at the 16-layer marker, whose value is also drawn as a guide line.
  Figs/<run>/iterations_to_match_<run>.png   iterations RNNProp (per
      width) and FISTA need to first reach the 16-layer LISTA / ALISTA R_f,
      vs. test sparsity (combined two-panel overview).
  Figs/<run>/iterations_to_match_<run>_{lista,alista}.png   the same two
      panels as standalone images, plus
  Figs/multifigure/iterations_to_match_<run>_figure.tex   a subcaption snippet
      referencing them (input it from the thesis; needs subcaption).
  Benchmarking/compute_matched_<run>.md   the numbers behind both figures.

Usage:
  python analysis/plot_compute_matched_rf.py --run exp2_try2
  python analysis/plot_compute_matched_rf.py --run exp2_try3_210x300
  python analysis/plot_compute_matched_rf.py --run exp2_try3_210x300 --overlay none
      -> compute_matched_rf_<run>_rnnprop_only.png: the RNNProp curves alone,
         for the "flattens by ~300 steps and is identical across widths"
         statement; --overlay classical adds ISTA/FISTA (_classical.png).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Benchmarking/ is this file's parent, not its own directory, so it is not on
# sys.path automatically when this runs as `python analysis/<script>.py`. This
# must precede every core.*/data.* import below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_specs  # noqa: E402
from core.seedsets import filter_to_complete_seeds  # noqa: E402
from core.lasso_metrics import _GAP_TOL, _xstar_cache_path, lasso_objective  # noqa: E402
from data.lasso import _test_filename  # noqa: E402

THESIS_FIGS_DIR = os.path.join("..", "Figs")
MULTIFIG_DIR = os.path.join(THESIS_FIGS_DIR, "multifigure")  # all subfigure snippets live here
NUM_FISTA_ITERS = 50_000  # every Experiment-2 run's x* budget (cache keys verified)

# Same width palette as every other Experiment-2 figure (blue / orange / aqua).
NARROW_C, MEDIUM_C, WIDE_C = "#2a78d6", "#eb6834", "#1baf7a"
# (key, label, colour, linestyle) -- same linestyle-per-width convention as the
# other Experiment-2 figures, needed here because the three RNNProp curves lie
# almost exactly on top of each other (that overlap IS the finding).
WIDTHS = [("narrow", "Narrow", NARROW_C, "-"), ("medium", "Medium", MEDIUM_C, "--"), ("wide", "Wide", WIDE_C, ":")]
ISTA_C, FISTA_C = "#b8b6ae", "#5d5b55"           # neutral greys: classical references
LISTA_C, ALISTA_C = "#0b0b0b", "#8a3ffc"          # ink + violet: the two unrolled networks

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"

MODEL_BASED_DEPTH = 16  # LISTA / ALISTA layers in every Experiment-2 run
BUDGETS = (16, 100, 1000)
# FISTA reaches the certified optimum (R_f ~ 1e-12) within a few hundred
# iterations on the sparsest instances; letting the log axis follow that tail
# squashes every other curve into the top decade. Anything below this is
# "converged" for the purposes of the comparison, and the caption says so.
RF_FLOOR = 1e-5

# Run identity -- results paths, dimensions, method lists, width triads --
# comes from the shared registry, so a run is defined in exactly one place.
# The ORDER of the keys below and the subtitle wording are presentation,
# and stay local: the order fixes panel/figure order in this script's own
# output, and each script phrases the run differently.
_ORDER = ["exp2_try2", "exp2_try3_210x300", "exp2_real",
          # Model-based only (LISTA / ALISTA re-trained with the
          # scoring-consistent loss); has no model-free rows, so only the
          # model-based readers below and plot_modelbased_seed_curves.py use it.
          "exp2_try3_210x300_scoringonly"]
RUNS = {k: {"subtitle": run_specs.RUN_SPECS[k].subtitle} for k in _ORDER}


def _eval_dirname(p: float) -> str:
    # experiment2*.py names eval dirs with the same 2-decimal format as the
    # test files (0.175 -> "0.17", 0.1125 -> "0.11", 0.2375 -> "0.24").
    return "eval_p{:.2f}".format(p)


def load_results(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_fstar(run_dir: str, seed: int, p: float, lam: float, m: int) -> float:
    """Exact mean f* for one test file, from the run's own x* cache."""
    seed_dir = os.path.join(run_dir, "data", "seed{}".format(seed))
    width_dir = os.path.join(seed_dir, sorted(
        w for w in os.listdir(seed_dir) if os.path.isdir(os.path.join(seed_dir, w)))[0])
    a = np.load(os.path.join(width_dir, "A.npy")).astype(np.float32)
    data = np.load(os.path.join(width_dir, _test_filename(p)))
    b = data[:, :m]
    path = _xstar_cache_path(os.path.join(run_dir, "xstar_cache"), a, b, lam,
                             NUM_FISTA_ITERS, _GAP_TOL)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no cached x* for seed {} p={} ({}); results.jsonl rows for this "
            "test file would have created it -- check the run finished this "
            "cell".format(seed, p, path))
    x_star = np.load(path)
    return float(np.mean(lasso_objective(a.astype(np.float64), x_star,
                                         b.astype(np.float64), lam)))


def rnnprop_rf_curves(run_dir: str, rows: List[dict], lam: float, m: int,
                      method: str = "l2o-rnnprop") -> Dict[Tuple[str, float], np.ndarray]:
    """{(width, p): R_f[k] median over seeds, k = 0..num_steps-1}."""
    per_cell: Dict[Tuple[str, float], List[np.ndarray]] = defaultdict(list)
    fstar_cache: Dict[Tuple[int, float], float] = {}
    for r in rows:
        if r["method"] != method:
            continue
        seed, width, p = int(r["seed"]), r["width"], float(r["test_sparsity"])
        pkl = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, method,
                           _eval_dirname(p), "eval", "L2L_eval_loss_record.pickle-lasso_dataset")
        if not os.path.exists(pkl):
            print("  missing loss record, skipping:", pkl)
            continue
        with open(pkl, "rb") as f:
            traj = np.asarray(pickle.load(f), dtype=np.float64)
        key = (seed, p)
        if key not in fstar_cache:
            fstar_cache[key] = mean_fstar(run_dir, seed, p, lam, m)
        rf = traj / fstar_cache[key] - 1.0
        # Guard that the content-addressed x* cache key resolved to THIS test
        # file: the record's last entry is f(x_999) and the row's R_f is scored
        # on x_1000, so once the trajectory has flattened they must agree.
        # A wrong m-slice or width dir would give a plausible curve at the
        # wrong scale, and this is the only place that would show.
        row_rf = float(r["modified_relative_loss"])
        if abs(rf[-1] - row_rf) > 0.05 * max(abs(row_rf), 1e-9) + 1e-4:
            print("  WARNING: R_f mismatch for seed {} {} p={}: record {:.4g} vs row {:.4g}".format(
                seed, width, p, rf[-1], row_rf))
        per_cell[(width, p)].append(rf)
    out = {}
    for key, curves in per_cell.items():
        n = min(len(c) for c in curves)
        out[key] = np.median(np.stack([c[:n] for c in curves]), axis=0)
    return out


def classical_curves(run_dir: str, seeds: List[int], ps: List[float], lam: float, m: int
                     ) -> Dict[Tuple[str, float], np.ndarray]:
    """{(method, p): R_f[k] median over seeds, k = 0..max_iters}."""
    traj_dir = os.path.join(run_dir, "classical_trajectories")
    out = {}
    for method in ("ista", "fista"):
        for p in ps:
            curves = []
            for seed in seeds:
                path = os.path.join(traj_dir, "{}_seed{}_p{:.4f}.npy".format(method, seed, p))
                if not os.path.exists(path):
                    continue
                traj = np.load(path)
                curves.append(traj / mean_fstar(run_dir, seed, p, lam, m) - 1.0)
            if curves:
                n = min(len(c) for c in curves)
                out[(method, p)] = np.median(np.stack([c[:n] for c in curves]), axis=0)
    return out


def model_based_rf(rows: List[dict]) -> Dict[Tuple[str, str, float], float]:
    """{(method, width, p): median R_f over seeds} for lista / alista."""
    acc: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        if r["method"] in ("lista", "alista"):
            acc[(r["method"], r["width"], float(r["test_sparsity"]))].append(
                float(r["modified_relative_loss"]))
    return {k: float(np.median(v)) for k, v in acc.items()}


def model_based_layer_curves(run_dir: str, rows: List[dict], lam: float, m: int
                             ) -> Dict[Tuple[str, str, float], np.ndarray]:
    """{(method, width, p): R_f[k] median over seeds, k = 0..16} for lista / alista.

    k = 0 is f(x0) (the shared seeded start), k = i the output of layer i.
    Source: each cell's `all_test_metrics_perfile.json` (`lasso_layer{i}` =
    objective of the iterate after i layers, i = 0..15 -- layer_id i slices
    y_pred = [b, x0, x1, ..., x16] at M + i*N, so index 0 is x0), written by
    `regenerate_modelbased_trajectories.py` after the 2026-09-17 fix (the
    library's own per-layer values were running means across test files and
    are NOT usable). k = 16 is the row's own `modified_relative_loss`, which
    is scored on the same final iterate. Cells without the regenerated file
    are skipped (a warning is printed) so a partially regenerated run still
    plots what it has.
    """
    per_cell: Dict[Tuple[str, str, float], List[np.ndarray]] = defaultdict(list)
    fstar_cache: Dict[Tuple[int, float], float] = {}
    missing = 0
    for r in rows:
        if r["method"] not in ("lista", "alista"):
            continue
        seed, width, p = int(r["seed"]), r["width"], float(r["test_sparsity"])
        pattern = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, r["method"],
                               "checkpoints", "models", "*", "replicate_1", "all_test_metrics_perfile.json")
        hits = glob.glob(pattern)
        if not hits:
            missing += 1
            continue
        with open(hits[0]) as f:
            per_file = json.load(f)
        vals = per_file.get(_test_filename(p))
        if vals is None:
            missing += 1
            continue
        key = (seed, p)
        if key not in fstar_cache:
            fstar_cache[key] = mean_fstar(run_dir, seed, p, lam, m)
        f_star = fstar_cache[key]
        layers = [vals["lasso_layer{}".format(i)] for i in range(MODEL_BASED_DEPTH)]
        rf = np.array(layers, dtype=np.float64) / f_star - 1.0
        rf = np.append(rf, float(r["modified_relative_loss"]))   # k = 16
        per_cell[(r["method"], width, p)].append(rf)
    if missing:
        print("  {} model-based cells lack all_test_metrics_perfile.json -- run "
              "regenerate_modelbased_trajectories.py --run {} for full per-layer curves".format(missing, run_dir))
    return {k: np.median(np.stack(v), axis=0) for k, v in per_cell.items()}


def first_reach(curve: np.ndarray, target: float) -> Optional[int]:
    hit = np.nonzero(curve <= target)[0]
    return int(hit[0]) if hit.size else None


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def save(fig, run_key: str, name: str):
    out_dir = os.path.join(THESIS_FIGS_DIR, run_key)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, name)
    fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("  wrote", stem + ".png")
    plt.close(fig)


def fmt_k(k: Optional[int]) -> str:
    return "never" if k is None else str(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--method", default="l2o-rnnprop")
    ap.add_argument("--overlay", choices=["none", "classical", "full"], default="full",
                    help="what to draw besides the RNNProp curves in the R_f-vs-iteration figure: "
                         "'none' = RNNProp only (the width-invariance / plateau figure), "
                         "'classical' = + ISTA/FISTA, 'full' = + LISTA/ALISTA per-layer curves and "
                         "16-layer markers (the compute-matched figure). Output name gets a suffix "
                         "for the reduced variants; the iterations-to-match figure and tables are unaffected.")
    args = ap.parse_args()

    run_key = args.run
    run_dir = os.path.join("runs", run_key)
    rows = load_results(run_dir)
    # Only seeds present in EVERY (method, width) cell: the matched-budget
    # comparison is model-free vs model-based, so an incomplete seed would put
    # the two sides of that comparison on different seed sets
    # (see core/seedsets.py).
    rows, _, _ = filter_to_complete_seeds(
        rows, methods=[args.method, "lista", "alista"], label=run_dir)
    seeds_all = sorted({int(r["seed"]) for r in rows})
    with open(os.path.join(run_dir, "data", "seed{}".format(seeds_all[0]), "metadata.json")) as f:
        meta = json.load(f)
    lam, m = float(meta["lam"]), int(meta["m"])
    ps = [float(p) for p in meta["test_sparsities"]]

    mf_seeds = sorted({int(r["seed"]) for r in rows if r["method"] == args.method})
    mb_seeds = sorted({int(r["seed"]) for r in rows if r["method"] in ("lista", "alista")})
    print("run {}: {} seeds {}, model-based seeds {}".format(run_key, args.method, mf_seeds, mb_seeds))

    rf_mf = rnnprop_rf_curves(run_dir, rows, lam, m, args.method)
    rf_cl = classical_curves(run_dir, seeds_all, ps, lam, m)
    rf_mb = model_based_rf(rows)
    rf_mb_layers = model_based_layer_curves(run_dir, rows, lam, m)
    if not rf_mf:
        print("  no valid {} rows in {} -- plotting classical + model-based only".format(args.method, run_dir))
    if not rf_cl:
        raise SystemExit("no classical trajectories -- run classical_baselines.py --run {} first".format(run_dir))

    widths_present = [w for w in WIDTHS if any(k[0] == w[0] for k in rf_mf)]
    n_seeds_note = len(mf_seeds)
    n_steps = min(len(c) for c in rf_mf.values()) if rf_mf else min(len(c) for c in rf_cl.values()) - 1
    # R_f at the full 1000-step budget for the model-free rows comes from
    # results.jsonl (scored on the saved final iterate x_1000); the loss record
    # only covers k = 0..999.
    _acc: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == args.method:
            _acc[(r["width"], float(r["test_sparsity"]))].append(float(r["modified_relative_loss"]))
    rf_mf_final = {k: float(np.median(v)) for k, v in _acc.items()}

    # ---------------------------------------------------------------- figure 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY})
    ncols = 4
    # Grid sized on the panels alone; the legend takes a spare cell when the
    # grid has one (11 panels -> 12 cells) and goes below the figure otherwise
    # (8 panels -> exactly 2 x 4, no spare cell).
    nrows = int(np.ceil(len(ps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    k_axis = np.arange(n_steps)
    for ax, p in zip(axes, ps):
        style_axes(ax)
        for method, label, color, ls in ((("ista", "ISTA", ISTA_C, "-"), ("fista", "FISTA", FISTA_C, "-"))
                                         if args.overlay != "none" else ()):
            c = rf_cl.get((method, p))
            if c is not None:
                kk = np.arange(len(c))
                (h,) = ax.plot(kk[1:], np.maximum(c[1:], RF_FLOOR), color=color, lw=2, ls=ls, zorder=2)
                handles[label] = h
        for key, label, color, ls in widths_present:
            c = rf_mf.get((key, p))
            if c is not None:
                (h,) = ax.plot(k_axis[1:], np.maximum(c[1:n_steps], RF_FLOOR), color=color, lw=2, ls=ls, zorder=3)
                handles["RNNProp, {} width".format(label.lower())] = h
        for method, label, color, marker in ((("lista", "LISTA (16 layers)", LISTA_C, "s"),
                                              ("alista", "ALISTA (16 layers)", ALISTA_C, "D"))
                                             if args.overlay == "full" else ()):
            vals = [rf_mb[(method, w, p)] for w, _, _, _ in WIDTHS if (method, w, p) in rf_mb]
            if not vals:
                continue
            # Per-layer trajectory, k = 1..16, one thin curve per training width
            # (same linestyle-per-width convention as the RNNProp curves), ending
            # at the marker. k = 0 is f(x0), identical for every method, and is
            # off the log x-axis anyway.
            for wkey, _, _, wls in WIDTHS:
                c = rf_mb_layers.get((method, wkey, p))
                if c is not None:
                    kk = np.arange(1, len(c))
                    (hl,) = ax.plot(kk, np.maximum(c[1:], RF_FLOOR), color=color, lw=1.2, ls=wls,
                                    alpha=0.85, zorder=3)
                    handles.setdefault(label.replace("(16 layers)", "per layer, one curve per width"), hl)
            med = float(np.median(vals))
            ax.axhline(med, color=color, lw=1, ls=(0, (2, 2)), zorder=1)
            ax.plot([MODEL_BASED_DEPTH, MODEL_BASED_DEPTH], [min(vals), max(vals)], color=color, lw=1.5, zorder=4)
            h = ax.scatter([MODEL_BASED_DEPTH], [med], marker=marker, s=36, facecolor=SURFACE,
                           edgecolor=color, linewidth=1.5, zorder=5)
            handles[label] = h
        ax.set_title("test $p = {:g}$".format(p), fontsize=9, color=INK_PRIMARY, pad=4)
        ax.set_xlim(1, max(n_steps, 1000))
        ax.set_ylim(RF_FLOOR, None)
    for ax in axes[len(ps):]:
        ax.axis("off")
    legend_ax = axes[len(ps)] if len(ps) < len(axes) else None
    order = [l for l in ("RNNProp, narrow width", "RNNProp, medium width", "RNNProp, wide width",
                         "FISTA", "ISTA",
                         "LISTA per layer, one curve per width", "LISTA (16 layers)",
                         "ALISTA per layer, one curve per width", "ALISTA (16 layers)") if l in handles]
    if legend_ax is not None:
        legend_ax.legend([handles[l] for l in order], order, loc="center", frameon=False, fontsize=8)
    else:
        fig.legend([handles[l] for l in order], order, loc="lower center", ncol=len(order),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    for ax in axes[::ncols]:
        ax.set_ylabel("$R_f$ after $k$ iterations", fontsize=8)
    if args.overlay == "none":
        # RNNProp never gets below ~1e-3; the 1e-5 floor exists for FISTA's tail.
        for ax in axes[:len(ps)]:
            ax.set_ylim(min(c[1:].min() for c in rf_mf.values()) / 3, None)
        xlab = "learned steps $k$"
    else:
        xlab = "iterations $k$  (learned steps / unrolled layers / solver iterations)"
    fig.supxlabel(xlab, fontsize=8, y=0.035)
    if not rf_mf:
        mf_note = "no valid model-free rows in this run; "
    elif len(widths_present) > 1:
        mf_note = ("RNNProp: median over {} seed(s); the {} width curves agree to <1% for k<100 "
                   "and within ~2x on the plateau; ").format(n_seeds_note, len(widths_present))
    else:
        mf_note = "RNNProp ({} width only so far): median over {} seed(s); ".format(
            widths_present[0][1].lower(), n_seeds_note)
    if args.overlay == "full":
        tail = "LISTA/ALISTA: median over {} seed(s), thin curves = per layer, one per width, " \
               "bar = spread across widths; ".format(len(mb_seeds))
        title = "Modified relative loss vs. iteration budget -- {}"
    elif args.overlay == "classical":
        tail = "ISTA/FISTA from the same shared $x_0$ (median over seeds); "
        title = "Modified relative loss vs. iteration budget, RNNProp vs classical -- {}"
    else:
        tail = ""
        title = "RNNProp: modified relative loss vs. learned steps, by training width -- {}"
    floor_note = "" if args.overlay == "none" else "axis floor {:g} = converged".format(RF_FLOOR)
    fig.suptitle((title + "\n{}{}{}").format(RUNS[run_key]["subtitle"], mf_note, tail, floor_note).rstrip("; "),
                 fontsize=9, color=INK_PRIMARY, y=1.01)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    suffix = {"full": "", "classical": "_classical", "none": "_rnnprop_only"}[args.overlay]
    save(fig, run_key, "compute_matched_rf_{}{}".format(run_key, suffix))

    # ------------------------------------------------- iterations-to-match
    match: Dict[Tuple[str, str, float], Optional[int]] = {}
    for p in ps:
        for target_m in ("lista", "alista"):
            vals = [rf_mb[(target_m, w, p)] for w, _, _, _ in WIDTHS if (target_m, w, p) in rf_mb]
            if not vals:
                continue
            tgt = float(np.median(vals))
            for key, _, _, _ in widths_present:
                c = rf_mf.get((key, p))
                match[("rnnprop-" + key, target_m, p)] = None if c is None else first_reach(c, tgt)
            for cm in ("ista", "fista"):
                c = rf_cl.get((cm, p))
                match[(cm, target_m, p)] = None if c is None else first_reach(c, tgt)

    series = [("rnnprop-" + k, "RNNProp, {} width".format(l.lower()), c, ls, "o") for k, l, c, ls in widths_present]
    series += [("fista", "FISTA", FISTA_C, "-", "^"), ("ista", "ISTA", ISTA_C, "-", "v")]
    targets = (("lista", "LISTA", LISTA_C), ("alista", "ALISTA", ALISTA_C))
    never_note = "x above the {}-iteration line = never reached within the evaluation horizon".format(n_steps)

    def draw_match_panel(ax, target_m, color, legend):
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.axhline(MODEL_BASED_DEPTH, color=color, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.text(ps[0], MODEL_BASED_DEPTH * 1.15, "16 = matched budget", fontsize=7, color=color)
        for skey, label, scol, ls, marker in series:
            ys = [match.get((skey, target_m, p)) for p in ps]
            xs_hit = [p for p, y in zip(ps, ys) if y is not None]
            ys_hit = [y for y in ys if y is not None]
            xs_miss = [p for p, y in zip(ps, ys) if y is None]
            ax.plot(xs_hit, ys_hit, color=scol, lw=2, ls=ls, marker=marker, ms=5, label=label, zorder=3)
            if xs_miss:
                ax.scatter(xs_miss, [n_steps * 1.6] * len(xs_miss), marker="x", color=scol, s=30, zorder=4)
        ax.set_ylim(None, n_steps * 2.4)
        ax.set_xlabel("test sparsity $p$", fontsize=8)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        if legend:
            # Below the axes: inside, every corner collides with a curve, the
            # 16-line or the never-reached crosses in at least one run.
            h, l = ax.get_legend_handles_labels()
            ax.figure.legend(h, l, frameon=False, fontsize=7, loc="lower center",
                             ncol=3, bbox_to_anchor=(0.5, -0.01))

    # (a) combined two-panel overview
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (target_m, tlabel, color) in zip(axes, targets):
        draw_match_panel(ax, target_m, color, legend=False)
        ax.set_title("Iterations to reach {}'s 16-layer $R_f$".format(tlabel), fontsize=9)
    axes[0].set_ylabel("first $k$ with $R_f(k) \\leq$ target", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.09, never_note, ha="center", fontsize=7, color=INK_MUTED)
    fig.suptitle("Iteration budget needed to match the unrolled networks -- {}".format(RUNS[run_key]["subtitle"]),
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, run_key, "iterations_to_match_{}".format(run_key))

    # (b) one standalone image per target (no title: the LaTeX subcaption
    #     carries it; legend inside each so the images stand alone) + a
    #     subfigure snippet to \input from the thesis
    sub_entries = []
    for target_m, tlabel, color in targets:
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        fig.patch.set_facecolor(SURFACE)
        draw_match_panel(ax, target_m, color, legend=True)
        ax.set_ylabel("first $k$ with $R_f(k) \\leq$ target", fontsize=8)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        name = "iterations_to_match_{}_{}".format(run_key, target_m)
        save(fig, run_key, name)
        sub_entries.append((name, "Iterations to reach {}'s 16-layer $R_f$".format(tlabel),
                            "fig:iterations-to-match-{}-{}".format(run_key.replace("_", "-"), target_m)))
    mf_desc = ("RNNProp (median over {} seed{}; one curve per training width)".format(
        len(mf_seeds), "" if len(mf_seeds) == 1 else "s") if rf_mf else "no valid model-free rows in this run")
    fig_lines = [
        "% Multi-panel figure: iterations to match the unrolled networks, {}.".format(run_key),
        "% Generated by plot_compute_matched_rf.py --run {}. Requires \\usepackage{{graphicx}}".format(run_key),
        "% and \\usepackage{subcaption}. Paths are relative to the repo root (Figs/...);",
        "% adjust \\graphicspath or the paths below to match the Overleaf project.",
        "% Include with: \\input{{Figs/multifigure/iterations_to_match_{}_figure}}".format(run_key),
        r"\begin{figure}[htbp]", r"\centering"]
    for i, (name, cap, lab) in enumerate(sub_entries):
        fig_lines += [r"\begin{subfigure}[t]{0.48\linewidth}", r"  \centering",
                      r"  \includegraphics[width=\linewidth]{Figs/" + run_key + "/" + name + ".png}",
                      r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}"
                      + (r"\hfill" if i % 2 == 0 else "")]
    fig_lines += [
        r"\caption{Iteration budget needed to match the 16-layer unrolled networks on the modified relative "
        r"loss $R_f$ (Eq.~10), " + RUNS[run_key]["subtitle"] + r". For each test sparsity $p$, the first "
        r"iteration $k$ at which a method's $R_f(k)$ falls to or below the 16-layer LISTA (left) or ALISTA "
        r"(right) value (median over seeds and over the three training widths); " + mf_desc + r", and "
        r"textbook ISTA and FISTA from the same shared $x_0$ (median over seeds). The dotted line marks the "
        r"matched budget of 16 iterations; a cross above the " + str(n_steps) + r"-iteration line means the "
        r"target was never reached within the evaluation horizon. Iteration-matched is not compute-matched: "
        r"one learned step is an LSTM forward pass per coordinate, one ISTA/FISTA step a matrix product.}",
        r"\label{fig:iterations-to-match-" + run_key.replace("_", "-") + "}", r"\end{figure}"]
    os.makedirs(MULTIFIG_DIR, exist_ok=True)
    out_fig_tex = os.path.join(MULTIFIG_DIR, "iterations_to_match_{}_figure.tex".format(run_key))
    with open(out_fig_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(fig_lines) + "\n")
    print("  wrote", out_fig_tex)

    # --------------------------------------------------------------- tables
    lines = ["# Compute-matched R_f -- {}".format(run_key), "",
             "{}. {} seeds for RNNProp: {}; LISTA/ALISTA seeds: {}. All cells are medians over seeds; "
             "LISTA/ALISTA cells are additionally the median over the three training widths with the "
             "width range in brackets. R_f at iteration k uses the iterate after k learned steps / "
             "k solver iterations / k layers, all from the shared seeded x0.".format(
                 RUNS[run_key]["subtitle"].replace("$", ""), args.method, mf_seeds, mb_seeds), ""]

    def mb_cell(method, p):
        vals = [rf_mb[(method, w, p)] for w, _, _, _ in WIDTHS if (method, w, p) in rf_mb]
        if not vals:
            return "--"
        return "{} [{}-{}]".format(fmt_rf(float(np.median(vals))), fmt_rf(min(vals)), fmt_rf(max(vals)))

    def fmt_rf(v):
        # Below 1e-6 the value is FISTA's own convergence noise around the
        # certified x* (medians over seeds land at ~1e-18), not a result.
        if abs(v) < 1e-6:
            return "<1e-6"
        return "{:.4f}".format(v) if abs(v) >= 1e-3 else "{:.1e}".format(v)

    def mf_cell(p, k):
        cells = []
        for key, _, _, _ in widths_present:
            c = rf_mf.get((key, p))
            if c is not None and k < len(c):
                cells.append(fmt_rf(c[k]))
            elif (key, p) in rf_mf_final and k == n_steps:
                cells.append(fmt_rf(rf_mf_final[(key, p)]))
            else:
                cells.append("--")
        return " / ".join(cells) if cells else "--"

    def cl_cell(method, p, k):
        c = rf_cl.get((method, p))
        return "--" if c is None or k >= len(c) else fmt_rf(c[k])

    wl = "/".join(l.lower() for _, l, _, _ in widths_present)
    for k in BUDGETS:
        lines += ["## R_f at k = {}".format(k), "",
                  "| test p | ISTA-{k} | FISTA-{k} | RNNProp-{k} ({wl}) | LISTA-16 | ALISTA-16 |".format(k=k, wl=wl),
                  "|---|---|---|---|---|---|"]
        for p in ps:
            lines.append("| {:g} | {} | {} | {} | {} | {} |".format(
                p, cl_cell("ista", p, k), cl_cell("fista", p, k), mf_cell(p, k),
                mb_cell("lista", p), mb_cell("alista", p)))
        lines.append("")
    lines += ["## Iterations to first reach the 16-layer R_f (median target across widths)", "",
              "| test p | RNNProp -> LISTA ({wl}) | FISTA -> LISTA | ISTA -> LISTA | RNNProp -> ALISTA ({wl}) | FISTA -> ALISTA | ISTA -> ALISTA |".format(wl=wl),
              "|---|---|---|---|---|---|---|"]
    for p in ps:
        def mfm(t):
            return " / ".join(fmt_k(match.get(("rnnprop-" + key, t, p))) for key, _, _, _ in widths_present)
        lines.append("| {:g} | {} | {} | {} | {} | {} | {} |".format(
            p, mfm("lista"), fmt_k(match.get(("fista", "lista", p))), fmt_k(match.get(("ista", "lista", p))),
            mfm("alista"), fmt_k(match.get(("fista", "alista", p))), fmt_k(match.get(("ista", "alista", p)))))
    lines.append("")
    lines.append("'never' = not reached within {} iterations.".format(n_steps))
    out_md = "compute_matched_{}.md".format(run_key)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", out_md)


if __name__ == "__main__":
    main()
