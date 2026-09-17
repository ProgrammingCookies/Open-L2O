"""
Experiment 2, RQ2a: the paired ID-cost / OOD-benefit of widening the training
sparsity distribution, for LISTA and ALISTA, deconfounded from (a) where each
width's training distribution is centred and (b) how hard the LASSO problem
is at each test sparsity.

Why this exists: the per-width NMSE tables and figures show the *raw* curves,
on which a 1-2 dB width effect is invisible against a 40-90 dB axis, and in
`exp2_real` the three widths differ in centre (0.100 / 0.1375 / 0.175) as well
as spread, so "wide is better at dense p" is partly "wide was trained nearer
dense p". This script measures the effect the way the claim is stated:

  * per seed (the same seed = the same dictionary A, the same test instances,
    the same x0 for every width), so differences are PAIRED across widths;
  * as a function of the OFFSET from each width's own training centre
    (offset = test p - centre_width), interpolated linearly in p onto a common
    offset grid, so narrow and wide are compared at the same distance from
    what they were trained on;
  * on EXCESS OVER THE LASSO-OPTIMAL FLOOR (method NMSE minus NMSE(x*, x_true)
    at the same test p) rather than raw NMSE, so a comparison at the same
    offset but different absolute p is not biased by the problem being harder
    at one of them. For runs whose three centres coincide (`exp2_try2`,
    `exp2_try3_210x300`) the floor cancels exactly and this equals the raw
    paired difference.

Reported quantities (all "wide minus narrow" and "medium minus narrow", dB,
positive = widening HURTS), matching how RQ2a is phrased -- ID quality vs
OOD *robustness*:
  ID cost              = paired difference at offset 0 (each width at its own
                         training centre)
  robustness benefit   = paired difference in DEGRADATION from own centre,
    at offset d          [excess_w(c_w + d) - excess_w(c_w)]
                         - [excess_n(c_n + d) - excess_n(c_n)]
                         (negative = the wider training degrades less)
  net at offset d      = paired difference of the absolute excess at offset d
                         = ID cost + robustness benefit
for d = -0.075 (sparse side) and d = +0.20 (dense side), with mean, 95%
t-interval and a paired two-sided t-test over seeds (Wilcoxon p also printed
to stdout). The offsets are the largest common range the test grids of all
three widths allow in every run. The figure plots the net curve; its value at
offset 0 is the ID cost and the curve minus that value is the robustness
benefit.

Inputs: `runs/<run>/results.jsonl` only (never summary.json -- stale).
Outputs:
  Figs/experiment2/width_tradeoff_nmse.png   combined overview, one panel
      per run x method: mean paired difference vs offset, 95% CI band, zero line
  Figs/experiment2/width_tradeoff_nmse_<run>_<method>.png   the same
      panels as standalone images (shared y-range, no title) for a LaTeX
      multi-panel figure
  Benchmarking/experiment2_width_tradeoff_figure.tex   subfigure snippet
      referencing those images (needs the subcaption package); input it
  Benchmarking/experiment2_width_tradeoff_table.tex   the numbers
  stdout: the same table in markdown plus the raw-NMSE-at-fixed-p variant
      used in the 2026-09-06 cross-run analysis, for continuity.

Runs with fewer than MIN_SEEDS seeds for a method are skipped (a paired test
on 2 seeds is not a result); re-run as exp2_try3_210x300 fills in.

Usage: python plot_experiment2_width_tradeoff.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

THESIS_FIGS_DIR = os.path.join("..", "Figs", "experiment2")
OUT_TEX = "experiment2_width_tradeoff_table.tex"
OUT_FIG_TEX = "experiment2_width_tradeoff_figure.tex"
MIN_SEEDS = 5

RUNS = [
    # (run key, label, m, n, noise label)
    ("exp2_try2", r"$(25,50)$, noiseless, centred widths", 25, 50),
    ("exp2_try3_210x300", r"$(210,300)$, noiseless, centred widths", 210, 300),
    ("exp2_real", r"$(210,300)$, SNR 40 dB, uncentred widths", 210, 300),
]
METHODS = [("lista", "LISTA"), ("alista", "ALISTA")]
WIDTHS = ["narrow", "medium", "wide"]
COMPARISONS = [("wide", "narrow", "wide $-$ narrow", "#1baf7a", "-"),
               ("medium", "narrow", "medium $-$ narrow", "#eb6834", "--")]

OFFSET_STEP = 0.025
# Cells where a width already recovers x_true to better than this are in the
# exact-recovery regime (ALISTA at (25,50) p <= 0.05: -60 to -97 dB, seed std
# 10-16 dB): both widths are numerically at zero error and their dB difference
# is float noise, not a width effect. Same threshold and rationale as
# plot_experiment2_metrics_modelbased.py. Masked from the curves and the table.
EXACT_RECOVERY_DB = -60.0
OOD_OFFSETS = [("sparse side, $d=-0.075$", -0.075), ("dense side, $d=+0.20$", 0.20)]
# Columns in the table / markdown: ID cost, then (robustness benefit, net) per offset.
REPORT_COLUMNS = ["ID cost"] + ["{}: {}".format(k, lab) for lab, _ in OOD_OFFSETS for k in ("robustness", "net")]

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK_PRIMARY, INK_MUTED = "#0b0b0b", "#898781"


def width_centres(run_dir: str, seed: int) -> Dict[str, float]:
    """Centre of each width's training-sparsity distribution: the fixed p for
    narrow, the midpoint of the uniform range otherwise (metadata.json)."""
    with open(os.path.join(run_dir, "data", "seed{}".format(seed), "metadata.json")) as f:
        spec = json.load(f)["train_sparsities"]
    out = {}
    for w, v in spec.items():
        out[w] = float(v) if not isinstance(v, list) else 0.5 * (float(v[0]) + float(v[1]))
    return out


def load_curves(run_dir: str):
    """{(method, width, seed): (p array, excess-over-floor array, raw NMSE array)}"""
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "results.jsonl")) if l.strip()]
    acc: Dict[Tuple[str, str, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        if r["method"] not in ("lista", "alista"):
            continue
        acc[(r["method"], r["width"], int(r["seed"]))].append(
            (float(r["test_sparsity"]),
             float(r["nmse_vs_true_mean"]) - float(r["lasso_optimal_recovery_error_mean"]),
             float(r["nmse_vs_true_mean"])))
    out = {}
    for k, v in acc.items():
        v.sort()
        p = np.array([a for a, _, _ in v]); ex = np.array([b for _, b, _ in v]); raw = np.array([c for _, _, c in v])
        out[k] = (p, ex, raw)
    return out


def common_offset_grid(ps: np.ndarray, centres: Dict[str, float]) -> np.ndarray:
    lo = max(ps.min() - c for c in centres.values())
    hi = min(ps.max() - c for c in centres.values())
    lo_g = np.ceil(lo / OFFSET_STEP) * OFFSET_STEP
    hi_g = np.floor(hi / OFFSET_STEP) * OFFSET_STEP
    return np.round(np.arange(lo_g, hi_g + 1e-9, OFFSET_STEP), 6)


def paired_diffs(curves, method, wa, wb, seeds, centres, grid):
    """(n_seeds, n_offsets) array of [wa - wb] excess-over-floor at each offset,
    plus a per-offset validity mask (False where either width's mean raw NMSE
    is in the exact-recovery regime)."""
    d, raw_a, raw_b = [], [], []
    for s in seeds:
        pa, ea, ra = curves[(method, wa, s)]
        pb, eb, rb = curves[(method, wb, s)]
        d.append(np.interp(grid + centres[wa], pa, ea) - np.interp(grid + centres[wb], pb, eb))
        raw_a.append(np.interp(grid + centres[wa], pa, ra))
        raw_b.append(np.interp(grid + centres[wb], pb, rb))
    valid = (np.mean(raw_a, axis=0) > EXACT_RECOVERY_DB) & (np.mean(raw_b, axis=0) > EXACT_RECOVERY_DB)
    return np.array(d), valid


def summarize(diff_col: np.ndarray):
    n = len(diff_col)
    mean = float(np.mean(diff_col))
    if n < 2:
        return mean, (np.nan, np.nan), np.nan, np.nan
    sem = stats.sem(diff_col)
    half = stats.t.ppf(0.975, n - 1) * sem
    p_t = stats.ttest_rel(diff_col, np.zeros_like(diff_col)).pvalue
    try:
        p_w = stats.wilcoxon(diff_col).pvalue
    except ValueError:
        p_w = np.nan
    return mean, (mean - half, mean + half), float(p_t), float(p_w)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def fmt_cell(mean, ci, p):
    if np.isnan(p):
        return "${:+.2f}$".format(mean)
    stars = "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
    return "${:+.2f}$ [{:+.2f}, {:+.2f}]{}".format(mean, ci[0], ci[1], stars)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY})
    panels = []      # (run label, method label, grid, {cmp label: (mean, lo, hi)}, n)
    table_rows = []  # (run label, method label, cmp label, n, [cells])
    md_rows = []
    for run_key, run_label, m, n in RUNS:
        run_dir = os.path.join("runs", run_key)
        if not os.path.exists(os.path.join(run_dir, "results.jsonl")):
            continue
        curves = load_curves(run_dir)
        for mkey, mlabel in METHODS:
            seeds = sorted({s for (mm, w, s) in curves if mm == mkey
                            and all((mm, ww, s) in curves for ww in WIDTHS)})
            if len(seeds) < MIN_SEEDS:
                print("skip {} {}: {} complete seed(s) < {}".format(run_key, mlabel, len(seeds), MIN_SEEDS))
                continue
            centres = width_centres(run_dir, seeds[0])
            ps = curves[(mkey, "narrow", seeds[0])][0]
            grid = common_offset_grid(ps, centres)
            panel = {}
            for wa, wb, clabel, color, ls in COMPARISONS:
                D, valid = paired_diffs(curves, mkey, wa, wb, seeds, centres, grid)
                means = np.where(valid, D.mean(axis=0), np.nan)
                half = stats.t.ppf(0.975, len(seeds) - 1) * stats.sem(D, axis=0)
                panel[clabel] = (means, means - half, means + half, color, ls)
                cells, md = [], []
                at0 = np.array([np.interp(0.0, grid, D[i]) for i in range(len(seeds))])

                def _add(col):
                    mean, ci, p_t, p_w = summarize(col)
                    cells.append(fmt_cell(mean, ci, p_t))
                    md.append("{:+.2f} [{:+.2f},{:+.2f}] p={:.3f} (W {:.3f})".format(mean, ci[0], ci[1], p_t, p_w))
                _add(at0)
                for _, off in OOD_OFFSETS:
                    i_off = int(np.argmin(np.abs(grid - off)))
                    if off < grid.min() - 1e-9 or off > grid.max() + 1e-9 or not valid[i_off]:
                        cells += ["n/a", "n/a"]; md += ["n/a (exact recovery)", "n/a"]; continue
                    net = np.array([np.interp(off, grid, D[i]) for i in range(len(seeds))])
                    _add(net - at0)   # robustness benefit: difference in degradation from own centre
                    _add(net)         # net: absolute paired difference at the offset
                table_rows.append((run_label, mlabel, clabel, len(seeds), cells))
                md_rows.append((run_key, mlabel, clabel.replace("$-$", "-"), len(seeds), md))
                # continuity with the 2026-09-06 analysis: raw NMSE at a fixed
                # in-support p (narrow's own training point) and at +0.20
                p_id = centres["narrow"]
                raw = [np.interp(p_id, curves[(mkey, wa, s)][0], curves[(mkey, wa, s)][2])
                       - np.interp(p_id, curves[(mkey, wb, s)][0], curves[(mkey, wb, s)][2]) for s in seeds]
                print("  [raw NMSE, fixed p={:.3f}] {} {} {}: {:+.2f} dB".format(
                    p_id, run_key, mlabel, clabel.replace("$-$", "-"), float(np.mean(raw))))
            panels.append((run_label, mlabel, grid, panel, len(seeds), centres, run_key, mkey))

    if not panels:
        raise SystemExit("nothing to plot")

    # ------------------------------------------------------------- figures
    run_labels = []
    for rl, *_ in panels:
        if rl not in run_labels:
            run_labels.append(rl)
    # One y-range for every panel, combined and standalone alike, so the four
    # standalone images stay comparable when placed side by side in LaTeX.
    y_lo = min(np.nanmin(lo) for _, _, _, panel, *_ in panels for (_, lo, _, _, _) in panel.values())
    y_hi = max(np.nanmax(hi) for _, _, _, panel, *_ in panels for (_, _, hi, _, _) in panel.values())
    pad = 0.05 * (y_hi - y_lo)
    ylim = (y_lo - pad, y_hi + pad)

    def draw_panel(ax, grid, panel, legend):
        style(ax)
        ax.axhline(0, color=INK_PRIMARY, lw=0.8, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
        for clabel, (means, lo, hi, color, ls) in panel.items():
            ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
            ax.plot(grid, means, color=color, lw=2, ls=ls, label=clabel, zorder=3)
        for _, off in OOD_OFFSETS:
            if grid.min() - 1e-9 <= off <= grid.max() + 1e-9:
                ax.axvline(off, color=INK_MUTED, lw=0.6, ls=(0, (1, 3)), zorder=1)
        ax.set_ylim(*ylim)
        ax.set_xlabel("offset of test $p$ from each width's own training centre", fontsize=8)
        if legend:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    os.makedirs(THESIS_FIGS_DIR, exist_ok=True)
    ylabel = "paired $\\Delta$ excess-over-floor NMSE (dB)\n$>0$: widening hurts"

    # (a) combined overview, kept for slides / quick looks
    nrows, ncols = len(run_labels), len(METHODS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.9 * nrows), sharex=False, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for rl, ml, grid, panel, nseeds, centres, run_key, mkey in panels:
        ax = axes[run_labels.index(rl)][[l for _, l in METHODS].index(ml)]
        draw_panel(ax, grid, panel, legend=False)
        ax.set_title("{} -- {}  ({} seeds)".format(ml, rl, nseeds), fontsize=8.5, loc="left")
    for r in range(nrows):
        axes[r][0].set_ylabel(ylabel, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Experiment 2: what widening the training-sparsity distribution costs in-distribution and buys out-of-distribution\n"
                 "mean over seeds with 95% t-interval; differences are paired by seed (same $A$, instances and $x_0$) and taken "
                 "against the LASSO-optimal floor at the same test $p$", fontsize=9, y=1.0)
    fig.tight_layout()
    stem = os.path.join(THESIS_FIGS_DIR, "width_tradeoff_nmse")
    fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("wrote", stem + ".png")
    plt.close(fig)

    # (b) one standalone image per panel (no title: the LaTeX subcaption carries
    #     it) + a subfigure snippet to \input from the thesis
    sub_entries = []
    for rl, ml, grid, panel, nseeds, centres, run_key, mkey in panels:
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        fig.patch.set_facecolor(SURFACE)
        draw_panel(ax, grid, panel, legend=True)
        ax.set_ylabel(ylabel, fontsize=8)
        fig.tight_layout()
        name = "width_tradeoff_nmse_{}_{}".format(run_key, mkey)
        stem = os.path.join(THESIS_FIGS_DIR, name)
        fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", stem + ".png")
        plt.close(fig)
        sub_entries.append((name, "{}, {} ({} seeds)".format(ml, rl, nseeds),
                            "fig:exp2-width-tradeoff-{}-{}".format(run_key.replace("_", "-"), mkey)))

    fig_lines = [
        "% Multi-panel figure for the Experiment-2 width tradeoff, generated by",
        "% plot_experiment2_width_tradeoff.py. Requires \\usepackage{graphicx} and",
        "% \\usepackage{subcaption}. Paths are relative to the repo root (Figs/...);",
        "% adjust \\graphicspath or the paths below to match the Overleaf project.",
        "% Include with: \\input{experiment2_width_tradeoff_figure}",
        r"\begin{figure}[htbp]", r"\centering"]
    for i, (name, cap, lab) in enumerate(sub_entries):
        fig_lines += [r"\begin{subfigure}[t]{0.48\linewidth}", r"  \centering",
                      r"  \includegraphics[width=\linewidth]{Figs/experiment2/" + name + ".png}",
                      r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}"
                      + (r"\hfill" if i % 2 == 0 else r"\\[6pt]")]
    fig_lines += [
        r"\caption{Experiment 2: what widening the training-sparsity distribution costs in-distribution "
        r"and buys out-of-distribution, for LISTA and ALISTA. Each curve is the mean over seeds of the paired "
        r"difference (wide $-$ narrow, medium $-$ narrow) in excess-over-floor NMSE, i.e.\ NMSE vs.\ "
        r"$x_\mathrm{true}$ minus the LASSO-optimal recovery error at the same test $p$, plotted against the "
        r"offset of the test sparsity from each width's own training centre; bands are 95\% $t$-intervals "
        r"over seeds. Positive values mean the wider training distribution is worse. The value at offset 0 is "
        r"the in-distribution cost of widening; the curve relative to that value is the robustness benefit. "
        r"Dotted verticals mark the offsets reported in Table~\ref{tab:exp2-width-tradeoff}. Points where a "
        r"width is in the exact-recovery regime (raw NMSE below $-60$ dB) are omitted.}",
        r"\label{fig:exp2-width-tradeoff}", r"\end{figure}"]
    with open(OUT_FIG_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(fig_lines) + "\n")
    print("wrote", OUT_FIG_TEX)

    # -------------------------------------------------------------- table
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Experiment 2, effect of widening the training-sparsity distribution on recovery "
             r"(NMSE vs.\ $x_\mathrm{true}$, dB), LISTA and ALISTA. Each cell is the mean paired difference "
             r"over seeds with its 95\% $t$-interval; differences are taken between widths at the same offset "
             r"from each width's own training centre and against the LASSO-optimal recovery floor at that "
             r"test $p$, so neither the centre of the training distribution nor the difficulty of the test "
             r"instances confounds the comparison. ID cost = difference at each width's own centre; "
             r"robustness = difference in degradation from own centre at offset $d$ (negative: the wider "
             r"training degrades less); net = ID cost $+$ robustness = difference at offset $d$. "
             r"Positive = widening hurts. "
             r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (paired two-sided $t$-test).}",
             r"\label{tab:exp2-width-tradeoff}",
             r"\begin{tabular}{@{} l l l r rr rr @{}}", r"\toprule",
             r"& & & & \multicolumn{2}{c}{" + OOD_OFFSETS[0][0] + r"} & \multicolumn{2}{c}{" + OOD_OFFSETS[1][0] + r"} \\",
             r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
             r"Run & Method & Comparison & ID cost & robustness & net & robustness & net \\", r"\midrule"]
    last_run = None
    for rl, ml, cl, nseeds, cells in table_rows:
        if last_run is not None and rl != last_run:
            lines.append(r"\addlinespace")
        lines.append("{} & {} & {} & {} \\\\".format(rl if rl != last_run else "", ml, cl, " & ".join(cells)))
        last_run = rl
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"Offsets are interpolated linearly between the test-sparsity grid points, so the OOD cells are "
              r"approximate to the grid resolution. n/a: at least one width is in the exact-recovery "
              r"regime there (raw NMSE below $-60$ dB), where the dB difference is numerical noise. Generated by \texttt{plot\_experiment2\_width\_tradeoff.py} "
              r"from \texttt{results.jsonl}; runs with fewer than " + str(MIN_SEEDS) + " complete seeds are omitted.",
              r"\end{minipage}", r"\end{table}"]
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_TEX)

    print("\n| run | method | comparison | n | " + " | ".join(REPORT_COLUMNS) + " |")
    print("|---|---|---|---|" + "---|" * len(REPORT_COLUMNS))
    for rk, ml, cl, nseeds, md in md_rows:
        print("| {} | {} | {} | {} | {} |".format(rk, ml, cl, nseeds, " | ".join(md)))


if __name__ == "__main__":
    main()
