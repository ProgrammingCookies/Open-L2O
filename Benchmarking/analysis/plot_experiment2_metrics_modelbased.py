"""Figures for Experiment 2's three remaining Table-4 metrics, model-based only:

  1. Suboptimality gap  g_q(x) = f_q(x) - f*_q            (pre-study Eq. 9)
  2. Modified relative loss  R_f                          (pre-study Eq. 10)
  3. NMSE vs. the true signal x_true, in dB               (pre-study Eq. 11)
  4. NMSE vs. the LASSO-optimal solution x*               (the recoverability floor)

for runs/exp2_real (m,n)=(210,300) SNR 40 dB, runs/exp2_try2 (m,n)=(25,50)
noiseless, and runs/exp2_try2_moredata (that same setting with the training set
doubled to 64 000 instances, seeds 21-23 only). Twelve figures:

    experiment2_modelbased_{subopt_gap,rel_loss,nmse_true}.png
    experiment2_try2_modelbased_{subopt_gap,rel_loss,nmse_true}.png
    experiment2_moredata_modelbased_{subopt_gap,rel_loss,nmse_true}.png
    experiment2_moredata_effect.png
    experiment2_moredata_effect_nmse_true.png
    experiment2_lasso_optimal_recovery_error.png

Scope: lista/alista only. Every model-free row in both runs was produced by an
evaluation that loaded a checkpoint written through the Keras-3 bare-`v.name`
collision, so those rows are invalid. Reads
results.jsonl directly and NOT summary.json -- both runs' summary.json predate
the 2026-09-06 pruning of those rows and still contain them.

Two deliberate departures from plot_ood_degradation.py (which in 2026-09 was a
pair of per-run forks), both consequences of what that script's own output got
wrong:

* SHARED Y-AXIS across the two method panels. The 2026-09-02 figures drew one
  standalone figure per method with independently autoscaled log axes, which
  made the width curves of whichever method had the smaller dynamic range look
  far more separated. Measured on exp2_try2's R_f, the two methods' width
  spreads are all but identical (LISTA 0.0614 vs ALISTA 0.0602 decades) yet
  filled 30.4% and 3.3% of their respective frames -- a 9x difference in
  apparent separation that is entirely an artefact of the autoscale. Panels
  here share one scale so the comparison is real; each panel is annotated with
  its own data range so the compression stays legible.

* ONE table-driven script rather than a per-run fork. The earlier fork was
  justified because exp2_try2 differs in analysis semantics; here those
  differences (widths, sparsity grid, seeds, labels) are exactly what RUNS
  parameterises, so a fork would duplicate the plotting code verbatim.

Same median + IQR (25th/75th) convention and palette as the existing OOD
figures, for visual consistency across the thesis's figure set.
"""
from __future__ import annotations

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

# Figures go to the thesis figures directory only (repo root, sibling of
# Benchmarking/). The older plot scripts also drop a copy alongside themselves;
# these don't -- Figs/ is the single destination.
THESIS_FIGS_DIR = os.path.join("..", "Figs")
MULTIFIG_DIR = os.path.join(THESIS_FIGS_DIR, "multifigure")  # all subfigure snippets live here

METHODS = [("lista", "LISTA"), ("alista", "ALISTA")]
# exp2_try3_210x300 also carries the post-checkpoint-fix RNNProp rows (valid,
# unlike exp2_real's), so that run gets a third panel. Per-run "methods"
# overrides METHODS; the default keeps every older figure unchanged.
METHODS_WITH_RNNPROP = METHODS + [("l2o-rnnprop", "L2O-RNNProp")]

# Colors: dataviz skill's validated 3-slot categorical order (blue/orange/aqua),
# unchanged from plot_ood_degradation.py so a width keeps one colour everywhere.
NARROW_C, MEDIUM_C, WIDE_C = "#2a78d6", "#eb6834", "#1baf7a"

TRY2_WIDTHS = [
    # (key, label, colour, linestyle, marker, training range, fixed point)
    ("narrow", r"Narrow ($p = 0.175$)", NARROW_C, "-", "o", None, 0.175),
    ("medium", r"Medium ($p \in [0.1125, 0.2375]$)", MEDIUM_C, "--", "s", (0.1125, 0.2375), None),
    ("wide", r"Wide ($p \in [0.05, 0.30]$)", WIDE_C, ":", "^", (0.05, 0.30), None),
]

# Run identity -- results path, output stem, width triad, method panels and
# band -- comes from the shared registry, so a run is defined in exactly one
# place. Only the subtitle wording stays here: this script states the training
# set size and seed count, which the other figures do not.
_SUBTITLES = {
    "exp2_real": r"$(m,n) = (210, 300)$, SNR 40 dB, 32 000 training instances, 10 seeds",
    "exp2_try2": r"$(m,n) = (25, 50)$, noiseless, 32 000 training instances, 10 seeds",
    # The deconfounded (210,300) run: exp2_try2's centred width scheme at the
    # pre-study's dims, noiseless; the thesis's headline run. Still growing, so
    # the subtitle reads the seed count from the file rather than stating it.
    "exp2_try3_210x300": r"$(m,n) = (210, 300)$, noiseless, 32 000 training instances, {n_seeds} seeds",
    # The lasso_train_objective ablation on the headline run: LISTA/ALISTA retrained
    # with --lasso_train_objective=scoring (0.5*||Ax-b||^2 + lam*||x||_1, the scored
    # objective) instead of the released LassoLoss (0.25*||Ax-b||^2 +
    # lam*||x||_1, an effective 2*lam -- experiments_analysis.md 2026-09-20).
    # Same protocol, data and test files as exp2_try3_210x300 (byte-identical,
    # verified); the paper arm is that run's own lista/alista rows. Its band is
    # "auto" in the registry: min-max while n < 5, IQR after.
    "exp2_try3_210x300_scoringonly":
        r"$(m,n) = (210, 300)$, noiseless, 32 000 training instances, {n_seeds} seeds, "
        r"scoring loss ($\frac{{1}}{{2}}\|Ax-b\|^2 + \lambda\|x\|_1$)",
    # The 2x-training-data follow-up, paired against exp2_try2 seed by seed.
    "exp2_try2_moredata":
        r"$(m,n) = (25, 50)$, noiseless, 64 000 training instances (2$\times$), 3 seeds",
}

RUNS = {
    key: {
        "results": run_specs.RUN_SPECS[key].results,
        "stem": run_specs.RUN_SPECS[key].figure_stem,
        "subtitle": subtitle,
        "band": run_specs.RUN_SPECS[key].band,
        "widths": run_specs.RUN_SPECS[key].plot_widths,
        "methods": run_specs.RUN_SPECS[key].panel_methods,
    }
    for key, subtitle in _SUBTITLES.items()
}

# The 2x-data comparison: (baseline run, doubled run). Paired seed-by-seed.
MOREDATA_PAIR = ("exp2_try2", "exp2_try2_moredata")

# Cells where the baseline already recovers x_true to better than this are in
# the exact-recovery regime: both runs are numerically at zero error, so the dB
# difference between them is dominated by float noise, not by anything the
# training set did. ALISTA hits this at p <= 0.10 on (25,50) noiseless
# (-61 to -97 dB); LISTA never does (its best is about -32 dB). Left in, these
# cells alone swing ALISTA-narrow's mean from ~0 to +4.4 dB and stretch the
# shared y-axis until every other curve reads as flat. Excluded from the NMSE
# effect figure and its summary, and the exclusion is stated on the figure.
EXACT_RECOVERY_DB = -60.0

# (results field, output suffix, y-axis label, range-note formatter, y scale)
# Eq. 11 is reported in dB, which is already a log quantity -- plotting it on a
# log axis would take the logarithm twice, so it gets a linear scale.
METRICS = [
    ("suboptimality_gap_mean", "subopt_gap",
     r"Suboptimality gap $g_q(x) = f_q(x) - f^*_q$ (Eq. 9)",
     lambda v: "{:.2e}".format(v), "log"),
    ("modified_relative_loss", "rel_loss",
     r"Modified relative loss $R_f$ (Eq. 10)",
     lambda v: "{:.4f}".format(v), "log"),
    ("nmse_vs_true_mean", "nmse_true",
     "NMSE vs. the true signal $x_\\mathrm{true}$, dB (Eq. 11)",
     lambda v: "{:.1f} dB".format(v), "linear"),
]

# dataviz skill chrome: light chart surface + recessive gridlines/ink.
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def load_rows(path, methods=METHODS):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keep = {m for m, _ in methods}
    dropped = sorted({r["method"] for r in rows} - keep)
    if dropped:
        print("  ignoring methods not plotted for this run: %s" % dropped)
    rows = [r for r in rows if r["method"] in keep]
    # Only seeds present in EVERY (method, width) cell -- a sweep stopped
    # mid-seed otherwise puts one method's panel on N seeds and another's on
    # N-1 within the same figure (see core/seedsets.py).
    rows, _, _ = filter_to_complete_seeds(rows, methods=sorted(keep), label=path)
    return rows


BAND_LABEL = {"iqr": "IQR band", "minmax": "min-max band"}


def _summarise(values, band):
    v = np.asarray(values, dtype=float)
    if band == "minmax":
        return {"median": float(np.median(v)), "lo": float(v.min()), "hi": float(v.max())}
    return {"median": float(np.median(v)),
            "lo": float(np.percentile(v, 25)),
            "hi": float(np.percentile(v, 75))}


def collapse(rows, field, key_fn, band="iqr"):
    """key -> {test_sparsity: {median, lo, hi}} over the per-seed values."""
    raw = {}
    for row in rows:
        raw.setdefault(key_fn(row), {}).setdefault(
            round(row["test_sparsity"], 4), []).append(row[field])
    return {key: {p: _summarise(v, band) for p, v in by_p.items()}
            for key, by_p in raw.items()}


def style_axes(ax, log=True):
    ax.set_facecolor(SURFACE)
    if log:
        ax.set_yscale("log")
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)


def figs_subdir(name):
    """The user keeps Figs/ sorted into per-run subfolders (2026-09-15):
    experiment2_try2_* -> Figs/exp2_try2, experiment2_moredata_* ->
    Figs/experiment2_moredata, everything else from this script ->
    Figs/experiment2. Mirror that so re-running never scatters copies at the
    Figs/ root."""
    if name.startswith("experiment2_try3_210x300_scoringonly"):
        return "exp2_try3_210x300_scoringonly"
    if name.startswith("experiment2_try3_210x300"):
        return "exp2_try3_210x300"
    if name.startswith("experiment2_try2"):
        return "exp2_try2"
    if name.startswith("experiment2_moredata"):
        return "experiment2_moredata"
    return "experiment2"


def save(fig, name):
    out_dir = os.path.join(THESIS_FIGS_DIR, figs_subdir(name))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, name)
    fig.savefig(stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("  wrote", stem + ".png")
    plt.close(fig)


def write_subfigure_tex(stem_name, entries, caption, label):
    """A \\input-able figure with one subfigure per standalone panel image,
    same convention as plot_experiment2_width_tradeoff.py and
    plot_compute_matched_rf.py. Image paths are relative to the repo root."""
    sub = figs_subdir(stem_name)
    lines = ["% Multi-panel figure generated by plot_experiment2_metrics_modelbased.py.",
             "% Requires \\usepackage{graphicx} and \\usepackage{subcaption}. Paths are",
             "% relative to the repo root (Figs/...); adjust \\graphicspath or the paths",
             "% below to match the Overleaf project.",
             "% Include with: \\input{Figs/multifigure/" + stem_name + "_figure}",
             r"\begin{figure}[htbp]", r"\centering"]
    for i, (name, cap, lab) in enumerate(entries):
        lines += [r"\begin{subfigure}[t]{0.48\linewidth}", r"  \centering",
                  r"  \includegraphics[width=\linewidth]{Figs/" + sub + "/" + name + ".png}",
                  r"  \caption{" + cap + "}", r"  \label{" + lab + "}", r"\end{subfigure}"
                  + (r"\hfill" if i % 2 == 0 and i + 1 < len(entries) else "")]
    lines += [r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure}"]
    os.makedirs(MULTIFIG_DIR, exist_ok=True)
    out = os.path.join(MULTIFIG_DIR, stem_name + "_figure.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", out)


def plot_metric(run_key, cfg, rows, field, suffix, ylabel, fmt, scale):
    methods = cfg.get("methods", METHODS)
    n_seeds = len({r["seed"] for r in rows})
    band = cfg["band"]
    if band == "auto":   # IQR needs enough seeds to mean anything; see exp2_try2_moredata
        band = "minmax" if n_seeds < 5 else "iqr"
    subtitle = cfg["subtitle"].format(n_seeds=n_seeds)
    index = collapse(rows, field, lambda r: (r["method"], r["width"]), band)

    # One shared y-range over both panels' bands -- see module docstring.
    lo = min(s["lo"] for by_p in index.values() for s in by_p.values())
    hi = max(s["hi"] for by_p in index.values() for s in by_p.values())
    if scale == "log":
        pad = (hi / lo) ** 0.06
        ylim = (lo / pad, hi * pad)
    else:
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)

    def draw_panel(ax, method_key, method_label, ylabel_on):
        style_axes(ax, log=(scale == "log"))

        for _, _, color, _, _, train_range, _ in cfg["widths"]:
            if train_range is not None:
                ax.axvspan(train_range[0], train_range[1], color=color, alpha=0.07,
                           zorder=1, linewidth=0)

        panel_lo, panel_hi = np.inf, -np.inf
        for width_key, width_label, color, ls, marker, _, fixed_p in cfg["widths"]:
            by_p = index.get((method_key, width_key), {})
            ps = sorted(by_p)
            med = np.array([by_p[p]["median"] for p in ps])
            band_lo = np.array([by_p[p]["lo"] for p in ps])
            band_hi = np.array([by_p[p]["hi"] for p in ps])
            panel_lo, panel_hi = min(panel_lo, med.min()), max(panel_hi, med.max())

            ax.plot(ps, med, color=color, linestyle=ls, marker=marker, markersize=5,
                    linewidth=2, zorder=3, label=width_label)
            ax.fill_between(ps, band_lo, band_hi, color=color, alpha=0.15, zorder=2,
                            linewidth=0)
            if fixed_p is not None:
                ax.axvline(fixed_p, color=color, linestyle=ls, linewidth=1, alpha=0.5,
                           zorder=1)

        ax.set_ylim(*ylim)
        ax.set_title(method_label, fontsize=12, color=INK_PRIMARY, loc="left")
        # Panels share a scale, so state each one's own span -- otherwise the
        # compressed panel reads as "no variation" rather than "less variation".
        # It goes in the right-aligned title slot rather than inside the axes:
        # anywhere in the data area it eventually collides with the legend or a
        # curve, and which corner is free changes from figure to figure.
        ax.set_title("range {} to {}".format(fmt(panel_lo), fmt(panel_hi)),
                     fontsize=8, color=INK_MUTED, loc="right")
        ax.set_xlabel("Test sparsity $p$", color=INK_PRIMARY)
        if ylabel_on:
            ax.set_ylabel(ylabel + "\n" + "median, {}{}".format(
                BAND_LABEL[band], " (log scale)" if scale == "log" else ""),
                color=INK_PRIMARY, fontsize=9)
        return panel_lo, panel_hi

    # (a) combined two-panel figure, as before
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, (method_key, method_label)) in enumerate(zip(axes, methods)):
        draw_panel(ax, method_key, method_label, ylabel_on=(i == 0))
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle(subtitle, fontsize=10, color=INK_MUTED, x=0.005, ha="left")
    fig.tight_layout()
    stem_name = "{}_{}".format(cfg["stem"], suffix)
    save(fig, stem_name)

    # (b) one standalone image per method panel (same shared y-range, so the
    #     two images stay comparable side by side) + a subfigure snippet
    entries = []
    for method_key, method_label in methods:
        fig, ax = plt.subplots(figsize=(5.2, 4.3))
        fig.patch.set_facecolor(SURFACE)
        draw_panel(ax, method_key, method_label, ylabel_on=True)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()
        name = "{}_{}".format(stem_name, method_key)
        save(fig, name)
        entries.append((name, method_label,
                        "fig:{}-{}".format(stem_name.replace("_", "-"), method_key)))
    caption = (ylabel.replace("\\", "\\") + r" vs.\ test sparsity $p$, " + subtitle +
               r". Median over seeds with " + BAND_LABEL[band] + r"; one curve per training-sparsity "
               r"width, with each width's training support shaded in its colour and the fixed narrow "
               r"training point marked by a vertical line. Both panels share the y-axis; each panel's "
               r"own data range is stated in its top-right corner.")
    write_subfigure_tex(stem_name, entries, caption, "fig:" + stem_name.replace("_", "-"))


def plot_recovery_floor():
    """NMSE(x*, x_true): depends only on (A, b, x_true, lam) via the shared
    FISTA-solved x*, so it is identical across every method and width. One
    curve per run, not per (method, width)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    style_axes(ax, log=False)

    # Underscores are plain text here (only the $...$ spans are mathtext), so
    # they must NOT be backslash-escaped -- matplotlib would render the
    # backslash literally.
    styles = {
        "exp2_real": ("exp2_real: $(m,n)=(210,300)$, SNR 40 dB", "#2a78d6", "-", "o"),
        "exp2_try2": ("exp2_try2: $(m,n)=(25,50)$, noiseless", "#eb6834", "--", "s"),
    }
    # exp2_try2_moredata is deliberately absent: its test files are byte-
    # identical to exp2_try2's for the seeds it shares, so its floor curve is
    # the same curve. Training-set size cannot move a quantity computed from
    # the test data alone.
    for run_key in ("exp2_real", "exp2_try2"):
        cfg = RUNS[run_key]
        rows = load_rows(cfg["results"])
        # Collapse over method+width too: the value is constant across them, so
        # every (seed, p) contributes the same number once per (method, width).
        by_p = collapse(rows, "lasso_optimal_recovery_error_mean",
                        lambda r: "all", cfg["band"])["all"]
        ps = sorted(by_p)
        med = np.array([by_p[p]["median"] for p in ps])
        band_lo = np.array([by_p[p]["lo"] for p in ps])
        band_hi = np.array([by_p[p]["hi"] for p in ps])
        label, color, ls, marker = styles[run_key]
        ax.plot(ps, med, color=color, linestyle=ls, marker=marker, markersize=5,
                linewidth=2, zorder=3, label=label)
        ax.fill_between(ps, band_lo, band_hi, color=color, alpha=0.15, zorder=2,
                        linewidth=0)

    # x = 0 scores exactly 0 dB under Eq. 11, so the line is the do-nothing
    # predictor: at a sparsity where the floor approaches it, the LASSO model
    # itself cannot recover the signal and no optimizer could.
    ax.axhline(0.0, color=INK_MUTED, linestyle="-.", linewidth=1, zorder=1)
    ax.text(0.235, -32, "0 dB = the $x = 0$ predictor;\nfloor approaching it means the\n"
            "sparsity level is unrecoverable",
            fontsize=8, color=INK_MUTED, ha="left", va="top")

    ax.set_xlabel("Test sparsity $p$", color=INK_PRIMARY)
    ax.set_ylabel("NMSE of the LASSO optimum $x^*$ vs. $x_\\mathrm{true}$, dB\n"
                  "median, IQR band", color=INK_PRIMARY, fontsize=9)
    ax.set_title("Recoverability floor", fontsize=12, color=INK_PRIMARY, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "experiment2_lasso_optimal_recovery_error")


def plot_moredata_effect(field, mode, suffix, ylabel, summary_label):
    """Did doubling the training set help? Paired seed-by-seed comparison.

    The two runs share seeds 21-23 and their test files are byte-identical
    there, so each (method, width, seed, p) cell has an exact counterpart and
    this is a true paired measurement -- not a comparison of two
    independently-drawn samples.

    mode "ratio": 64k / 32k on a log axis; below 1 means more data helped.
    mode "diff":  64k - 32k in dB on a linear axis; below 0 means more data
                  helped (NMSE is better when lower).

    Eq. 9 gets no figure of its own here: suboptimality_gap_mean =
    R_f x mean(f*_q), and mean(f*_q) depends solely on the test file, so within
    a (seed, p) the two are exactly proportional (verified: the ratio is
    constant across every method and width) and their paired ratios are
    identical.
    """
    base_key, more_key = MOREDATA_PAIR
    base, more = {}, {}
    for store, key in ((base, base_key), (more, more_key)):
        for r in load_rows(RUNS[key]["results"]):
            store[(r["method"], r["width"], r["seed"],
                   round(r["test_sparsity"], 4))] = r[field]

    seeds = sorted({k[2] for k in more})

    # Exclude whole (method, test sparsity) columns, not individual cells: at
    # the threshold's boundary only some seeds fall past it, and keeping the
    # rest would build that point from a biased subset of seeds.
    excluded = set()
    if mode == "diff":
        excluded = {(k[0], k[3]) for k in more
                    if k in base and base[k] < EXACT_RECOVERY_DB}

    ratios = {}
    for k, v in more.items():
        if k not in base or (k[0], k[3]) in excluded:
            continue
        if mode == "ratio":
            if base[k] <= 0:
                continue
            value = v / base[k]
        else:
            value = v - base[k]
        ratios.setdefault((k[0], k[1]), {}).setdefault(k[3], []).append(value)
    if excluded:
        print("    (excluding %d exact-recovery columns, baseline NMSE < %.0f dB: %s)"
              % (len(excluded), EXACT_RECOVERY_DB,
                 sorted({"%s p=%g" % e for e in excluded})))

    neutral = 1.0 if mode == "ratio" else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    print("  paired %s (64k vs 32k), over seeds %s:" % (summary_label, seeds))
    for ax, (method_key, method_label) in zip(axes, METHODS):
        style_axes(ax, log=(mode == "ratio"))
        ax.axhline(neutral, color=INK_MUTED, linestyle="-.", linewidth=1, zorder=1)

        for _, _, color, _, _, train_range, _ in TRY2_WIDTHS:
            if train_range is not None:
                ax.axvspan(train_range[0], train_range[1], color=color, alpha=0.07,
                           zorder=1, linewidth=0)

        for width_key, width_label, color, ls, marker, _, fixed_p in TRY2_WIDTHS:
            by_p = ratios.get((method_key, width_key), {})
            ps = sorted(by_p)
            med = np.array([float(np.median(by_p[p])) for p in ps])
            lo = np.array([float(np.min(by_p[p])) for p in ps])
            hi = np.array([float(np.max(by_p[p])) for p in ps])
            ax.plot(ps, med, color=color, linestyle=ls, marker=marker, markersize=5,
                    linewidth=2, zorder=3, label=width_label)
            ax.fill_between(ps, lo, hi, color=color, alpha=0.15, zorder=2, linewidth=0)
            if fixed_p is not None:
                ax.axvline(fixed_p, color=color, linestyle=ls, linewidth=1, alpha=0.5,
                           zorder=1)
            agg = (float(np.exp(np.mean(np.log(med)))) if mode == "ratio"
                   else float(np.mean(med)))
            print("    %-7s %-7s %s = %.3f" % (method_key, width_key,
                                               "geo. mean over p" if mode == "ratio"
                                               else "mean over p, dB", agg))

        ax.set_title(method_label, fontsize=12, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Test sparsity $p$", color=INK_PRIMARY)

    axes[0].set_ylabel(ylabel, color=INK_PRIMARY, fontsize=9)
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    note = "below {} = doubling the training set helped".format(
        "1.0" if mode == "ratio" else "0 dB")
    if excluded:
        note += ("\nomits {} ALISTA points at $p \\leq 0.10$: both runs already "
                 "recover $x_\\mathrm{{true}}$\nto better than {:.0f} dB there, so the "
                 "dB difference is float noise".format(len(excluded), EXACT_RECOVERY_DB))
    axes[0].annotate(note, xy=(0.02, 0.03), xycoords="axes fraction", ha="left",
                     va="bottom", fontsize=8, color=INK_MUTED)
    fig.suptitle(r"Effect of doubling the training set -- $(m,n) = (25, 50)$, noiseless, "
                 r"seeds 21-23 paired", fontsize=10, color=INK_MUTED, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, suffix)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    for run_key, cfg in RUNS.items():
        print("%s:" % run_key)
        rows = load_rows(cfg["results"], cfg.get("methods", METHODS))
        for field, suffix, ylabel, fmt, scale in METRICS:
            plot_metric(run_key, cfg, rows, field, suffix, ylabel, fmt, scale)

    print("recoverability floor (method- and width-independent):")
    plot_recovery_floor()

    print("2x-training-data effect (paired against exp2_try2):")
    plot_moredata_effect(
        "modified_relative_loss", "ratio", "experiment2_moredata_effect",
        "Modified relative loss $R_f$ with 64 000 training instances,\n"
        "relative to 32 000 (median, min-max over 3 paired seeds)",
        "R_f ratio")
    plot_moredata_effect(
        "nmse_vs_true_mean", "diff", "experiment2_moredata_effect_nmse_true",
        "Change in NMSE vs. $x_\\mathrm{true}$ from doubling the training set, dB\n"
        "(median, min-max over 3 paired seeds)",
        "NMSE change, dB")


if __name__ == "__main__":
    main()
