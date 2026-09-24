"""
One place where each Experiment-2 run's identity lives: paths, method columns,
width triad, sparsity grid, and the LaTeX chrome that names the run.

Why this exists. `generate_result_tables.py`, `generate_result_tables_try2_modelbased.py`
and `generate_result_tables_try3.py` were three copies of one ~290-line script
whose only real differences were the values below; likewise
`plot_ood_degradation.py` and its `_try2_modelbased` fork. Each fork's docstring
argued, correctly, that the runs differ in more than a results-dir swap -- but
those differences are *data*, and the arguments belong on the data. They are
preserved as comments on the specs here.

A run's spec is the single definition of:
  * where its rows are (`results`, `baselines`),
  * which methods are legitimate columns for it and why the others are not,
  * the width triad and test-sparsity grid it was generated on,
  * how its tables and figures are named, labelled and captioned.

Adding a run = one entry. Nothing else changes.

Not part of the platform (`core/`, `adapters/`): this is thesis-specific
bookkeeping about particular sweeps, the same way `experiments/` is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

TABLES_DIR = os.path.join("..", "Figs", "tables")      # all result tables live here
MULTIFIG_DIR = os.path.join("..", "Figs", "multifigure")  # subfigure .tex snippets
FIGS_DIR = os.path.join("..", "Figs")

# Table-4's four headline metrics, in the order they are reported.
METRIC_FIELDS = [
    "suboptimality_gap_mean",
    "modified_relative_loss",
    "nmse_vs_true_mean",
    "lasso_optimal_recovery_error_mean",
]

# Classical reference columns from classical_baselines.py (2026-09-13): plain
# FISTA at 16 iterations (= LISTA/ALISTA's depth) and at 1000 (= the model-free
# evaluation horizon). Width-independent, so they are keyed with width "all".
BASELINE_KEYS = {"fista@16", "fista@1000"}
FISTA_COLUMNS = [("fista@16", "FISTA-16"), ("fista@1000", "FISTA-1000")]

MODEL_BASED = [("lista", "LISTA"), ("alista", "ALISTA")]
# Model-free arm evaluated at 1000 learned steps (the pre-study's own
# per-protocol budget). Only legitimate in runs trained AND evaluated after the
# 2026-09-06 checkpoint fix.
RNNPROP = [("l2o-rnnprop", "L2O-RNNProp")]

# The two width schemes. The pre-study's original triad is centred on p=0.10 for
# narrow, which puts narrow's centre outside medium's and wide's ranges and so
# confounds "width" with "centre"; data/lasso.py's 2026-08-31 update re-centres
# all three on p=0.175 so that only the width differs.
UNCENTRED_WIDTHS = [
    ("narrow", r"Narrow ($p = 0.10$, fixed)"),
    ("medium", r"Medium ($p \sim \mathcal{U}(0.075, 0.20)$)"),
    ("wide", r"Wide ($p \sim \mathcal{U}(0.05, 0.30)$)"),
]
CENTRED_WIDTHS = [
    ("narrow", r"Narrow ($p = 0.175$, fixed)"),
    ("medium", r"Medium ($p \sim \mathcal{U}(0.1125, 0.2375)$)"),
    ("wide", r"Wide ($p \sim \mathcal{U}(0.05, 0.3)$)"),
]

# The uncentred design predates the 11-point grid; the centred one adds each
# width's own exact centre point, which is what makes an ID-cost reading possible.
UNCENTRED_SPARSITIES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40]
CENTRED_SPARSITIES = [0.02, 0.05, 0.10, 0.1125, 0.15, 0.175, 0.20, 0.2375, 0.30, 0.35, 0.40]

# Colours: the dataviz palette's validated 3-slot categorical order
# (blue/orange/aqua), which passes CVD and normal-vision separation on the light
# chart surface. Same mapping in every figure, so a width means one colour
# throughout the thesis.
NARROW_C, MEDIUM_C, WIDE_C = "#2a78d6", "#eb6834", "#1baf7a"

# (key, label, colour, linestyle, marker, training range to shade, fixed centre)
PLOT_WIDTHS_UNCENTRED = [
    ("narrow", r"Narrow ($p = 0.10$)", NARROW_C, "-", "o", None, 0.10),
    ("medium", r"Medium ($p \in [0.075, 0.20]$)", MEDIUM_C, "--", "s", (0.075, 0.20), None),
    ("wide", r"Wide ($p \in [0.05, 0.30]$)", WIDE_C, ":", "^", (0.05, 0.30), None),
]
PLOT_WIDTHS_CENTRED = [
    ("narrow", r"Narrow ($p = 0.175$)", NARROW_C, "-", "o", None, 0.175),
    ("medium", r"Medium ($p \in [0.1125, 0.2375]$)", MEDIUM_C, "--", "s", (0.1125, 0.2375), None),
    ("wide", r"Wide ($p \in [0.05, 0.30]$)", WIDE_C, ":", "^", (0.05, 0.30), None),
]


@dataclass(frozen=True)
class RunSpec:
    """Everything that distinguishes one Experiment-2 run from another."""

    key: str                       # the runs/ directory name
    methods: Sequence[Tuple[str, str]]   # (result key, column/panel label)
    widths: Sequence[Tuple[str, str]]    # (width key, LaTeX label) for tables
    plot_widths: Sequence[tuple]         # the richer tuple used by figures
    sparsities: Sequence[float]
    sparsity_round: int            # test_sparsity rounding: 2 decimals for the
                                   # uncentred grid, 4 for the centred one
                                   # (0.1125 and 0.2375 collide at 2)
    tex_stem: str                  # stem for this run's result tables and OOD figures
    label_prefix: str              # LaTeX \label prefix, e.g. "exp2try3"
    scale_caption: str             # "(210,300) scale, noiseless, centred widths"
    method_caption: str            # what the columns are, for the table caption
    subtitle: str                  # one-line run description for figure subtitles
    fig_stem: str = ""             # stem for the Table-4 metric figures, which do
                                   # not always match tex_stem: exp2_real's tables
                                   # are "experiment2_*" while its figures are
                                   # "experiment2_modelbased_*". Defaults to tex_stem.
    seed_count_from_data: bool = False   # caption reads n from the rows (growing run)
    m: int = 0                     # measurement dimension; 0 = not recorded
    n: int = 0                     # signal dimension
    train_size: int = 0            # training instances per width; 0 = not recorded
    standalone_tables: bool = True # False = a companion run that only appears
                                   # inside another figure (no table/OOD of its own)
    model_free_arm: str = ""       # a model-free method with VALID rows in this run,
                                   # even if it has too few seeds to be a table
                                   # column. `methods` answers "what may be a
                                   # column"; this answers "what may be plotted at
                                   # all", and they genuinely differ: exp2_try2's
                                   # l2o-rnnprop has 2 post-checkpoint-fix seeds --
                                   # not enough for a mean (std) column, but exactly
                                   # what the per-seed spread figure is for.
    band: str = "iqr"              # figure band: "iqr" or "minmax"
    notes: str = ""                # why this run exists / what it must not be used for

    @property
    def run_dir(self) -> str:
        return os.path.join("runs", self.key)

    @property
    def results(self) -> str:
        return os.path.join(self.run_dir, "results.jsonl")

    @property
    def baselines(self) -> str:
        return os.path.join(self.run_dir, "classical_baselines.jsonl")

    @property
    def tables_path(self) -> str:
        return os.path.join(TABLES_DIR, self.tex_stem + "_results_tables.tex")

    def format_p(self, p: float) -> str:
        """Renders a test sparsity for the first column of a table.

        The two width schemes need different formats and the tables must not
        silently change: the uncentred grid is 2-decimal throughout, so "0.10"
        reads consistently; the centred grid contains 0.1125 and 0.2375, which
        "{:.2f}" would collapse to 0.11 / 0.24 and make two rows look like
        duplicates, so it uses "{:.4g}". Tied to `sparsity_round` so the two
        cannot drift apart.
        """
        return ("{:.2f}" if self.sparsity_round <= 2 else "{:.4g}").format(p)

    @property
    def figure_stem(self) -> str:
        return self.fig_stem or self.tex_stem

    @property
    def learned_methods(self) -> List[str]:
        """Method keys that are actual sweep rows (not classical reference columns)."""
        return [m for m, _ in self.methods if m not in BASELINE_KEYS]

    @property
    def panel_methods(self) -> List[Tuple[str, str]]:
        """(key, label) for the learned methods only.

        Tables carry FISTA-16/FISTA-1000 as reference columns; per-method figure
        panels do not -- a classical solver has no training width, so it has no
        width story to plot.
        """
        return [(k, lab) for k, lab in self.methods if k not in BASELINE_KEYS]

    def label(self, suffix: str) -> str:
        return "tab:{}-{}".format(self.label_prefix, suffix)


RUN_SPECS: Dict[str, RunSpec] = {}


def _add(spec: RunSpec) -> None:
    RUN_SPECS[spec.key] = spec


_add(RunSpec(
    key="exp2_real",
    fig_stem="experiment2_modelbased",
    m=210, n=300, train_size=32000,
    methods=list(MODEL_BASED) + FISTA_COLUMNS,
    widths=UNCENTRED_WIDTHS,
    plot_widths=PLOT_WIDTHS_UNCENTRED,
    sparsities=UNCENTRED_SPARSITIES,
    sparsity_round=2,
    tex_stem="experiment2",
    label_prefix="exp2",
    scale_caption=r"(210,300) scale, 40 dB SNR",
    method_caption=r"model-based methods plus plain-FISTA reference columns (see note)",
    subtitle=r"$(m,n) = (210, 300)$, SNR 40 dB, uncentred widths",
    notes=(
        "The pre-study's original Experiment 2: (210,300) at 40 dB SNR on the "
        "uncentred width triad, 10 seeds, completed 2026-08-20.\n"
        "Two things it must not be used for:\n"
        "  * l2o-dm / l2o-rnnprop columns. This run's model-free rows were produced "
        "    through the checkpoint save/load bug and were pruned from results.jsonl "
        "    on 2026-09-06; they survive only in a stale summary.json, which nothing "
        "    here reads.\n"
        "  * a clean width reading. Narrow is centred on p=0.10 while medium and wide "
        "    are centred on ~0.1375 and 0.175, so 'wide vs narrow' here also moves the "
        "    centre. That is why exp2_try3_210x300 exists. Appendix material at most."
    ),
))

_add(RunSpec(
    key="exp2_try2",
    fig_stem="experiment2_try2_modelbased",
    m=25, n=50, train_size=32000,
    model_free_arm="l2o-rnnprop",
    methods=list(MODEL_BASED) + FISTA_COLUMNS,
    widths=CENTRED_WIDTHS,
    plot_widths=PLOT_WIDTHS_CENTRED,
    sparsities=CENTRED_SPARSITIES,
    sparsity_round=4,
    tex_stem="experiment2_try2_modelbased",
    label_prefix="exp2try2mb",
    scale_caption=r"(25,50) scale, noiseless",
    method_caption=r"model-based methods plus plain-FISTA reference columns (see note)",
    subtitle=r"$(m,n) = (25, 50)$, noiseless, centred widths",
    notes=(
        "Experiment 2 re-run at the Primer's own (25,50) noiseless scale on the "
        "re-centred width triad. The cross-scale companion to exp2_try3_210x300.\n"
        "Scoped to LISTA/ALISTA: they are the 10-seed-complete subset. This run's "
        "l2o-rnnprop has post-checkpoint-fix rows for 2 seeds only and l2o-dm for 1, "
        "so neither is a column here (the model-free story is told from "
        "exp2_try3_210x300 and the compute-matched view instead)."
    ),
))

_add(RunSpec(
    key="exp2_try3_210x300",
    fig_stem="experiment2_try3_210x300",
    m=210, n=300, train_size=32000,
    model_free_arm="l2o-rnnprop",
    methods=list(MODEL_BASED) + RNNPROP + FISTA_COLUMNS,
    widths=CENTRED_WIDTHS,
    plot_widths=PLOT_WIDTHS_CENTRED,
    sparsities=CENTRED_SPARSITIES,
    sparsity_round=4,
    tex_stem="experiment2_try3",
    label_prefix="exp2try3",
    scale_caption=r"(210,300) scale, noiseless, centred widths",
    method_caption=(r"LISTA/ALISTA at 16 layers, L2O-RNNProp at 1000 steps, "
                    r"plus plain-FISTA reference columns (see note)"),
    subtitle=r"$(m,n) = (210, 300)$, noiseless, centred widths",
    seed_count_from_data=True,
    notes=(
        "THE HEADLINE RUN: the pre-study's dimensions with exp2_try2's centred width "
        "triad, so width varies and centre does not. Four method columns -- the "
        "model-free arm is valid here because it was trained and evaluated after the "
        "2026-09-06 checkpoint fix.\n"
        "Still growing (seeds 21-25 complete, 26-30 pending), so captions read the "
        "seed count from the rows and core.seedsets drops any seed that is not "
        "present in every cell. Quote it at 5 seeds until 26-30 land."
    ),
))

_add(RunSpec(
    key="exp2_try3_210x300_scoringonly",
    band="auto",
    fig_stem="experiment2_try3_210x300_scoringonly",
    m=210, n=300, train_size=32000,
    methods=list(MODEL_BASED) + FISTA_COLUMNS,
    widths=CENTRED_WIDTHS,
    plot_widths=PLOT_WIDTHS_CENTRED,
    sparsities=CENTRED_SPARSITIES,
    sparsity_round=4,
    tex_stem="experiment2_try3_scoringonly",
    label_prefix="exp2try3so",
    scale_caption=r"(210,300) scale, noiseless, centred widths, scoring loss",
    method_caption=r"model-based methods plus plain-FISTA reference columns (see note)",
    subtitle=r"$(m,n) = (210, 300)$, noiseless, centred widths, scoring loss",
    notes=(
        "The lasso_train_objective ablation: LISTA/ALISTA retrained with the scored "
        "objective (0.5||Ax-b||^2 + lam||x||_1) instead of the released LassoLoss "
        "(0.25||Ax-b||^2 + lam||x||_1, an effective 2*lam). Same protocol, data and "
        "test files as exp2_try3_210x300, verified byte-identical; that run's own "
        "lista/alista rows are the paper arm.\n"
        "The paper arm's 2*lam bias is a width-INDEPENDENT R_f offset, so it "
        "compresses the wide/narrow ratio -- this is the arm to cite for the ID cost "
        "of width. Model-based only; it has no model-free rows."
    ),
))


_add(RunSpec(
    key="exp2_try2_moredata",
    fig_stem="experiment2_moredata_modelbased",
    m=25, n=50, train_size=64000,
    standalone_tables=False,
    methods=list(MODEL_BASED),
    widths=CENTRED_WIDTHS,
    plot_widths=PLOT_WIDTHS_CENTRED,
    sparsities=CENTRED_SPARSITIES,
    sparsity_round=4,
    tex_stem="experiment2_moredata_modelbased",
    label_prefix="exp2moredata",
    scale_caption=r"(25,50) scale, noiseless, 2$	imes$ training data",
    method_caption=r"model-based methods",
    subtitle=r"$(m,n) = (25, 50)$, noiseless, 64 000 training instances (2$	imes$)",
    band="minmax",
    notes=(
        "The 2x-training-data follow-up: identical to exp2_try2 in every respect "
        "(dims, noise, widths, test grid, and -- verified -- byte-identical test "
        "files) except train_size 64 000 vs 32 000. Only seeds 21-23 were run, and "
        "with n=3 an IQR is not a meaningful summary, so its band is the full "
        "per-seed min-max.\n"
        "A companion run, not a standalone result: it exists to be paired against "
        "exp2_try2 seed-by-seed in the more-data effect figure, so it gets no "
        "result table or OOD figure of its own (standalone_tables=False). Its "
        "finding is negative -- the own-centre gap did not close with 2x data, if "
        "anything it widened -- which is evidence against a data-scarcity "
        "explanation."
    ),
))


def get(run_key: str) -> RunSpec:
    """Looks up a run spec, failing with the list of valid keys rather than a KeyError."""
    try:
        return RUN_SPECS[run_key]
    except KeyError:
        raise SystemExit("unknown run %r -- known runs: %s"
                         % (run_key, ", ".join(sorted(RUN_SPECS))))


def choices(standalone_only: bool = True) -> List[str]:
    """Valid --run values, for argparse.

    Defaults to the runs that have results of their own to report. Companion
    runs (`standalone_tables=False`) are registered so their paths and grids
    live here too, but they only appear inside another run's figure, so
    offering them as `--run` would produce output nothing cites.
    """
    return sorted(k for k, v in RUN_SPECS.items()
                  if v.standalone_tables or not standalone_only)
