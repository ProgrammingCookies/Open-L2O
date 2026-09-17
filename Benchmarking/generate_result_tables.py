"""Generates Benchmarking/experiment2_results_tables.tex from
runs/exp2_real/results.jsonl + runs/exp2_real/classical_baselines.jsonl --
one LaTeX table per Table-4 headline metric, rows grouped by
training-sparsity width (narrow/medium/wide) with test sparsity within each
group, columns = method. Cells are "mean (std)" aggregated across the 10
seeds, computed here from the row files (np.std, ddof=0, same as
experiment2.py's summarize()).

Source change, 2026-09-13: this script used to read summary.json. It no
longer does -- runs/exp2_real/summary.json is stale (written 2026-08-20,
before the checkpoint-bug-invalidated l2o-dm / l2o-rnnprop rows were pruned
from results.jsonl on 2026-09-06) and still carries those invalid columns.
Consequently the model-free columns are gone from this table (see METHODS)
and two classical reference columns were added: FISTA-16 and FISTA-1000 from
classical_baselines.py (plain Beck-Teboulle FISTA, NOT the restart /
certified-gap FISTA that computes x* -- see FISTA_NOTE, which is also
printed under every table).

Not part of the Benchmarking platform itself -- a one-off analysis script,
not imported anywhere. Rerun after any change to results.jsonl or
classical_baselines.jsonl (run classical_baselines.py --run runs/exp2_real
first if the latter is missing).
"""
from __future__ import annotations

import json
import os

import numpy as np

RESULTS_PATH = os.path.join("runs", "exp2_real", "results.jsonl")
BASELINES_PATH = os.path.join("runs", "exp2_real", "classical_baselines.jsonl")
SPARSITY_ROUND = 2
OUT_PATH = "experiment2_results_tables.tex"

METHODS = [
    # l2o-dm / l2o-rnnprop are deliberately NOT columns any more: this run's
    # model-free rows were produced through the checkpoint save/load bug
    # (see RESULTS_VALIDITY.md) and were pruned from results.jsonl on
    # 2026-09-06; they survive only in the stale summary.json, which this
    # script no longer reads.
    ("lista", "LISTA"),
    ("alista", "ALISTA"),
    # Classical reference columns from classical_baselines.py (2026-09-13):
    # plain FISTA at 16 iterations (= LISTA/ALISTA's depth) and at 1000
    # (= the model-free evaluation horizon). See FISTA_NOTE for how this
    # differs from the FISTA that computes x*.
    ("fista@16", "FISTA-16"),
    ("fista@1000", "FISTA-1000"),
]
BASELINE_KEYS = {"fista@16", "fista@1000"}
METRIC_FIELDS = ["suboptimality_gap_mean", "modified_relative_loss",
                 "nmse_vs_true_mean", "lasso_optimal_recovery_error_mean"]
WIDTHS = [
    ("narrow", r"Narrow ($p = 0.10$, fixed)"),
    ("medium", r"Medium ($p \sim \mathcal{U}(0.075, 0.20)$)"),
    ("wide", r"Wide ($p \sim \mathcal{U}(0.05, 0.30)$)"),
]
SPARSITIES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40]

FISTA_NOTE = (
    "\\textbf{FISTA-$k$ columns}: textbook FISTA (Beck \\& Teboulle, 2009; fixed step $1/L$, $L = \\lambda_{\\max}(A^\\top A)$, no restart), run for exactly $k$ iterations from the same shared seeded $x_0$ every learned method starts from; width-independent, so the same values repeat in each width block. This is \\emph{not} the solver that produced the LASSO optimum $x^*$ behind every metric: $x^*$ comes from FISTA with gradient-scheme adaptive restart (O'Donoghue \\& Cand\\`es, 2015) and certified-duality-gap early stopping (relative gap $10^{-10}$, cap 50\\,000 iterations) started from $x = 0$ -- a reference solver, not a fixed-budget baseline. FISTA-1000's $R_f$ is therefore not zero by construction; it measures how far plain FISTA still is from the certified optimum after 1000 iterations. Values below $10^{-9}$ are inside the $x^*$ certification tolerance and shown as a bound."
)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(results_path, baselines_path, sparsity_round):
    """Aggregates results.jsonl (+ classical_baselines.jsonl) into the same
    {(method, width, p): {<metric>_seed_mean, <metric>_seed_std}} shape that
    summary.json used to provide -- computed here, from the row files, on
    purpose: both runs' summary.json are STALE (written before the 2026-09-06
    pruning of the checkpoint-bug-invalidated model-free rows) and reading
    them would silently re-emit invalid columns. np.std with ddof=0 matches
    experiment2.py's summarize(). Baseline rows are keyed by
    "<method>@<iters>" with width "all" (they do not depend on the training
    width); the lookup in the render functions maps every width to "all" for
    those keys.
    """
    rows = _load_jsonl(results_path)
    if os.path.exists(baselines_path):
        for r in _load_jsonl(baselines_path):
            r = dict(r)
            r["method"] = "{}@{}".format(r["method"], r["iters"])
            rows.append(r)
    else:
        print("note: {} not found -- run classical_baselines.py first; FISTA columns will be '--'"
              .format(baselines_path))
    groups = {}
    for r in rows:
        key = (r["method"], r["width"], round(float(r["test_sparsity"]), sparsity_round))
        groups.setdefault(key, []).append(r)
    index = {}
    for key, grp in groups.items():
        entry = {"n_seeds": len({g["seed"] for g in grp})}
        for k in METRIC_FIELDS:
            vals = [float(g[k]) for g in grp if k in g]
            if vals:
                entry[k + "_seed_mean"] = float(np.mean(vals))
                entry[k + "_seed_std"] = float(np.std(vals))
        index[key] = entry
    return index


def lookup(index, method_key, width_key, p, sparsity_round):
    if method_key in BASELINE_KEYS:
        width_key = "all"
    return index.get((method_key, width_key, round(p, sparsity_round)))


# (results.jsonl field, display name, label suffix, formatter)
# x* is certified to a relative duality gap of 1e-10, so a suboptimality
# below ~1e-9 (only plain FISTA at 1000 iterations gets there) is inside the
# reference's own tolerance and can come out as a tiny negative number. It is
# shown as a bound rather than as float noise with a sign.
SCI_FLOOR = 1e-9


def fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < SCI_FLOOR:
        return r"<\!10^{-9}"
    mantissa_exp = "{:.2e}".format(x)
    mantissa, exp = mantissa_exp.split("e")
    exp_i = int(exp)
    return r"{}\!\times\!10^{{{}}}".format(mantissa, exp_i)


def fmt_fixed2(x: float) -> str:
    return "{:.2f}".format(x)


# Method- and width-dependent metrics: full 4-column x (3-width-grouped,
# 8-sparsity) grid.
GRID_METRICS = [
    ("suboptimality_gap_mean", "Suboptimality Gap $g_q$ (Eq.~9)",
     "tab:exp2-subopt-gap", fmt_sci),
    ("modified_relative_loss", "Modified Relative Loss $R_f$ (Eq.~10)",
     "tab:exp2-rel-loss", fmt_sci),
    ("nmse_vs_true_mean", "NMSE vs.\\ True Signal, dB (Eq.~11)",
     "tab:exp2-nmse-true", fmt_fixed2),
]

# Method- and width-INdependent metric: depends only on (A, b, x_true, lam)
# via the shared FISTA-solved x*, not on any method's prediction, and the
# test files themselves are shared across all 3 widths -- so this is
# constant across all 12 (method, width) combinations at a given test
# sparsity. Collapsed to a single column instead of needlessly repeating the
# same 8 numbers 12 times. (Confirmed identical between LISTA and ALISTA as
# a built-in consistency check when this platform was first verified --
# see project_experiment2_orchestrator memory.)
SCALAR_METRIC = ("lasso_optimal_recovery_error_mean",
                 "NMSE vs.\\ LASSO-Optimal Solution, dB",
                 "tab:exp2-nmse-lassoopt", fmt_fixed2)


def render_grid_table(summary_index, field, display_name, label, fmt):
    mean_key = field + "_seed_mean"
    std_key = field + "_seed_std"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + display_name +
                 r": mean (std.\ dev.) across 10 seeds, by training-sparsity width and test sparsity $p$. (210,300) scale, 40 dB SNR; model-based methods plus plain-FISTA reference columns (see note).}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{@{} l " + "r" * len(METHODS) + r" @{}}")
    lines.append(r"\toprule")
    header = "Test $p$ & " + " & ".join(m[1] for m in METHODS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for width_key, width_label in WIDTHS:
        lines.append(r"\multicolumn{" + str(len(METHODS) + 1) +
                     r"}{@{}l}{\textit{" + width_label + r"}} \\")
        for p in SPARSITIES:
            cells = []
            for method_key, _ in METHODS:
                row = lookup(summary_index, method_key, width_key, p, SPARSITY_ROUND)
                if row is None:
                    cells.append("--")
                    continue
                mean_v = row[mean_key]
                std_v = row[std_key]
                cells.append("${}\\ ({})$".format(fmt(mean_v), fmt(std_v)))
            lines.append("{:.2f} & {} \\\\".format(p, " & ".join(cells)))
        if width_key != WIDTHS[-1][0]:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\begin{minipage}{\linewidth}\footnotesize")
    lines.append(FISTA_NOTE)
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_scalar_table(summary_index, field, display_name, label, fmt):
    mean_key = field + "_seed_mean"
    std_key = field + "_seed_std"

    # Sanity-check the collapse assumption before hiding any data: every
    # (method, width) combination at a given sparsity must agree, or this
    # table would be silently wrong.
    for p in SPARSITIES:
        values = set()
        for method_key, _ in METHODS:
            for width_key, _ in WIDTHS:
                row = lookup(summary_index, method_key, width_key, p, SPARSITY_ROUND)
                if row is not None:
                    values.add((round(row[mean_key], 6), round(row[std_key], 6)))
        if len(values) > 1:
            raise ValueError(
                "{} is not actually method/width-independent at p={}: {} distinct "
                "(mean,std) pairs found -- either classical_baselines.jsonl covers a "
                "different seed set than results.jsonl (re-run classical_baselines.py "
                "to completion), or the metric really is not method/width-independent "
                "(then revert to render_grid_table for it).".format(field, p, values))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + display_name +
                 r": mean (std.\ dev.) across 10 seeds. Identical across every "
                 r"method/width combination at a given test sparsity, so "
                 r"collapsed to a single column.}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{@{} l r @{}}")
    lines.append(r"\toprule")
    lines.append(r"Test $p$ & NMSE vs.\ LASSO-optimal (dB) \\")
    lines.append(r"\midrule")
    for p in SPARSITIES:
        # Any (method, width) key works -- all confirmed identical above.
        row = lookup(summary_index, METHODS[0][0], WIDTHS[0][0], p, SPARSITY_ROUND)
        cell = "${}\\ ({})$".format(fmt(row[mean_key]), fmt(row[std_key]))
        lines.append("{:.2f} & {} \\\\".format(p, cell))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\begin{minipage}{\linewidth}\footnotesize")
    lines.append(r"Depends only on the shared dictionary $A$, measurements $b$, "
                 r"and true signal $x^*_{\text{true}}$ via the FISTA-solved LASSO "
                 r"optimum, not on any method's prediction; the test files are "
                 r"shared across all three training-sparsity widths. Confirmed "
                 r"identical, not assumed -- see the sanity check in "
                 r"\texttt{render\_scalar\_table} (\texttt{generate\_result\_tables.py}).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    index = build_index(RESULTS_PATH, BASELINES_PATH, SPARSITY_ROUND)

    chunks = [render_grid_table(index, *m) for m in GRID_METRICS]
    chunks.append(render_scalar_table(index, *SCALAR_METRIC))

    header_comment = (
        "% Experiment 2 results tables, generated by generate_result_tables.py\n"
        "% from runs/exp2_real/results.jsonl (LISTA/ALISTA, 10 seeds, completed\n"
        "% 2026-08-20; the run's l2o-dm/l2o-rnnprop rows are invalid per\n"
        "% RESULTS_VALIDITY.md and are not shown) plus\n"
        "% runs/exp2_real/classical_baselines.jsonl (FISTA-16 / FISTA-1000,\n"
        "% plain FISTA from the shared x0 -- not the x* solver, see the note\n"
        "% under each table). One table per Table-4 headline metric;\n"
        "% rows grouped by training-sparsity width, columns = method, EXCEPT\n"
        "% NMSE-vs-LASSO-optimal, which is method/width-independent by\n"
        "% construction and collapsed to a single column (see\n"
        "% render_scalar_table's sanity check below).\n"
        "% Requires: \\usepackage{booktabs} \\usepackage{amsmath}\n"
        "% Include with: \\input{experiment2_results_tables}\n\n"
    )
    with open(OUT_PATH, "w") as f:
        f.write(header_comment)
        f.write("\n\n".join(chunks))
        f.write("\n")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
