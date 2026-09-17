"""Generates Benchmarking/experiment2_results_tables.tex from
runs/exp2_real/summary.json -- one LaTeX table per Table-4 headline metric,
rows grouped by training-sparsity width (narrow/medium/wide) with test
sparsity within each group, columns = method. Cells are "mean (std)"
aggregated across the 10 seeds (summary.json's *_seed_mean/_seed_std).

Not part of the Benchmarking platform itself -- a one-off analysis script,
not imported anywhere. Rerun after any change to runs/exp2_real/summary.json.
"""
from __future__ import annotations

import json
import os

SUMMARY_PATH = os.path.join("runs", "exp2_real", "summary.json")
OUT_PATH = "experiment2_results_tables.tex"

METHODS = [
    ("l2o-dm", "L2O-DM"),
    ("l2o-rnnprop", "L2O-RNNProp"),
    ("lista", "LISTA"),
    ("alista", "ALISTA"),
]
WIDTHS = [
    ("narrow", r"Narrow ($p = 0.10$, fixed)"),
    ("medium", r"Medium ($p \sim \mathcal{U}(0.075, 0.20)$)"),
    ("wide", r"Wide ($p \sim \mathcal{U}(0.05, 0.30)$)"),
]
SPARSITIES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40]

# (summary.json field prefix, display name, label suffix, formatter)
def fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
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
                 r": mean (std.\ dev.) across 10 seeds, by training-sparsity width and test sparsity $p$.}")
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
                row = summary_index.get((method_key, width_key, round(p, 2)))
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
                row = summary_index.get((method_key, width_key, round(p, 2)))
                if row is not None:
                    values.add((round(row[mean_key], 6), round(row[std_key], 6)))
        if len(values) > 1:
            raise ValueError(
                "{} is not actually method/width-independent at p={}: {} distinct "
                "(mean,std) pairs found -- do not collapse to a single column, "
                "revert to render_grid_table for this metric.".format(field, p, values))

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
        row = summary_index[(METHODS[0][0], WIDTHS[0][0], round(p, 2))]
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
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    index = {}
    for row in summary:
        index[(row["method"], row["width"], round(row["test_sparsity"], 2))] = row

    chunks = [render_grid_table(index, *m) for m in GRID_METRICS]
    chunks.append(render_scalar_table(index, *SCALAR_METRIC))

    header_comment = (
        "% Experiment 2 results tables, generated by generate_result_tables.py\n"
        "% from runs/exp2_real/summary.json (full 120-run sweep, 10 seeds,\n"
        "% completed 2026-08-20). One table per Table-4 headline metric;\n"
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
