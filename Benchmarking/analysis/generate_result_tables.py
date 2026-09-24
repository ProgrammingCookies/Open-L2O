"""Generates one LaTeX table per Table-4 headline metric for an Experiment-2 run.

    python analysis/generate_result_tables.py --run exp2_try3_210x300

Rows are grouped by training-sparsity width (narrow/medium/wide) with test
sparsity within each group; columns are methods. Cells are "mean (std)"
aggregated across seeds, computed here from `runs/<run>/results.jsonl` (np.std,
ddof=0, matching the orchestrator's `summarize()`) plus
`runs/<run>/classical_baselines.jsonl` for the FISTA reference columns. Output
goes to `Figs/tables/<stem>_results_tables.tex`.

Everything that differs between runs -- paths, which methods are legitimate
columns, the width triad, the sparsity grid, LaTeX labels and captions -- lives
in `run_specs.py`, including the reasons. Run
`python analysis/generate_result_tables.py --list` to see the runs and their caveats.

This script used to be three near-identical files (`generate_result_tables.py`,
`..._try2_modelbased.py`, `..._try3.py`). They shared every function and differed
only in those values; consolidating them removed ~580 duplicated lines and the
risk of fixing a bug in one copy and not the others.

It does NOT read `summary.json`. Every run's `summary.json` is stale -- written
before the 2026-09-06 pruning of the checkpoint-bug-invalidated model-free rows --
and reading it would silently re-emit invalid columns.

Rerun after any change to `results.jsonl` or `classical_baselines.jsonl` (run
`classical_baselines.py --run runs/<run>` first if the latter is missing or is
behind on seeds).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sys

# Benchmarking/ is this file's parent, not its own directory, so it is not on
# sys.path automatically when this runs as `python analysis/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_specs
from core.seedsets import filter_to_complete_seeds

FISTA_NOTE = (
    "\\textbf{FISTA-$k$ columns}: textbook FISTA (Beck \\& Teboulle, 2009; fixed step $1/L$, $L = \\lambda_{\\max}(A^\\top A)$, no restart), run for exactly $k$ iterations from the same shared seeded $x_0$ every learned method starts from; width-independent, so the same values repeat in each width block. This is \\emph{not} the solver that produced the LASSO optimum $x^*$ behind every metric: $x^*$ comes from FISTA with gradient-scheme adaptive restart (O'Donoghue \\& Cand\\`es, 2015) and certified-duality-gap early stopping (relative gap $10^{-10}$, cap 50\\,000 iterations) started from $x = 0$ -- a reference solver, not a fixed-budget baseline. FISTA-1000's $R_f$ is therefore not zero by construction; it measures how far plain FISTA still is from the certified optimum after 1000 iterations. Values below $10^{-9}$ are inside the $x^*$ certification tolerance and shown as a bound."
)

# Values below ~1e-9 (only plain FISTA at 1000 iterations gets there) are inside
# the reference solver's own tolerance and can come out as a tiny negative
# number. Shown as a bound rather than as float noise with a sign.
SCI_FLOOR = 1e-9


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < SCI_FLOOR:
        return r"<\!10^{-9}"
    mantissa, exp = "{:.2e}".format(x).split("e")
    return r"{}\!\times\!10^{{{}}}".format(mantissa, int(exp))


def fmt_fixed2(x: float) -> str:
    return "{:.2f}".format(x)


# Method- and width-dependent metrics: the full (3-width-grouped, N-sparsity) grid.
GRID_METRICS = [
    ("suboptimality_gap_mean", "Suboptimality Gap $g_q$ (Eq.~9)", "subopt-gap", fmt_sci),
    ("modified_relative_loss", "Modified Relative Loss $R_f$ (Eq.~10)", "rel-loss", fmt_sci),
    ("nmse_vs_true_mean", "NMSE vs.\\ True Signal, dB (Eq.~11)", "nmse-true", fmt_fixed2),
]

# Method- and width-INdependent metric: depends only on (A, b, x_true, lam) via
# the shared FISTA-solved x*, not on any method's prediction, so it collapses to
# one column. The collapse is checked, not assumed -- see render_scalar_table.
SCALAR_METRIC = ("lasso_optimal_recovery_error_mean",
                 "NMSE vs.\\ LASSO-Optimal Solution, dB", "nmse-lassoopt", fmt_fixed2)


def build_index(spec):
    """Aggregates the row files into {(method, width, p): {<metric>_seed_mean/std}}.

    Baseline rows are keyed "<method>@<iters>" with width "all" (they do not
    depend on the training width); `lookup` maps every width to "all" for them.
    Returns (index, n_seeds).
    """
    rows = _load_jsonl(spec.results)
    # Only seeds present in EVERY (method, width) cell. A sweep stopped mid-seed
    # otherwise puts one method column on N seeds and another on N-1 inside the
    # same table, which also breaks render_scalar_table's independence check.
    rows, complete, _ = filter_to_complete_seeds(
        rows, methods=spec.learned_methods, label=spec.results)

    if os.path.exists(spec.baselines):
        keep = set(complete)
        for r in _load_jsonl(spec.baselines):
            if int(r["seed"]) not in keep:
                continue      # same seed set as the learned columns, not the union
            r = dict(r)
            r["method"] = "{}@{}".format(r["method"], r["iters"])
            rows.append(r)
    else:
        print("note: {} not found -- run classical_baselines.py first; "
              "FISTA columns will be '--'".format(spec.baselines))

    groups = {}
    for r in rows:
        key = (r["method"], r["width"], round(float(r["test_sparsity"]), spec.sparsity_round))
        groups.setdefault(key, []).append(r)

    index = {}
    for key, grp in groups.items():
        entry = {"n_seeds": len({g["seed"] for g in grp})}
        for k in run_specs.METRIC_FIELDS:
            vals = [float(g[k]) for g in grp if k in g]
            if vals:
                entry[k + "_seed_mean"] = float(np.mean(vals))
                entry[k + "_seed_std"] = float(np.std(vals))
        index[key] = entry
    return index, len(complete)


def lookup(index, spec, method_key, width_key, p):
    if method_key in run_specs.BASELINE_KEYS:
        width_key = "all"
    return index.get((method_key, width_key, round(float(p), spec.sparsity_round)))


def render_grid_table(index, spec, n_seeds, field, display_name, label_suffix, fmt):
    mean_key, std_key = field + "_seed_mean", field + "_seed_std"
    methods, widths = spec.methods, spec.widths

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + display_name +
             r": mean (std.\ dev.) across " + str(n_seeds) +
             r" seeds, by training-sparsity width and test sparsity $p$. " +
             spec.scale_caption + "; " + spec.method_caption + r".}",
             r"\label{" + spec.label(label_suffix) + "}",
             r"\begin{tabular}{@{} l " + "r" * len(methods) + r" @{}}",
             r"\toprule",
             "Test $p$ & " + " & ".join(m[1] for m in methods) + r" \\",
             r"\midrule"]

    for width_key, width_label in widths:
        lines.append(r"\multicolumn{" + str(len(methods) + 1) +
                     r"}{@{}l}{\textit{" + width_label + r"}} \\")
        for p in spec.sparsities:
            cells = []
            for method_key, _ in methods:
                row = lookup(index, spec, method_key, width_key, p)
                if row is None or mean_key not in row:
                    cells.append("--")
                    continue
                cells.append("${}\\ ({})$".format(fmt(row[mean_key]), fmt(row[std_key])))
            lines.append("{} & {} \\\\".format(spec.format_p(p), " & ".join(cells)))
        if width_key != widths[-1][0]:
            lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
              r"\begin{minipage}{\linewidth}\footnotesize", FISTA_NOTE,
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


def render_scalar_table(index, spec, n_seeds, field, display_name, label_suffix, fmt):
    mean_key, std_key = field + "_seed_mean", field + "_seed_std"

    # Sanity-check the collapse assumption before hiding any data.
    for p in spec.sparsities:
        values = set()
        for method_key, _ in spec.methods:
            for width_key, _ in spec.widths:
                row = lookup(index, spec, method_key, width_key, p)
                if row is not None and mean_key in row:
                    values.add((round(row[mean_key], 6), round(row[std_key], 6)))
        if len(values) > 1:
            raise ValueError(
                "{} is not actually method/width-independent at p={}: {} distinct "
                "(mean,std) pairs found -- either classical_baselines.jsonl covers a "
                "different seed set than results.jsonl (re-run classical_baselines.py "
                "to completion), or the metric really is not method/width-independent "
                "(then revert to render_grid_table for it).".format(field, p, values))

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{" + display_name +
             r": mean (std.\ dev.) across " + str(n_seeds) +
             r" seeds. Identical across every method/width combination at a given "
             r"test sparsity, so collapsed to a single column. " +
             spec.scale_caption + r".}",
             r"\label{" + spec.label(label_suffix) + "}",
             r"\begin{tabular}{@{} l r @{}}", r"\toprule",
             r"Test $p$ & NMSE vs.\ LASSO-optimal (dB) \\", r"\midrule"]
    for p in spec.sparsities:
        row = lookup(index, spec, spec.methods[0][0], spec.widths[0][0], p)
        if row is None or mean_key not in row:
            continue
        lines.append("{} & ${}\\ ({})$ \\\\".format(
            spec.format_p(p), fmt(row[mean_key]), fmt(row[std_key])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"Depends only on the shared dictionary $A$, measurements $b$, "
              r"and true signal $x^*_{\text{true}}$ via the FISTA-solved LASSO "
              r"optimum, not on any method's prediction; the test files are "
              r"shared across all three training-sparsity widths. Confirmed "
              r"identical, not assumed -- see the sanity check in "
              r"\texttt{render\_scalar\_table} "
              r"(\texttt{generate\_result\_tables.py}).",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


def build_header(spec, n_seeds):
    return (
        "% Experiment 2 results tables for run '{}' ({}),\n".format(spec.key, spec.scale_caption)
        + "% generated by generate_result_tables.py --run {}\n".format(spec.key)
        + "% from {} ({}; {} seeds)\n".format(
            spec.results, ", ".join(l for _, l in spec.methods
                                    if _ not in run_specs.BASELINE_KEYS), n_seeds)
        + "% plus {} (FISTA-16 / FISTA-1000, plain\n".format(spec.baselines)
        + "% FISTA from the shared x0 -- not the x* solver, see the note under each\n"
        "% table). One table per Table-4 headline metric; rows grouped by\n"
        "% training-sparsity width, columns = method, EXCEPT NMSE-vs-LASSO-optimal,\n"
        "% which is method/width-independent by construction and collapsed to a\n"
        "% single column (see render_scalar_table's sanity check).\n"
        "% Requires: \\usepackage{booktabs} \\usepackage{amsmath}\n"
        + "% Include with: \\input{{Figs/tables/{}_results_tables}}\n\n".format(spec.tex_stem)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=run_specs.choices(),
                    help="which Experiment-2 run to tabulate")
    ap.add_argument("--list", action="store_true",
                    help="print the known runs with their caveats and exit")
    args = ap.parse_args()

    if args.list:
        for key in run_specs.choices():
            spec = run_specs.RUN_SPECS[key]
            print("\n=== %s -> %s" % (key, spec.tables_path))
            print(spec.notes)
        return
    if not args.run:
        ap.error("--run is required (or --list)")

    spec = run_specs.get(args.run)
    index, n_seeds = build_index(spec)

    chunks = [render_grid_table(index, spec, n_seeds, *m) for m in GRID_METRICS]
    chunks.append(render_scalar_table(index, spec, n_seeds, *SCALAR_METRIC))

    os.makedirs(run_specs.TABLES_DIR, exist_ok=True)
    with open(spec.tables_path, "w") as f:
        f.write(build_header(spec, n_seeds))
        f.write("\n\n".join(chunks))
        f.write("\n")
    print("wrote", spec.tables_path)


if __name__ == "__main__":
    main()
