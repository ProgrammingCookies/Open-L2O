"""
Experiment-2 model-free scale-convergence diagnostic, ported onto
``experiments/sweep.py`` (via the ported ``thesis/experiment2.py``).

Not a full experiment -- a cheap pre-check before committing to a full
model-free hyperparameter search at Table 3 scale (m=210, n=300). Runs the
standard Experiment 2 orchestrator (thesis/experiment2.py's OrchestratorConfig/
run_experiment, reused as-is) at a sequence of (m, n) scales, width="narrow"
only (training sparsity fixed at p=0.1), evaluated in-distribution at that same
p=0.1. This isolates "does this method converge at all, and does the gap to
model-based grow with dimension" from Experiment 2's actual question (does
training-distribution width trade ID quality for OOD robustness, and does
that differ by method family) -- OOD/width is deliberately out of scope here.

Every other hyperparameter (train_size, num_epochs, num_steps, unroll_length,
lr, ...) stays at the real Table 3 values (OrchestratorConfig's defaults) so
scale is the only thing varying between runs -- if it were varying training
budget too, a convergence failure at large n wouldn't tell you whether n or
budget was the cause.

PORTING NOTE. Each (m, n) still gets its own results_dir and its own sweep,
exactly as before -- the dims are NOT folded into one sweep axis, because
``_print_comparison`` below reads one results.jsonl per dim and merging them
would change the on-disk layout for no benefit. What the port removes is this
file's hand-copied CLI-override block (an ``add_argument`` per field plus a
``setattr`` loop, then three more ``if args.x:`` lines): the overrides are now
derived from OrchestratorConfig, so every scalar field is reachable instead of
the six that were hand-listed.

Usage:
    # Step 1: does l2o-dm/l2o-rnnprop converge at all, smallest scale only.
    python experiments/thesis/scale_diagnostic.py --results_dir runs/scale_diag --dims 25x50

    # Step 2: full scale sweep (run only after Step 1 looks sane).
    python experiments/thesis/scale_diagnostic.py --results_dir runs/scale_diag \
        --dims 25x50 50x100 100x150 210x300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

_THESIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_THESIS_DIR)
_BENCHMARKING_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
# Order matters and must be enforced, not assumed. Python puts the script's own
# directory (thesis/) on sys.path automatically, but a plain insert(0, ...) of
# the others pushes it back, and `import experiment2` would then resolve
# against experiments/ or Benchmarking/ rather than this directory. Removing
# each path before re-inserting makes the final order exactly thesis/ >
# experiments/ > Benchmarking/, whatever was there before. (Until 2026-09-24
# this also guarded against same-named pre-port originals one level up; those
# are gone, but the ordering guarantee is still the thing being asserted.)
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _THESIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from sweep import add_overrides, apply_overrides  # noqa: E402
from experiment2 import ALL_METHODS, OrchestratorConfig, run_experiment  # noqa: E402


def _parse_dims(spec: str) -> Tuple[int, int]:
    m_str, n_str = spec.lower().split("x")
    return int(m_str), int(n_str)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--dims", nargs="+", default=["25x50", "50x100", "100x150", "210x300"],
                   help="m x n pairs, smallest first so a diagnostic-worthy "
                        "failure (or a per-run cost surprise) surfaces before "
                        "the expensive end of the sweep runs.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Fewer than Table 3's 10 -- this is a diagnostic, not "
                        "the final protocol.")
    p.add_argument("--methods", nargs="+", default=["l2o-dm", "l2o-rnnprop", "lista"],
                   choices=list(ALL_METHODS),
                   help="alista omitted by default (extra LP-solve cost for "
                        "its W.npy, and lista alone is enough as the "
                        "model-based reference point); add it back with "
                        "--methods l2o-dm l2o-rnnprop lista alista if wanted.")
    p.add_argument("--in_distribution_p", type=float, default=0.1,
                   help="Single test sparsity, matching narrow width's fixed "
                        "training sparsity p=0.1 -- deliberately ID-only, one "
                        "point, not the full 8-sparsity test battery.")
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise a failing cell instead of recording it in "
                        "errors.jsonl and continuing. For development.")
    # Budget overrides, same escape hatch experiment2.py itself offers --
    # None means "use OrchestratorConfig's real Table-3 default", not "off".
    # Derived from the dataclass now, so all of them are reachable, not the
    # six this file used to hand-list.
    scalar_fields = add_overrides(p, OrchestratorConfig, help_texts={
        "model_free_curriculum": "Enable --if_cl (curriculum learning on unroll "
                                 "length) -- Chen et al. 2020's fix for truncation-"
                                 "bias instability in fixed-short-unroll training.",
        "model_free_imitation": "Enable --if_mt (imitation learning from analytical "
                                "optimizers) -- Chen et al. 2020's second fix.",
        "model_free_profile": "Write a per-epoch (loss, meta-grad-norm) trajectory "
                              "to each run's checkpoint dir as profile.jsonl.",
    })
    # m/n are the axis this file sweeps via --dims, so they must not also be
    # settable per-run; drop them from the override set.
    for name in ("m", "n"):
        if name in scalar_fields:
            scalar_fields.remove(name)
    return p.parse_args(), scalar_fields


def main() -> None:
    args, scalar_fields = _parse_args()
    dims = [_parse_dims(d) for d in args.dims]

    for m, n in dims:
        scale_dir = os.path.join(args.results_dir, "{}x{}".format(m, n))
        print("=" * 80)
        print("[scale] m={} n={} -> {}".format(m, n, scale_dir))
        print("=" * 80)
        cfg = OrchestratorConfig(
            results_dir=scale_dir,
            seeds=args.seeds,
            methods=args.methods,
            widths=["narrow"],
            m=m, n=n,
            test_sparsities=[args.in_distribution_p],
        )
        apply_overrides(cfg, args, scalar_fields)
        run_experiment(cfg, stop_on_error=args.stop_on_error)

    _print_comparison(args.results_dir, dims, args.methods)


def _print_comparison(results_dir: str, dims: List[Tuple[int, int]], methods: List[str]) -> None:
    print()
    print("=" * 80)
    print("Scale-diagnostic summary: mean min_loss across seeds "
          "(width=narrow, in-distribution test sparsity)")
    print("=" * 80)
    print("{:>12s}".format("(m,n)") + "".join("{:>16s}".format(m) for m in methods))
    for m, n in dims:
        results_path = os.path.join(results_dir, "{}x{}".format(m, n), "results.jsonl")
        label = "{}x{}".format(m, n)
        if not os.path.exists(results_path):
            print("{:>12s}  (missing: {})".format(label, results_path))
            continue
        rows = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        cells = []
        for method_name in methods:
            vals = [r["min_loss"] for r in rows if r["method"] == method_name]
            cells.append("{:16.4f}".format(sum(vals) / len(vals)) if vals
                         else "{:>16s}".format("n/a"))
        print("{:>12s}".format(label) + "".join(cells))


if __name__ == "__main__":
    main()
