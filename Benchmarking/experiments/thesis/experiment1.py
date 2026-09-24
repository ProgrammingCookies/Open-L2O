"""Experiment 1 orchestrator (DA233X pre-study, Table 1/2): "Model-free L2O
Scalability and Truncation", ported onto ``experiments/sweep.py``.

Sweeps L2O-DM, L2O-RNNProp, L2O-Scale x 4 LASSO dimensions x 9 unroll lengths
T x N seeds ("10 runs/config" per Table 1), by driving the *existing*
Benchmarking platform's adapters directly (``core.registry.get_method``),
plus this repo's own LASSO dataset primitives (``data.lasso``). Writes a
flat per-run results table (JSON Lines) and a seed-aggregated summary
(mean/std per (method, dim, unroll_length)), matching Table 2's "Mean, Std
Dev across the 10 runs".

PORTING NOTE. This is a 1:1 port of the pre-sweep original ``experiments/experiment1.py``: same
dimensions, unroll lengths, budgets, dataset generation, and metrics. The
~175 lines of provenance capture this file used to own (``_run_git``,
``_head_from_git_dir``, ``_git_state``, ``_append_provenance_record``) were
the best in the repo, so they moved into ``sweep.py`` verbatim and now serve
every experiment rather than only this one. The grid loop, resume, error
capture and summarization moved with them.

Rows gain a ``cell_sec`` field (sweep-driver bookkeeping, never summarized), and
the config snapshot's key is ``cells_already_done_when_written`` rather than
``rows_already_done_when_written`` -- this experiment emits exactly one row
per cell so the two counts coincide here, but the sweep driver serves fan-out
sweeps where they do not, and the old name was simply wrong for those.

Resume is compatible with results.jsonl files written by the pre-port
``experiment1.py``: the key is read from each row's own
method/dim/unroll_length/seed fields.

*** Read before running this for real ***

1. No GPU here. This dev machine's TF build has no GPU device at all, so
   ``peak_allocated_mb``/``peak_reserved_mb`` will be null in every row
   produced on this machine -- the ``profiling.py`` peak-memory capture
   could only be exercised end-to-end (verified to run without crashing),
   not validated against real numbers. Table 2 also wants two separate
   memory columns (PyTorch's allocated vs reserved); TF's
   ``get_memory_info()`` exposes a single 'peak' figure, so both columns
   here report that same number (this is a user-confirmed simplification,
   not a bug).

2. Horizon control. Unroll length T is the axis under study, so total
   optimization horizon (= T * num_unrolls) is held to a FIXED
   ``num_unrolls`` across all T (via ``--fix_num_unrolls`` on
   train_dm.py/train_rnnprop.py, and ``--fix_num_steps = T * num_unrolls``
   on L2O-Scale's train.py) -- NOT a fixed total step count, which would
   silently truncate the horizon differently per T via integer division
   (100 // 75 = 1 vs 100 // 2 = 50) and confound "effect of T" with
   "effect of total steps". ``num_unrolls`` itself (default 10) is a
   placeholder, like Experiment 2's training-budget defaults -- Table 1
   does not specify it.

3. Dataset size per dimension. ``train_size`` defaults scale DOWN with
   dimension (see DEFAULT_TRAIN_SIZE_BY_DIM) -- reusing Experiment 2's flat
   32k would be hundreds of MB/split at the larger dims just for the raw
   .npy file, before any batching. These are placeholders; override via
   ``Experiment1Config.train_size_by_dim`` for a real run.

4. Modified relative loss (Eq 10) requires a per-run FISTA solve of the
   LASSO optimum on that run's own (A, b) batch. A same-(A,b) cache is used
   opportunistically (``solve_xstar``'s disk cache, keyed by content hash):
   L2O-Scale's problem (``problems/problem_generator.py``'s ``pg.Lasso``)
   draws its one fixed batch from a numpy RNG state that's fully determined
   by ``--seed``, so for L2O-Scale the same (dim, seed) genuinely hits cache
   across all 9 T values. train_{dm,rnnprop}.py have a ``--seed`` flag too,
   but their *evaluation* batch (which is what recovery.npz/the FISTA cache
   key is drawn from) is picked by a fresh ``np.random.randint`` call each
   eval pass (problems.py's ``lasso_from_dataset``), independent of that
   seed -- so it is still not reproducible run-to-run and the cache still
   will not hit for them -- each of their runs pays its own FISTA solve.

5. Full sweep is 3 methods x 4 dims x 9 unroll lengths x 10 seeds = 1080
   runs. Use --smoke first.

6. (5000, 10000) was DROPPED from DIMS (2026-09-01), not just left as a
   placeholder to tune later. All three methods here give every optimizee
   coordinate its own RNN state, so GPU memory scales with
   ``batch_size * n * T`` (backprop-through-time keeps every unrolled
   step's activations) -- at n=10000 this OOMs an 8GB card (RTX 2070
   SUPER) for ALL THREE methods at the T values Table 1 cares about, not
   just the coordinatewise LSTM ones (L2O-DM/RNNProp OOM'd even at T=2,
   batch=128; L2O-Scale's HierarchicalRNN OOM'd too once T=100, and still
   OOM'd at batch=8 -- only batch=1 fit). Table 1's "10 runs/config" at
   batch_size=1 would be a materially different (and non-comparable,
   across the dim axis) experiment, not a truncated version of the same
   one, so this dim was excluded rather than silently run at batch=1.

7. Excluding (5000, 10000) does NOT fully retire point 6's memory ceiling
   -- two of the remaining four dims cross it too, at T=100 specifically
   (the largest T in UNROLL_LENGTHS; smaller T fits at the flat default).
   Empirically (RTX 2070 SUPER, T=100, ``fix_num_unrolls=1``): DM OOMs at
   batch*n=32000 for both (250,500) (batch=64) and (500,1000) (batch=32),
   but succeeds at batch*n=16000 for both ((250,500) batch=32; (500,1000)
   batch=16); L2O-Scale confirmed to also succeed at (500,1000) batch=16.
   Since batch_size must stay fixed across T within a dim for the same
   horizon-control reason as point 2 (varying it by T would confound the
   T-scaling comparison), DEFAULT_BATCH_SIZE_BY_DIM sets batch_size=32/16
   for these two dims (sized for the T=100 worst case, used at every T);
   (25,50)/(50,100) keep the flat 128 default (128*100=12800, already
   under the ~16000 safe threshold). A smaller batch means a noisier
   per-run meta-gradient estimate for these two dims specifically -- a
   methodological cost, not a correctness bug, and it does not affect
   comparability across methods (verified for both architectures) or
   across seeds within a dim (batch_size is fixed per dim, so every seed
   at (250,500) uses batch=32, etc.) -- only across the dim axis itself,
   which Table 1 does not claim is apples-to-apples on batch size anyway.

Usage:
    python experiments/thesis/experiment1.py --results_dir runs/exp1_smoke --seeds 0 --smoke
    python experiments/thesis/experiment1.py --results_dir runs/exp1 --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

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

import adapters  # noqa: F401,E402  (import side effect: registers all methods)
from core.registry import get_method  # noqa: E402
from core.lasso_metrics import lasso_objective, solve_xstar  # noqa: E402
from data.lasso import generate_split, sample_dictionary, _x0_filename  # noqa: E402

from sweep import Axis, SweepSpec, add_overrides, apply_overrides, run_sweep  # noqa: E402

METHODS = ("l2o-dm", "l2o-rnnprop", "l2o-scale")
# (5000, 10000) deliberately excluded -- see module docstring point 6.
DIMS: List[Tuple[int, int]] = [(25, 50), (50, 100), (250, 500), (500, 1000)]
UNROLL_LENGTHS: List[int] = [2, 4, 8, 16, 20, 30, 50, 75, 100]

# Placeholders (see module docstring point 3) -- not from the pre-study.
DEFAULT_TRAIN_SIZE_BY_DIM: Dict[Tuple[int, int], int] = {
    (25, 50): 12800,
    (50, 100): 12800,
    (250, 500): 6400,
    (500, 1000): 3200,
}

# GPU-memory-driven, not a placeholder (see module docstring point 7):
# empirically the largest safe batch_size*n product at T=100 (the largest
# unroll length in UNROLL_LENGTHS) on an 8GB card is ~16000 for all three
# methods -- (25,50)/(50,100) fit the flat default (128*100=12800) with
# margin, so only the two larger dims need a reduced batch_size here.
DEFAULT_BATCH_SIZE_BY_DIM: Dict[Tuple[int, int], int] = {
    (250, 500): 32,
    (500, 1000): 16,
}


def dim_key(dim: Tuple[int, int]) -> str:
    return "{}x{}".format(dim[0], dim[1])


def generate_experiment1_dataset(dim: Tuple[int, int], seed: int, train_size: int, lam: float,
                                 snr_db: float, sparsity: float, x0_stddev: float,
                                 out_dir: str) -> str:
    """One fixed-sparsity LASSO training set per (dim, seed). Experiment 1 has
    no train-distribution-width axis (that's Experiment 2's concern), so this
    reuses data/lasso.py's sample_dictionary/generate_split primitives
    directly rather than the width-shaped generate_experiment2_dataset.

    Also writes the seeded ``train_data_x0.npy`` sibling (see
    data/lasso.py's ``_x0_filename``/"x0 alignment" decision): DM/RNNProp's
    ``lasso_from_dataset`` problem (Model_Free_L2O/.../problems.py) now hard-
    requires this file to exist next to ``train_data.npy``.

    If ``out_dir/seed{seed}/{dim}/metadata.json`` already exists (written last,
    so its presence proves a complete prior generation) and its params match
    this call's, generation is skipped and the existing dir is reused -- see
    data/lasso.py's generate_experiment2_dataset for the identical rationale.
    A param mismatch is a hard error, not a silent overwrite/reuse.
    """
    m, n = dim
    dim_dir = os.path.join(out_dir, "seed{}".format(seed), dim_key(dim))
    metadata_path = os.path.join(dim_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            existing = json.load(f)
        wanted = {"m": m, "n": n, "lam": lam, "train_size": train_size,
                 "snr_db": snr_db, "sparsity": sparsity, "x0_stddev": x0_stddev}
        mismatches = {k: (existing.get(k), v) for k, v in wanted.items() if existing.get(k) != v}
        if mismatches:
            raise RuntimeError(
                "{} already exists but was generated with different parameters than "
                "the current config -- refusing to silently overwrite or reuse it. "
                "Mismatched fields (existing, requested): {}. Use a different out_dir "
                "or delete the stale dim directory if this change was intentional."
                .format(metadata_path, mismatches))
        return dim_dir

    rng = np.random.default_rng(seed)
    a = sample_dictionary(m, n, rng)
    data, _, x0 = generate_split(a, train_size, sparsity, snr_db, x0_stddev, rng)

    os.makedirs(dim_dir, exist_ok=True)
    np.save(os.path.join(dim_dir, "A.npy"), a)
    np.save(os.path.join(dim_dir, "train_data.npy"), data)
    np.save(os.path.join(dim_dir, _x0_filename("train_data.npy")), x0)
    with open(os.path.join(dim_dir, "metadata.json"), "w") as f:
        json.dump({
            "seed": seed, "m": m, "n": n, "lam": lam, "train_size": train_size,
            "snr_db": snr_db, "sparsity": sparsity, "x0_stddev": x0_stddev,
            "row_layout": "[b (M,); x_true (N,)], length M+N",
        }, f, indent=2)
    return dim_dir


@dataclass
class Experiment1Config:
    results_dir: str
    seeds: List[int]
    methods: List[str] = field(default_factory=lambda: list(METHODS))
    dims: List[Tuple[int, int]] = field(default_factory=lambda: list(DIMS))
    unroll_lengths: List[int] = field(default_factory=lambda: list(UNROLL_LENGTHS))

    lam: float = 0.005
    snr_db: float = 40.0
    train_sparsity: float = 0.1
    # Matches Experiment 2's default (Benchmarking/data/lasso.py's
    # Experiment2Config.x0_stddev) -- the seeded starting point every
    # method's recovery trajectory begins from.
    x0_stddev: float = 0.01
    train_size_by_dim: Dict[Tuple[int, int], int] = field(
        default_factory=lambda: dict(DEFAULT_TRAIN_SIZE_BY_DIM))
    batch_size: int = 128
    # Per-dim override (see module docstring point 7 / DEFAULT_BATCH_SIZE_BY_DIM)
    # -- falls back to `batch_size` above for any dim not listed here.
    batch_size_by_dim: Dict[Tuple[int, int], int] = field(
        default_factory=lambda: dict(DEFAULT_BATCH_SIZE_BY_DIM))

    # Horizon control (see module docstring point 2) -- placeholder, not
    # from the pre-study.
    num_unrolls: int = 10

    # Training-budget placeholders -- see module docstring's runtime note.
    dm_rnnprop_num_epochs: int = 50
    scale_num_meta_iterations: int = 50
    # l2o-scale's optimizer architecture -- config['train']['optimizer'/'cell_cls'/
    # 'cell_size'/'num_cells'] are required by ScaleFamilyMethod (no adapter-level
    # default), so this experiment pins them explicitly. Not a placeholder in the
    # same sense as the epoch/iteration counts above: these were already what the
    # adapter defaulted to, kept here unchanged since Experiment 1 studies unroll
    # length T, not optimizer architecture.
    scale_optimizer: str = "HierarchicalRNN"
    scale_cell_cls: str = "GRUCell"
    scale_cell_size: int = 20
    scale_num_cells: int = 2

    num_fista_iters: int = 2000

    def __post_init__(self):
        for m in self.methods:
            if m not in METHODS:
                raise ValueError("unknown method {!r} (expected one of {})".format(m, METHODS))

    def train_size(self, dim: Tuple[int, int]) -> int:
        return self.train_size_by_dim.get(dim, 12_800)

    def batch_size_for(self, dim: Tuple[int, int]) -> int:
        return self.batch_size_by_dim.get(dim, self.batch_size)


def _build_train_config(method_name: str, cfg: Experiment1Config, dim: Tuple[int, int],
                        dim_dir: str, unroll_length: int, seed: int,
                        output_dir: str) -> Dict[str, Any]:
    profile_path = os.path.join(output_dir, "profile.jsonl")
    total_steps = unroll_length * cfg.num_unrolls
    batch_size = cfg.batch_size_for(dim)

    if method_name in ("l2o-dm", "l2o-rnnprop"):
        return {
            "method": method_name,
            "problem": "lasso_dataset",
            "output_dir": output_dir,
            "train": {
                "lasso_data_dir": dim_dir,
                "lasso_lam": cfg.lam,
                "lasso_batch_size": batch_size,
                "num_epochs": cfg.dm_rnnprop_num_epochs,
                "unroll_length": unroll_length,
                "fix_num_unrolls": cfg.num_unrolls,
                "profile_path": profile_path,
                # One evaluation at the very end (writes recovery.npz once
                # against the final trained state) -- Experiment 1 doesn't
                # need the periodic-eval/early-stopping machinery.
                "evaluation_period": cfg.dm_rnnprop_num_epochs,
                "evaluation_epochs": 1,
            },
        }
    # l2o-scale
    return {
        "method": method_name,
        "seed": seed,
        "output_dir": output_dir,
        "train": {
            "optimizer": cfg.scale_optimizer,
            "cell_cls": cfg.scale_cell_cls,
            "cell_size": cfg.scale_cell_size,
            "num_cells": cfg.scale_num_cells,
            "include_lasso_problems": True,
            "lasso_data_dir": dim_dir,
            "lasso_split": "train_data.npy",
            "lasso_batch_size": batch_size,
            "lasso_lam": cfg.lam,
            "num_problems": 1,
            "num_meta_iterations": cfg.scale_num_meta_iterations,
            "fix_unroll": True,
            "fix_unroll_length": unroll_length,
            "fix_num_steps": total_steps,
            "fix_num_steps_eval": total_steps,
            "evaluation_period": cfg.scale_num_meta_iterations,
            "evaluation_epochs": 1,
            "profile_path": profile_path,
        },
    }


def _read_profile(run_dir: str) -> Dict[str, Any]:
    """Reads profile_summary.json (see profiling.py in each model-free
    library). Only `*_final` fields and `avg_time_per_optimizee_step_s` are
    safe to compare ACROSS methods: DM/RNNProp log one point per
    meta-training epoch, L2O-Scale logs one point per unroll segment, so
    `meta_loss_mean`/`grad_norm_mean`/`avg_time_per_logged_point_s` average
    over different-sized populations per method (see profiling.py's module
    docstring) -- kept here for within-method diagnostics only, not for a
    cross-method Table 2 comparison.
    """
    path = os.path.join(run_dir, "profile_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        summary = json.load(f)
    return {
        "meta_loss_final": summary.get("final_loss"),
        "meta_loss_mean": summary.get("mean_loss"),
        "grad_norm_final": summary.get("final_grad_norm"),
        "grad_norm_mean": summary.get("mean_grad_norm"),
        "grad_norm_post_clip_final": summary.get("final_grad_norm_post_clip"),
        "grad_norm_post_clip_mean": summary.get("mean_grad_norm_post_clip"),
        # TF exposes one 'peak' figure -- both Table 2 memory columns report
        # it (see module docstring point 1).
        "peak_allocated_mb": summary.get("peak_mem_mb"),
        "peak_reserved_mb": summary.get("peak_mem_mb"),
        "total_time_s": summary.get("total_time_s"),
        # Table 2's "Avg Time/Iteration": the one timing figure comparable
        # across methods, since it's normalized by optimizee steps rather
        # than by how often each method logs (see profiling.py).
        "avg_time_per_iteration_s": summary.get("avg_time_per_optimizee_step_s"),
    }


def _read_modified_relative_loss(run_dir: str, dim_dir: str, cfg: Experiment1Config,
                                 results_dir: str) -> Dict[str, Any]:
    recovery_path = os.path.join(run_dir, "recovery.npz")
    if not os.path.exists(recovery_path):
        return {}
    rec = np.load(recovery_path)
    a = np.load(os.path.join(dim_dir, "A.npy")).astype(np.float32)
    b, x_pred = rec["b"], rec["x_pred"]

    # Disk-backed (see module docstring point 4 and core/lasso_metrics.py's
    # solve_xstar): survives across process restarts, unlike a plain
    # in-memory dict, and shares hits with experiment2.py's own cache dir
    # scheme if pointed at the same results tree.
    cache_dir = os.path.join(results_dir, "xstar_cache")
    x_star = solve_xstar(a, b, cfg.lam, cfg.num_fista_iters, cache_dir)
    f_star = lasso_objective(a, x_star, b, cfg.lam)

    f_pred = lasso_objective(a, x_pred, b, cfg.lam)
    modified_relative_loss = float(np.mean(f_pred - f_star) / np.mean(f_star))
    return {"modified_relative_loss": modified_relative_loss}


def run_cell(cfg: Experiment1Config, cell: Dict[str, Any],
             context: Dict[str, Any]) -> Dict[str, Any]:
    """One (seed, dim, T, method) run. Returns a single row.

    method/dim/unroll_length/seed are stamped in by the sweep driver, so they are
    deliberately absent here. ``train_sec`` is still recorded explicitly
    because Table 2 reports it; the sweep driver's own ``cell_sec`` measures the
    same thing here (this experiment does no separate eval step) but is
    bookkeeping rather than a reported column.
    """
    method_name, dim, unroll_length, seed = (
        cell["method"], cell["dim"], cell["unroll_length"], cell["seed"])
    dim_dir = context["dim_dir"]
    method = get_method(method_name)
    run_name = "{}__{}__T{}__s{}".format(method_name, dim_key(dim), unroll_length, seed)
    output_dir = os.path.join(
        context["checkpoints_dir"], "s{}".format(seed), dim_key(dim),
        "T{}".format(unroll_length), method_name)
    config = _build_train_config(method_name, cfg, dim, dim_dir, unroll_length, seed, output_dir)
    config["run_name"] = run_name
    config.setdefault("seed", seed)

    t0 = time.time()
    train_out = method.train(config, output_dir)
    train_sec = time.time() - t0

    # profile.jsonl/profile_summary.json land wherever --profile_path points
    # (output_dir, set above in _build_train_config); recovery.npz lands
    # alongside the model checkpoint itself (save_path for DM/RNNProp,
    # logdir for L2O-Scale) -- these are two different directories.
    checkpoint_dir = train_out.metrics["logdir"] if method_name == "l2o-scale" else train_out.checkpoint

    row: Dict[str, Any] = {"train_sec": train_sec}
    row.update(_read_profile(output_dir))
    row.update(_read_modified_relative_loss(
        checkpoint_dir, dim_dir, cfg, context["results_dir"]))
    return row


def make_dataset(cell: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """One dataset per (seed, dim), shared by every (T, method) cell under it."""
    cfg: Experiment1Config = context["experiment_config"]
    dim, seed = cell["dim"], cell["seed"]
    dim_dir = generate_experiment1_dataset(
        dim, seed, cfg.train_size(dim), cfg.lam, cfg.snr_db, cfg.train_sparsity,
        cfg.x0_stddev, context["data_dir"])
    print("[seed {} dim {}] dataset ready: {}".format(seed, dim_key(dim), dim_dir))
    return {"dim_dir": dim_dir}


def run_experiment(cfg: Experiment1Config, stop_on_error: bool = False) -> None:
    spec = SweepSpec(
        results_dir=cfg.results_dir,
        # Same nesting as the pre-port file: seed -> dim -> T -> method,
        # so one dataset is generated per (seed, dim) and reused across all
        # nine T values and all three methods.
        axes=[Axis("seed", cfg.seeds),
              Axis("dim", cfg.dims, label=dim_key),
              Axis("unroll_length", cfg.unroll_lengths),
              Axis("method", cfg.methods)],
        prepare=[(("seed", "dim"), make_dataset)],
        context={"experiment_config": cfg},
        stop_on_error=stop_on_error,
    )
    run_sweep(spec, lambda cell, ctx: run_cell(cfg, cell, ctx), config_snapshot=cfg,
              # Table 2 groups by (method, dim, unroll_length); the sweep driver
              # default would order these dim-first.
              group_by=["method", "dim", "unroll_length"])


def _smoke_config(results_dir: str, seeds: List[int], methods: List[str]) -> Experiment1Config:
    tiny_dims = [(20, 30), (30, 40)]
    return Experiment1Config(
        results_dir=results_dir, seeds=seeds, methods=methods,
        dims=tiny_dims, unroll_lengths=[2, 4],
        train_size_by_dim={d: 200 for d in tiny_dims},
        batch_size=16, num_unrolls=2,
        dm_rnnprop_num_epochs=3, scale_num_meta_iterations=3,
        num_fista_iters=50,
    )


def _parse_dim(s: str) -> Tuple[int, int]:
    m, n = s.lower().split("x")
    return (int(m), int(n))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--dims", type=_parse_dim, nargs="+", default=None,
                   help="Filter to a subset of dims, e.g. --dims 500x1000. "
                        "Format MxN, matching a pair from DIMS.")
    p.add_argument("--unroll_lengths", type=int, nargs="+", default=None,
                   help="Filter to a subset of unroll lengths T.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budgets to sanity-check the whole "
                        "pipeline in well under a minute. NOT a real Experiment-1 run.")
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise a failing cell instead of recording it in "
                        "errors.jsonl and continuing. For development.")
    scalar_fields = add_overrides(p, Experiment1Config)
    return p.parse_args(), scalar_fields


def main() -> None:
    args, scalar_fields = _parse_args()
    if args.smoke:
        cfg = _smoke_config(args.results_dir, args.seeds, args.methods)
    else:
        cfg = Experiment1Config(results_dir=args.results_dir, seeds=args.seeds,
                                methods=args.methods)
        apply_overrides(cfg, args, scalar_fields)
    if args.dims is not None:
        unknown = [d for d in args.dims if d not in DIMS]
        if unknown:
            raise ValueError("--dims {} not in DIMS {}".format(unknown, DIMS))
        cfg.dims = args.dims
    if args.unroll_lengths is not None:
        unknown = [t for t in args.unroll_lengths if t not in UNROLL_LENGTHS]
        if unknown:
            raise ValueError("--unroll_lengths {} not in UNROLL_LENGTHS {}".format(
                unknown, UNROLL_LENGTHS))
        cfg.unroll_lengths = args.unroll_lengths
    run_experiment(cfg, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
