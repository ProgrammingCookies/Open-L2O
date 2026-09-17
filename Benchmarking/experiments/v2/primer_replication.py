"""
Primer/JMLR-paper LASSO replication -- L2O-RNNprop only. Ported onto
``experiments/sweep.py``.

Replicates Chen et al., "Learning to Optimize: A Primer and a Benchmark"
(JMLR 2022), section 4.1.2's LASSO-minimization experiment as closely as
practical, for RNNprop alone, as a correctness sanity check: if this
reimplementation is right, we should see roughly the recovery quality the
paper's own Figure 6 reports at (m,n)=(25,50) -- "L2O-RNNprop can converge
faster than ISTA and comparable to FISTA" -- not the catastrophic divergence
seen in Benchmarking/experiments/experiment2*.py's own (different-protocol)
runs.

Differences from experiment2.py/experiment2_try2.py, all deliberate, all
paper-sourced (see project_model_free_nonconvergence_investigation memory
for the full trail):
  - Bernoulli(0.1) sparsity (data.lasso.sample_sparse_signal_bernoulli), not
    Experiment 2's exact-round(p*n)-count model.
  - No width/OOD axis -- a single fixed sparsity for train AND test (the
    width axis is a DA233X pre-study addition, absent from the primer's own
    experiment).
  - train_size=12,800 (paper-stated, section 4.1.2), not Table 3's 32,000.
  - Evaluation = average over 10 independent random starting points
    (x0_mode="random"), not Experiment 2's cross-method x0 alignment --
    matches the paper's own "average performance over 10 random starting
    points" protocol.
  - --last_step_loss enabled (the confirmed paper-correct RNNprop training
    objective: w_T=1, w_t=0 otherwise, vs. this repo's prior default of
    summing the loss over every step -- DM's convention, not RNNprop's).

Also computes every Experiment 2 metric (suboptimality gap, modified
relative loss, NMSE vs x_true, LASSO-optimal recovery error) per replicate,
then averages across the 10 -- so despite the different generation/eval
protocol, the resulting numbers are directly the same metrics used
everywhere else in Experiment 2.

PORTING NOTE. This one is not a grid sweep in the usual sense: it trains
ONCE and then evaluates that single checkpoint from 10 random starting
points. That maps onto the sweep driver as a one-axis sweep over ``replicate``,
with training done in a ``prepare`` hook registered against the empty axis
prefix -- so it runs once, lazily, before the first replicate that actually
needs it. Two real gains over the pre-port file, which had none of this:

  * **Per-replicate resume.** Before, resumability was all-or-nothing ("if
    results.jsonl exists, skip everything"), so an interruption after
    replicate 7 either threw away the training run or required hand-editing.
    Now replicates 1-7 are skipped and 8-10 continue.
  * **Provenance.** This file previously recorded no commit, no working-tree
    state, and no config snapshot at all -- the weakest in ``experiments/``.
    It now gets the same capture as everything else.

Schema change to be aware of: ``summary.json`` is now the sweep driver's list-of-
groups with ``_seed_mean``/``_seed_std`` suffixes (one group, since replicate
is the only axis and it is averaged over), where the pre-port file wrote a
flat dict with ``_mean``/``_std``. Nothing in the repo reads this file, so
nothing breaks -- but a hand-written analysis snippet expecting the old shape
would need updating.

Usage:
    python experiments/v2/primer_replication.py --results_dir runs/primer_replication --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_V2_DIR)
_BENCHMARKING_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
# Order matters and must be enforced, not assumed. Python puts the script's own
# directory (v2/) on sys.path automatically, but a plain insert(0, ...) of the
# others pushes it back -- and since these files share their names with the
# pre-port originals one level up, `import experiment2` would then silently
# resolve to experiments/experiment2.py instead of this directory's. Removing
# each path before re-inserting makes the final order exactly V2 > experiments
# > Benchmarking, whatever was there before.
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _V2_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import adapters  # noqa: F401,E402  (import side effect: registers all methods)
from core.registry import get_method  # noqa: E402
from data.lasso import generate_primer_lasso_dataset  # noqa: E402

from sweep import Axis, SweepSpec, run_sweep  # noqa: E402

M, N = 25, 50
LAM = 0.005
SPARSITY_P = 0.1        # Ber(0.1)*N(0,1) -- primer sections 4.1.1/4.1.2's own stated convention
TRAIN_SIZE = 12_800     # paper-stated, section 4.1.2 ("12,800 pairs... for training")
VAL_SIZE = 1_280
TEST_SIZE = 1_280       # paper-stated ("1,280 pairs for validation and testing")
SNR_DB = float("inf")   # noiseless, section 4.1.2 ("the samples are noiseless")
NUM_EPOCHS = 100
NUM_STEPS = 1_000       # both training-steps-per-epoch and eval horizon, section 4.1.2
UNROLL_LENGTH = 20
LEARNING_RATE = 1e-3
EVALUATION_PERIOD = 5
EVALUATION_EPOCHS = 20
BETA1 = 0.95
BETA2 = 0.95
NUM_REPLICATES = 10     # "average performance over 10 random starting points"
NUM_FISTA_ITERS = 50_000


def _train_config(data_dir: str, seed: int, output_dir: str) -> Dict[str, Any]:
    return {
        "method": "l2o-rnnprop",
        "problem": "lasso_dataset",
        "seed": seed,
        "run_name": "l2o-rnnprop__primer_replication",
        "output_dir": output_dir,
        "train": {
            "lasso_data_dir": data_dir,
            "lasso_split": "train_data.npy",
            "lasso_lam": LAM,
            "lasso_batch_size": 128,
            "lasso_x0_mode": "random",
            "num_epochs": NUM_EPOCHS,
            "num_steps": NUM_STEPS,
            "unroll_length": UNROLL_LENGTH,
            "learning_rate": LEARNING_RATE,
            "evaluation_period": EVALUATION_PERIOD,
            "evaluation_epochs": EVALUATION_EPOCHS,
            "beta1": BETA1,
            "beta2": BETA2,
            "last_step_loss": True,
        },
    }


def train_once(cell: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the dataset and train the single checkpoint every replicate
    evaluates. Registered against the empty axis prefix, so it fires exactly
    once -- and lazily, so resuming a finished run does no work at all.
    """
    seed = context["dataset_seed"]
    data_dir = generate_primer_lasso_dataset(
        seed=seed, m=M, n=N, lam=LAM, p=SPARSITY_P,
        train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
        snr_db=SNR_DB, out_dir=context["data_dir"])
    print("dataset ready:", data_dir, flush=True)

    output_dir = os.path.join(context["checkpoints_dir"], "l2o-rnnprop")
    config = _train_config(data_dir, seed, output_dir)
    t0 = time.time()
    train_out = get_method("l2o-rnnprop").train(config, output_dir)
    print("train done in {:.1f}s".format(time.time() - t0), flush=True)
    return {"data_dir_for_run": data_dir, "checkpoint": train_out.checkpoint,
            "train_output_dir": output_dir, "train_config": config}


def run_cell(cell: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the one trained checkpoint from this replicate's random x0."""
    replicate = cell["replicate"]
    eval_config = dict(context["train_config"])
    eval_config["seed"] = replicate  # controls x0_mode="random"'s draw per replicate
    eval_config["eval"] = {
        "optimizer": "L2L",
        "lasso_data_dir": context["data_dir_for_run"],
        "lasso_split": "test_data.npy",
        "lasso_batch_size": 128,
        "lasso_x0_mode": "random",
        "num_steps": NUM_STEPS,
        "lasso_lam": LAM,
        "num_fista_iters": NUM_FISTA_ITERS,
        "xstar_cache_dir": os.path.join(context["results_dir"], "xstar_cache"),
        "beta1": BETA1,
        "beta2": BETA2,
    }
    eval_output_dir = os.path.join(context["train_output_dir"],
                                   "eval_replicate{}".format(replicate))
    t1 = time.time()
    eval_out = get_method("l2o-rnnprop").evaluate(
        eval_config, context["checkpoint"], eval_output_dir)
    eval_sec = time.time() - t1
    row = dict(eval_out.metrics)
    row["eval_sec"] = eval_sec
    print("[replicate {}] done in {:.1f}s: nmse_vs_true_mean={:.2f}dB "
          "modified_relative_loss={:.2%}".format(
              replicate, eval_sec, row.get("nmse_vs_true_mean", float("nan")),
              row.get("modified_relative_loss", float("nan"))), flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seed", type=int, default=0, help="Dataset-generation seed.")
    p.add_argument("--num_replicates", type=int, default=NUM_REPLICATES)
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise a failing replicate instead of recording it "
                        "in errors.jsonl and continuing. For development.")
    args = p.parse_args()

    spec = SweepSpec(
        results_dir=args.results_dir,
        axes=[Axis("replicate", list(range(1, args.num_replicates + 1)))],
        # Empty prefix: fires once for the whole sweep, before the first
        # replicate that needs it.
        prepare=[((), train_once)],
        context={"dataset_seed": args.seed},
        # replicate IS the repetition axis here, so it is averaged out rather
        # than grouped by -- the summary is one row, matching the paper's
        # "average performance over 10 random starting points".
        seed_axis="replicate",
        stop_on_error=args.stop_on_error,
    )
    run_sweep(spec, run_cell, config_snapshot={
        "dataset_seed": args.seed, "num_replicates": args.num_replicates,
        "m": M, "n": N, "lam": LAM, "sparsity_p": SPARSITY_P,
        "train_size": TRAIN_SIZE, "val_size": VAL_SIZE, "test_size": TEST_SIZE,
        "snr_db": SNR_DB, "num_epochs": NUM_EPOCHS, "num_steps": NUM_STEPS,
        "unroll_length": UNROLL_LENGTH, "learning_rate": LEARNING_RATE,
        "evaluation_period": EVALUATION_PERIOD, "evaluation_epochs": EVALUATION_EPOCHS,
        "beta1": BETA1, "beta2": BETA2, "num_fista_iters": NUM_FISTA_ITERS,
        "last_step_loss": True,
    })


if __name__ == "__main__":
    main()
