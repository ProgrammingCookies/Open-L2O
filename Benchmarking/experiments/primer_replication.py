"""
Primer/JMLR-paper LASSO replication -- L2O-RNNprop only.

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

Usage:
    python primer_replication.py --results_dir runs/primer_replication --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKING_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKING_ROOT)

import adapters  # noqa: E402
from core.registry import get_method  # noqa: E402
from data.lasso import generate_primer_lasso_dataset  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seed", type=int, default=0, help="Dataset-generation seed.")
    p.add_argument("--num_replicates", type=int, default=NUM_REPLICATES)
    args = p.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    data_root = os.path.join(results_dir, "data")
    checkpoints_root = os.path.join(results_dir, "checkpoints")

    data_dir = generate_primer_lasso_dataset(
        seed=args.seed, m=M, n=N, lam=LAM, p=SPARSITY_P,
        train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
        snr_db=SNR_DB, out_dir=data_root)
    print("dataset ready:", data_dir, flush=True)

    method = get_method("l2o-rnnprop")
    output_dir = os.path.join(checkpoints_root, "l2o-rnnprop")
    train_config = {
        "method": "l2o-rnnprop",
        "problem": "lasso_dataset",
        "seed": args.seed,
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

    if os.path.exists(os.path.join(results_dir, "results.jsonl")):
        print("results.jsonl already exists -- skipping straight to summary "
             "recompute (delete it to force a full rerun)", flush=True)
    else:
        t0 = time.time()
        train_out = method.train(train_config, output_dir)
        train_sec = time.time() - t0
        print("train done in {:.1f}s".format(train_sec), flush=True)

        xstar_cache_dir = os.path.join(results_dir, "xstar_cache")
        replicate_metrics = []
        for replicate in range(1, args.num_replicates + 1):
            eval_config = dict(train_config)
            eval_config["seed"] = replicate  # controls x0_mode="random"'s draw per replicate
            eval_config["eval"] = {
                "optimizer": "L2L",
                "lasso_data_dir": data_dir,
                "lasso_split": "test_data.npy",
                "lasso_batch_size": 128,
                "lasso_x0_mode": "random",
                "num_steps": NUM_STEPS,
                "lasso_lam": LAM,
                "num_fista_iters": NUM_FISTA_ITERS,
                "xstar_cache_dir": xstar_cache_dir,
                "beta1": BETA1,
                "beta2": BETA2,
            }
            eval_output_dir = os.path.join(output_dir, "eval_replicate{}".format(replicate))
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint, eval_output_dir)
            eval_sec = time.time() - t1
            row = dict(eval_out.metrics)
            row["replicate"] = replicate
            row["eval_sec"] = eval_sec
            replicate_metrics.append(row)
            print("[replicate {}] done in {:.1f}s: nmse_vs_true_mean={:.2f}dB "
                 "modified_relative_loss={:.2%}".format(
                     replicate, eval_sec, row.get("nmse_vs_true_mean", float("nan")),
                     row.get("modified_relative_loss", float("nan"))), flush=True)

        results_path = os.path.join(results_dir, "results.jsonl")
        with open(results_path, "w") as f:
            for row in replicate_metrics:
                f.write(json.dumps(row) + "\n")
        print("wrote", results_path, flush=True)

    with open(os.path.join(results_dir, "results.jsonl")) as f:
        replicate_metrics = [json.loads(line) for line in f if line.strip()]

    metric_keys = sorted({k for row in replicate_metrics for k in row
                          if k not in ("replicate", "eval_sec")})
    summary = {"num_replicates": len(replicate_metrics)}
    for k in metric_keys:
        vals = [r[k] for r in replicate_metrics if isinstance(r.get(k), (int, float))]
        if vals:
            summary[k + "_mean"] = float(np.mean(vals))
            summary[k + "_std"] = float(np.std(vals))
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary to", summary_path, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
