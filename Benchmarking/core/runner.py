"""Generic orchestrator: turn a config into a :class:`BenchmarkResult`.

This is the concrete realization of the ``train_evaluate`` pseudocode that lived
in ``Benchmarking/train_evaluate.py``. It knows nothing about any specific
library -- it just drives whatever adapter the config names.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from core.registry import get_method
from core.result import BenchmarkResult


def _run_dir(config: Dict[str, Any]) -> str:
    """Return the directory where results for this config should be written."""
    base = config.get("output_dir", os.path.join("results"))
    run_name = config.get("run_name") or "{}__{}".format(
        config.get("method", "method"), config.get("problem", "problem"))
    return os.path.join(base, run_name)


def train_evaluate(config: Dict[str, Any]) -> BenchmarkResult:
    """Run one benchmark: (optionally) train the optimizer, then evaluate it.

    ``config`` keys:
        method      (str)  registry name of the L2O method / classical optimizer
        problem     (str)  canonical problem name understood by that adapter
        run_name    (str)  optional; names the results subdirectory
        output_dir  (str)  optional; base directory for results
        train       (dict) optional; raw flags forwarded to the training script
        eval        (dict) optional; raw flags forwarded to the evaluation script
        skip_train  (bool) optional; evaluate an existing checkpoint only
        checkpoint  (str)  optional; explicit checkpoint to evaluate

    Returns a :class:`BenchmarkResult` and writes it under the run directory.
    """
    method = get_method(config["method"])
    out_dir = _run_dir(config)
    os.makedirs(out_dir, exist_ok=True)

    # Thread a top-level config["seed"] into eval, where every adapter's underlying
    # script accepts --seed (l2o-swarm, l2o-dm/rnnprop, and the l2o-scale family).
    # Record whatever seed was requested in `provenance` so a saved result.json is
    # self-documenting.
    # Not threaded into config["train"] here: each adapter (swarm.py, dm_rnnprop.py,
    # scale_family.py) already reads config["seed"] itself and forwards it to its
    # own train flags, so doing it again at this level would be redundant.
    seed = config.get("seed")
    if seed is not None:
        config.setdefault("eval", {}).setdefault("seed", seed)
    provenance: Dict[str, Any] = {"timings": {}, "commands": {}, "seed": seed}

    # ---- Train -----------------------------------------------------------
    checkpoint = config.get("checkpoint")
    train_metrics: Dict[str, Any] = {}
    train_artifacts: Dict[str, str] = {}
    if method.trainable and not config.get("skip_train", False):
        t0 = time.time()
        train_out = method.train(config, out_dir)
        provenance["timings"]["train_sec"] = time.time() - t0
        provenance["commands"]["train"] = train_out.command
        checkpoint = train_out.checkpoint or checkpoint
        train_metrics = train_out.metrics
        train_artifacts = train_out.artifacts

    # ---- Evaluate --------------------------------------------------------
    t0 = time.time()
    eval_out = method.evaluate(config, checkpoint, out_dir)
    provenance["timings"]["eval_sec"] = time.time() - t0
    provenance["commands"]["eval"] = eval_out.command

    result = BenchmarkResult(
        method=config["method"],
        problem=config.get("problem", ""),
        config=config,
        checkpoint=checkpoint,
        train_metrics=train_metrics,
        eval_metrics=eval_out.metrics,
        trajectory=eval_out.trajectory,
        artifacts={**train_artifacts, **eval_out.artifacts},
        provenance=provenance,
    )
    result.save(out_dir)
    return result


def run_suite(configs: List[Dict[str, Any]]) -> List[BenchmarkResult]:
    """Run several benchmarks and return their results (for a leaderboard)."""
    results = []
    for cfg in configs:
        results.append(train_evaluate(cfg))
    return results


def leaderboard(results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
    """Flatten results into rows sorted by final loss (lower is better)."""
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "problem": r.problem,
            "final_loss": r.eval_metrics.get("final_loss"),
            "min_loss": r.eval_metrics.get("min_loss"),
            "checkpoint": r.checkpoint,
        })
    rows.sort(key=lambda row: (row["final_loss"] is None, row["final_loss"]))
    return rows
