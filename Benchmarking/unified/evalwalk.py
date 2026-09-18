"""The one evaluation walk.

Replaces the per-library evaluation loops (``evaluate_dm.py:83-152``,
``evaluate_rnnprop.py``, L2O-Scale ``evaluate.py::_evaluate_lasso_split``,
and the layer-as-iteration read-back in ``adapters/model_based.py:246-293``)
with one function that runs a solver over every batch the problem's
``walk()`` yields and records:

  * ``trajectory[k] = f(x_k)`` for k = 0 .. T-1  -- note x_T is NOT in the
    trajectory (the scripts record the objective *before* each step), and
  * ``x_pred = x_T`` -- the iterate the problem's recovery metrics score.

``reduction`` selects how per-batch objective values are combined, because
the library scripts differ and parity requires matching:
  * ``"batch_mean"``: each chunk contributes its batch-mean objective; the
    trajectory is the float64 mean over chunks (model-free LASSO scripts,
    which also gives the last, possibly smaller, chunk equal weight).
  * ``"concat"``: every chunk's per-step costs appended (``evaluate_dm.py``'s
    non-LASSO branch: ``loss_record.extend``).
  * ``"sample_mean"``: Keras ``Mean``-metric semantics -- float32 running
    sum of per-sample objectives / float32 count (Model_Base ``train.py``).

``on_first_batch`` runs once, on the first batch, before the solver touches
it: the scripts build the network on the first problem instance and *then*
restore weights, and for library-generated problems that order is what
keeps the random-number stream identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from core.fingerprint import METRIC_KEY as FINGERPRINT_KEY
from core.result import EvalOutput, summarize_trajectory
from unified.problem import Batch, Problem
from unified.solver import IterativeSolver


@dataclass
class SplitResult:
    label: str
    trajectory: List[float]
    x_pred: np.ndarray
    batches: List[Batch] = field(default_factory=list)   # in walk order
    chunk_trajectories: List[List[float]] = field(default_factory=list)   # per chunk, "batch_mean"/"concat" only


def evaluate_split(make_solver: Callable[[], IterativeSolver], problem: Problem, num_steps: int,
                   reduction: str, label: str = "", on_first_batch: Optional[Callable[[Batch], None]] = None
                   ) -> SplitResult:
    import tensorflow as tf

    if reduction not in ("batch_mean", "concat", "sample_mean"):
        raise ValueError("reduction must be 'batch_mean', 'concat' or 'sample_mean', got {!r}".format(reduction))

    sum_costs = np.zeros(num_steps, dtype=np.float64)
    concat: List[float] = []
    chunk_trajectories: List[List[float]] = []
    sum_per_sample = [tf.zeros((), tf.float32) for _ in range(num_steps)]
    count = tf.zeros((), tf.float32)
    x_pred_chunks, batches = [], []

    for i, batch in enumerate(problem.walk()):
        if i == 0 and on_first_batch is not None:
            on_first_batch(batch)
        solver = make_solver()
        solver.new_run()
        costs = []
        x_final = None
        for k, x_k in enumerate(solver.iterate(batch, num_steps)):
            if k < num_steps:
                if reduction == "sample_mean":
                    sum_per_sample[k] = sum_per_sample[k] + tf.reduce_sum(problem.per_sample(x_k, batch))
                else:
                    costs.append(float(problem.objective(x_k, batch)))
            else:
                x_final = x_k
        if x_final is None:
            raise RuntimeError("solver yielded fewer than num_steps + 1 iterates")
        if reduction == "batch_mean":
            sum_costs += np.array(costs, dtype=np.float64)
            chunk_trajectories.append(list(costs))
        elif reduction == "concat":
            concat.extend(costs)
            chunk_trajectories.append(list(costs))
        else:
            count = count + tf.cast(tf.shape(batch.x0)[0], tf.float32)
        x_pred_chunks.append(np.asarray(x_final))
        batches.append(batch)

    if reduction == "batch_mean":
        trajectory = (sum_costs / problem.num_chunks()).tolist()
    elif reduction == "concat":
        trajectory = list(concat)
    else:
        trajectory = [float(s / count) for s in sum_per_sample]
    return SplitResult(label=label, trajectory=trajectory,
                       x_pred=np.concatenate(x_pred_chunks, axis=0), batches=batches,
                       chunk_trajectories=chunk_trajectories)


def split_metrics(result: SplitResult, problem: Problem, eval_section: Dict) -> Dict:
    """summarize_trajectory + the problem's fingerprint and recovery metrics (if any)."""
    metrics = summarize_trajectory(result.trajectory)
    fingerprint = problem.fingerprint(result.batches)
    if fingerprint is not None:
        metrics[FINGERPRINT_KEY] = fingerprint
    metrics.update(problem.recovery_metrics(result.batches, result.x_pred, eval_section))
    return metrics


def to_eval_output(results: List[SplitResult], problems: List[Problem], eval_section: Dict,
                   artifacts: Dict[str, str], command: Optional[str] = None) -> EvalOutput:
    """First split = headline metrics/trajectory; every split under ``per_seed[label]``
    (the model-based adapter's convention that experiment2's fan-out reads)."""
    per_seed = {r.label: split_metrics(r, p, eval_section) for r, p in zip(results, problems)}
    head = results[0]
    return EvalOutput(metrics=dict(per_seed[head.label]), trajectory=list(head.trajectory),
                      per_seed=per_seed, artifacts=artifacts, command=command)
