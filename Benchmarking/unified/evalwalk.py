"""The one evaluation walk over a LASSO split.

Replaces the per-library evaluation loops (``evaluate_dm.py:83-152``,
``evaluate_rnnprop.py``, L2O-Scale ``evaluate.py::_evaluate_lasso_split``,
and the layer-as-iteration read-back in ``adapters/model_based.py:246-293``)
with one function that walks every row of a split exactly once, in file
order, and records:

  * ``trajectory[k] = f(x_k)`` for k = 0 .. T-1  -- note x_T is NOT in the
    trajectory (the scripts record the objective *before* each step), and
  * ``x_pred = x_T`` -- the iterate the Table-4 recovery metrics score.

``reduction`` selects how per-chunk objective values are combined, because
the two adapter families do it differently and parity requires matching:
  * ``"batch_mean"``: each chunk contributes its batch-mean objective; the
    trajectory is the float64 mean over chunks (model-free scripts, which
    also gives the last, possibly smaller, chunk equal weight).
  * ``"sample_mean"``: Keras ``Mean``-metric semantics -- float32 running
    sum of per-sample objectives / float32 count (Model_Base ``train.py``).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from core.fingerprint import METRIC_KEY as FINGERPRINT_KEY, fingerprint_arrays
from core.lasso_metrics import evaluate_lasso_recovery
from core.result import EvalOutput, summarize_trajectory
from unified.instances import LassoInstances
from unified.objective import LassoObjective
from unified.solver import IterativeSolver


@dataclass
class SplitResult:
    split: str
    trajectory: List[float]
    x_pred: np.ndarray
    b: np.ndarray
    x_true: np.ndarray
    x0: np.ndarray

    def fingerprint(self) -> Dict:
        return fingerprint_arrays(b=self.b, x_true=self.x_true, x0=self.x0)


def evaluate_split(make_solver: Callable[[], IterativeSolver], instances: LassoInstances,
                   objective: LassoObjective, batch_size: int, num_steps: int,
                   reduction: str = "batch_mean") -> SplitResult:
    import tensorflow as tf

    if reduction not in ("batch_mean", "sample_mean"):
        raise ValueError("reduction must be 'batch_mean' or 'sample_mean', got {!r}".format(reduction))

    num_chunks = instances.num_chunks(batch_size)
    sum_costs = np.zeros(num_steps, dtype=np.float64)
    sum_per_sample = [tf.zeros((), tf.float32) for _ in range(num_steps)]
    count = tf.zeros((), tf.float32)
    x_pred_chunks, b_chunks, x_true_chunks, x0_chunks = [], [], [], []

    for batch in instances.walk(batch_size):
        solver = make_solver()
        solver.new_run()
        b = tf.constant(batch.b)
        costs = []
        x_final = None
        for k, x_k in enumerate(solver.iterate(batch, num_steps)):
            if k < num_steps:
                if reduction == "batch_mean":
                    costs.append(float(objective.batch_mean(x_k, b)))
                else:
                    sum_per_sample[k] = sum_per_sample[k] + tf.reduce_sum(
                        objective.per_sample(x_k, b))
            else:
                x_final = x_k
        if x_final is None:
            raise RuntimeError("solver yielded fewer than num_steps + 1 iterates")
        if reduction == "batch_mean":
            sum_costs += np.array(costs, dtype=np.float64)
        else:
            count = count + tf.cast(tf.shape(b)[0], tf.float32)
        x_pred_chunks.append(np.asarray(x_final))
        b_chunks.append(batch.b)
        x_true_chunks.append(batch.x_true)
        x0_chunks.append(batch.x0)

    if reduction == "batch_mean":
        trajectory = (sum_costs / num_chunks).tolist()
    else:
        trajectory = [float(s / count) for s in sum_per_sample]

    return SplitResult(
        split=instances.split, trajectory=trajectory,
        x_pred=np.concatenate(x_pred_chunks, axis=0),
        b=np.concatenate(b_chunks, axis=0),
        x_true=np.concatenate(x_true_chunks, axis=0),
        x0=np.concatenate(x0_chunks, axis=0))


def split_metrics(result: SplitResult, a: np.ndarray, lam: float,
                  num_fista_iters: Optional[int], xstar_cache_dir: Optional[str]) -> Dict:
    """summarize_trajectory + fingerprint + the four Table-4 recovery metrics."""
    metrics = summarize_trajectory(result.trajectory)
    metrics[FINGERPRINT_KEY] = result.fingerprint()
    if num_fista_iters is not None:
        metrics.update(evaluate_lasso_recovery(
            a, result.b, result.x_true, result.x_pred, lam,
            num_fista_iters=num_fista_iters, xstar_cache_dir=xstar_cache_dir).to_dict())
    return metrics


def write_model_free_artifacts(result: SplitResult, output_path: str,
                               optimizer_tag: str, problem: str = "lasso_dataset") -> Dict[str, str]:
    """Same files/layout as evaluate_{dm,rnnprop}.py so parity can diff artifacts."""
    os.makedirs(output_path, exist_ok=True)
    pickle_path = os.path.join(output_path, "{}_eval_loss_record.pickle-{}".format(optimizer_tag, problem))
    with open(pickle_path, "wb") as f:
        pickle.dump(list(result.trajectory), f)
    recovery_path = os.path.join(output_path, "{}_recovery-{}.npz".format(optimizer_tag, problem))
    np.savez(recovery_path, x_pred=result.x_pred, x_true=result.x_true, b=result.b, x0=result.x0)
    return {"eval_pickle": pickle_path, "recovery": recovery_path}


def to_eval_output(results: List[SplitResult], a: np.ndarray, lam: float,
                   num_fista_iters: Optional[int], xstar_cache_dir: Optional[str],
                   artifacts: Dict[str, str], command: Optional[str] = None) -> EvalOutput:
    """First split = headline metrics/trajectory; every split under ``per_seed[filename]``
    (the model-based adapter's convention that experiment2's fan-out reads)."""
    per_seed = {r.split: split_metrics(r, a, lam, num_fista_iters, xstar_cache_dir) for r in results}
    head = results[0]
    return EvalOutput(metrics=dict(per_seed[head.split]), trajectory=list(head.trajectory),
                      per_seed=per_seed, artifacts=artifacts, command=command)
