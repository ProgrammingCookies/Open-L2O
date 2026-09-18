"""LASSO on the datasets written by ``Benchmarking/data/lasso.py``.

    f(x) = 0.5 * ||A x - b||_2^2 + lam * ||x||_1        (per instance)

One ``Problem`` replacing three copies of the same file-reading and objective
code: DM/RNNProp ``problems.lasso_from_dataset``, L2O-Scale ``pg.Lasso`` and
Model_Base ``data_preprocessing.input_fn`` / ``utils.LassoObjective``. Layout
read:

  <data_dir>/A.npy              (m, n) shared dictionary
  <data_dir>/<split>.npy        rows ``[b (m,); x_true (n,)]`` float32
  <data_dir>/<split>_x0.npy     (num_samples, n) seeded starting point
  <data_dir>/W.npy              ALISTA's analytic weight (optional)

Two dialects, both frozen by the existing experiment configs:

* ``"lasso_dataset"`` (model-free): ``lasso_data_dir``, ``lasso_split``,
  ``lasso_batch_size``, ``lasso_lam``, ``lasso_x0_mode`` in ``train``/``eval``;
  ``eval.test_files`` evaluates several splits at once.
* ``"lasso"`` (unrolled models): ``train.data_dir`` + ``train.lasso_lam`` (an
  *identity* field: evaluation reads it from ``train`` too, so a checkpoint is
  always scored with the lambda it was trained for), ``eval.data_dir``,
  ``eval.test_files``; instances come from ``train_data.npy`` / ``val_data.npy``.

Evaluation hooks reproduce the model-free scripts exactly: the deterministic
``ceil(N / batch)`` cursor walk, the ``batch_mean`` reduction over chunks (the
last, smaller chunk weighs the same), the ``instances_fingerprint`` over
``b + x0 + x_true``, the four Table-4 recovery metrics via
``core.lasso_metrics.evaluate_lasso_recovery``, and the
``<tag>_eval_loss_record.pickle-lasso_dataset`` / ``<tag>_recovery-lasso_dataset.npz``
artifact files.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from core.fingerprint import fingerprint_arrays
from unified.problem import Batch, Problem, ProblemDialect
from unified.schema import ConfigSchema


def x0_filename(split: str) -> str:
    """``test_sparsity_0.10.npy`` -> ``test_sparsity_0.10_x0.npy`` (data/lasso.py's rule)."""
    if not split.endswith(".npy"):
        raise ValueError("split must be a .npy filename, got {!r}".format(split))
    return split[:-len(".npy")] + "_x0.npy"


class LassoProblem(Problem):
    eval_reduction = "batch_mean"

    def __init__(self, data_dir: str, split: str, lam: float, batch_size: int,
                 x0_mode: str = "aligned", x0_stddev: float = 0.01, dtype=np.float32):
        if x0_mode not in ("aligned", "random"):
            raise ValueError("x0_mode must be 'aligned' or 'random', got {!r}".format(x0_mode))
        self.data_dir, self.split, self.lam = data_dir, split, float(lam)
        self.batch_size, self.x0_mode, self.x0_stddev = int(batch_size), x0_mode, x0_stddev

        self.a = np.load(os.path.join(data_dir, "A.npy")).astype(dtype)
        data = np.load(os.path.join(data_dir, split)).astype(dtype)
        self.m, self.n = self.a.shape
        if data.shape[1] != self.m + self.n:
            raise ValueError("{} has row width {}, expected {} (= m={} + n={} from {}/A.npy)".format(
                os.path.join(data_dir, split), data.shape[1], self.m + self.n, self.m, self.n, data_dir))
        self._data = data
        self.num_samples = data.shape[0]

        self._x0: Optional[np.ndarray] = None
        if x0_mode == "aligned":
            x0_path = os.path.join(data_dir, x0_filename(split))
            if not os.path.exists(x0_path):
                raise FileNotFoundError(
                    "{} not found -- x0_mode='aligned' requires the seeded x0 sibling file generated "
                    "by Benchmarking/data/lasso.py alongside {}.".format(x0_path, split))
            self._x0 = np.load(x0_path).astype(dtype)
            if self._x0.shape != (self.num_samples, self.n):
                raise ValueError("{} has shape {}, expected {}".format(
                    x0_path, self._x0.shape, (self.num_samples, self.n)))

        w_path = os.path.join(data_dir, "W.npy")
        self.w_alista = np.load(w_path).astype(dtype) if os.path.exists(w_path) else None
        self._a_const = None   # tf.constant, created on first use (keeps the import lazy)

    # ---- instances -------------------------------------------------------
    def _batch(self, idx: np.ndarray) -> Batch:
        rows = self._data[idx]
        if self.x0_mode == "aligned":
            x0 = self._x0[idx]
        else:
            import tensorflow as tf   # same op the scripts draw with, so seeded runs agree
            x0 = tf.random.normal((len(idx), self.n), stddev=self.x0_stddev, dtype=tf.float32).numpy()
        return Batch(x0=x0, data={"idx": idx, "b": rows[:, :self.m], "x_true": rows[:, self.m:]})

    def sample(self) -> Batch:
        # problems.py:233 -- with replacement, from the GLOBAL numpy RNG
        return self._batch(np.random.randint(0, self.num_samples, size=self.batch_size))

    def walk(self) -> Iterator[Batch]:
        for start in range(0, self.num_samples, self.batch_size):
            yield self._batch(np.arange(start, min(start + self.batch_size, self.num_samples)))

    def num_chunks(self) -> int:
        return -(-self.num_samples // self.batch_size)

    # ---- objective -------------------------------------------------------
    def _a(self):
        import tensorflow as tf
        if self._a_const is None:
            self._a_const = tf.constant(self.a)
        return self._a_const

    def per_sample(self, x, batch: Batch):
        import tensorflow as tf
        residual = tf.matmul(x, self._a(), transpose_b=True) - tf.constant(batch.data["b"])
        left_term = 0.5 * tf.reduce_sum(residual ** 2, axis=1)
        other_term = self.lam * tf.norm(x, ord=1, axis=1)   # problems.py's op, not reduce_sum(abs)
        return left_term + other_term

    # ---- hooks -----------------------------------------------------------
    @staticmethod
    def _stack(batches: List[Batch], key: str) -> np.ndarray:
        return np.concatenate([b.data[key] for b in batches], axis=0)

    def fingerprint(self, batches: List[Batch]) -> Dict[str, Any]:
        return fingerprint_arrays(b=self._stack(batches, "b"), x_true=self._stack(batches, "x_true"),
                                  x0=np.concatenate([b.x0 for b in batches], axis=0))

    def recovery_metrics(self, batches: List[Batch], x_pred: np.ndarray,
                         eval_section: Dict[str, Any]) -> Dict[str, Any]:
        from core.lasso_metrics import evaluate_lasso_recovery
        if eval_section.get("num_fista_iters") is None:
            return {}
        return evaluate_lasso_recovery(
            self.a, self._stack(batches, "b"), self._stack(batches, "x_true"), x_pred, self.lam,
            num_fista_iters=int(eval_section["num_fista_iters"]),
            xstar_cache_dir=eval_section.get("xstar_cache_dir")).to_dict()

    def write_eval_artifacts(self, result, output_path: str, tag: str) -> Dict[str, str]:
        """Same files/layout as evaluate_{dm,rnnprop}.py so parity can diff artifacts."""
        os.makedirs(output_path, exist_ok=True)
        pickle_path = os.path.join(output_path, "{}_eval_loss_record.pickle-lasso_dataset".format(tag))
        with open(pickle_path, "wb") as f:
            pickle.dump(list(result.trajectory), f)
        recovery_path = os.path.join(output_path, "{}_recovery-lasso_dataset.npz".format(tag))
        np.savez(recovery_path, x_pred=result.x_pred, x_true=self._stack(result.batches, "x_true"),
                 b=self._stack(result.batches, "b"), x0=np.concatenate([b.x0 for b in result.batches]))
        return {"eval_pickle": pickle_path, "recovery": recovery_path}

    def training_recovery(self, batch: Batch, x_pred: np.ndarray) -> Dict[str, np.ndarray]:
        return {"x_pred": x_pred, "x_true": batch.data["x_true"], "b": batch.data["b"], "x0": batch.x0}


# ---------------------------------------------------------------- dialects

class LassoDatasetDialect(ProblemDialect):
    """``problem="lasso_dataset"`` -- the DM/RNNProp (and Scale) vocabulary."""

    name = "lasso_dataset"
    kind = "model_free"
    train_schema = ConfigSchema.of(
        required=("lasso_data_dir",),
        defaults={"lasso_split": "train_data.npy", "lasso_batch_size": 128, "lasso_lam": 0.005,
                  "lasso_x0_mode": "aligned"})
    eval_schema = ConfigSchema.of(
        required=("lasso_data_dir",),
        allowed=("test_files", "num_fista_iters", "xstar_cache_dir"),
        defaults={"lasso_split": "val_data.npy", "lasso_batch_size": 128, "lasso_lam": 0.005,
                  "lasso_x0_mode": "aligned"})

    @staticmethod
    def _make(section: Dict[str, Any], split: str) -> LassoProblem:
        return LassoProblem(section["lasso_data_dir"], split, section["lasso_lam"],
                            section["lasso_batch_size"], x0_mode=section["lasso_x0_mode"])

    def train_problem(self, config: Dict[str, Any]) -> Problem:
        tr = config["train"]
        return self._make(tr, tr["lasso_split"])

    def eval_problems(self, config: Dict[str, Any]) -> List[Tuple[str, Problem]]:
        ev = config["eval"]
        splits = list(ev["test_files"]) if ev.get("test_files") else [ev["lasso_split"]]
        return [(split, self._make(ev, split)) for split in splits]


class LassoUnrolledDialect(ProblemDialect):
    """``problem="lasso"`` -- the Model_Base_L2O vocabulary used by LISTA / ALISTA."""

    name = "lasso"
    kind = "unrolled"
    # lasso_lam is an identity field of the trained model: read from train in both phases
    train_schema = ConfigSchema.of(required=("data_dir", "lasso_lam"))
    eval_schema = ConfigSchema.of(
        required=("test_files",),
        allowed=("data_dir", "num_fista_iters", "xstar_cache_dir", "lasso_lam"),
        defaults={"val_batch_size": 1024})

    @staticmethod
    def _lam(config: Dict[str, Any]) -> float:
        return float(config["train"]["lasso_lam"])

    def train_problem(self, config: Dict[str, Any]) -> Problem:
        tr = config["train"]
        return LassoProblem(tr["data_dir"], "train_data.npy", self._lam(config), int(tr.get("train_batch_size", 128)))

    def val_problem(self, config: Dict[str, Any]) -> Problem:
        tr = config["train"]
        return LassoProblem(tr["data_dir"], "val_data.npy", self._lam(config), int(tr.get("val_batch_size", 1024)))

    def eval_problems(self, config: Dict[str, Any]) -> List[Tuple[str, Problem]]:
        ev = config["eval"]
        data_dir = ev.get("data_dir") or config["train"].get("data_dir")
        if data_dir is None:
            raise ValueError("problem 'lasso' requires config['eval']['data_dir']")
        if "lasso_lam" in ev and float(ev["lasso_lam"]) != self._lam(config):
            logging.warning("problem 'lasso': config['eval']['lasso_lam'] ignored; the identity field "
                            "config['train']['lasso_lam']=%r is used", self._lam(config))
        return [(split, LassoProblem(data_dir, split, self._lam(config), int(ev["val_batch_size"])))
                for split in ev["test_files"]]
