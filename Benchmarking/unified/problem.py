"""The problem protocol: what a learned optimizer (or an unrolled network) is
run on, and how its result is scored.

The training loop and the evaluation walk touch exactly two things: a
``Batch``'s starting point ``x0`` and the problem's objective / gradient.
Everything problem-specific -- where instances come from, what else a batch
carries (LASSO: ``b``, ``x_true``; quadratic: ``W``, ``y``), how instances
are fingerprinted, which recovery metrics exist, what artifact files the
library scripts wrote -- is behind the optional hooks below, so a problem
that has none of it (a library-generated toy) implements only the first
five methods.

A problem is reached from a config through a ``ProblemDialect``: the
``config["problem"]`` name plus the config keys that dialect understands.
One problem class may have several dialects (LASSO has the model-free
``"lasso_dataset"`` and the model-based ``"lasso"`` vocabularies, both
frozen by the existing experiment configs). ``register_problem`` fills the
registry that ``unified/__init__.py`` populates; adding a problem is one
module plus one line there.

No TensorFlow at import time: hooks import it inside the function body.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from unified.schema import ConfigSchema


@dataclass
class Batch:
    x0: np.ndarray                                 # (B, ...) starting point -- all the loops use
    data: Dict[str, Any] = field(default_factory=dict)   # problem-owned instance data


class Problem(abc.ABC):
    """One instance source + objective. ``batch_size`` is the problem's (the
    library scripts fix it per problem, e.g. quadratic's 128)."""

    batch_size: int
    #: how the model-free evaluation reduces per-chunk objective values, matching
    #: the library script for this problem: "batch_mean" (mean over chunks),
    #: "concat" (per-step costs of every chunk appended), "sample_mean"
    eval_reduction: str = "batch_mean"

    # ---- instances
    @abc.abstractmethod
    def sample(self) -> Batch:
        """A training batch, drawn the way the library script draws it (global RNG)."""

    @abc.abstractmethod
    def walk(self) -> Iterator[Batch]:
        """The evaluation pass: every instance once, in the script's order."""

    @abc.abstractmethod
    def num_chunks(self) -> int:
        """How many batches ``walk()`` yields."""

    # ---- objective
    @abc.abstractmethod
    def per_sample(self, x, batch: Batch):
        """(B,) objective values."""

    def objective(self, x, batch: Batch):
        """Batch-mean scalar (what the model-free scripts differentiate)."""
        import tensorflow as tf
        return tf.reduce_mean(self.per_sample(x, batch))

    def value_and_grad(self, x, batch: Batch, second_order: bool = False):
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            fx = self.objective(x, batch)
        grad = tape.gradient(fx, x)
        if not second_order:
            grad = tf.stop_gradient(grad)
        return fx, grad

    # ---- optional evaluation / training hooks
    def fingerprint(self, batches: List[Batch]) -> Optional[Dict[str, Any]]:
        """``core.fingerprint`` dict over the instances walked, or None if the
        problem has no file-backed instances to hash."""
        return None

    def recovery_metrics(self, batches: List[Batch], x_pred: np.ndarray,
                         eval_section: Dict[str, Any]) -> Dict[str, Any]:
        """Problem-specific metrics of the final iterate (LASSO: the four Table-4 metrics)."""
        return {}

    def write_eval_artifacts(self, result, output_path: str, tag: str) -> Dict[str, str]:
        """Files in the library script's own layout, for diff-based parity tests."""
        return {}

    def training_recovery(self, batch: Batch, x_pred: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Arrays for the training scripts' ``recovery.npz`` (written after each
        in-training evaluation), or None if the script writes none for this problem."""
        return None


class ProblemDialect(abc.ABC):
    """A ``config["problem"]`` name with its config vocabulary and how it turns a
    config into problems."""

    name: str
    #: "model_free" dialects are used by learned optimizers (train/eval sections
    #: carry the problem keys), "unrolled" by unrolled networks (data_dir/test_files)
    kind: str
    train_schema: ConfigSchema = ConfigSchema()
    eval_schema: ConfigSchema = ConfigSchema()

    @abc.abstractmethod
    def train_problem(self, config: Dict[str, Any]) -> Problem: ...

    def val_problem(self, config: Dict[str, Any]) -> Optional[Problem]:
        """Validation instances for curricula that need them (unrolled models)."""
        return None

    @abc.abstractmethod
    def eval_problems(self, config: Dict[str, Any]) -> List[Tuple[str, Problem]]:
        """``[(label, problem)]`` -- one per evaluation split/file; the first is the headline."""


PROBLEMS: Dict[str, ProblemDialect] = {}


def register_problem(dialect: ProblemDialect) -> None:
    if dialect.name in PROBLEMS:
        raise ValueError("problem dialect {!r} already registered".format(dialect.name))
    PROBLEMS[dialect.name] = dialect


def get_dialect(name: str, kind: str, method_name: str) -> ProblemDialect:
    if name not in PROBLEMS:
        raise NotImplementedError("{}: no unified implementation of problem {!r} (known: {})".format(
            method_name, name, sorted(PROBLEMS)))
    dialect = PROBLEMS[name]
    if dialect.kind != kind:
        raise ValueError("{}: problem {!r} is a {} dialect, but this is a {} method".format(
            method_name, name, dialect.kind, kind))
    return dialect
