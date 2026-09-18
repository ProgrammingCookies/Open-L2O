"""``quadratic``: L2O-DM's built-in toy problem, ``f(x) = ||W x - y||^2``.

Port of ``Model_Free_L2O/L2O-DM and L2O-RNNProp/problems.py::quadratic``
(``batch_size=128, num_dims=10, stddev=0.01``, fixed by ``util.get_config``).
A library-generated problem: no data files, no ground truth, no recovery
metrics, no fingerprint -- the second problem of the unified framework and
the one that exercises every hook LASSO does not.

Bit-exact parity with the scripts depends on two things that must not be
"simplified":

* RNG order per ``build()``: ``x0 ~ tf.random.normal(stddev)``, then
  ``W ~ tf.random.uniform([B, d, d])``, then ``y ~ tf.random.uniform([B, d])``,
  all from the global TF generator after ``tf.random.set_seed(seed)``.
* Op sequence: ``product = tf.squeeze(tf.matmul(W, tf.expand_dims(x, -1)))``,
  ``tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))``.

Evaluation follows ``evaluate_dm.py``'s non-LASSO branch: ``eval.num_epochs``
(default 1) fresh problems, per-step costs of every epoch *appended*
(``loss_record.extend``) rather than averaged, no artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from unified.problem import Batch, Problem, ProblemDialect
from unified.schema import ConfigSchema


class QuadraticProblem(Problem):
    eval_reduction = "concat"

    def __init__(self, batch_size: int = 128, num_dims: int = 10, stddev: float = 0.01, num_epochs: int = 1):
        self.batch_size, self.num_dims, self.stddev = int(batch_size), int(num_dims), float(stddev)
        self.num_epochs = int(num_epochs)

    def sample(self) -> Batch:
        import tensorflow as tf
        x = tf.random.normal([self.batch_size, self.num_dims], stddev=self.stddev, dtype=tf.float32)
        w = tf.random.uniform([self.batch_size, self.num_dims, self.num_dims], dtype=tf.float32)
        y = tf.random.uniform([self.batch_size, self.num_dims], dtype=tf.float32)
        return Batch(x0=x.numpy(), data={"w": w, "y": y})

    def walk(self) -> Iterator[Batch]:
        for _ in range(self.num_epochs):
            yield self.sample()

    def num_chunks(self) -> int:
        return self.num_epochs

    def per_sample(self, x, batch: Batch):
        import tensorflow as tf
        product = tf.squeeze(tf.matmul(batch.data["w"], tf.expand_dims(x, -1)))
        return tf.reduce_sum((product - batch.data["y"]) ** 2, 1)


class QuadraticDialect(ProblemDialect):
    name = "quadratic"
    kind = "model_free"
    train_schema = ConfigSchema()
    eval_schema = ConfigSchema.of(defaults={"num_epochs": 1})

    def train_problem(self, config: Dict[str, Any]) -> Problem:
        return QuadraticProblem()

    def eval_problems(self, config: Dict[str, Any]) -> List[Tuple[str, Problem]]:
        return [("quadratic", QuadraticProblem(num_epochs=int(config["eval"].get("num_epochs", 1))))]
