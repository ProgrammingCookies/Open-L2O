"""The one LASSO objective / gradient oracle.

    f(x) = 0.5 * ||A x - b||_2^2 + lam * ||x||_1        (per instance)

Every library computes this with the same ops; the copies it replaces:
  * DM/RNNProp ``problems.lasso_from_dataset``'s ``loss_fn`` (batch mean)
  * Scale ``pg.Lasso.objective`` (batch mean)
  * Model_Base ``utils.LassoObjective`` metric (per-sample, then Keras Mean)
  * ``core.lasso_metrics.lasso_objective`` (numpy, scoring)

The model-free libraries differentiate the BATCH-MEAN objective and feed that
gradient to the learned optimizer; ``value_and_grad`` reproduces exactly that
(including the per-batch 1/B factor in the gradient).
"""

from __future__ import annotations

import numpy as np


class LassoObjective:
    def __init__(self, a: np.ndarray, lam: float):
        import tensorflow as tf
        self.a = tf.constant(np.asarray(a, dtype=np.float32))
        self.lam = float(lam)
        self.m, self.n = a.shape

    def per_sample(self, x, b):
        """(B,) objective values -- ``tf.norm(ord=1)`` is what problems.py uses."""
        import tensorflow as tf
        residual = tf.matmul(x, self.a, transpose_b=True) - b
        left_term = 0.5 * tf.reduce_sum(residual ** 2, axis=1)
        other_term = self.lam * tf.norm(x, ord=1, axis=1)
        return left_term + other_term

    def batch_mean(self, x, b):
        import tensorflow as tf
        return tf.reduce_mean(self.per_sample(x, b))

    def value_and_grad(self, x, b, second_order: bool = False):
        """``(f(x) batch-mean, df/dx)`` with the gradient detached unless
        ``second_order`` (the libraries' ``second_derivatives`` flag)."""
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            fx = self.batch_mean(x, b)
        grad = tape.gradient(fx, x)
        if not second_order:
            grad = tf.stop_gradient(grad)
        return fx, grad
