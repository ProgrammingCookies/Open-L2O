"""The one interface both method kinds implement.

A learned optimizer (DM / RNNProp / ...) and an unrolled network (LISTA /
ALISTA) are both *iterative solvers*: given a batch of instances and a
starting point they produce x_0, x_1, ..., x_T. What differs is only how
x_{k+1} is computed -- a policy step on the gradient, or the k-th layer.
"""

from __future__ import annotations

import abc
from typing import Iterator

from unified.problem import Batch, Problem


class IterativeSolver(abc.ABC):
    @abc.abstractmethod
    def iterate(self, batch: Batch, num_steps: int) -> Iterator:
        """Yield exactly ``num_steps + 1`` tensors x_0, ..., x_T with x_0 = batch.x0."""

    def new_run(self) -> None:
        """Per-batch reset hook (default: nothing to reset)."""


class PolicySolver(IterativeSolver):
    """Model-free: x_{t+1} = policy.step(x_t, grad f(x_t)).

    Mirrors ``evaluate_dm.py`` / ``evaluate_rnnprop.py``'s inner loop: the
    objective is evaluated at x_t, its gradient fed to the policy, and the
    step counter ``t`` starts at 1 on the first update (RNNProp's Adam bias
    correction depends on that).
    """

    def __init__(self, policy, problem: Problem, second_order: bool = False):
        self.policy = policy
        self.problem = problem
        self.second_order = second_order

    def iterate(self, batch: Batch, num_steps: int) -> Iterator:
        import tensorflow as tf
        x = tf.identity(tf.constant(batch.x0))
        state = self.policy.initial_state(x)
        yield x
        for t in range(1, num_steps + 1):
            _, grad = self.problem.value_and_grad(x, batch, self.second_order)
            x, state = self.policy.step(x, grad, state, t)
            yield x
