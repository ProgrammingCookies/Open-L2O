"""Learned-optimizer *policies*: the "x_{t+1} = x_t + net(features(grad))" step.

Ported from the DM/RNNProp lineage's four near-duplicate ``MetaOptimizer``
classes (``meta.py``, ``meta_dm_train.py``, ``meta_rnnprop_train.py``,
``meta_rnnprop_eval.py``) collapsed to one class per optimizer:

  * ``DMPolicy``       -- ``meta.py:100-118`` ``_apply_step``: delta = net(g)
  * ``RNNPropPolicy``  -- ``meta_rnnprop_eval.py:93-125``: Adam-normalised
                          (m~, g~) features with bias correction at step ``t``
  * ``ClassicalAdamPolicy`` -- the ``--optimizer Adam`` baseline branch of
                          ``evaluate_dm.py:87-96``

All LASSO configurations use a single network over the single optimizee
tensor ``x`` (``net_assignments=None`` in ``util.get_config``), so the
per-subset bookkeeping of the originals is not reproduced.

Network hyper-parameters are the ones ``util.get_config`` wires for
``problem="lasso_dataset"``: NOT ``meta.py``'s LogAndSign/0.01 defaults --
that config would have a different tensor count and cannot load the
checkpoints Experiment 2 produced.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Tuple

import numpy as np

from unified import nets_dm


class OptimizerPolicy(abc.ABC):
    #: checkpoint file stem ("cw" -> cw.l2l-<idx>, "rp" -> rp.l2l-<idx>), from util.get_config
    net_key: str = ""

    @abc.abstractmethod
    def initial_state(self, x) -> Any: ...

    @abc.abstractmethod
    def step(self, x, grad, state, t: int) -> Tuple[Any, Any]:
        """Return ``(x_next, state_next)``; ``t`` is 1 on the first call."""

    @property
    def trainable_variables(self):
        return []

    def build(self, x_example) -> None:
        """Run one throw-away step so Keras creates the weights (needed before load)."""
        import tensorflow as tf
        x = tf.identity(x_example)
        self.step(x, tf.zeros_like(x), self.initial_state(x), 1)

    def checkpoint_file(self, path: str, index: int = 0) -> str:
        return os.path.join(path, "{}.l2l-{}".format(self.net_key, index))

    def save(self, path: str, index: int) -> None:
        raise NotImplementedError

    def restore(self, path: str, index: int = 0) -> None:
        raise NotImplementedError


class _NetPolicy(OptimizerPolicy):
    """Common save/restore for policies wrapping one ``nets_dm`` network."""

    net = None

    @property
    def trainable_variables(self):
        return list(self.net.trainable_variables)

    def save(self, path: str, index: int) -> None:
        os.makedirs(path, exist_ok=True)
        nets_dm.save(self.net, self.checkpoint_file(path, index))

    def restore(self, path: str, index: int = 0) -> None:
        filename = self.checkpoint_file(path, index)
        if not os.path.exists(filename):
            # evaluate_{dm,rnnprop}.py silently evaluate an UNTRAINED net when
            # the file is missing -- exactly the failure class RESULTS_VALIDITY.md
            # documents. The unified path refuses instead.
            raise FileNotFoundError(
                "{}: no checkpoint {} -- refusing to evaluate an untrained "
                "optimizer".format(type(self).__name__, filename))
        nets_dm.load(self.net, filename)


class DMPolicy(_NetPolicy):
    net_key = "cw"

    def __init__(self, layers=(20, 20)):
        # util.get_config "lasso_dataset": CoordinateWiseDeepLSTM(layers=(20,20))
        # with StandardDeepLSTM defaults preprocess="identity", scale=1.0,
        # tanh_output=False.
        self.net = nets_dm.CoordinateWiseDeepLSTM(layers=tuple(layers))

    def initial_state(self, x):
        return self.net.initial_state_for_inputs(x)

    def step(self, x, grad, state, t):
        delta, state_next = self.net(grad, state)
        return x + delta, state_next


class RNNPropPolicy(_NetPolicy):
    net_key = "rp"

    def __init__(self, beta1=0.95, beta2=0.95, layers=(20, 20)):
        # util.get_config net_name="RNNprop"
        self.net = nets_dm.RNNprop(layers=tuple(layers), preprocess_name="fc",
                                   preprocess_options={"dim": 20}, scale=0.01,
                                   tanh_output=True)
        self.beta1 = beta1
        self.beta2 = beta2

    def initial_state(self, x):
        import tensorflow as tf
        return (self.net.initial_state_for_inputs(x), tf.zeros_like(x), tf.zeros_like(x))

    def step(self, x, grad, state, t):
        import tensorflow as tf
        rnn_state, mt, vt = state
        b1 = tf.cast(self.beta1, tf.float32)
        b2 = tf.cast(self.beta2, tf.float32)
        t_f = tf.cast(t, tf.float32)
        mt_next = b1 * mt + (1.0 - b1) * grad
        mt_hat = mt_next / (1.0 - tf.pow(b1, t_f))
        vt_next = b2 * vt + (1.0 - b2) * grad * grad
        vt_hat = vt_next / (1.0 - tf.pow(b2, t_f))
        mt_tilde = mt_hat / (tf.sqrt(vt_hat) + 1e-8)
        gt_tilde = grad / (tf.sqrt(vt_hat) + 1e-8)
        delta, rnn_next = self.net(mt_tilde, gt_tilde, rnn_state)
        return x + delta, (rnn_next, mt_next, vt_next)


class ClassicalAdamPolicy(OptimizerPolicy):
    """``tf.keras.optimizers.Adam`` driven through the same step interface.

    Keras optimizers update a ``tf.Variable`` in place, so the policy owns one
    and copies the functional ``x`` through it each step -- this keeps the
    trajectory bit-identical to ``evaluate_dm.py``'s Adam branch while
    fitting the ``IterativeSolver`` contract.
    """

    net_key = "adam"

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self._var = None
        self._opt = None

    def initial_state(self, x):
        import tensorflow as tf
        # Fresh optimizer per run: Keras 3 optimizers cannot be re-built on
        # new variables, and every chunk of the eval walk is a new run.
        self._var = tf.Variable(x)
        self._opt = tf.keras.optimizers.Adam(self.learning_rate)
        return None

    def step(self, x, grad, state, t):
        import tensorflow as tf
        self._var.assign(x)
        self._opt.apply_gradients([(grad, self._var)])
        return tf.identity(self._var), state

    def build(self, x_example) -> None:
        pass

    def restore(self, path, index=0):
        pass
