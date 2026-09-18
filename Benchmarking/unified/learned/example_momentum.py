"""Worked example: adding a learned optimizer to the framework.

A minimal *learned* optimizer -- momentum with a learnable log step size and a
learnable momentum coefficient (two scalars) -- written the way a new method
should be. It is deliberately NOT registered in ``unified/__init__.py``;
``tests/test_unified_example.py`` imports it, trains it on ``quadratic`` and
evaluates it on LASSO, so this file is a recipe that is executed, not just
described. Copy it, rename, register.

The recipe:

1. A network whose trainable variables are created on the first call
   (Keras' lazy build), so ``NetPolicy.save/restore`` -- the index-keyed
   pickle contract shared with the library scripts -- works unchanged.
2. An ``OptimizerPolicy`` subclass with ``initial_state`` / ``step``. The
   policy owns its state layout; the loop only threads it through.
3. A ``PolicySpec``: the method name, ``make_policy`` (TF imported inside),
   the config keys the method accepts, the training-loop defaults and which
   classical baselines its ``eval.optimizer`` may name.
4. (In ``unified/__init__.py``) one line in ``METHODS``.

Everything else -- meta-training, evaluation on every registered problem,
the classical baselines, checkpoint save/load, determinism and fairness
tests -- comes from the framework.
"""

from __future__ import annotations

from unified.learned.classical import DM_SCRIPT_CLASSICAL
from unified.policy import MODEL_FREE_EVAL_SCHEMA, MODEL_FREE_TRAIN_SCHEMA, NetPolicy, PolicySpec
from unified.schema import ConfigSchema


def _make_net(init_log_lr: float, init_momentum: float):
    import tensorflow as tf

    class MomentumNet(tf.keras.layers.Layer):
        """x' = x - exp(log_lr) * v,  v = sigmoid(mu_logit) * v_prev + grad."""

        def build(self, input_shape):
            self.log_lr = self.add_weight(name="log_lr", shape=(), initializer=tf.constant_initializer(init_log_lr))
            self.mu_logit = self.add_weight(name="mu_logit", shape=(),
                                            initializer=tf.constant_initializer(init_momentum))

        def call(self, grad, velocity):
            v = tf.sigmoid(self.mu_logit) * velocity + grad
            return -tf.exp(self.log_lr) * v, v

    return MomentumNet(name="momentum_net")


class LearnedMomentumPolicy(NetPolicy):
    net_key = "mom"

    def __init__(self, init_log_lr: float = -3.0, init_momentum: float = 0.0):
        self.net = _make_net(init_log_lr, init_momentum)

    def initial_state(self, x):
        import tensorflow as tf
        return tf.zeros_like(x)

    def step(self, x, grad, state, t):
        delta, velocity = self.net(grad, state)
        return x + delta, velocity


EXAMPLE_SPEC = PolicySpec(
    name="unified-example-momentum",
    make_policy=lambda section: LearnedMomentumPolicy(float(section["init_log_lr"]),
                                                      float(section["init_momentum"])),
    train_schema=MODEL_FREE_TRAIN_SCHEMA.merge(ConfigSchema.of(defaults={"init_log_lr": -3.0, "init_momentum": 0.0})),
    eval_schema=MODEL_FREE_EVAL_SCHEMA.merge(ConfigSchema.of(defaults={"init_log_lr": -3.0, "init_momentum": 0.0})),
    train_defaults={"evaluation_period": 10},
    classical=DM_SCRIPT_CLASSICAL,    # the Adam baseline the DM/RNNProp scripts provide
)
