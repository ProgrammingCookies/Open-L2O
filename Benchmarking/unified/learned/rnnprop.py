"""L2O-RNNProp (Lv, Jiang & Li 2017) as a policy.

Port of ``meta_rnnprop_eval.py:93-125``: the network sees Adam-normalised
``(m~, g~)`` computed from running moments ``mt, vt`` (beta1 = beta2 = 0.95)
with bias correction at the global step ``t`` (1 on the first update), and
``x' = x + net(m~, g~)``. The moments are part of the policy state.
Network: ``RNNprop(layers=(20, 20), preprocess="fc"(20), scale=0.01, tanh)``
(``util.get_config``, ``net_name="RNNprop"``).
"""

from __future__ import annotations

from unified.policy import (MODEL_FREE_EVAL_SCHEMA, MODEL_FREE_TRAIN_SCHEMA, NetPolicy,
                            PolicySpec)
from unified.learned.classical import DM_SCRIPT_CLASSICAL
from unified.schema import ConfigSchema


class RNNPropPolicy(NetPolicy):
    net_key = "rp"

    def __init__(self, beta1=0.95, beta2=0.95, layers=(20, 20)):
        from unified.learned import nets_dm
        self.net = nets_dm.RNNprop(layers=tuple(layers), preprocess_name="fc",
                                   preprocess_options={"dim": 20}, scale=0.01, tanh_output=True)
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


_BETAS = ConfigSchema.of(defaults={"beta1": 0.95, "beta2": 0.95})

RNNPROP_SPEC = PolicySpec(
    classical=DM_SCRIPT_CLASSICAL,
    name="unified-rnnprop",
    make_policy=lambda section: RNNPropPolicy(beta1=float(section["beta1"]), beta2=float(section["beta2"])),
    # --last_step_loss and --lasso_x0_mode exist only in the RNNProp scripts
    train_schema=MODEL_FREE_TRAIN_SCHEMA.merge(_BETAS).merge(
        ConfigSchema.of(defaults={"last_step_loss": False}, allowed=("lasso_x0_mode",))),
    eval_schema=MODEL_FREE_EVAL_SCHEMA.merge(_BETAS).merge(ConfigSchema.of(allowed=("lasso_x0_mode",))),
    train_defaults={"evaluation_period": 10},
)
