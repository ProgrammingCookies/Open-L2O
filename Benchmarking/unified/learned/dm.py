"""L2O-DM (Andrychowicz et al. 2016) as a policy: ``x' = x + net(grad)``.

Port of ``meta.py:100-118`` (``_apply_step``). The network is
``CoordinateWiseDeepLSTM(layers=(20, 20))`` with ``StandardDeepLSTM`` defaults
(identity preprocessing, scale 1.0, no tanh) -- what ``util.get_config`` wires
for every problem the unified slice supports, NOT ``meta.py``'s LogAndSign/0.01
defaults (a different tensor count that could not load Experiment 2's
checkpoints).
"""

from __future__ import annotations

from unified.policy import (MODEL_FREE_EVAL_SCHEMA, MODEL_FREE_TRAIN_SCHEMA, NetPolicy,
                            PolicySpec)
from unified.schema import ConfigSchema


class DMPolicy(NetPolicy):
    net_key = "cw"

    def __init__(self, layers=(20, 20)):
        from unified.learned import nets_dm
        self.net = nets_dm.CoordinateWiseDeepLSTM(layers=tuple(layers))

    def initial_state(self, x):
        return self.net.initial_state_for_inputs(x)

    def step(self, x, grad, state, t):
        delta, state_next = self.net(grad, state)
        return x + delta, state_next


# train_dm.py / evaluate_dm.py have no --beta1/--beta2/--last_step_loss/--lasso_x0_mode
# flags: argparse would have rejected them, so the unified method does too.
_RNNPROP_ONLY = ("beta1", "beta2", "last_step_loss", "lasso_x0_mode")

DM_SPEC = PolicySpec(
    name="unified-dm",
    make_policy=lambda section: DMPolicy(),
    train_schema=MODEL_FREE_TRAIN_SCHEMA.merge(ConfigSchema.of(forbidden=_RNNPROP_ONLY)),
    eval_schema=MODEL_FREE_EVAL_SCHEMA.merge(ConfigSchema.of(forbidden=_RNNPROP_ONLY)),
    train_defaults={"evaluation_period": 100},   # train_dm.py's default (rnnprop: 10)
)
