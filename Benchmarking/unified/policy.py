"""The learned-optimizer protocol and its declarative registration record.

A learned optimizer is one ``OptimizerPolicy`` subclass: it owns its network
and its state layout, and exposes ``x_{t+1}, state' = step(x_t, grad, state, t)``.
The training loop (``meta_train``) and the evaluation walk (``evalwalk``)
know nothing else about it. A ``PolicySpec`` is what ``unified/__init__.py``
registers: the method name, how to build the policy from a config section,
the config keys the method understands, and the training-loop defaults the
corresponding library script uses.

Bit-exactness with the library scripts rests on two habits every policy
keeps: the network is created lazily on the first ``step`` (Keras draws its
initial weights then, in the scripts' order), and checkpoints use the
index-keyed pickle contract of ``learned/nets_dm.py``.

No TensorFlow at import time.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

from unified.schema import ConfigSchema


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


class NetPolicy(OptimizerPolicy):
    """Common save/restore for policies wrapping one ``nets_dm`` network."""

    net = None

    @property
    def trainable_variables(self):
        return list(self.net.trainable_variables)

    def save(self, path: str, index: int) -> None:
        from unified.learned import nets_dm
        os.makedirs(path, exist_ok=True)
        nets_dm.save(self.net, self.checkpoint_file(path, index))

    def restore(self, path: str, index: int = 0) -> None:
        from unified.learned import nets_dm
        filename = self.checkpoint_file(path, index)
        if not os.path.exists(filename):
            # evaluate_{dm,rnnprop}.py silently evaluate an UNTRAINED net when
            # the file is missing -- exactly the failure class RESULTS_VALIDITY.md
            # documents. The unified path refuses instead.
            raise FileNotFoundError("{}: no checkpoint {} -- refusing to evaluate an untrained "
                                    "optimizer".format(type(self).__name__, filename))
        nets_dm.load(self.net, filename)


@dataclass(frozen=True)
class PolicySpec:
    """Everything the framework needs to register a learned optimizer."""
    name: str                                          # e.g. "unified-rnnprop"
    make_policy: Callable[[Dict[str, Any]], OptimizerPolicy]   # from the train/eval section
    #: method-level config keys (the problem dialect contributes its own)
    train_schema: ConfigSchema = ConfigSchema()
    eval_schema: ConfigSchema = ConfigSchema()
    #: MetaTrainSpec field defaults where the library script's differ from the loop's
    train_defaults: Dict[str, Any] = field(default_factory=dict)
    #: tag used in artifact file names for the learned path ("L2L" in the DM/RNNProp scripts)
    learned_tag: str = "L2L"


# Keys of train_{dm,rnnprop}.py that belong to the method (problem keys live in the dialects).
MODEL_FREE_TRAIN_SCHEMA = ConfigSchema.of(
    required=("num_epochs",),
    allowed=("save_path", "seed", "problem"),
    defaults={"num_steps": 100, "unroll_length": 20, "learning_rate": 0.001, "second_derivatives": False,
              "evaluation_period": 10, "evaluation_epochs": 20},
    unported=("if_cl", "if_mt", "if_scale", "fix_num_unrolls"),
    ignored=("profile_path",),
)
# The unported flags' companions are accepted (a script would) but meaningless without them.
MODEL_FREE_TRAIN_SCHEMA = MODEL_FREE_TRAIN_SCHEMA.merge(ConfigSchema.of(
    allowed=("rd_scale_bound", "min_num_eval", "num_mt", "optimizers", "mt_ratio", "mt_ratios", "k")))

MODEL_FREE_EVAL_SCHEMA = ConfigSchema.of(
    required=("optimizer", "num_steps"),
    allowed=("path", "output_path", "seed", "problem", "num_epochs"),
    defaults={"learning_rate": 0.001},
)
