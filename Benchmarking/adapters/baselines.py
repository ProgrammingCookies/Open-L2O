"""Adapters for Classical optimizers (Adam / SGD / Adagrad)

A classical optimizer is evaluated on a hosts code configuration so it uses on the exact same optimizee problem and metric as
the learned optimizer it is being compared against. The config selects the host:

    {"method": "adam", "host": "l2o-scale", "problem": "mnist_mlp", ...}

Note: the L2O-Scale has already support for SGD/Adam/Adagrad
While the DM/RNNPROP and Swarm code only have an Adam classical optimizer atm (see each adapter's run_classical_optimizer_eval).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.method import L2OMethod
from core.registry import get_method
from core.result import EvalOutput


class ClassicalOptimizer(L2OMethod):
    trainable = False

    def __init__(self, name: str, classical: str):
        self.classical = classical
        self.name = name

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        host_name = config.get("host")
        if not host_name:
            raise ValueError(
                "classical optimizer {!r} requires a 'host' key naming the L2O "
                "library whose eval harness to run it through "
                "(e.g. \"host\": \"l2o-scale\")".format(self.name))
        host = get_method(host_name)
        if not hasattr(host, "run_classical_optimizer_eval"):
            raise ValueError(
                "host {!r} does not support classical optimizers".format(host_name))
        return host.run_classical_optimizer_eval(config, self.classical, output_dir)
