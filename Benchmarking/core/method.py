"""The interface every benchmark adapter implements.

Two flavors of method exist in this codebase, and both fit the same interface:

* **Learned optimizers** (``trainable = True``): L2O-Scale, L2O-Entropy,
  L2O-Jacobian, L2O-DM, L2O-RNNProp, L2O-Swarm. ``train`` produces a checkpoint;
  ``evaluate`` runs that checkpoint on held-out problems and records a loss
  trajectory.

* **Classical optimizers** (``trainable = False``): Adam / SGD / Adagrad. They
  have nothing to train, so ``train`` is a no-op; ``evaluate`` runs the classical
  optimizer through a *host* library's evaluation harness so the numbers are
  directly comparable to the learned optimizer measured on the same problem.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from core.result import EvalOutput, TrainOutput


class L2OMethod(abc.ABC):
    """Base class for anything that can be benchmarked."""

    #Name of method, for example: "l2o-scale", "l2o-dm", "adam"
    name: str = ""

    #: whether ``train`` does real work. Classical optimizers set this to false.
    trainable: bool = True

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        """Train the optimizer described by ``config``; 
        return a checkpoint handle.

        Non-trainable methods inherit this no-op.
        """
        return TrainOutput(checkpoint=None)

    @abc.abstractmethod
    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        """
        Evaluates on the config's problem.
        return normalized eval metrics.
        """
        raise NotImplementedError
