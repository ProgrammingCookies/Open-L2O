"""Run one (entry, method, seed) live through the platform, for the slow tests."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import adapters  # noqa: F401  -- registers every method
import unified  # noqa: F401  -- registers the in-process unified-* methods
from core.registry import get_method
from core.result import EvalOutput
from tests.fairness_matrix import Entry, Fixture


def run_live(entry: Entry, method_name: str, fixture: Optional[Fixture], seed: int,
             output_dir: str) -> EvalOutput:
    """Train (if the builder gave a ``train`` section) and evaluate, exactly as
    ``core.runner.train_evaluate`` would, but returning the EvalOutput."""
    method = get_method(method_name)
    config: Dict[str, Any] = entry.methods[method_name](fixture, seed)
    config.update({
        "method": method_name,
        "seed": seed,
        "run_name": "{}__{}__s{}".format(entry.name, method_name, seed),
        "output_dir": output_dir,
    })
    # Mirror runner.train_evaluate: the top-level seed is also the eval seed.
    config.setdefault("eval", {}).setdefault("seed", seed)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = None
    if "train" in config and method.trainable:
        checkpoint = method.train(config, output_dir).checkpoint
    return method.evaluate(config, checkpoint, output_dir)
