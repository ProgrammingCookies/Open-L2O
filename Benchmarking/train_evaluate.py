"""Bootstrap shim: puts Benchmarking/ on sys.path, registers all methods via
``adapters``, and re-exports ``train_evaluate`` from ``core.runner`` so it can be
imported as ``from train_evaluate import train_evaluate`` from anywhere.
    Example usage:
    from train_evaluate import train_evaluate
    result = train_evaluate({
        "method": "l2o-scale",
        "problem": "mnist_mlp",
        "train": {"num_meta_iterations": 5},
        "eval":  {"num_testing_itrs": 100},
    })
    print(result.eval_metrics["final_loss"])
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adapters  # noqa: F401  (registers all methods on import)
from core.runner import train_evaluate  # noqa: F401  (re-exported)

__all__ = ["train_evaluate"] 
