"""Registration record for unrolled networks (LISTA-style models), the
second kind of method the framework hosts.

An unrolled network is not an ``OptimizerPolicy``: it is a fixed-depth
network whose layer k produces iterate x_k, trained by its own procedure
(a Keras curriculum here) rather than by ``meta_train``. What the framework
needs from it is declared in a ``ModelSpec``: how to build it from a
problem, how to load its checkpoint, how to train it, how to wrap it as an
``IterativeSolver``, and which config keys it understands. The *identity*
keys are the ones read from ``config["train"]`` in both phases, so an
evaluation can never use a different architecture than was trained.

No TensorFlow at import time: the callables import it lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from unified.schema import ConfigSchema


@dataclass(frozen=True)
class ModelSpec:
    name: str                                      # e.g. "unified-lista"
    problem: str                                   # the problem dialect it solves ("lasso")
    #: (problem, identity, create_cells) -> keras model
    build: Callable[[Any, Dict[str, Any], bool], Any]
    #: (model, model_dir, num_layers) -> {variable name: checkpoint key}
    load_checkpoint: Callable[[Any, str, int], Dict[str, str]]
    #: (model, train_problem, val_problem, model_dir, train_section, identity, log) -> per-layer history
    train: Callable[..., Any]
    #: model -> IterativeSolver
    make_solver: Callable[[Any], Any]
    #: keys read from config["train"] in BOTH phases, with defaults
    identity_keys: Tuple[str, ...]
    identity_defaults: Dict[str, Any]
    train_schema: ConfigSchema = ConfigSchema()
    eval_schema: ConfigSchema = ConfigSchema()

    def identity(self, train_section: Dict[str, Any]) -> Dict[str, Any]:
        ident = {}
        for k in self.identity_keys:
            if k in train_section:
                ident[k] = train_section[k]
            elif k in self.identity_defaults:
                ident[k] = self.identity_defaults[k]
            else:
                raise ValueError("{} requires config['train'][{!r}] (identity field, must match between "
                                 "train and evaluate)".format(self.name, k))
        return ident
