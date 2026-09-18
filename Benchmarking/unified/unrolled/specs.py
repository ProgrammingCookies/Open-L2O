"""``ModelSpec`` records for LISTA and ALISTA (TF-free; everything heavy is
imported inside the callables)."""

from __future__ import annotations

from typing import Any, Dict

from unified.model import ModelSpec
from unified.schema import ConfigSchema


def _build(model_name):
    def build(problem, ident: Dict[str, Any], create_cells: bool = True):
        from unified.tasks.lasso import LassoProblem
        from unified.unrolled.lista import build_model
        if not isinstance(problem, LassoProblem):
            raise TypeError("{} solves LASSO only (got {})".format(model_name, type(problem).__name__))
        return build_model(model_name, problem.a, int(ident["num_layers"]), float(ident["model_lam"]),
                           w_alista=problem.w_alista, ss_q_per_layer=float(ident["ss_q_per_layer"]),
                           ss_maxq=float(ident["ss_maxq"]), share_W=bool(ident["share_W"]),
                           create_cells=create_cells, ident=ident)
    return build


def _load(model, model_dir: str, num_layers: int):
    from unified.unrolled.loader import load_layerwise_checkpoint
    return load_layerwise_checkpoint(model, model_dir, num_layers)


def _train(model, train_problem, val_problem, model_dir: str, section: Dict[str, Any],
           ident: Dict[str, Any], log):
    from unified.unrolled.curriculum import train_curriculum
    return train_curriculum(
        model, train_problem, val_problem, model_dir,
        lasso_lam=float(ident["lasso_lam"]),
        epochs=int(section["epochs"]), patience=int(section["patience"]), base_lr=float(section["base_lr"]),
        train_batch_size=int(section["train_batch_size"]), val_batch_size=int(section["val_batch_size"]),
        keras_train_loss=section["keras_train_loss"], log=log)


def _solver(model):
    from unified.unrolled.lista import UnrolledSolver
    return UnrolledSolver(model)


# Model_Base_L2O/train.py flag defaults
_TRAIN_SCHEMA = ConfigSchema.of(
    defaults={"epochs": 100000, "patience": 8, "base_lr": 0.0005, "train_batch_size": 128,
              "val_batch_size": 1024, "keras_train_loss": "paper"},
    allowed=("seed", "training_curve_dir", "training_curve_test_file"),
    unported=("track_training_curve",),
)
_EVAL_SCHEMA = ConfigSchema.of(allowed=("seed",))
_IDENTITY = ("num_layers", "lasso_lam", "model_lam", "ss_q_per_layer", "ss_maxq", "share_W")
_IDENTITY_DEFAULTS = {"model_lam": 0.4, "ss_q_per_layer": 1.2, "ss_maxq": 13.0, "share_W": False}


def _spec(name, model_name, extra_identity=(), extra_defaults=None):
    identity = _IDENTITY + tuple(extra_identity)
    defaults = {**_IDENTITY_DEFAULTS, **(extra_defaults or {})}
    return ModelSpec(
        name=name, problem="lasso", build=_build(model_name), load_checkpoint=_load, train=_train,
        make_solver=_solver, identity_keys=identity, identity_defaults=defaults,
        train_schema=_TRAIN_SCHEMA.merge(ConfigSchema.of(allowed=identity)),
        eval_schema=_EVAL_SCHEMA)


LISTA_SPEC = _spec("unified-lista", "lista")
ALISTA_SPEC = _spec("unified-alista", "alista")
# x0_at_layer0 is an identity key (see unrolled/lfista.py): False = the library's
# behaviour (parity target), True = the platform's start-point fairness.
LFISTA_SPEC = _spec("unified-lfista", "lfista", extra_identity=("x0_at_layer0",),
                    extra_defaults={"x0_at_layer0": False})
