"""LASSO: the problem specs and recovery metrics.

Two ``config["problem"]`` vocabularies name the same problem, because the
libraries do: the model-free adapters (L2O-DM/RNNProp, L2O-Scale) call it
``"lasso_dataset"`` and read it from ``config["eval"]``, the model-based
adapter (LISTA/ALISTA) calls it ``"lasso"`` and keeps ``lasso_lam`` in
``config["train"]`` (it is an identity field of the trained model there). Both
specs share the fingerprint components and the default metric, so every
method's ``x_pred`` is scored by the identical code against the identical
FISTA-computed x*.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from core.config import require
from core.lasso_metrics import compute_lasso_metrics
from core.metrics import register_metric
from core.problems import EvalRecord, ProblemSpec, register_problem

#: Arrays a LASSO evaluation provides, instances along axis 0.
LASSO_ARRAYS = ("x_pred", "b", "x_true")
#: What identifies the instances evaluated: the measurements, the ground
#: truth and the starting point, in the order the script walked them.
LASSO_FINGERPRINT = ("b", "x_true", "x0")


def _load_a(data_dir: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, "A.npy")).astype(np.float32)


def _lasso_dataset_params(config: Dict[str, Any], who: str) -> Dict[str, Any]:
    eval_cfg = config.get("eval", {})
    data_dir = require(eval_cfg, "lasso_data_dir",
        "{} requires config['eval']['lasso_data_dir'] when problem='lasso_dataset'".format(who))
    lam = require(eval_cfg, "lasso_lam",
        "{} requires config['eval']['lasso_lam'] when problem='lasso_dataset' -- must "
        "match the lambda used to generate the dataset (see Benchmarking/data/lasso.py)"
        .format(who))
    return {"a": _load_a(data_dir), "lam": lam}


def _lasso_model_based_params(config: Dict[str, Any], who: str) -> Dict[str, Any]:
    data_dir = config.get("eval", {}).get("data_dir") or config.get("data_dir")
    if data_dir is None:
        raise ValueError("{} requires config['eval']['data_dir'], see "
                         "Benchmarking/data/lasso.py".format(who))
    # config["train"], not config["eval"]: the lambda the checkpoint was trained
    # for (adapters/model_based.py warns about and ignores an eval override).
    lam = require(config.get("train", {}), "lasso_lam",
        "{} requires config['train']['lasso_lam'] and must match between train and "
        "evaluate ".format(who))
    return {"a": _load_a(data_dir), "lam": lam}


@register_metric("lasso_recovery", config_keys=("num_fista_iters", "xstar_cache_dir"))
def lasso_recovery(record: EvalRecord) -> Dict[str, Any]:
    """The four Experiment-2 metrics (core/lasso_metrics.py): suboptimality
    gap, modified relative loss, NMSE vs x_true, LASSO-optimal recovery error."""
    num_fista_iters = require(record.eval_cfg, "num_fista_iters",
        "{} requires config['eval']['num_fista_iters'] (FISTA iteration count for the x* "
        "reference solve -- see core/lasso_metrics.py)".format(record.who))
    arr = record.arrays
    return compute_lasso_metrics(
        record.params["a"], arr["b"], arr["x_true"], arr["x_pred"], record.params["lam"],
        num_fista_iters=num_fista_iters,
        xstar_cache_dir=record.eval_cfg.get("xstar_cache_dir")).to_dict()


register_problem(ProblemSpec(
    name="lasso_dataset",
    required_arrays=LASSO_ARRAYS,
    fingerprint_arrays=LASSO_FINGERPRINT,
    load_params=_lasso_dataset_params,
    default_metrics=("lasso_recovery",),
    description="LASSO on a Benchmarking/data/lasso.py dataset (model-free adapters)",
))

register_problem(ProblemSpec(
    name="lasso",
    required_arrays=LASSO_ARRAYS,
    fingerprint_arrays=LASSO_FINGERPRINT,
    load_params=_lasso_model_based_params,
    default_metrics=("lasso_recovery",),
    description="LASSO on a Benchmarking/data/lasso.py dataset (model-based adapter)",
))
