"""The extension point of the fairness tests: one ``Entry`` per problem.

The tests in this directory do not know what a problem instance is. They know
three things about an ``Entry``: how to build a tiny fixture for it, how to
run each method on that fixture, and what the run must satisfy. To make a
new problem fairness-testable, add an ``Entry`` here (and, if the problem
needs generated data, a ``make_fixture`` function); see ``CONTRIBUTING.md``
§12. Nothing else in ``tests/`` changes.

Each method builder returns the config the adapter needs, *without*
``method``/``seed``/``run_name``/``output_dir`` (the test fills those in). A
config with a ``train`` section is trained first; one without is evaluated
directly, which is how the model-free adapters run their classical-optimizer
path with no training at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from data.lasso import Experiment2Config, generate_experiment2_dataset, _test_filename


@dataclass
class Fixture:
    """A generated dataset the methods of one ``Entry`` are run on."""
    name: str
    root: str                       # directory the data lives in
    num_instances: int              # rows every method must evaluate exactly once
    parts: Dict[str, Any] = field(default_factory=dict)  # whatever the builders need


@dataclass
class Entry:
    name: str
    #: method name -> builder(fixture, seed) -> config dict (see module docstring)
    methods: Dict[str, Callable[[Optional[Fixture], int], Dict[str, Any]]]
    #: (root_dir, seed) -> Fixture, or None when the problem is built by the
    #: library itself from the seed (e.g. L2O-DM's "quadratic").
    make_fixture: Optional[Callable[[str, int], Fixture]] = None
    #: True when the evaluation seed selects the instances (library-generated
    #: problems); False when instances come from files and the seed acts at
    #: dataset generation (checked by test_dataset_generation.py instead).
    seed_selects_eval_instances: bool = False
    #: When True, every method above must report an instances fingerprint and
    #: they must all agree. False disables the cross-method check (Tier B) and
    #: keeps only determinism (Tier A).
    fingerprint_required: bool = True
    #: Tolerance for the same-seed trajectory comparison. 0.0 = bit-exact.
    trajectory_atol: float = 0.0


# --------------------------------------------------------------------------
# LASSO (Experiment 2's problem): tiny instance of Benchmarking/data/lasso.py
# --------------------------------------------------------------------------

_LASSO_M, _LASSO_N, _LASSO_LAM = 10, 20, 0.005
_LASSO_TEST_P = 0.10
_LASSO_TEST_SIZE = 64
_LASSO_BATCH = 32
_FISTA_ITERS = 500      # tiny problem: converges in far fewer; the cap is irrelevant


def make_lasso_tiny(root: str, seed: int) -> Fixture:
    seed_dir = generate_experiment2_dataset(
        Experiment2Config(seed=seed, m=_LASSO_M, n=_LASSO_N, lam=_LASSO_LAM,
                          train_size=256, val_size=64, test_size=_LASSO_TEST_SIZE,
                          train_sparsities={"narrow": _LASSO_TEST_P},
                          test_sparsities=[_LASSO_TEST_P],
                          compute_alista_w=True),
        os.path.join(root, "lasso_tiny"))
    width_dir = os.path.join(seed_dir, "narrow")
    return Fixture(name="lasso_tiny", root=seed_dir, num_instances=_LASSO_TEST_SIZE,
                   parts={"width_dir": width_dir, "test_file": _test_filename(_LASSO_TEST_P),
                          "lam": _LASSO_LAM})


def _lasso_model_free(fixture: Fixture, seed: int, extra_eval: Dict[str, Any]) -> Dict[str, Any]:
    # Classical Adam through the DM/RNNProp harness: no checkpoint, no training.
    return {
        "problem": "lasso_dataset",
        "eval": {
            "optimizer": "Adam",
            "num_steps": 10,
            "lasso_data_dir": fixture.parts["width_dir"],
            "lasso_split": fixture.parts["test_file"],
            "lasso_batch_size": _LASSO_BATCH,
            "lasso_lam": fixture.parts["lam"],
            "num_fista_iters": _FISTA_ITERS,
            **extra_eval,
        },
    }


def _lasso_rnnprop(fixture: Fixture, seed: int) -> Dict[str, Any]:
    return _lasso_model_free(fixture, seed, {"beta1": 0.95, "beta2": 0.95})


def _lasso_dm(fixture: Fixture, seed: int) -> Dict[str, Any]:
    return _lasso_model_free(fixture, seed, {})


def _lasso_scale(fixture: Fixture, seed: int) -> Dict[str, Any]:
    # Classical Adam through L2O-Scale's evaluate.py on the same split.
    return {
        "problem": "lasso_dataset",
        "eval": {
            "test_optimizer": "Adam",
            "num_testing_itrs": 10,
            "lasso_data_dir": fixture.parts["width_dir"],
            "lasso_split": fixture.parts["test_file"],
            "lasso_batch_size": _LASSO_BATCH,
            "lasso_lam": fixture.parts["lam"],
            "num_fista_iters": _FISTA_ITERS,
        },
    }


def _lasso_model_based(fixture: Fixture, seed: int) -> Dict[str, Any]:
    # LISTA/ALISTA have no classical path: train one layer for one epoch.
    return {
        "problem": "lasso",
        "train": {
            "data_dir": fixture.parts["width_dir"],
            "num_layers": 1,
            "lasso_lam": fixture.parts["lam"],
            "model_lam": fixture.parts["lam"],
            "epochs": 1,
            "patience": 1,
            "base_lr": 5e-4,
            "train_batch_size": _LASSO_BATCH,
        },
        "eval": {
            "data_dir": fixture.parts["width_dir"],
            "test_files": [fixture.parts["test_file"]],
            "num_fista_iters": _FISTA_ITERS,
        },
    }


# --------------------------------------------------------------------------
# "quadratic": L2O-DM's built-in toy problem; W, y and x0 are drawn from
# tf.random, so here the evaluation seed is what selects the instances. No
# files, no fingerprint (the adapter has nothing to hash), Tier A only.
# ("simple" would not do: it is f(x) = x^2 from x = 1, no randomness at all.)
# --------------------------------------------------------------------------

def _quadratic_rnnprop(fixture: Optional[Fixture], seed: int) -> Dict[str, Any]:
    return {"problem": "quadratic",
            "eval": {"optimizer": "Adam", "num_steps": 20, "beta1": 0.95, "beta2": 0.95}}


ENTRIES: List[Entry] = [
    Entry(
        name="lasso_tiny",
        make_fixture=make_lasso_tiny,
        methods={
            "l2o-rnnprop": _lasso_rnnprop,
            "l2o-dm": _lasso_dm,
            "l2o-scale": _lasso_scale,
            "lista": _lasso_model_based,
            "alista": _lasso_model_based,
        },
        seed_selects_eval_instances=False,
        fingerprint_required=True,
    ),
    Entry(
        name="quadratic",
        make_fixture=None,
        methods={"l2o-rnnprop": _quadratic_rnnprop, "unified-rnnprop": _quadratic_rnnprop},
        seed_selects_eval_instances=True,
        fingerprint_required=False,
    ),
    # The in-process unified-* methods (Benchmarking/unified) on the same tiny
    # LASSO fixture, with one adapter-backed method as the reference so Tier B
    # proves the two code paths score the very same instances. Kept as a
    # separate entry so the adapter-only ``lasso_tiny`` results stay comparable
    # across time.
    Entry(
        name="lasso_tiny_unified",
        make_fixture=make_lasso_tiny,
        methods={
            "l2o-rnnprop": _lasso_rnnprop,
            "unified-rnnprop": _lasso_rnnprop,
            "unified-dm": _lasso_dm,
            "unified-lista": _lasso_model_based,
            "unified-alista": _lasso_model_based,
        },
        seed_selects_eval_instances=False,
        fingerprint_required=True,
    ),
]


def entry_by_name(name: str) -> Entry:
    for e in ENTRIES:
        if e.name == name:
            return e
    raise KeyError(name)
