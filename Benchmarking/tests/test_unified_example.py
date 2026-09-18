"""The framework's extension recipe, executed: ``unified/learned/example_momentum.py``
is a learned optimizer written exactly as a new method should be, NOT
registered anywhere. These tests instantiate it through the generic
``UnifiedModelFreeMethod`` and put it through training on one problem and
evaluation on another -- everything a new method gets for free."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

import adapters  # noqa: F401
import unified  # noqa: F401
from core.fingerprint import METRIC_KEY
from unified.learned.example_momentum import EXAMPLE_SPEC
from unified.methods import UnifiedModelFreeMethod
from tests.fairness_matrix import make_lasso_tiny, _FISTA_ITERS, _LASSO_BATCH


def test_example_spec_is_tf_free_and_validates():
    if "tensorflow" in sys.modules:
        pytest.skip("tensorflow already imported by another test module")
    method = UnifiedModelFreeMethod(EXAMPLE_SPEC)
    assert method.name == "unified-example-momentum"
    section = EXAMPLE_SPEC.train_schema.validate({"num_epochs": 1, "init_log_lr": -2.0}, "x")
    assert section["init_log_lr"] == -2.0 and section["init_momentum"] == 0.0
    with pytest.raises(ValueError, match="unknown config keys"):
        EXAMPLE_SPEC.train_schema.validate({"num_epochs": 1, "typo": 1}, "x")
    assert "tensorflow" not in sys.modules


@pytest.mark.slow
def test_example_trains_on_quadratic_and_evaluates_on_lasso(fixture_root, tmp_path):
    method = UnifiedModelFreeMethod(EXAMPLE_SPEC)
    # train on the library-generated problem
    t_out = method.train({"method": method.name, "problem": "quadratic", "seed": 7,
                          "train": {"num_epochs": 2, "num_steps": 10, "unroll_length": 5,
                                    "evaluation_period": 1, "evaluation_epochs": 1}},
                         os.path.join(str(tmp_path), "train"))
    with open(os.path.join(t_out.checkpoint, "mom.l2l-0"), "rb") as f:
        weights = pickle.load(f)
    assert sorted(weights) == ["0:log_lr", "1:mu_logit"]
    assert np.isfinite(t_out.metrics["final_training_loss"])

    # evaluate the checkpoint on the file-backed problem: fingerprint + recovery metrics appear
    fixture = make_lasso_tiny(fixture_root, 7)
    ev = {"optimizer": "L2L", "num_steps": 10, "lasso_data_dir": fixture.parts["width_dir"],
          "lasso_split": fixture.parts["test_file"], "lasso_batch_size": _LASSO_BATCH,
          "lasso_lam": fixture.parts["lam"], "num_fista_iters": _FISTA_ITERS,
          "xstar_cache_dir": os.path.join(fixture.root, "xstar_cache")}
    e_out = method.evaluate({"method": method.name, "problem": "lasso_dataset", "seed": 7, "eval": ev},
                            t_out.checkpoint, os.path.join(str(tmp_path), "eval"))
    assert len(e_out.trajectory) == 10 and np.all(np.isfinite(e_out.trajectory))
    assert e_out.metrics[METRIC_KEY]["num_instances"] == fixture.num_instances
    assert "nmse_vs_true_mean" in e_out.metrics
    # and the classical baseline path works for it too
    c_out = method.run_classical_optimizer_eval(
        {"method": method.name, "problem": "lasso_dataset", "seed": 7, "eval": ev}, "Adam",
        os.path.join(str(tmp_path), "adam"))
    assert e_out.metrics[METRIC_KEY]["sha256"] == c_out.metrics[METRIC_KEY]["sha256"]
