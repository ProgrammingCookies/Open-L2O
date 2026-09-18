"""Training parity: the unified meta-training loop reproduces the library
training scripts, and its checkpoints are interchangeable with theirs.

Both sides run on this machine (CPU), so the comparison is strict:
``train_{dm,rnnprop}.py`` and ``unified.meta_train`` consume random numbers in
the same order and create the same Keras variables at the same moment, which
makes a same-seed run bit-for-bit identical -- every saved tensor, the
in-training ``recovery.npz`` and every logged loss.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

import adapters  # noqa: F401
import unified  # noqa: F401
from core.registry import get_method
from tests.fairness_matrix import make_lasso_tiny, _FISTA_ITERS, _LASSO_BATCH

_VARIANTS = [
    ("rnnprop", {"beta1": 0.95, "beta2": 0.95}),
    ("rnnprop", {"beta1": 0.95, "beta2": 0.95, "last_step_loss": True}),
    ("dm", {}),
]
_IDS = ["rnnprop", "rnnprop-last_step_loss", "dm"]
_TINY = {"lasso_batch_size": _LASSO_BATCH, "num_epochs": 3, "num_steps": 10, "unroll_length": 5,
         "evaluation_period": 1, "evaluation_epochs": 2, "learning_rate": 1e-3}


def _train_both(variant, extra, fixture, tmp_path, seed=7):
    train = {"lasso_data_dir": fixture.parts["width_dir"], "lasso_lam": fixture.parts["lam"], **_TINY, **extra}
    outs = {}
    for name in ("l2o-" + variant, "unified-" + variant):
        cfg = {"method": name, "problem": "lasso_dataset", "seed": seed, "train": dict(train)}
        outs[name] = get_method(name).train(cfg, os.path.join(str(tmp_path), name))
    return outs


def _logged_losses(path):
    with open(path) as f:
        return [l.strip() for l in f if l.startswith("training_loss=") or l.startswith("epoch=")]


def _train_both_quadratic(variant, extra, tmp_path, seed=7):
    train = {k: v for k, v in _TINY.items() if k != "lasso_batch_size"}
    train.update(extra)
    outs = {}
    for name in ("l2o-" + variant, "unified-" + variant):
        cfg = {"method": name, "problem": "quadratic", "seed": seed, "train": dict(train)}
        outs[name] = get_method(name).train(cfg, os.path.join(str(tmp_path), name))
    return outs


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", _VARIANTS, ids=_IDS)
@pytest.mark.parametrize("problem", ["lasso_dataset", "quadratic"])
def test_same_seed_training_is_bit_identical(variant, extra, problem, fixture_root, tmp_path):
    if problem == "quadratic":
        outs = _train_both_quadratic(variant, extra, tmp_path)
    else:
        fixture = make_lasso_tiny(fixture_root, 7)
        outs = _train_both(variant, extra, fixture, tmp_path)
    key = "rp" if variant == "rnnprop" else "cw"
    weights = {}
    for name, out in outs.items():
        with open(os.path.join(out.checkpoint, key + ".l2l-0"), "rb") as f:
            weights[name] = pickle.load(f)
    a, u = weights["l2o-" + variant], weights["unified-" + variant]
    assert list(a.keys()) == list(u.keys())
    for k in a:
        assert np.array_equal(a[k], u[k]), "tensor {} differs (max |diff| {:.3e})".format(
            k, float(np.abs(a[k] - u[k]).max()))
    if problem == "lasso_dataset":
        ra = np.load(os.path.join(outs["l2o-" + variant].checkpoint, "recovery.npz"))
        ru = np.load(os.path.join(outs["unified-" + variant].checkpoint, "recovery.npz"))
        assert np.array_equal(ra["x_pred"], ru["x_pred"]) and np.array_equal(ra["b"], ru["b"])
    else:   # the scripts write no recovery.npz for problems without ground truth
        assert not os.path.exists(os.path.join(outs["unified-" + variant].checkpoint, "recovery.npz"))
    assert _logged_losses(outs["l2o-" + variant].artifacts["train_log"]) == \
        _logged_losses(outs["unified-" + variant].artifacts["train_log"])


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", _VARIANTS[:1] + _VARIANTS[2:], ids=[_IDS[0], _IDS[2]])
def test_unified_checkpoint_loads_in_adapter(variant, extra, fixture_root, tmp_path):
    """Cross-direction: the ADAPTER evaluates a unified-trained checkpoint and
    gets what the unified evaluator gets."""
    fixture = make_lasso_tiny(fixture_root, 7)
    train = {"lasso_data_dir": fixture.parts["width_dir"], "lasso_lam": fixture.parts["lam"], **_TINY, **extra}
    uni, adapter = get_method("unified-" + variant), get_method("l2o-" + variant)
    ckpt = uni.train({"method": uni.name, "problem": "lasso_dataset", "seed": 7, "train": train},
                     os.path.join(str(tmp_path), "train")).checkpoint
    ev = {"optimizer": "L2L", "num_steps": 10, "lasso_data_dir": fixture.parts["width_dir"],
          "lasso_split": fixture.parts["test_file"], "lasso_batch_size": _LASSO_BATCH,
          "lasso_lam": fixture.parts["lam"], "num_fista_iters": _FISTA_ITERS,
          "xstar_cache_dir": os.path.join(fixture.root, "xstar_cache"),
          **{k: v for k, v in extra.items() if k in ("beta1", "beta2")}}
    cfg = {"method": adapter.name, "problem": "lasso_dataset", "seed": 7, "eval": ev}
    a_out = adapter.evaluate(cfg, ckpt, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(cfg, ckpt, os.path.join(str(tmp_path), "unified"))
    assert np.array_equal(np.asarray(a_out.trajectory), np.asarray(u_out.trajectory))
    assert a_out.metrics["nmse_vs_true_mean"] == pytest.approx(u_out.metrics["nmse_vs_true_mean"], rel=1e-6)


_UNROLLED_TINY = {"num_layers": 2, "epochs": 2, "patience": 1, "base_lr": 5e-4, "train_batch_size": _LASSO_BATCH,
                  "val_batch_size": 64}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["lista", "alista"])
def test_unified_curriculum_checkpoint_loads_in_adapter(model_name, fixture_root, tmp_path):
    """Unified layer-wise training writes the library's checkpoint layout AND
    object graph: ``Model_Base_L2O/train.py --test`` (via the adapter) restores
    it and reports the same per-layer values and x_pred as the unified
    evaluator. Training itself is not bit-reproducible (tf.data
    ``experimental_deterministic=False``), so only the round trip is strict."""
    fixture = make_lasso_tiny(fixture_root, 7)
    uni, adapter = get_method("unified-" + model_name), get_method(model_name)
    train = {"data_dir": fixture.parts["width_dir"], "lasso_lam": fixture.parts["lam"],
             "model_lam": fixture.parts["lam"], **_UNROLLED_TINY}
    run_name = "{}__tiny__s7".format(model_name)
    t_out = uni.train({"method": uni.name, "problem": "lasso", "seed": 7, "run_name": run_name, "train": train},
                      os.path.join(str(tmp_path), "train"))
    assert len(t_out.metrics["layers"]) == 2
    assert all(1 <= e <= 2 for h in t_out.metrics["layers"] for e in h["stage_epochs"])
    assert np.isfinite(t_out.metrics["final_val_lasso"])

    ev = {"data_dir": fixture.parts["width_dir"], "test_files": [fixture.parts["test_file"]],
          "num_fista_iters": _FISTA_ITERS, "xstar_cache_dir": os.path.join(fixture.root, "xstar_cache")}
    cfg = {"method": adapter.name, "problem": "lasso", "seed": 7, "run_name": run_name, "train": train, "eval": ev}
    # The library script writes its outputs next to the checkpoint, so give the
    # adapter its own copy of the checkpoint tree.
    import shutil
    ckpt_copy = os.path.join(str(tmp_path), "ckpt_for_adapter")
    shutil.copytree(t_out.checkpoint, ckpt_copy)
    a_out = adapter.evaluate(cfg, ckpt_copy, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(cfg, t_out.checkpoint, os.path.join(str(tmp_path), "unified"))
    assert len(u_out.trajectory) == 2
    assert np.allclose(a_out.trajectory, u_out.trajectory, rtol=1e-5, atol=1e-6), (a_out.trajectory, u_out.trajectory)
    ra = np.load(a_out.artifacts["final_output__" + fixture.parts["test_file"][:-4]])
    ru = np.load(u_out.artifacts["{}:final_output".format(fixture.parts["test_file"])])
    assert np.allclose(ra, ru, atol=1e-5)
    # the untrained model would give a different first layer than the trained one
    assert a_out.metrics["nmse_vs_true_mean"] == pytest.approx(u_out.metrics["nmse_vs_true_mean"], rel=1e-5)


@pytest.mark.slow
def test_unported_flags_raise(fixture_root, tmp_path):
    fixture = make_lasso_tiny(fixture_root, 7)
    train = {"lasso_data_dir": fixture.parts["width_dir"], "lasso_lam": fixture.parts["lam"], **_TINY, "if_cl": True}
    with pytest.raises(NotImplementedError, match="if_cl"):
        get_method("unified-dm").train({"method": "unified-dm", "problem": "lasso_dataset", "seed": 7,
                                        "train": train}, str(tmp_path))
