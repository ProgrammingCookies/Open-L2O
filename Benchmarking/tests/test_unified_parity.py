"""Checkpoint-load parity: the unified in-process methods reproduce what the
subprocess adapters produced from the SAME checkpoint on the SAME split.

This is the acceptance test of the unified slice. Two sources of checkpoints:

* **Real runs** (``runs/exp2_try3_210x300`` RNNProp, ``runs/exp2_try2`` DM):
  the stored ``L2L_eval_loss_record.pickle-*`` / ``L2L_recovery-*.npz`` of one
  cell is compared against the unified evaluation of that cell's checkpoint.
  Skipped when the run directories are not on this machine. Only the first
  ``--parity-steps`` trajectory points are compared by default (a prefix does
  not depend on the total step count); pass ``--parity-steps 1000`` for the
  full trajectory + ``x_pred`` comparison (~5 min CPU per method).
  Everything is read-only against ``runs/``; outputs go to ``tmp_path``.

* **Tiny CPU checkpoints** trained through the *existing* adapters on the
  ``lasso_tiny`` fixture, then evaluated by both the adapter and the unified
  method.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import adapters  # noqa: F401
import unified  # noqa: F401
from core.fingerprint import METRIC_KEY, fingerprints_agree
from core.registry import get_method
from unified.methods import UnifiedModelFreeMethod
from tests.fairness_matrix import make_lasso_tiny, _FISTA_ITERS, _LASSO_BATCH

_RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
_RTOL, _ATOL = 1e-5, 1e-6

# (variant, run, seed, width, test split, extra eval keys) -- one stored cell each.
_REAL_CELLS = [
    ("rnnprop", "exp2_try3_210x300", 21, "narrow", "test_sparsity_0.10.npy",
     {"beta1": 0.95, "beta2": 0.95}),
    ("dm", "exp2_try2", 21, "narrow", "test_sparsity_0.10.npy", {}),
]


def _real_cell_paths(variant, run, seed, width, split):
    run_dir = os.path.join(_RUNS, run)
    cell = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, "l2o-" + variant)
    p = split[len("test_sparsity_"):-len(".npy")]
    return {
        "checkpoint": os.path.join(cell, "checkpoint"),
        "data_dir": os.path.join(run_dir, "data", "seed{}".format(seed), width),
        "stored": os.path.join(cell, "eval_p{}".format(p), "eval"),
    }


def _assert_close(name, got, want, rtol=_RTOL, atol=_ATOL):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, "{}: shape {} vs stored {}".format(name, got.shape, want.shape)
    finite = np.isfinite(want)
    assert np.array_equal(finite, np.isfinite(got)), "{}: non-finite pattern differs".format(name)
    diff = np.abs(got[finite] - want[finite])
    tol = atol + rtol * np.abs(want[finite])
    assert np.all(diff <= tol), "{}: max |diff| {:.3e} at index {} (stored {:.6g})".format(
        name, float(diff.max()), int(np.argmax(diff)), float(want[finite][np.argmax(diff)]))


# The stored real-run evaluations were computed on the container's GPU; this
# replay runs on CPU. The library scripts themselves, replayed on CPU, differ
# from their own GPU output (up to ~1e-2 per x_pred entry after 1000 float32
# recurrent steps; for the unrolled nets an occasional soft-threshold /
# support flip on one coordinate of one instance) while agreeing with the
# unified path bit-for-bit (verified 2026-09-17: adapter-CPU vs unified-CPU
# max |diff| = 0.0). So against GPU-stored targets x_pred is compared per row
# (at most 1 % of rows may differ) plus the objective it achieves; the strict
# comparison lives in the tiny-checkpoint tests, where both sides run here.
_XPRED_ROW_ATOL, _XPRED_MAX_ROW_FRAC, _OBJ_RTOL = 1e-3, 0.01, 1e-4


def _assert_xpred_vs_gpu_stored(name, got, want, a, b, lam):
    from core.lasso_metrics import lasso_objective
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, name
    rows_off = int((np.abs(got - want).max(axis=1) > _XPRED_ROW_ATOL).sum())
    assert rows_off <= _XPRED_MAX_ROW_FRAC * want.shape[0], (
        "{}: {} of {} rows differ by more than {}".format(name, rows_off, want.shape[0], _XPRED_ROW_ATOL))
    f_got, f_want = lasso_objective(a, got, b, lam).mean(), lasso_objective(a, want, b, lam).mean()
    assert abs(f_got - f_want) <= _OBJ_RTOL * abs(f_want), (
        "{}: mean objective {:.6g} vs stored {:.6g}".format(name, f_got, f_want))


@pytest.mark.slow
@pytest.mark.parametrize("variant,run,seed,width,split,extra", _REAL_CELLS,
                         ids=[c[0] + "-" + c[1] for c in _REAL_CELLS])
def test_real_checkpoint_parity(variant, run, seed, width, split, extra, parity_steps, tmp_path):
    paths = _real_cell_paths(variant, run, seed, width, split)
    ckpt_file = os.path.join(paths["checkpoint"], ("rp" if variant == "rnnprop" else "cw") + ".l2l-0")
    traj_file = os.path.join(paths["stored"], "L2L_eval_loss_record.pickle-lasso_dataset")
    rec_file = os.path.join(paths["stored"], "L2L_recovery-lasso_dataset.npz")
    for f in (ckpt_file, traj_file, rec_file, os.path.join(paths["data_dir"], "A.npy")):
        if not os.path.exists(f):
            pytest.skip("real run artefact missing: {}".format(f))
    # The DM/RNNProp scripts overwrite <key>.l2l-0 as "best" on every
    # improvement, so a checkpoint newer than the stored eval belongs to a
    # later, interrupted continuation of that cell -- not the eval's weights.
    if os.path.getmtime(ckpt_file) > os.path.getmtime(rec_file) + 60:
        pytest.skip("checkpoint {} is newer than the stored eval it would be compared to".format(ckpt_file))

    with open(traj_file, "rb") as f:
        stored_traj = np.asarray(pickle.load(f), np.float64)
    stored_rec = np.load(rec_file)
    steps = min(int(parity_steps), len(stored_traj))

    config = {
        "method": "unified-" + variant, "problem": "lasso_dataset", "seed": seed,
        "eval": {"optimizer": "L2L", "lasso_data_dir": paths["data_dir"], "lasso_split": split,
                 "lasso_batch_size": 128, "num_steps": steps, "lasso_lam": 0.005, **extra},
    }
    out = get_method("unified-" + variant).evaluate(config, paths["checkpoint"], str(tmp_path))

    _assert_close("trajectory", out.trajectory, stored_traj[:steps])
    rec = np.load(out.artifacts["recovery"])
    assert np.array_equal(rec["b"], stored_rec["b"])
    assert np.array_equal(rec["x_true"], stored_rec["x_true"])
    if "x0" in stored_rec.files:
        assert np.array_equal(rec["x0"], stored_rec["x0"])
    if steps == len(stored_traj):
        a = np.load(os.path.join(paths["data_dir"], "A.npy")).astype(np.float32)
        _assert_xpred_vs_gpu_stored("x_pred", rec["x_pred"], stored_rec["x_pred"], a, rec["b"], 0.005)
    assert out.metrics[METRIC_KEY]["num_instances"] == stored_rec["b"].shape[0]


# --------------------------------------------------------------------------
# LISTA / ALISTA real checkpoints (exp2_try3_210x300, seed 21, narrow)
# --------------------------------------------------------------------------

_UNROLLED_CELLS = [("lista", "exp2_try3_210x300", 21, "narrow"),
                   ("alista", "exp2_try3_210x300", 21, "narrow")]
_TRY3_TEST_FILES = ["test_sparsity_{:.2f}.npy".format(p) for p in
                    (0.02, 0.05, 0.10, 0.11, 0.15, 0.17, 0.20, 0.24, 0.30, 0.35, 0.40)]


@pytest.mark.slow
@pytest.mark.parametrize("model_name,run,seed,width", _UNROLLED_CELLS, ids=[c[0] for c in _UNROLLED_CELLS])
def test_real_unrolled_checkpoint_parity(model_name, run, seed, width, tmp_path):
    """x_pred must match the stored ``<file>_final_output.npy`` for every test
    file, and the per-layer trajectory must match the stored ``lasso_layer{i}``
    -- after undoing a defect in ``Model_Base_L2O/train.py``'s test loop: its
    Keras ``Mean`` metric objects are created once and reused across the
    ``compile``/``evaluate`` of every test file, so the stored value for the
    k-th file is the running mean over files 1..k (all files have the same row
    count, so ``f_k = k*stored_k - (k-1)*stored_{k-1}``). Only the first file's
    stored trajectory is the per-file value."""
    run_dir = os.path.join(_RUNS, run)
    base = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, model_name, "checkpoints")
    model_dir = os.path.join(base, "models", "{}__{}__s{}".format(model_name, width, seed), "replicate_1")
    data_dir = os.path.join(run_dir, "data", "seed{}".format(seed), width)
    stored_path = os.path.join(model_dir, "all_test_metrics.json")
    if not (os.path.exists(stored_path) and os.path.exists(os.path.join(data_dir, "A.npy"))):
        pytest.skip("real run artefacts missing under {}".format(model_dir))
    import json
    with open(stored_path) as f:
        stored = json.load(f)
    files = [s for s in _TRY3_TEST_FILES if s in stored]
    assert files[:1] == _TRY3_TEST_FILES[:1], "expected the sweep's first test file to be present"

    config = {
        "method": "unified-" + model_name, "problem": "lasso", "seed": seed,
        "run_name": "{}__{}__s{}".format(model_name, width, seed),
        "train": {"data_dir": data_dir, "num_layers": 16, "lasso_lam": 0.005, "model_lam": 0.005,
                  "ss_q_per_layer": 1.2, "ss_maxq": 13.0},
        "eval": {"data_dir": data_dir, "test_files": files},
    }
    out = get_method("unified-" + model_name).evaluate(config, base, str(tmp_path))

    with open(out.artifacts["all_test_metrics"]) as f:
        unified = json.load(f)
    prev = None
    for k, split in enumerate(files, start=1):
        traj = np.array([unified[split]["lasso_layer{}".format(i)] for i in range(16)])
        stored_k = np.array([stored[split]["lasso_layer{}".format(i)] for i in range(16)])
        expected = stored_k if prev is None else k * stored_k - (k - 1) * prev
        # De-accumulating amplifies the stored float32 rounding by ~k, hence the looser tolerance.
        assert np.allclose(traj, expected, rtol=2e-4 * k, atol=1e-5 * k), (
            "{}: per-layer trajectory differs from the de-accumulated stored values".format(split))
        prev = stored_k
        x_stored = np.load(os.path.join(model_dir, split[:-len(".npy")] + "_final_output.npy"))
        x_unified = np.load(out.artifacts["{}:final_output".format(split)])
        rows = np.load(os.path.join(data_dir, split))
        a = np.load(os.path.join(data_dir, "A.npy")).astype(np.float32)
        _assert_xpred_vs_gpu_stored("x_pred " + split, x_unified, x_stored, a, rows[:, :a.shape[0]], 0.005)
        assert out.per_seed[split][METRIC_KEY]["num_instances"] == x_stored.shape[0]


# --------------------------------------------------------------------------
# Tiny checkpoints through the existing adapters
# --------------------------------------------------------------------------

_TINY_TRAIN = {"num_epochs": 2, "num_steps": 10, "unroll_length": 5, "evaluation_period": 1,
               "evaluation_epochs": 1, "lasso_batch_size": _LASSO_BATCH, "learning_rate": 1e-3}


def _tiny_eval(fixture, optimizer, extra):
    return {"optimizer": optimizer, "num_steps": 10, "lasso_data_dir": fixture.parts["width_dir"],
            "lasso_split": fixture.parts["test_file"], "lasso_batch_size": _LASSO_BATCH,
            "lasso_lam": fixture.parts["lam"], "num_fista_iters": _FISTA_ITERS,
            "xstar_cache_dir": os.path.join(fixture.root, "xstar_cache"), **extra}


def _compare_outputs(adapter_out, unified_out, exact=False):
    if exact:
        assert np.array_equal(np.asarray(adapter_out.trajectory), np.asarray(unified_out.trajectory)), (
            "trajectories differ: max |diff| {:.3e}".format(float(np.max(np.abs(
                np.asarray(adapter_out.trajectory) - np.asarray(unified_out.trajectory))))))
    else:
        _assert_close("trajectory", unified_out.trajectory, adapter_out.trajectory)
    assert fingerprints_agree(adapter_out.metrics[METRIC_KEY], unified_out.metrics[METRIC_KEY])
    ra, ru = np.load(adapter_out.artifacts["recovery"]), np.load(unified_out.artifacts["recovery"])
    _assert_close("x_pred", ru["x_pred"], ra["x_pred"])
    for key in ("suboptimality_gap_mean", "modified_relative_loss", "nmse_vs_true_mean"):
        assert abs(adapter_out.metrics[key] - unified_out.metrics[key]) <= 1e-6 + 1e-5 * abs(adapter_out.metrics[key]), key


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", [("rnnprop", {"beta1": 0.95, "beta2": 0.95}), ("dm", {})],
                         ids=["rnnprop", "dm"])
def test_tiny_adapter_checkpoint_parity(variant, extra, fixture_root, tmp_path):
    seed = 7
    fixture = make_lasso_tiny(fixture_root, seed)
    adapter, uni = get_method("l2o-" + variant), get_method("unified-" + variant)
    train_config = {"method": adapter.name, "problem": "lasso_dataset", "seed": seed,
                    "train": {"lasso_data_dir": fixture.parts["width_dir"],
                              "lasso_lam": fixture.parts["lam"], **_TINY_TRAIN, **extra}}
    ckpt = adapter.train(train_config, os.path.join(str(tmp_path), "train")).checkpoint

    eval_config = {"method": adapter.name, "problem": "lasso_dataset", "seed": seed,
                   "eval": _tiny_eval(fixture, "L2L", extra)}
    a_out = adapter.evaluate(eval_config, ckpt, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(eval_config, ckpt, os.path.join(str(tmp_path), "unified"))
    _compare_outputs(a_out, u_out)


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", [("rnnprop", {"beta1": 0.95, "beta2": 0.95}), ("dm", {})],
                         ids=["rnnprop", "dm"])
def test_classical_adam_parity(variant, extra, fixture_root, tmp_path):
    """No checkpoint at all: the Adam baseline path must be bit-identical."""
    seed = 7
    fixture = make_lasso_tiny(fixture_root, seed)
    adapter, uni = get_method("l2o-" + variant), get_method("unified-" + variant)
    eval_config = {"method": adapter.name, "problem": "lasso_dataset", "seed": seed,
                   "eval": _tiny_eval(fixture, "Adam", extra)}
    a_out = adapter.evaluate(eval_config, None, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(eval_config, None, os.path.join(str(tmp_path), "unified"))
    _compare_outputs(a_out, u_out, exact=True)


@pytest.mark.slow
def test_missing_checkpoint_is_refused(fixture_root, tmp_path):
    """evaluate_*.py silently evaluates an untrained net when the file is
    missing (the RESULTS_VALIDITY.md failure class); the unified path refuses."""
    fixture = make_lasso_tiny(fixture_root, 7)
    config = {"method": "unified-rnnprop", "problem": "lasso_dataset", "seed": 7,
              "eval": _tiny_eval(fixture, "L2L", {"beta1": 0.95, "beta2": 0.95})}
    with pytest.raises(FileNotFoundError):
        get_method("unified-rnnprop").evaluate(config, str(tmp_path / "nowhere"), str(tmp_path))


# --------------------------------------------------------------------------
# quadratic: the second problem (library-generated, no files) -- proves the
# Problem abstraction. Same machine on both sides, so bit-exact.
# --------------------------------------------------------------------------

_QUAD_VARIANTS = [("rnnprop", {"beta1": 0.95, "beta2": 0.95}), ("dm", {})]


def _quad_eval(optimizer, extra, num_steps=20, num_epochs=1):
    return {"optimizer": optimizer, "num_steps": num_steps, "num_epochs": num_epochs, **extra}


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", _QUAD_VARIANTS, ids=["rnnprop", "dm"])
@pytest.mark.parametrize("num_epochs", [1, 3], ids=["1epoch", "3epochs"])
def test_quadratic_classical_adam_parity(variant, extra, num_epochs, tmp_path):
    """No files, no checkpoint: the trajectory over ``num_epochs`` library-generated
    batches (per-step costs appended, as evaluate_dm.py does) must be bit-identical."""
    adapter, uni = get_method("l2o-" + variant), get_method("unified-" + variant)
    cfg = {"method": adapter.name, "problem": "quadratic", "seed": 7,
           "eval": _quad_eval("Adam", extra, num_epochs=num_epochs)}
    a_out = adapter.evaluate(cfg, None, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(cfg, None, os.path.join(str(tmp_path), "unified"))
    assert len(u_out.trajectory) == 20 * num_epochs
    assert np.array_equal(np.asarray(a_out.trajectory), np.asarray(u_out.trajectory)), (
        "max |diff| {:.3e}".format(float(np.max(np.abs(np.asarray(a_out.trajectory) - np.asarray(u_out.trajectory))))))
    assert METRIC_KEY not in u_out.metrics     # nothing file-backed to fingerprint


@pytest.mark.slow
@pytest.mark.parametrize("variant,extra", _QUAD_VARIANTS, ids=["rnnprop", "dm"])
def test_quadratic_tiny_adapter_checkpoint_parity(variant, extra, tmp_path):
    """Train on quadratic through the EXISTING adapter, then evaluate the
    checkpoint with both -- exercises the first-batch build/restore order on a
    problem whose batches consume random numbers."""
    adapter, uni = get_method("l2o-" + variant), get_method("unified-" + variant)
    train = {"num_epochs": 2, "num_steps": 10, "unroll_length": 5, "evaluation_period": 1,
             "evaluation_epochs": 1, "learning_rate": 1e-3, **extra}
    ckpt = adapter.train({"method": adapter.name, "problem": "quadratic", "seed": 7, "train": train},
                         os.path.join(str(tmp_path), "train")).checkpoint
    cfg = {"method": adapter.name, "problem": "quadratic", "seed": 7, "eval": _quad_eval("L2L", extra)}
    a_out = adapter.evaluate(cfg, ckpt, os.path.join(str(tmp_path), "adapter"))
    u_out = uni.evaluate(cfg, ckpt, os.path.join(str(tmp_path), "unified"))
    assert np.array_equal(np.asarray(a_out.trajectory), np.asarray(u_out.trajectory)), (
        "max |diff| {:.3e}".format(float(np.max(np.abs(np.asarray(a_out.trajectory) - np.asarray(u_out.trajectory))))))


# --------------------------------------------------------------------------
# L2O-Scale (unified-scale): the in-place, RNG-hungry optimizer. Same machine
# on both sides -> bit-exact, including the dummy-step preamble and the
# per-step dead random draws the library's Problem.gradients makes.
# --------------------------------------------------------------------------

_SCALE_IDENTITY = {"optimizer": "HierarchicalRNN", "cell_cls": "GRUCell", "cell_size": 20, "num_cells": 2}
_SCALE_TINY_TRAIN = {"lasso_split": "train_data.npy", "lasso_batch_size": _LASSO_BATCH,
                     "num_problems": 2, "num_meta_iterations": 2, "fix_unroll": True, "fix_unroll_length": 5,
                     "fix_num_steps": 10, "fix_num_steps_eval": 10, "evaluation_period": 1,
                     "evaluation_epochs": 1, "save_period": 1, "min_lr": 1e-3, "max_lr": 1e-1,
                     "select_checkpoint": "best"}


def _scale_train_section(fixture):
    return {**_SCALE_IDENTITY, "lasso_data_dir": fixture.parts["width_dir"], "lasso_lam": fixture.parts["lam"],
            **_SCALE_TINY_TRAIN}


def _scale_eval_section(fixture, test_optimizer, num_steps=10):
    return {"test_optimizer": test_optimizer, "num_testing_itrs": num_steps,
            "lasso_data_dir": fixture.parts["width_dir"], "lasso_split": fixture.parts["test_file"],
            "lasso_batch_size": _LASSO_BATCH, "lasso_lam": fixture.parts["lam"], "min_lr": 1e-3, "max_lr": 1e-1,
            "num_fista_iters": _FISTA_ITERS, "xstar_cache_dir": os.path.join(fixture.root, "xstar_cache")}


def _assert_scale_parity_per_chunk(a_out, cfg, ckpt, out_dir, label):
    """Before its 2026-09-18 fix, L2O-Scale/evaluate.py's per-chunk behaviour was
    allocator-dependent: a chunk's real variable either got a fresh optimizer
    slot or inherited the dummy step's (when CPython reused the freed dummy's
    id(); see RESULTS_VALIDITY.md). Both behaviours are implemented by
    ``ScalePolicy(dummy_slot=...)``, so parity is checked PER CHUNK -- the
    script's x_pred rows for a chunk must be bit-identical to one of the two
    unified variants and its averaged trajectory must equal the mean of the
    matched per-chunk trajectories -- and the caller then asserts that every
    chunk is the fresh one. Kept in this form so a regression of the fix is
    diagnosed, not just detected."""
    from unified.evalwalk import evaluate_split
    from unified.learned import scale as scale_mod
    from unified.problem import PROBLEMS
    from unified.solver import PolicySolver
    ev = scale_mod.SCALE_SPEC.eval_schema.merge(PROBLEMS["lasso_dataset"].eval_schema).validate(cfg["eval"], "eval")
    (label_, problem), = PROBLEMS["lasso_dataset"].eval_problems({**cfg, "eval": ev})
    variants = {}
    for mode in ("fresh", "inherit"):
        import tensorflow as tf
        tf.random.set_seed(cfg["seed"])
        variants[mode] = evaluate_split(
            lambda: PolicySolver(scale_mod.ScalePolicy(min_lr=float(ev["min_lr"]), max_lr=float(ev["max_lr"]),
                                                       checkpoint=ckpt, dummy_slot=mode), problem),
            problem, int(ev["num_testing_itrs"]), "batch_mean", label=label_)
    a_x = np.load(a_out.artifacts["recovery"])["x_pred"]
    sizes = [b.x0.shape[0] for b in variants["fresh"].batches]
    starts = np.cumsum([0] + sizes[:-1])
    chosen, expected = [], np.zeros(int(ev["num_testing_itrs"]), np.float64)
    for i, (start, size) in enumerate(zip(starts, sizes)):
        rows = a_x[start:start + size]
        match = [m for m in ("fresh", "inherit") if np.array_equal(rows, variants[m].x_pred[start:start + size])]
        assert match, "{} chunk {}: the script's x_pred rows match neither the fresh nor the inherited-slot " \
                      "variant (max |diff| fresh {:.3e}, inherit {:.3e})".format(
                          label, i, float(np.abs(rows - variants["fresh"].x_pred[start:start + size]).max()),
                          float(np.abs(rows - variants["inherit"].x_pred[start:start + size]).max()))
        chosen.append(match[0])
        expected += np.array(variants[match[0]].chunk_trajectories[i], np.float64)
    expected = expected / len(sizes)
    ta = np.asarray(a_out.trajectory, np.float64)
    assert np.array_equal(ta, expected), "{}: averaged trajectory differs from the per-chunk reconstruction " \
                                         "(chunks {}); max |diff| {:.3e}".format(label, chosen, float(np.abs(ta - expected).max()))
    return chosen


def _assert_bit_identical(a_out, u_out):
    ta, tu = np.asarray(a_out.trajectory), np.asarray(u_out.trajectory)
    assert ta.shape == tu.shape, (ta.shape, tu.shape)
    assert np.array_equal(ta, tu), "trajectory differs: max |diff| {:.3e} at step {}".format(
        float(np.max(np.abs(ta - tu))), int(np.argmax(np.abs(ta - tu))))
    ra, ru = np.load(a_out.artifacts["recovery"]), np.load(u_out.artifacts["recovery"])
    assert np.array_equal(ra["x_pred"], ru["x_pred"]), "x_pred differs: max |diff| {:.3e}".format(
        float(np.max(np.abs(ra["x_pred"] - ru["x_pred"]))))
    assert fingerprints_agree(a_out.metrics[METRIC_KEY], u_out.metrics[METRIC_KEY])


@pytest.fixture(scope="session")
def scale_tiny_checkpoint(fixture_root, tmp_path_factory):
    """A tiny L2O-Scale checkpoint trained through the EXISTING adapter (shared by the tests below)."""
    fixture = make_lasso_tiny(fixture_root, 7)
    out_dir = str(tmp_path_factory.mktemp("scale_tiny_train"))
    ckpt = get_method("l2o-scale").train(
        {"method": "l2o-scale", "problem": "lasso_dataset", "seed": 7, "train": _scale_train_section(fixture)},
        out_dir).checkpoint
    return fixture, ckpt


@pytest.mark.slow
def test_scale_tiny_adapter_checkpoint_parity(scale_tiny_checkpoint, tmp_path):
    fixture, ckpt = scale_tiny_checkpoint
    cfg = {"method": "l2o-scale", "problem": "lasso_dataset", "seed": 7,
           "train": _scale_train_section(fixture), "eval": _scale_eval_section(fixture, "L2o")}
    a_out = get_method("l2o-scale").evaluate(cfg, ckpt, os.path.join(str(tmp_path), "adapter"))
    u_out = get_method("unified-scale").evaluate(cfg, ckpt, os.path.join(str(tmp_path), "unified"))
    assert os.path.basename(u_out.artifacts["eval_pickle"]) == "seed7_eval_loss_record.pickle-benchmark"
    assert fingerprints_agree(a_out.metrics[METRIC_KEY], u_out.metrics[METRIC_KEY])
    chosen = _assert_scale_parity_per_chunk(a_out, cfg, ckpt, str(tmp_path), "tiny")
    # With the 2026-09-18 fix in L2O-Scale/evaluate.py every chunk is the fresh
    # variant, i.e. the registered unified-scale is bit-identical outright.
    assert all(c == "fresh" for c in chosen), chosen
    _assert_bit_identical(a_out, u_out)


@pytest.mark.slow
@pytest.mark.parametrize("classical", ["Adam", "SGD", "Adagrad"])
def test_scale_classical_parity(classical, fixture_root, tmp_path):
    fixture = make_lasso_tiny(fixture_root, 7)
    cfg = {"method": "l2o-scale", "problem": "lasso_dataset", "seed": 7,
           "train": _scale_train_section(fixture), "eval": _scale_eval_section(fixture, classical)}
    a_out = get_method("l2o-scale").evaluate(cfg, None, os.path.join(str(tmp_path), "adapter"))
    u_out = get_method("unified-scale").evaluate(cfg, None, os.path.join(str(tmp_path), "unified"))
    _assert_bit_identical(a_out, u_out)


@pytest.mark.slow
def test_scale_real_checkpoint_parity(parity_steps, tmp_path_factory):
    """runs/exp2_try2_scale_narrow's model-best.l2o: no stored evaluation exists,
    so the target is the ADAPTER replayed on this machine (a ``--parity-steps``
    prefix; costs are pre-step, so a prefix is independent of the total)."""
    run = os.path.join(_RUNS, "exp2_try2_scale_narrow")
    ckpt = os.path.join(run, "checkpoints", "s21", "narrow", "l2o-scale", "checkpoints",
                        "HierarchicalRNN_GRUCell_20_2", "model-best.l2o")
    data_dir = os.path.join(run, "data", "seed21", "narrow")
    if not (os.path.exists(ckpt) and os.path.exists(os.path.join(data_dir, "A.npy"))):
        pytest.skip("real Scale run artefacts missing")
    steps = int(parity_steps)
    out_dir = str(tmp_path_factory.mktemp("scale_real"))
    ev = {"test_optimizer": "L2o", "num_testing_itrs": steps, "lasso_data_dir": data_dir,
          "lasso_split": "test_sparsity_0.10.npy", "lasso_batch_size": 128, "lasso_lam": 0.005,
          "min_lr": 1e-3, "max_lr": 1e-1,
          # the adapter recomputes the recovery metrics and requires these; the
          # x* cache goes to the session tmp dir, never into runs/
          "num_fista_iters": 2000, "xstar_cache_dir": os.path.join(out_dir, "xstar_cache")}
    cfg = {"method": "l2o-scale", "problem": "lasso_dataset", "seed": 21, "train": {**_SCALE_IDENTITY},
           "eval": ev}
    a_out = get_method("l2o-scale").evaluate(cfg, ckpt, os.path.join(out_dir, "adapter"))
    u_out = get_method("unified-scale").evaluate(cfg, ckpt, os.path.join(out_dir, "unified"))
    _assert_bit_identical(a_out, u_out)


# --------------------------------------------------------------------------
# LFISTA (unified-lfista): the second unrolled family. The library's weights
# are untrackable tf.Variables that never train, so the only library target is
# its untrained forward pass -- computed in a Model_Base_L2O subprocess (that
# library's top-level modules must not be imported next to the others').
# --------------------------------------------------------------------------

_LFISTA_LIBRARY_FORWARD = textwrap.dedent("""
    import sys, numpy as np, tensorflow as tf
    import models
    data_dir, fname, out, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    A = np.load(data_dir + "/A.npy").astype(np.float32)
    m, n = A.shape
    split = np.load(data_dir + "/" + fname).astype(np.float32)
    x0 = np.load(data_dir + "/" + fname[:-4] + "_x0.npy").astype(np.float32)
    model = models.Lfista(A, T, 0.005)
    for k in range(T):
        model.create_cell(k)
    h = tf.concat([tf.constant(split[:, :m]), tf.constant(x0)], 1)   # [b, x0], as UnrolledSolver feeds it
    layers = [h[:, -n:].numpy()]
    for cell in model.layers:
        h = cell(h)
        layers.append(h[:, -n:].numpy())
    np.save(out, np.stack(layers))
    print("OK", len(layers))
""")


@pytest.mark.slow
def test_lfista_untrained_forward_matches_library(fixture_root, tmp_path):
    from core.shell import REPO_ROOT
    from unified.tasks.lasso import LassoProblem
    from unified.unrolled.lista import UnrolledSolver, build_model
    fixture = make_lasso_tiny(fixture_root, 7)
    T = 4
    out = os.path.join(str(tmp_path), "library_layers.npy")
    proc = subprocess.run(
        [sys.executable, "-c", _LFISTA_LIBRARY_FORWARD, fixture.parts["width_dir"], fixture.parts["test_file"], out, str(T)],
        cwd=os.path.join(str(REPO_ROOT), "Model_Base_L2O"), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    library = np.load(out)                                          # (T+1, rows, n)

    problem = LassoProblem(fixture.parts["width_dir"], fixture.parts["test_file"], lam=0.005, batch_size=1024)
    model = build_model("lfista", problem.a, T, 0.005, ident={"x0_at_layer0": False})
    batch = next(problem.walk())
    unified = np.stack([np.asarray(x) for x in UnrolledSolver(model).iterate(batch, T)])
    assert unified.shape == library.shape
    for k in range(T + 1):
        assert np.array_equal(unified[k], library[k]), "layer {} differs (max |diff| {:.3e})".format(
            k, float(np.abs(unified[k] - library[k]).max()))
    # and the x0-consuming variant is a different model at layers 0 and 1 only
    # in what it reads, so with x0 ~ N(0, 0.01^2) it must stay close but not equal
    fair = build_model("lfista", problem.a, T, 0.005, ident={"x0_at_layer0": True})
    unified_fair = np.stack([np.asarray(x) for x in UnrolledSolver(fair).iterate(batch, T)])
    assert not np.array_equal(unified_fair[1], unified[1])
    assert np.allclose(unified_fair[1], unified[1], atol=0.1)
