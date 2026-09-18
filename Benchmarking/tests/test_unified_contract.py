"""Fast contract checks for the ``unified`` package (no TensorFlow needed).

The unified methods sit beside the adapter-backed ones in the same registry;
these tests pin the two properties that keep that safe: nothing the adapters
registered is shadowed, and importing ``unified`` puts none of the library
module names (``networks``, ``problems``, ...) into ``sys.modules`` -- the
collision that forced the adapters to be subprocess-based in the first place.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import adapters  # noqa: F401
import unified  # noqa: F401
from core.registry import get_method, list_methods
from unified.tasks.lasso import LassoProblem, x0_filename

_LIB_MODULE_NAMES = ("networks", "problems", "util", "utils", "meta", "metaopt",
                     "optimizer", "preprocess", "evaluate", "train", "problem_generator")


def test_unified_names_registered_and_adapters_untouched():
    names = set(list_methods())
    for name in unified.METHOD_NAMES:
        assert name in names
        assert type(get_method(name)).__module__.startswith("unified.")
    # The adapter-backed methods are still the adapters' classes.
    for name in ("l2o-dm", "l2o-rnnprop", "l2o-scale", "lista", "alista"):
        assert type(get_method(name)).__module__.startswith("adapters.")


def test_no_library_module_names_leak():
    leaked = [n for n in _LIB_MODULE_NAMES if n in sys.modules]
    assert not leaked, "import unified put library module names on sys.modules: {}".format(leaked)
    leaked = [n for n in sys.modules if n.startswith("unified.")
              and n.split(".", 1)[1] in _LIB_MODULE_NAMES]
    assert not leaked, "unified has a submodule named like a library module: {}".format(leaked)


def test_unified_import_is_tf_free():
    # TensorFlow is imported lazily inside train()/evaluate(); the registry
    # import must stay cheap for the fast suite. Only meaningful when nothing
    # else in this process pulled TF in first.
    if "tensorflow" in sys.modules:
        pytest.skip("tensorflow already imported by another test module")
    assert "tensorflow" not in sys.modules


def test_x0_filename_rule():
    assert x0_filename("test_sparsity_0.10.npy") == "test_sparsity_0.10_x0.npy"
    with pytest.raises(ValueError):
        x0_filename("train_data")


def test_instances_walk_covers_split_once_in_order(tmp_path):
    m, n, rows = 3, 5, 10
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, n)).astype(np.float32)
    data = rng.standard_normal((rows, m + n)).astype(np.float32)
    x0 = rng.standard_normal((rows, n)).astype(np.float32)
    np.save(os.path.join(tmp_path, "A.npy"), a)
    np.save(os.path.join(tmp_path, "split.npy"), data)
    np.save(os.path.join(tmp_path, "split_x0.npy"), x0)

    inst = LassoProblem(str(tmp_path), "split.npy", lam=0.005, batch_size=4)
    batches = list(inst.walk())
    assert [len(b.data["idx"]) for b in batches] == [4, 4, 2]
    assert inst.num_chunks() == 3
    seen = np.concatenate([b.data["idx"] for b in batches])
    assert np.array_equal(seen, np.arange(rows))
    assert np.array_equal(np.concatenate([b.data["b"] for b in batches]), data[:, :m])
    assert np.array_equal(np.concatenate([b.data["x_true"] for b in batches]), data[:, m:])
    assert np.array_equal(np.concatenate([b.x0 for b in batches]), x0)


def test_instances_requires_aligned_x0(tmp_path):
    a = np.zeros((2, 3), np.float32)
    np.save(os.path.join(tmp_path, "A.npy"), a)
    np.save(os.path.join(tmp_path, "s.npy"), np.zeros((4, 5), np.float32))
    with pytest.raises(FileNotFoundError):
        LassoProblem(str(tmp_path), "s.npy", lam=0.005, batch_size=2)
    assert LassoProblem(str(tmp_path), "s.npy", lam=0.005, batch_size=2, x0_mode="random").num_samples == 4


def test_model_free_config_rejects_unknown_eval_keys():
    method = get_method("unified-rnnprop")
    with pytest.raises(ValueError, match="unknown config keys"):
        method.evaluate({"problem": "lasso_dataset",
                         "eval": {"optimizer": "Adam", "num_steps": 1, "lasso_data_dir": ".",
                                  "made_up_flag": 1}}, None, "unused")
    with pytest.raises(NotImplementedError):
        method.evaluate({"problem": "mnist", "eval": {}}, None, "unused")
    with pytest.raises(ValueError, match="unrolled dialect"):
        method.evaluate({"problem": "lasso", "eval": {}}, None, "unused")
    with pytest.raises(NotImplementedError, match="if_cl"):
        method.train({"problem": "lasso_dataset", "train": {"lasso_data_dir": ".", "num_epochs": 1,
                                                             "if_cl": True}}, "unused")
    with pytest.raises(ValueError, match="not accepted by this method"):
        get_method("unified-dm").evaluate({"problem": "lasso_dataset",
                                           "eval": {"optimizer": "Adam", "num_steps": 1, "lasso_data_dir": ".",
                                                    "beta1": 0.9}}, None, "unused")
