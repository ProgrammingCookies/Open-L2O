"""The property the model-based fingerprint relies on: at test time,
``Model_Base_L2O/data_preprocessing.input_fn(is_training=False)`` yields the
split's rows in file order, each with its ``x0`` appended, so the ``b``,
``x_true`` and ``x0`` an adapter reads from the files are exactly what the
network saw, in the order its outputs are written.

Run as a subprocess in ``Model_Base_L2O/`` because that library's top-level
modules (``utils.py`` etc.) must not be imported next to the other libraries'.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

from core.shell import REPO_ROOT
from tests.fairness_matrix import make_lasso_tiny

_CHECK = textwrap.dedent("""
    import sys, numpy as np, tensorflow as tf
    import data_preprocessing as dp
    data_dir, fname = sys.argv[1], sys.argv[2]
    A = np.load(data_dir + "/A.npy")
    m, n = A.shape
    ds = dp.input_fn(False, data_dir, batch_size=7, task="lasso", A=A, filename=fname)
    inputs, labels = [], []
    for x, y in ds:
        inputs.append(x.numpy()); labels.append(y.numpy())
    inputs = np.concatenate(inputs); labels = np.concatenate(labels)
    split = np.load(data_dir + "/" + fname)
    x0 = np.load(data_dir + "/" + fname[:-4] + "_x0.npy")
    assert labels.shape[0] == split.shape[0], (labels.shape, split.shape)
    assert np.array_equal(labels, split.astype(labels.dtype)), "labels not in file order"
    assert np.array_equal(inputs[:, :m], split[:, :m].astype(inputs.dtype)), "b not in file order"
    assert np.array_equal(inputs[:, m:], x0.astype(inputs.dtype)), "x0 not appended in file order"
    print("OK", labels.shape[0])
""")


def test_model_based_test_input_is_file_order(fixture_root):
    fixture = make_lasso_tiny(fixture_root, seed=3)
    lib_dir = os.path.join(str(REPO_ROOT), "Model_Base_L2O")
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK, fixture.parts["width_dir"], fixture.parts["test_file"]],
        cwd=lib_dir, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert "OK {}".format(fixture.num_instances) in proc.stdout
