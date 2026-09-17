"""The data layer: a generated dataset is a pure function of (seed, config).

For file-based problems this is where the seed acts (evaluation then walks
the files deterministically), so it is tested here rather than through the
evaluation seed. A new generator for another problem should get the same
three checks: same seed -> identical bytes, different seed -> different data,
changed config on an existing directory -> refused.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from data.lasso import Experiment2Config, generate_experiment2_dataset

_CFG = dict(m=10, n=20, lam=0.005, train_size=64, val_size=16, test_size=16,
            train_sparsities={"narrow": 0.1, "wide": (0.05, 0.3)},
            test_sparsities=[0.1, 0.2], compute_alista_w=True)


def _file_hashes(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".npy"):
                path = os.path.join(dirpath, f)
                with open(path, "rb") as fh:
                    out[os.path.relpath(path, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def test_same_seed_identical_files(tmp_path):
    d1 = generate_experiment2_dataset(Experiment2Config(seed=5, **_CFG), str(tmp_path / "a"))
    d2 = generate_experiment2_dataset(Experiment2Config(seed=5, **_CFG), str(tmp_path / "b"))
    h1, h2 = _file_hashes(d1), _file_hashes(d2)
    assert h1.keys() == h2.keys()
    differing = sorted(k for k in h1 if h1[k] != h2[k])
    assert not differing, "same seed, different files: {}".format(differing)
    assert any(k.endswith("_x0.npy") for k in h1), "no x0 sibling files were generated"


def test_different_seed_different_data(tmp_path):
    d1 = generate_experiment2_dataset(Experiment2Config(seed=5, **_CFG), str(tmp_path / "a"))
    d2 = generate_experiment2_dataset(Experiment2Config(seed=6, **_CFG), str(tmp_path / "b"))
    a1 = np.load(os.path.join(d1, "narrow", "A.npy"))
    a2 = np.load(os.path.join(d2, "narrow", "A.npy"))
    assert not np.array_equal(a1, a2)


def test_widths_share_dictionary_and_test_sets(tmp_path):
    d = generate_experiment2_dataset(Experiment2Config(seed=5, **_CFG), str(tmp_path / "a"))
    for f in ("A.npy", "test_sparsity_0.10.npy", "test_sparsity_0.10_x0.npy", "W.npy"):
        a = np.load(os.path.join(d, "narrow", f))
        b = np.load(os.path.join(d, "wide", f))
        assert np.array_equal(a, b), "{} differs between widths".format(f)
    tr_n = np.load(os.path.join(d, "narrow", "train_data.npy"))
    tr_w = np.load(os.path.join(d, "wide", "train_data.npy"))
    assert not np.array_equal(tr_n, tr_w), "training sets should differ between widths"


def test_changed_config_on_existing_dir_is_refused(tmp_path):
    generate_experiment2_dataset(Experiment2Config(seed=5, **_CFG), str(tmp_path / "a"))
    changed = dict(_CFG, test_size=_CFG["test_size"] + 1)
    with pytest.raises(RuntimeError, match="different parameters"):
        generate_experiment2_dataset(Experiment2Config(seed=5, **changed), str(tmp_path / "a"))
