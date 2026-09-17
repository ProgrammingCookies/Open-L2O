"""Tier A: the same (method, problem, seed) evaluated twice gives the same
result. Needs no contract beyond ``EvalOutput.trajectory``, so it applies to
every registered method on every problem in the matrix."""

from __future__ import annotations

import os

import numpy as np
import pytest

from core.fingerprint import METRIC_KEY, describe, fingerprints_agree
from tests.fairness_matrix import ENTRIES
from tests.live import run_live

_CASES = [(e, m) for e in ENTRIES for m in e.methods]
_IDS = ["{}-{}".format(e.name, m) for e, m in _CASES]
_SEED_CASES = [(e, m) for e, m in _CASES if e.seed_selects_eval_instances]
_SEED_IDS = [i for (e, m), i in zip(_CASES, _IDS) if e.seed_selects_eval_instances]


def _fixture_for(entry, root, seed):
    return entry.make_fixture(root, seed) if entry.make_fixture else None


@pytest.mark.slow
@pytest.mark.parametrize("entry,method_name", _CASES, ids=_IDS)
def test_same_seed_same_result(entry, method_name, fixture_root, tmp_path):
    seed = 7
    fixture = _fixture_for(entry, fixture_root, seed)
    a = run_live(entry, method_name, fixture, seed, os.path.join(str(tmp_path), "a"))
    b = run_live(entry, method_name, fixture, seed, os.path.join(str(tmp_path), "b"))

    ta = np.asarray(a.trajectory, dtype=np.float64)
    tb = np.asarray(b.trajectory, dtype=np.float64)
    assert ta.shape == tb.shape, "trajectory length differs between two same-seed runs"
    assert ta.size > 0, "empty trajectory -- the adapter read nothing back"
    if entry.trajectory_atol == 0.0:
        assert np.array_equal(ta, tb), (
            "same seed, different trajectory (max |diff| {:.3e})".format(
                float(np.max(np.abs(ta - tb)))))
    else:
        assert np.allclose(ta, tb, atol=entry.trajectory_atol, rtol=0.0)

    fa, fb = a.metrics.get(METRIC_KEY), b.metrics.get(METRIC_KEY)
    if fa or fb:
        assert fingerprints_agree(fa, fb), (
            "same seed, different instances: {} vs {}".format(describe(fa), describe(fb)))


@pytest.mark.slow
@pytest.mark.parametrize("entry,method_name", _SEED_CASES, ids=_SEED_IDS)
def test_seed_selects_instances(entry, method_name, fixture_root, tmp_path):
    """Where the evaluation seed is what picks the instances, two seeds must
    not produce the same run."""
    outs = []
    for seed in (7, 8):
        fixture = _fixture_for(entry, fixture_root, seed)
        outs.append(run_live(entry, method_name, fixture, seed,
                             os.path.join(str(tmp_path), "s{}".format(seed))))
    fa, fb = outs[0].metrics.get(METRIC_KEY), outs[1].metrics.get(METRIC_KEY)
    if fa and fb:
        assert not fingerprints_agree(fa, fb), "seeds 7 and 8 evaluated identical instances"
    else:
        ta, tb = np.asarray(outs[0].trajectory), np.asarray(outs[1].trajectory)
        assert not (ta.shape == tb.shape and np.array_equal(ta, tb)), (
            "seeds 7 and 8 produced identical trajectories -- the seed is not "
            "reaching the problem")
