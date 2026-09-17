"""Tier B: every method in an entry is scored on the same instances from the
same starting points. Rests on the ``instances_fingerprint`` contract
(``core/fingerprint.py``): an adapter that reports nothing fails here, which
is what obliges a new adapter to implement it."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import pytest

from core.fingerprint import METRIC_KEY, describe, fingerprints_agree
from tests.fairness_matrix import ENTRIES
from tests.live import run_live

_ENTRIES_B = [e for e in ENTRIES if e.fingerprint_required]


@pytest.mark.slow
@pytest.mark.parametrize("entry", _ENTRIES_B, ids=[e.name for e in _ENTRIES_B])
def test_methods_share_instances(entry, fixture_root, tmp_path):
    seed = 7
    fixture = entry.make_fixture(fixture_root, seed) if entry.make_fixture else None
    fps = {}
    for method_name in entry.methods:
        out = run_live(entry, method_name, fixture, seed,
                       os.path.join(str(tmp_path), method_name))
        fps[method_name] = out.metrics.get(METRIC_KEY)

    missing = sorted(m for m, fp in fps.items() if not fp)
    assert not missing, (
        "no instances fingerprint from {} -- the adapter must put "
        "core.fingerprint.fingerprint_arrays(...) under EvalOutput.metrics[{!r}] "
        "(CONTRIBUTING.md section 12)".format(missing, METRIC_KEY))

    if fixture is not None:
        wrong = {m: fp["num_instances"] for m, fp in fps.items()
                 if fp["num_instances"] != fixture.num_instances}
        assert not wrong, "expected {} instances, got {}".format(fixture.num_instances, wrong)

    ref_name = next(iter(fps))
    ref = fps[ref_name]
    disagree = {m: describe(fp) for m, fp in fps.items() if not fingerprints_agree(ref, fp)}
    assert not disagree, "instances differ from {} = {}: {}".format(
        ref_name, describe(ref), disagree)


# --------------------------------------------------------------------------
# Post hoc: any sweep whose rows carry the fingerprint.
# --------------------------------------------------------------------------

_NON_AXIS_KEYS = {"method", "train_sec", "eval_sec", "cell_sec", METRIC_KEY}
_MAX_FLOAT_AXIS_VALUES = 32


def _infer_axes(rows):
    """Cell-identifying fields, recovered without knowing the experiment.

    A field is taken as an axis when (a) it is a string/int, or a float with
    few distinct values (a grid like ``test_sparsity``, not a metric);
    (b) it varies within at least one method (a per-method constant such as
    ``num_points`` is a property of the method, not of the cell); and (c) at
    least one of its values is shared by rows of different methods (metrics
    are not). Pass ``--group-by`` to override when a sweep does not fit.
    """
    values = defaultdict(set)                       # key -> distinct values
    per_method = defaultdict(lambda: defaultdict(set))  # key -> method -> values
    shared = defaultdict(lambda: defaultdict(set))  # key -> value -> methods
    for r in rows:
        for k, v in r.items():
            if k in _NON_AXIS_KEYS or isinstance(v, bool) or not isinstance(v, (str, int, float)):
                continue
            values[k].add(v)
            per_method[k][r.get("method")].add(v)
            shared[k][v].add(r.get("method"))
    axes = []
    for k in values:
        if isinstance(next(iter(values[k])), float) and len(values[k]) > _MAX_FLOAT_AXIS_VALUES:
            continue
        if all(len(vs) == 1 for vs in per_method[k].values()):
            continue
        if not any(len(ms) > 1 for ms in shared[k].values()):
            continue
        axes.append(k)
    return sorted(axes)


def test_run_dir_rows_agree(run_dir, group_by):
    # A backfilled copy (backfill_fingerprints_exp2.py) takes precedence over
    # the raw rows; sweeps run after the contract existed carry the key natively.
    path = os.path.join(run_dir, "results_with_fingerprints.jsonl")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "results.jsonl")
    with open(path) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    rows = [r for r in rows if r.get(METRIC_KEY)]
    if not rows:
        pytest.skip("no row in {} carries {!r} (sweep predates the fingerprint; "
                    "see backfill_fingerprints_exp2.py)".format(run_dir, METRIC_KEY))

    axes = group_by or _infer_axes(rows)
    assert axes, "could not infer any grid axis from the rows; pass --group-by"
    missing = [k for k in axes if not any(k in r for r in rows)]
    assert not missing, "--group-by field(s) {} not present in any row".format(missing)

    groups = defaultdict(dict)
    for r in rows:
        groups[tuple(r.get(k) for k in axes)][r.get("method")] = r[METRIC_KEY]

    problems = []
    compared = 0
    for key, fps in groups.items():
        if len(fps) < 2:
            continue
        compared += 1
        ref_name, ref = next(iter(fps.items()))
        for m, fp in fps.items():
            if not fingerprints_agree(ref, fp):
                problems.append("{}: {}={} vs {}={}".format(
                    dict(zip(axes, key)), ref_name, describe(ref), m, describe(fp)))
    assert compared > 0, "no cell has more than one method; nothing to compare (axes: {})".format(axes)
    assert not problems, "{} of {} cells disagree:\n{}".format(
        len(problems), compared, "\n".join(problems))
