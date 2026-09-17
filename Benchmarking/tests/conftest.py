"""pytest wiring for the platform's fairness/reproducibility tests.

Run from anywhere with the ``thesis_updated`` env active::

    pytest Benchmarking/tests -m "not slow"        # seconds: contract + data tests
    pytest Benchmarking/tests -m slow               # minutes: live runs on the tiny fixture
    pytest Benchmarking/tests --run-dir runs/<run>  # post-hoc: check an existing sweep

The live tests need every registered method's library to run in this
interpreter's environment (same requirement as ``run.py``).
"""

from __future__ import annotations

import os
import sys

import pytest

_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKING_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKING_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--run-dir", action="store", default=None,
        help="An existing sweep directory (containing results.jsonl) to check post hoc.")
    parser.addoption(
        "--group-by", action="store", default=None,
        help="Comma-separated row fields that identify one cell of that sweep, excluding "
             "'method' (e.g. seed,width,test_sparsity). Inferred from the rows if omitted.")


@pytest.fixture(scope="session")
def run_dir(request):
    value = request.config.getoption("--run-dir")
    if not value:
        pytest.skip("pass --run-dir <sweep dir> to check an existing run")
    path = os.path.abspath(value)
    if not os.path.exists(os.path.join(path, "results.jsonl")):
        pytest.fail("{} has no results.jsonl".format(path))
    return path


@pytest.fixture(scope="session")
def group_by(request):
    value = request.config.getoption("--group-by")
    return [k.strip() for k in value.split(",") if k.strip()] if value else None


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    """One scratch directory per session for generated fixture data."""
    return str(tmp_path_factory.mktemp("fairness_fixtures"))
