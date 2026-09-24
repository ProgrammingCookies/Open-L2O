"""Compute ``instances_fingerprint`` for the rows of an Experiment-2 sweep that
predates the fingerprint contract (``core/fingerprint.py``), so that
``pytest tests --run-dir <run>`` can check it post hoc.

Experiment-2-specific by necessity: it knows the ``experiments/(v2/)experiment2*.py``
artifact layout. Sweeps run after 2026-09-17 carry the fingerprint in their
rows natively and never need this.

    python analysis/backfill_fingerprints_exp2.py --run runs/exp2_try3_210x300

Writes ``<run>/results_with_fingerprints.jsonl`` (never touches
``results.jsonl``). Rows whose artifacts cannot be found are copied through
without the key and listed at the end.

Components: pre-2026-09-17 model-free ``.npz`` files hold ``b``/``x_true`` but
not ``x0``, while the model-based fingerprint reads ``x0`` from the data
files. For a like-for-like comparison every row is fingerprinted over the
same component set, ``--components`` (default ``b,x_true``). Pass
``b,x_true,x0`` for a sweep whose model-free artifacts include ``x0``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.fingerprint import METRIC_KEY, fingerprint_arrays  # noqa: E402

_MODEL_FREE = ("l2o-dm", "l2o-rnnprop", "l2o-scale")
_MODEL_FREE_NPZ = ("L2L_recovery-lasso_dataset.npz", "L2o_recovery-lasso_dataset.npz")


def _test_fname(p):
    return "test_sparsity_{:.2f}.npy".format(p)


def _row_fingerprint(run, row, components):
    seed, width, method, p = row["seed"], row["width"], row["method"], row["test_sparsity"]
    cell = os.path.join(run, "checkpoints", "s{}".format(seed), width, method)
    data_dir = os.path.join(run, "data", "seed{}".format(seed), width)
    arrays = {}
    if method in _MODEL_FREE:
        eval_dir = os.path.join(cell, "eval_p{:.2f}".format(p), "eval")
        for name in _MODEL_FREE_NPZ:
            path = os.path.join(eval_dir, name)
            if os.path.exists(path):
                rec = np.load(path)
                for c in components:
                    if c not in rec.files:
                        raise KeyError("{} lacks {!r}; drop it from --components".format(path, c))
                    arrays[c] = rec[c]
                return fingerprint_arrays(**arrays)
        raise FileNotFoundError("no recovery .npz under {}".format(eval_dir))
    # model-based: x_pred from the run, b/x_true/x0 from the data files
    matches = glob.glob(os.path.join(cell, "**", _test_fname(p)[:-4] + "_final_output.npy"),
                        recursive=True)
    if not matches:
        raise FileNotFoundError("no final_output for p={} under {}".format(p, cell))
    x_pred = np.load(matches[0])
    a = np.load(os.path.join(data_dir, "A.npy"))
    m = a.shape[0]
    split = np.load(os.path.join(data_dir, _test_fname(p))).astype(np.float32)
    if x_pred.shape[0] != split.shape[0]:
        raise ValueError("{} has {} rows, split has {}".format(matches[0], x_pred.shape[0],
                                                              split.shape[0]))
    source = {"b": lambda: split[:, :m], "x_true": lambda: split[:, m:],
              "x0": lambda: np.load(os.path.join(data_dir, _test_fname(p)[:-4] + "_x0.npy"))}
    for c in components:
        arrays[c] = source[c]()
    return fingerprint_arrays(**arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--components", default="b,x_true")
    args = ap.parse_args()
    components = [c.strip() for c in args.components.split(",") if c.strip()]

    src = os.path.join(args.run, "results.jsonl")
    dst = os.path.join(args.run, "results_with_fingerprints.jsonl")
    with open(src) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]

    failed = []
    with open(dst, "w") as out:
        for i, row in enumerate(rows):
            try:
                row[METRIC_KEY] = _row_fingerprint(args.run, row, components)
            except Exception as exc:  # noqa: BLE001 -- report and continue
                failed.append((i, row.get("method"), row.get("seed"), row.get("width"),
                               row.get("test_sparsity"), str(exc)))
            out.write(json.dumps(row) + "\n")
    print("wrote {} ({} rows, {} fingerprinted over {})".format(
        dst, len(rows), len(rows) - len(failed), "+".join(components)))
    for item in failed:
        print("  skipped row {} {}: {}".format(item[0], item[1:5], item[5]))


if __name__ == "__main__":
    main()
