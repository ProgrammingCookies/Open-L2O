"""Instances fingerprint: the platform's evidence that methods were scored on
the same problem instances from the same starting points (x0).

An adapter's ``evaluate()`` hashes the arrays that define what its script
evaluated (for LASSO: ``b``, ``x_true`` and the starting point ``x0``, in the
order the script walked them) and stores the result under
``EvalOutput.metrics[METRIC_KEY]``. Because every experiment already spreads
``eval_out.metrics`` into its result rows, the fingerprint lands in
``results.jsonl`` with no experiment-side change, and ``sweep.summarize``
skips it (non-numeric). ``tests/test_fairness.py`` then asserts that every
method in a group reports the same fingerprint.

The fingerprint is adaptable to new problem types: it only knows named arrays. What the
arrays *are* is the adapter's business, a new problem's adapter chooses the
components that identify one of its instances (for a neural-net optimizee
that might be the initial parameters and the data batches consumed) and
calls :func:`fingerprint_arrays` with them.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

#: Key under which adapters store the fingerprint in ``EvalOutput.metrics``.
METRIC_KEY = "instances_fingerprint"


def fingerprint_arrays(**arrays: Any) -> Dict[str, Any]:
    """Hash named arrays into ``{"sha256", "num_instances", "components"}``.

    Components are hashed in sorted name order so the result does not depend
    on keyword order; each is normalised to C-contiguous float32 so that the
    same numbers loaded as float64 by one adapter and float32 by another
    still agree. ``num_instances`` is the leading dimension shared by all
    components (if a mismatch raises it means the caller paired arrays that
    were not walked together).
    """
    if not arrays:
        raise ValueError("fingerprint_arrays needs at least one named array")
    h = hashlib.sha256()
    num_instances: Optional[int] = None
    components = []
    for name in sorted(arrays):
        value = arrays[name]
        if value is None:
            continue
        arr = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("component {!r} is a scalar; expected (num_instances, ...)".format(name))
        n = int(arr.shape[0])
        if num_instances is None:
            num_instances = n
        elif n != num_instances:
            raise ValueError(
                "component {!r} has {} instances but an earlier component has {}; "
                "fingerprint components must be row-aligned".format(name, n, num_instances))
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
        components.append(name)
    if num_instances is None:
        raise ValueError("fingerprint_arrays: every component was None")
    return {"sha256": h.hexdigest(), "num_instances": num_instances, "components": components}


def fingerprints_agree(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    """True iff both fingerprints exist and hash the same components to the same digest."""
    if not a or not b:
        return False
    return (a.get("sha256") == b.get("sha256")
            and list(a.get("components", [])) == list(b.get("components", []))
            and a.get("num_instances") == b.get("num_instances"))


def describe(fp: Optional[Dict[str, Any]]) -> str:
    """Short human-readable form for assertion messages."""
    if not fp:
        return "<none>"
    return "{}[n={}, {}]".format(fp.get("sha256", "?")[:12], fp.get("num_instances"),
                                 "+".join(fp.get("components", [])))
