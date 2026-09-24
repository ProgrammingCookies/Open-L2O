"""
Shared helper for aggregating a still-growing sweep without silently mixing seed counts.

A sweep writes one result row per (method, width, seed, test_sparsity) cell, and a
stopped or in-flight sweep leaves those cells at different seed counts. Every
cross-method table and figure then aggregates one method over N seeds and another
over N-1, which is not a comparison: the difference it reports contains a
seed-composition difference as well as the effect being measured.

Usage::

    from core.seedsets import complete_seeds
    seeds, dropped = complete_seeds(rows, methods=["lista", "alista"])
    if dropped:
        print("seed-incomplete, excluded: %s" % dropped)
    rows = [r for r in rows if r["seed"] in seeds]
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def seeds_by_cell(rows: Iterable[Dict[str, Any]],
                  methods: Sequence[str] | None = None,
                  cell_keys: Sequence[str] = ("method", "width"),
                  ) -> Dict[Tuple, Set[int]]:
    """Maps each (method, width, ...) cell to the set of seeds present for it."""
    out: Dict[Tuple, Set[int]] = collections.defaultdict(set)
    for row in rows:
        if methods is not None and row.get("method") not in methods:
            continue
        out[tuple(row[k] for k in cell_keys)].add(int(row["seed"]))
    return dict(out)


def complete_seeds(rows: Iterable[Dict[str, Any]],
                   methods: Sequence[str] | None = None,
                   cell_keys: Sequence[str] = ("method", "width"),
                   ) -> Tuple[List[int], List[int]]:
    """Returns sorted lists of which seeds are completed across configurations and whic are not.
    Returns (complete seeds, dropped seeds). Raises ValueError if no rows are present for the given methods.
    Example: 
    A completed seeds: [1, 2, 3]
    B completed seeds: [1, 2, 3, 4] 
    Returns: ([1, 2, 3], [4])
    """
    cells = seeds_by_cell(rows, methods=methods, cell_keys=cell_keys)
    if not cells:
        raise ValueError(
            "no result rows for methods=%r -- check the method keys against the "
            "run's results.jsonl" % (list(methods) if methods else methods,))
    union: Set[int] = set().union(*cells.values())
    complete: Set[int] = set.intersection(*cells.values())
    return sorted(complete), sorted(union - complete)


def filter_to_complete_seeds(rows: Sequence[Dict[str, Any]],
                             methods: Sequence[str] | None = None,
                             cell_keys: Sequence[str] = ("method", "width"),
                             label: str = "",
                             verbose: bool = True,
                             ) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """`complete_seeds` plus the filtering, announcing what it dropped.
    Returns (kept rows, complete seeds, dropped seeds).
    """
    complete, dropped = complete_seeds(rows, methods=methods, cell_keys=cell_keys)
    if not complete:
        raise ValueError(
            "no seed is present in every %s cell%s, nothing can be aggregated "
            "fairly. Seeds seen per cell: %r"
            % ("/".join(cell_keys), (" of " + label) if label else "",
               {k: sorted(v) for k, v in
                seeds_by_cell(rows, methods, cell_keys).items()}))
    if verbose and dropped:
        print("[seedsets] %s: using %d complete seeds %s; EXCLUDED %s "
              "(present for some methods/widths but not all -- the sweep is "
              "mid-seed or was stopped early)"
              % (label or "run", len(complete), complete, dropped))
    kept = [r for r in rows if int(r["seed"]) in set(complete)]
    return kept, complete, dropped
