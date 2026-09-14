"""Shared sweep driver for the scripts in ``experiments/``.

WHAT STAYS YOUR JOB
-------------------
The sweep driver owns the *mechanics*. It deliberately does not own:

* your config dataclass -- keep it per-experiment and keep its comments. The
  justification text in ``experiment2.py``'s and ``experiment2_try2.py``'s
  ``OrchestratorConfig`` is the actual content of those files; two configs
  with the same value for different reasons are not duplication;
* translating your hyperparameters into a specific library's flags -- flag
  names differ per library (there are three conventions in this repo, see
  ``CONTRIBUTING.md`` §9), so this is irreducibly per-experiment;
* what one cell of the sweep actually runs.

QUICK START
-----------
Copy ``experiment_template.py`` and edit it. The shape is::

    spec = SweepSpec(
        results_dir=cfg.results_dir,
        axes=[Axis("seed", cfg.seeds), Axis("method", cfg.methods)],
        prepare=[(("seed",), make_dataset)],
    )
    run_sweep(spec, run_cell, config_snapshot=cfg)

``run_cell(cell, context)`` gets one grid point and returns a result row (or a
list of rows). Everything else -- resume, errors, provenance, summary -- is
handled for you.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCHMARKING_ROOT, ".."))

_ALWAYS_NON_METRIC = ("train_sec", "eval_sec", "cell_sec")

#: How ``summarize`` reduces each metric's values within a group. The dict KEY
#: is the suffix appended to the metric name in summary.json, so the name and
#: the function that produces it stay together, and insertion order fixes the
#: field order in the output.
#:
#: To add a statistic rather than replace these two, spread this dict:
#:
#:     summarize(..., aggregators={**DEFAULT_AGGREGATORS, "seed_max": np.max})
#:
#: Note ``np.std`` is the POPULATION standard deviation (ddof=0). That is what
#: every summary.json in this repo was produced with; pass
#: ``lambda v: np.std(v, ddof=1)`` under your own suffix if you want the sample
#: estimate rather than silently changing the meaning of an existing field.
DEFAULT_AGGREGATORS: "Dict[str, Callable[[Sequence[float]], Any]]" = {
    "seed_mean": np.mean,
    "seed_std": np.std,
}


def _json_scalar(value: Any) -> Any:
    """Make an aggregator's return value json-dumpable.

    ``float(...)`` alone was enough while the only aggregators were mean and
    std, but an aggregator may legitimately return an int (``len``) or an
    array (``np.percentile`` with several quantiles), and a numpy scalar is
    not json-serializable at all. Anything else is passed through untouched so
    a custom aggregator returning, say, a plain dict still works.
    """
    if isinstance(value, np.generic):     # np.float64, np.int64 etc
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Sweep class
@dataclass
class Axis:
    """One dimension of the sweep grid.

    ``values`` are passed to ``run_cell`` unchanged, so they can be any Python
    object your experiment finds convenient.

    ``label`` maps a value to how it appears in ``results.jsonl`` and in the
    resume key. It exists because those two places need something JSON-round-
    trippable and human-readable. It must return a str, int
    or float -- a tuple would survive ``json.dump`` as a list and then fail to
    compare equal to the original tuple on resume, silently re-running every
    completed cell.
    """
    name: str
    values: Sequence[Any]
    label: Optional[Callable[[Any], Any]] = None

    def label_of(self, value: Any) -> Any:
        labelled = self.label(value) if self.label is not None else value
        if not isinstance(labelled, (str, int, float)) or isinstance(labelled, bool):
            raise TypeError(
                "Axis {!r} labelled {!r} as {!r}; a label must be a str/int/float so it "
                "survives the JSON round-trip through results.jsonl and still compares "
                "equal on resume. Pass label= to Axis to convert it."
                .format(self.name, value, labelled))
        return labelled


@dataclass
class SweepSpec:
    """Everything the driver needs that is not the experiment itself."""

    results_dir: str
    #: Grid axes, outermost loop first. Their names become row fields.
    axes: List[Axis]

    #: Row fields ``run_cell`` adds that are NOT part of the resume key,
    #: because one cell legitimately produces several rows that differ in
    #: them. ``experiment2.py`` trains once per (seed, width, method) and then
    #: evaluates that one checkpoint at eight test sparsities, so
    #: ``fanout_keys=("test_sparsity",)``: the sweep resumes at
    #: checkpoint granularity, but the results table has one row per sparsity.
    fanout_keys: Tuple[str, ...] = ()

    #: Per-prefix setup, as ``(axis_name_prefix, fn)`` pairs. ``fn(cell,
    #: context) -> dict`` is merged into the context handed to ``run_cell``,
    #: and is called at most once per distinct prefix -- this is where dataset
    #: generation goes, so a dataset shared by every cell under ``seed=0`` is
    #: built once rather than once per cell.
    #:
    #: Hooks fire lazily, on the first cell under that prefix that actually
    #: needs to run. Resuming a finished sweep therefore does no dataset work
    #: at all. (The pre-port scripts generated eagerly; the generators are
    #: idempotent, so this changes cost and log ordering, not results.)
    prepare: List[Tuple[Tuple[str, ...], Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]] = \
        field(default_factory=list)

    #: Static extras merged into every cell's context, before the prepare
    #: hooks run. This is where your config object goes, so ``run_cell`` and
    #: the hooks can reach it without a closure or a module global.
    context: Dict[str, Any] = field(default_factory=dict)

    #: Which axis is the repetition axis. Excluded from the default summary
    #: grouping, since the summary's whole job is averaging over it.
    seed_axis: str = "seed"

    #: Re-raise instead of appending to errors.jsonl. Off by default because
    #: the real sweeps are unattended and multi-day -- one bad cell must not
    #: cost the other 1079. Turn it ON while developing, where a silent
    #: errors.jsonl that you only read afterwards is the worse failure.
    stop_on_error: bool = False

    def __post_init__(self):
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate axis names: {}".format(names))
        clash = set(names) & set(self.fanout_keys)
        if clash:
            raise ValueError(
                "{} appears both as an axis and in fanout_keys -- a field is either part "
                "of the resume key or it varies within a cell, not both.".format(sorted(clash)))

    @property
    def axis_names(self) -> List[str]:
        return [a.name for a in self.axes]


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def _grid(axes: List[Axis]) -> Iterable[List[Tuple[Axis, Any]]]:
    """Cartesian product of the axes, first axis varying slowest."""
    if not axes:
        yield []
        return
    head, rest = axes[0], axes[1:]
    for value in head.values:
        for tail in _grid(rest):
            yield [(head, value)] + tail


def _load_completed(results_path: str, axis_names: List[str]) -> set:
    """Resume keys already present in results.jsonl.

    The key is read out of the row itself rather than recomputed, which is what
    makes resume robust: whatever the row says it is, is what counts as done.
    """
    completed = set()
    if not os.path.exists(results_path):
        return completed
    with open(results_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [n for n in axis_names if n not in row]
            if missing:
                raise ValueError(
                    "{}:{} has no {} field(s), so this sweep cannot tell which cells are "
                    "already done. The results file was probably written by a different "
                    "experiment or an older axis set -- point --results_dir somewhere else."
                    .format(results_path, lineno, missing))
            completed.add(tuple(row[n] for n in axis_names))
    return completed


def _stamp_axes(row: Dict[str, Any], labels: Dict[str, Any], where: str) -> Dict[str, Any]:
    """Write the cell's axis labels into the row.

    Done by the sweep driver rather than by ``run_cell`` so a row can never disagree
    with the cell that produced it -- the class of bug where a results table
    records ``seed=3`` for a run that actually used seed 4, which is invisible
    afterwards and invalidates everything downstream.
    """
    for name, label in labels.items():
        if name in row and row[name] != label:
            raise ValueError(
                "{} returned a row with {}={!r}, but this cell is {}={!r}. The sweep driver "
                "stamps axis fields itself; remove it from your row."
                .format(where, name, row[name], name, label))
        row[name] = label
    return row


def run_sweep(spec: SweepSpec,
              run_cell: Callable[[Dict[str, Any], Dict[str, Any]], Any],
              config_snapshot: Any = None,
              group_by: Optional[Sequence[str]] = None,
              aggregators: Optional[Dict[str, Callable[[Sequence[float]], Any]]] = None) -> str:
    """Run the grid, appending to ``results.jsonl``. Returns its path.

    ``run_cell(cell, context)`` receives:
      * ``cell``   -- ``{axis_name: raw value}`` for this grid point (raw, not
        the label: your code gets the ``(25, 50)`` tuple, the results table
        gets ``"25x50"``);
      * ``context`` -- ``results_dir``/``data_dir``/``checkpoints_dir`` plus
        whatever the ``prepare`` hooks returned.

    and returns one row dict, or a list of them (one per fan-out value).

    ``config_snapshot`` is your experiment's config object; it is recorded once
    per results_dir in ``config.json`` alongside the code state, so a
    ``summary.json`` produced today is still traceable after your defaults
    change next month.

    ``group_by`` and ``aggregators`` shape the summary written at the end; see
    :func:`summarize`. Both default to something sensible, so pass them only
    when the defaults are wrong for your experiment.
    """
    # abspath first: adapters launch library scripts with cwd=<that library's
    # own directory>, so a relative results_dir resolves against the wrong
    # place the moment training starts.
    spec.results_dir = os.path.abspath(spec.results_dir)
    os.makedirs(spec.results_dir, exist_ok=True)

    results_path = os.path.join(spec.results_dir, "results.jsonl")
    errors_path = os.path.join(spec.results_dir, "errors.jsonl")
    axis_names = spec.axis_names
    completed = _load_completed(results_path, axis_names)

    # Keep the caller's config in step with the abspath above, so config.json
    # records the directory the sweep actually wrote to and not the relative
    # path that would resolve elsewhere under an adapter's cwd.
    if config_snapshot is not None and hasattr(config_snapshot, "results_dir"):
        config_snapshot.results_dir = spec.results_dir

    git_state = capture_provenance(spec.results_dir)
    write_config_snapshot(spec.results_dir, config_snapshot, git_state, len(completed))
    append_provenance_record(spec.results_dir, git_state, len(completed))
    if not git_state.get("git_binary_available"):
        print("[provenance] WARNING: no `git` binary here -- recorded the commit from "
              ".git/ but NOT the working-tree state. Install git in the image "
              "(Dockerfile) to capture it.")

    # results_dir here is the absolute one, so a run_cell that stores it back
    # onto its config records the path the sweep actually used rather than the
    # relative one the caller typed.
    base_context = {
        "results_dir": spec.results_dir,
        "data_dir": os.path.join(spec.results_dir, "data"),
        "checkpoints_dir": os.path.join(spec.results_dir, "checkpoints"),
        **spec.context,
    }
    prepared: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}

    for point in _grid(spec.axes):
        cell = {axis.name: value for axis, value in point}
        labels = {axis.name: axis.label_of(value) for axis, value in point}
        key = tuple(labels[n] for n in axis_names)
        if key in completed:
            print("[skip, already done] {}".format(key))
            continue

        # Hooks run only now, once we know this cell has work to do.
        context = dict(base_context)
        for prefix, fn in spec.prepare:
            unknown = [n for n in prefix if n not in cell]
            if unknown:
                raise ValueError(
                    "prepare hook prefix {} names {} which is not an axis ({})."
                    .format(prefix, unknown, axis_names))
            memo_key = (",".join(prefix), tuple(labels[n] for n in prefix))
            if memo_key not in prepared:
                prepared[memo_key] = fn({n: cell[n] for n in prefix}, dict(context))
            context.update(prepared[memo_key])

        print("[running] {}".format(key))
        t0 = time.time()
        try:
            produced = run_cell(cell, context)
        except Exception as exc:  # noqa: BLE001 -- one bad cell must not end the sweep
            with open(errors_path, "a") as f:
                f.write(json.dumps({
                    **labels,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
            print("[FAILED] {} -- {} (see {})".format(key, exc, errors_path))
            if spec.stop_on_error:
                raise
            continue

        rows = [produced] if isinstance(produced, dict) else list(produced)
        if not rows:
            # Recording nothing would leave the cell looking un-run forever, so
            # a resume would repeat it indefinitely. Treat it as a failure.
            with open(errors_path, "a") as f:
                f.write(json.dumps({
                    **labels, "error": "run_cell returned no rows",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
            print("[FAILED] {} -- run_cell returned no rows".format(key))
            if spec.stop_on_error:
                raise RuntimeError("run_cell returned no rows for {}".format(key))
            continue

        cell_sec = time.time() - t0
        # Build every row before writing any: a fan-out key missing from row 3
        # must not leave rows 1-2 on disk, which would mark the cell done and
        # make the failure unreachable on resume.
        stamped = []
        for row in rows:
            row = _stamp_axes(dict(row), labels, "run_cell")
            missing = [k for k in spec.fanout_keys if k not in row]
            if missing:
                raise ValueError(
                    "run_cell returned a row for {} without fan-out key(s) {}. Every row must "
                    "carry them, or the summary groups every row under None and silently "
                    "averages across values that were never comparable."
                    .format(key, missing))
            row.setdefault("cell_sec", cell_sec)
            stamped.append(row)
        with open(results_path, "a") as f:
            for row in stamped:
                f.write(json.dumps(row) + "\n")
        completed.add(key)
        print("[done] {} in {:.1f}s ({} row{})".format(
            key, cell_sec, len(rows), "" if len(rows) == 1 else "s"))

    if group_by is None:
        group_by = [n for n in axis_names if n != spec.seed_axis] + list(spec.fanout_keys)
    non_metric = tuple(axis_names) + spec.fanout_keys + _ALWAYS_NON_METRIC
    summarize(results_path, os.path.join(spec.results_dir, "summary.json"),
              group_by=group_by, non_metric_keys=non_metric, aggregators=aggregators)
    return results_path


def summarize(results_path: str, summary_path: str, group_by: Sequence[str],
              non_metric_keys: Sequence[str],
              aggregators: Optional[Dict[str, Callable[[Sequence[float]], Any]]] = None) -> None:
    """Reduce results.jsonl to one entry per ``group_by`` combination.

    Every numeric field that is not in ``non_metric_keys`` is aggregated, so a
    metric added to your rows later shows up in the summary without touching
    this function.

    ``aggregators`` maps a field-name suffix to the function that computes it,
    defaulting to :data:`DEFAULT_AGGREGATORS` (mean and population std). A
    metric ``final_loss`` with the default therefore yields
    ``final_loss_seed_mean`` and ``final_loss_seed_std``. Supply your own when
    the mean is the wrong summary for what you are studying -- worst case
    across seeds, a median that ignores one diverged run, an interquartile
    spread:

        summarize(..., aggregators={**DEFAULT_AGGREGATORS,
                                    "seed_median": np.median,
                                    "seed_max": np.max})

    Spread ``DEFAULT_AGGREGATORS`` like that rather than passing a bare dict
    unless you genuinely want mean/std gone: the repo's own analysis scripts
    (``generate_result_tables.py`` and friends) look up ``<field>_seed_mean``
    and ``<field>_seed_std`` by name.
    """
    if aggregators is None:
        aggregators = DEFAULT_AGGREGATORS
    if not os.path.exists(results_path):
        return
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    non_metric = set(non_metric_keys)
    metric_keys = sorted({k for row in rows for k in row if k not in non_metric})
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in group_by), []).append(row)

    summary = []
    # Plain sort, not a stringified one: each key position always holds the
    # same axis and therefore the same type, so this is safe -- and it keeps
    # numeric axes in numeric order (a stringified sort puts 10 before 2).
    for key, group_rows in sorted(groups.items()):
        entry: Dict[str, Any] = dict(zip(group_by, key))
        # Named num_seeds for continuity with the existing summary.json files,
        # though it counts rows in the group, which for a fan-out sweep is
        # one row per seed at that fan-out value.
        entry["num_seeds"] = len(group_rows)
        for k in metric_keys:
            vals = [r[k] for r in group_rows
                    if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            if not vals:
                continue
            for suffix, aggregate in aggregators.items():
                try:
                    value = aggregate(vals)
                except Exception as exc:  # noqa: BLE001 -- re-raised with context below
                    raise RuntimeError(
                        "aggregator {!r} raised on metric {!r} (group {}, {} value(s)): {}. "
                        "NOTHING IS LOST -- {} is already written and complete; the sweep "
                        "itself finished. Fix the aggregator and call summarize() on that "
                        "file again, no recompute needed."
                        .format(suffix, k, key, len(vals), exc, results_path)) from exc
                entry["{}_{}".format(k, suffix)] = _json_scalar(value)
        summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary ({} groups) to {}".format(len(summary), summary_path))


# ---------------------------------------------------------------------------
# Provenance
#
# Ported from experiment1.py, which has the only complete version in the repo.
# A commit hash alone does not identify the code that ran here: much of what a
# sweep executes is uncommitted (core/, the library training scripts) or
# entirely untracked -- including, in exp2_try2's case, the orchestrator
# itself. That is exactly why those runs' config.json had to be rebuilt by
# hand. Sharing it here means every experiment gets the good version.
# ---------------------------------------------------------------------------

def _run_git(args: List[str]) -> Optional[str]:
    """``git <args>`` at the repo root, or None if it fails for any reason.

    Everything provenance-related is best-effort by construction: this is a
    multi-day unattended sweep, and a missing/broken git must never be able to
    kill it. stderr is discarded because git chatters about CRLF conversion on
    a Windows checkout; ``encoding=`` rather than ``text=`` so a non-UTF-8 byte
    in a diff can't raise instead of just being replaced.

    The two ``-c`` overrides matter for the Docker runs, where Linux git reads
    a checkout made by Windows git:
      * core.autocrlf -- set in this machine's *system* gitconfig, which a
        container never sees, so without pinning it here Linux git treats the
        CRLF working tree as differing from every LF blob (measured on this
        repo: 1820 "modified" paths and a multi-MB junk diff, versus 97 with it
        set). No-op on a checkout that is already LF.
      * safe.directory -- a bind-mounted repo is owned by a different uid than
        the container user, which makes git refuse every repo operation.
    """
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-c", "core.autocrlf=true"] + args,
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace")
    except Exception:
        return None


def _head_from_git_dir() -> Dict[str, Any]:
    """Commit/branch read straight out of .git/, for when the ``git`` binary is
    unavailable -- which is the normal case for the real runs: the open-l2o
    Docker image does not install git, while the repo (including .git/) is
    bind-mounted in. Without this, provenance in the environment that actually
    produces the results would silently be all-None.

    Dirty-tree state genuinely needs git, so it stays unknown here.
    """
    git_dir = os.path.join(_REPO_ROOT, ".git")
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
    except OSError:
        return {"commit": None, "branch": None}
    if not head.startswith("ref:"):
        return {"commit": head, "branch": None}  # detached HEAD
    ref = head.split(":", 1)[1].strip()
    # Strip only the refs/heads/ prefix, not every path segment -- a branch
    # named "feature/x" must not be reported as "x" (git rev-parse
    # --abbrev-ref, which the non-degraded path uses, keeps the slash).
    branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
    try:  # loose ref
        with open(os.path.join(git_dir, *ref.split("/")), encoding="utf-8") as f:
            return {"commit": f.read().strip(), "branch": branch}
    except OSError:
        pass
    try:  # packed-refs (a ref that's been gc'd into the pack file)
        with open(os.path.join(git_dir, "packed-refs"), encoding="utf-8") as f:
            for line in f:
                if line.startswith(("#", "^")):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return {"commit": parts[0], "branch": branch}
    except OSError:
        pass
    return {"commit": None, "branch": branch}


def capture_provenance(results_dir: str) -> Dict[str, Any]:
    """Commit *and* working-tree state as of this moment.

    Records, beyond the commit hash:
      * ``git status --porcelain`` -- the full list of dirty/untracked paths;
      * ``git diff HEAD`` -- written to a sidecar .diff file, since a 20-40 KB
        diff does not belong inline in a JSON record. Covers modified TRACKED
        files only;
      * sha256 of every untracked ``*.py`` -- the gap the diff cannot cover,
        and the specific thing that made exp2_try2 unreproducible (its
        orchestrator was itself untracked). This is also what keeps *this*
        file honest while it is untracked.
    """
    state: Dict[str, Any] = {"git_binary_available": _run_git(["--version"]) is not None}

    def _degrade(why: str) -> Dict[str, Any]:
        state.update(_head_from_git_dir())
        state["dirty"] = None
        state["degraded_note"] = (
            why + " Commit/branch were read directly from .git/ instead, and no "
            "working-tree state could be captured. dirty=null means UNKNOWN, "
            "not clean -- do not read the absence of a diff here as evidence "
            "the tree was clean.")
        return state

    if not state["git_binary_available"]:
        return _degrade("No `git` binary in this environment.")

    commit = _run_git(["rev-parse", "HEAD"])
    if commit is None:
        # git exists but can't operate on this repo at all (unreadable .git, a
        # refused bind mount, ...). Degrading here rather than trusting the
        # `git_binary_available: true` branch is what keeps this from silently
        # recording an all-null state with no explanation attached.
        return _degrade("`git` is installed but could not read this repository.")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    state["commit"] = commit.strip()
    state["branch"] = branch.strip() if branch else None

    # -uall: without it git collapses a new untracked directory to a single
    # "?? dir/" entry, so .py files inside it would never be fingerprinted
    # below. Safe to enumerate fully because Benchmarking/.gitignore already
    # excludes runs/ (else this would list every checkpoint).
    porcelain = _run_git(["status", "--porcelain", "-uall"])
    if porcelain is None:
        state["dirty"] = None
        return state
    entries = [ln for ln in porcelain.splitlines() if ln.strip()]
    state["dirty"] = bool(entries)
    state["status_porcelain"] = entries

    diff = _run_git(["diff", "HEAD"])
    if diff:
        raw = diff.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        # Named by content hash so repeated invocations over a stopped/restarted
        # sweep share one file when the tree hasn't moved, instead of piling up
        # identical copies; a resume that *has* drifted gets its own file.
        name = "working_tree_{}.diff".format(digest[:12])
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            # Binary write: on Windows text mode would translate \n -> \r\n, so
            # the recorded sha256 would not match the bytes actually on disk.
            with open(path, "wb") as f:
                f.write(raw)
        state["diff_file"] = name
        state["diff_sha256"] = digest
        state["diff_bytes"] = len(raw)
    state["diff_note"] = ("`git diff HEAD` excludes untracked files; those are the "
                          "'??' entries in status_porcelain, hashed in untracked_py_sha256.")

    untracked: Dict[str, str] = {}
    for entry in entries:
        if not entry.startswith("??"):
            continue
        rel = entry[3:].strip().strip('"')
        if not rel.endswith(".py"):
            continue
        try:
            with open(os.path.join(_REPO_ROOT, rel), "rb") as f:
                untracked[rel] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            continue
    if untracked:
        state["untracked_py_sha256"] = untracked
    return state


def _jsonable(obj: Any) -> Any:
    """Make a config dataclass JSON-dumpable.

    ``dataclasses.asdict`` is not enough on its own: this repo's configs hold
    dicts keyed by ``(m, n)`` tuples (``train_size_by_dim``,
    ``batch_size_by_dim``), which ``json.dump`` rejects outright. Tuple keys
    are joined with "x" because that is already this repo's spelling for a
    dimension pair everywhere else -- ``_dim_key``, the row field, and the
    ``--dims 25x50`` CLI format.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = "x".join(str(part) for part in k)
            elif not isinstance(k, str):
                k = str(k)
            out[k] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_config_snapshot(results_dir: str, config: Any, git_state: Dict[str, Any],
                          cells_already_done: int) -> None:
    """Write ``config.json`` once per results_dir.

    Write-once, not overwritten on resume, for the same reason results.jsonl is
    append-only: a resumed run's CLI flags are not necessarily the ones that
    produced the rows already on disk, and silently replacing the snapshot with
    the later invocation's would misattribute them.

    ``cells_already_done_when_written`` records how many completed cells predate this
    snapshot, so a results_dir that was running before the snapshot mechanism
    existed can never be misread as fully described by it.
    """
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        return
    snapshot = {
        "config": _jsonable(config),
        "git": git_state,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "cells_already_done_when_written": cells_already_done,
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print("Wrote config snapshot to {}".format(path))


def append_provenance_record(results_dir: str, git_state: Dict[str, Any],
                             cells_already_done: int) -> None:
    """Append one record per ``run_sweep`` invocation to ``provenance.jsonl``.

    ``config.json`` above is deliberately write-once, which is right for the
    config but wrong for code state: these sweeps are resumable and get stopped
    and restarted over days (to free the GPU for other jobs), and a restart is
    precisely when the working tree has drifted. Since results.jsonl is
    append-only, recording how many cells were already complete when each invocation
    started is what lets a row be mapped back to the code state that produced
    it.
    """
    record = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "cells_already_done": cells_already_done,
        "argv": sys.argv,
        "git": git_state,
    }
    with open(os.path.join(results_dir, "provenance.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# CLI overrides
#
# Today every tunable is written three times: once as a dataclass field, once
# as an add_argument call, once in a setattr loop in main(). That is not just
# tedious, it silently rots -- experiment2.py's main() applies 6 of its
# OrchestratorConfig's ~25 fields, so most --flags a reader would expect to
# exist simply do not. Deriving the flags from the dataclass keeps them in
# sync by construction.
# ---------------------------------------------------------------------------

def add_overrides(parser, config_cls: type, fields: Optional[Sequence[str]] = None,
                  help_texts: Optional[Dict[str, str]] = None) -> List[str]:
    """Add one ``--field`` flag per scalar field of ``config_cls``.

    Every flag defaults to ``None``, meaning "leave the dataclass default
    alone" -- so ``--flag`` present and absent are distinguishable, and the
    dataclass stays the single source of truth for what the default *is*.

    Only ``int``/``float``/``str``/``bool`` fields are handled. List and dict
    fields (seeds, dims, per-dim tables) need bespoke parsing and are skipped;
    add those to your parser by hand. Bools become ``store_true``, matching the
    existing scripts' convention that a flag can turn a feature on but not off.

    A field whose flag the caller already declared by hand is skipped, and left
    out of the returned list, so the hand-written one wins -- otherwise a
    required positional-ish flag like ``--results_dir``, which callers declare
    themselves, would collide here and the script could not start at all.

    Returns the field names it added, to pass straight to ``apply_overrides``.
    """
    help_texts = help_texts or {}
    added: List[str] = []
    for f in dataclasses.fields(config_cls):
        if fields is not None and f.name not in fields:
            continue
        # With ``from __future__ import annotations`` in the defining module,
        # f.type is the string "int"/"List[int]"/...; without it, the type
        # object. Handle both, and skip anything that isn't a plain scalar.
        kind = f.type if isinstance(f.type, type) else {
            "int": int, "float": float, "str": str, "bool": bool}.get(str(f.type))
        if kind not in (int, float, str, bool):
            continue
        flag = "--{}".format(f.name)
        try:
            if kind is bool:
                parser.add_argument(flag, action="store_true", default=None,
                                    help=help_texts.get(f.name))
            else:
                parser.add_argument(flag, type=kind, default=None,
                                    help=help_texts.get(f.name))
        except argparse.ArgumentError:
            continue  # already declared by hand; that one wins
        added.append(f.name)
    return added


def apply_overrides(config: Any, args, fields: Sequence[str]) -> Any:
    """Copy every non-None ``args.<field>`` onto ``config``. Returns config."""
    for name in fields:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    return config
