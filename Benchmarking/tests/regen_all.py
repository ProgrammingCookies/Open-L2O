"""Regenerates every thesis figure and result table, and snapshots the output.

Two jobs in one file, because they are the same list of commands:

1. **Documentation.** ``--list`` prints the exact invocation behind every
   figure and table in the thesis. That list did not previously exist
   anywhere; it was spread across the module docstrings of 16 scripts.
2. **Regression safety net.** ``--snapshot DIR`` runs them all and copies
   every produced artifact into DIR; ``--compare DIR`` runs them again and
   diffs against that snapshot. Capture a snapshot BEFORE refactoring an
   analysis script and any byte that changes afterwards is a bug, not an
   improvement -- this is how the three ``generate_result_tables*.py`` forks
   and the two ``plot_ood_degradation*.py`` forks were proved to collapse into
   one ``--run``-parameterized script each without moving a single number.

Run from the Benchmarking/ directory::

    python tests/regen_all.py --list
    python tests/regen_all.py --snapshot tests/golden
    python tests/regen_all.py --compare tests/golden

Not a pytest module (it shells out to 20+ matplotlib scripts and takes
minutes); it lives in tests/ because that is where the other verification
tooling lives.

CUDA is disabled for every child process: these are analysis scripts that
only read .jsonl result rows, and a GPU sweep is often running alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BENCH_DIR)
FIGS_DIR = os.path.join(REPO_ROOT, "Figs")

# Importable whether this is run as `python tests/regen_all.py` (which puts
# tests/ on sys.path, not Benchmarking/) or via pytest.
for _p in (BENCH_DIR, os.path.join(BENCH_DIR, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every Experiment-2 run, from the shared registry. Kept in sync automatically:
# adding a spec to run_specs.py adds it to this regeneration list.
import run_specs  # noqa: E402  (needs the sys.path line above)

MODEL_BASED_RUNS = run_specs.choices()
# Runs that declare a model-free arm with valid (post-checkpoint-fix) rows.
# Not the same as having a model-free table column: exp2_try2 has 2 valid
# l2o-rnnprop seeds, too few to tabulate but enough to plot the spread.
MODEL_FREE_RUNS = [k for k in MODEL_BASED_RUNS
                   if run_specs.RUN_SPECS[k].model_free_arm]

# (label, argv). Label is what --list prints and what EXIT_CODES.txt records.
COMMANDS = []

# --- Experiment 2: result tables --------------------------------------------
COMMANDS += [("tables/" + _r, ["analysis/generate_result_tables.py", "--run", _r])
             for _r in MODEL_BASED_RUNS]

# --- Experiment 2: OOD degradation -------------------------------------------
COMMANDS += [("ood/" + _r, ["analysis/plot_ood_degradation.py", "--run", _r])
             for _r in MODEL_BASED_RUNS]

# --- Experiment 2: multi-run analyses (these loop over runs internally) ------
COMMANDS += [
    ("metrics/all", ["analysis/plot_experiment2_metrics_modelbased.py"]),
    ("width_ratio_rf/all", ["analysis/plot_experiment2_width_ratio_rf.py"]),
    ("width_tradeoff/all", ["analysis/plot_experiment2_width_tradeoff.py"]),
]

# --- Experiment 2: per-run analyses ------------------------------------------
for _run in MODEL_BASED_RUNS:
    COMMANDS.append(("compute_matched/" + _run,
                     ["analysis/plot_compute_matched_rf.py", "--run", _run]))
    COMMANDS.append(("seed_curves_modelbased/" + _run,
                     ["analysis/plot_modelbased_seed_curves.py", "--run", _run]))
for _run in MODEL_FREE_RUNS:
    COMMANDS.append(("seed_curves_rnnprop/" + _run,
                     ["analysis/plot_rnnprop_seed_curves.py", "--run", _run]))

# --- Experiment 1 and the grad-norm diagnostics ------------------------------
COMMANDS += [
    ("exp1/cost", ["analysis/plot_experiment1_cost.py"]),
    ("exp1/quality", ["analysis/plot_experiment1_quality.py"]),
    ("exp1/gradnorm", ["analysis/plot_experiment1_gradnorm.py"]),
    ("gradnorm/dm_instability", ["analysis/plot_gradnorm_instability.py"]),
    ("gradnorm/lista", ["analysis/plot_gradnorm_lista.py"]),
    ("gradnorm/alista", ["analysis/plot_gradnorm_alista.py"]),
]


def watched_files():
    """Every path these scripts can write, relative to the repo root.

    Figs/ recursively (the PNG-only + Figs/tables/ + Figs/multifigure/
    convention), plus the top level of Benchmarking/ -- several plot scripts
    save "alongside this script" as well as into Figs/, and
    plot_compute_matched_rf.py emits compute_matched_<run>.md there.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(FIGS_DIR):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            out.append(os.path.relpath(os.path.join(dirpath, name), REPO_ROOT))
    for name in os.listdir(BENCH_DIR):
        if name.endswith((".png", ".pdf", ".tex")) or name.startswith("compute_matched_"):
            path = os.path.join(BENCH_DIR, name)
            if os.path.isfile(path):
                out.append(os.path.relpath(path, REPO_ROOT))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_all(only=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    statuses = []
    for label, argv in COMMANDS:
        if only and only not in label:
            continue
        print("--- %s: %s" % (label, " ".join(argv)), flush=True)
        # Retried once: saving a PNG on this Windows host intermittently fails
        # with OSError errno 22 (an antivirus/indexer holding the handle), which
        # has nothing to do with the code under test. Observed twice; a genuine
        # failure reproduces on the retry, a race does not.
        for attempt in (1, 2):
            proc = subprocess.run([sys.executable] + argv, cwd=BENCH_DIR, env=env,
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                break
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
            print("    attempt %d FAILED (%d): %s"
                  % (attempt, proc.returncode, " | ".join(tail)), flush=True)
        statuses.append((label, proc.returncode))
    return statuses


def snapshot(dest, statuses):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    manifest = []
    for rel in watched_files():
        src = os.path.join(REPO_ROOT, rel)
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append("%s  %s" % (sha256(src), rel.replace(os.sep, "/")))
    with open(os.path.join(dest, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(dest, "EXIT_CODES.txt"), "w", encoding="utf-8") as f:
        for label, code in statuses:
            f.write("%-40s %d\n" % (label, code))
    print("\nsnapshot: %d files -> %s" % (len(manifest), dest))


def compare(ref, statuses=None):
    """Diff current outputs against a snapshot.

    `statuses` matters: a script that CRASHED wrote nothing, so its previous
    output is still on disk and hashes equal -- a file-only comparison reports
    a clean pass for a run that produced nothing. Any non-zero exit is a
    failure regardless of what the hashes say.
    """
    ref_manifest = {}
    with open(os.path.join(ref, "MANIFEST.sha256"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split("  ", 1)
            ref_manifest[rel] = digest
    now = {}
    for rel in watched_files():
        now[rel.replace(os.sep, "/")] = sha256(os.path.join(REPO_ROOT, rel))

    changed = sorted(r for r in ref_manifest if r in now and now[r] != ref_manifest[r])
    missing = sorted(r for r in ref_manifest if r not in now)
    added = sorted(r for r in now if r not in ref_manifest)

    for label, items in (("CHANGED", changed), ("MISSING", missing), ("NEW", added)):
        if items:
            print("\n%s (%d):" % (label, len(items)))
            for rel in items:
                print("   ", rel)
    failed = [lbl for lbl, code in (statuses or []) if code != 0]
    if failed:
        print("\n%d command(s) FAILED -- their outputs are stale, so matching "
              "hashes prove nothing:" % len(failed))
        for lbl in failed:
            print("   ", lbl)
        return 1
    if not (changed or missing):
        print("\nOK: %d files match the snapshot (%d new), all %d commands exited 0."
              % (len(ref_manifest), len(added), len(statuses or [])))
        return 0
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print the invocations, run nothing")
    g.add_argument("--snapshot", metavar="DIR", help="run all, then copy outputs into DIR")
    g.add_argument("--compare", metavar="DIR", help="run all, then diff outputs against DIR")
    g.add_argument("--run-only", action="store_true", help="run all, snapshot nothing")
    ap.add_argument("--only", metavar="SUBSTR", help="restrict to labels containing SUBSTR")
    ap.add_argument("--no-run", action="store_true",
                    help="with --snapshot/--compare: use the files already on disk")
    args = ap.parse_args()

    if args.list:
        for label, argv in COMMANDS:
            if not args.only or args.only in label:
                print("%-40s python %s" % (label, " ".join(argv)))
        return 0

    statuses = [] if args.no_run else run_all(args.only)
    if args.snapshot:
        snapshot(args.snapshot, statuses)
        return 0
    if args.compare:
        return compare(args.compare, statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
