"""Re-evaluate every L2O-Scale cell of a sweep with the deterministic in-process
implementation, replacing evaluations made before the 2026-09-18 fix of
``L2O-Scale/evaluate.py``.

The defect (see ``RESULTS_VALIDITY.md``, 2026-09-18): the script's per-chunk
dummy step left an optimizer slot keyed by the dummy variable's ``id()``; when
CPython reused that address for the real variable, the chunk silently started
from the dummy's post-step state. Which chunks were affected differed from
process to process, so pre-fix ``l2o-scale`` rows carry an unseeded random
component. Training is unaffected: the checkpoints are valid.

No retraining. ``unified-scale`` (``Benchmarking/unified``) evaluates each
cell's checkpoint exactly as the FIXED script would (bit-identical, see
``tests/test_unified_parity.py``), on CPU, with the flags recorded in the
cell's own ``eval_p*/eval.log`` command line. Each re-evaluation is
cross-checked against the stored pre-fix artifacts: same instances
(fingerprint), and every chunk's stored ``x_pred`` rows either identical to the
regenerated ones ("fresh" chunks) or not (the "inherited-slot" chunks the fix
removes) -- the count of the latter is reported per cell.

    python regenerate_scale_evaluations.py --run runs/exp2_try2

Writes, never touching the originals:
  <cell>/eval_p<p>_reeval/eval/     the fixed script's artifact layout (pickle + npz)
  <cell>/reeval.provenance.json
  <run>/results_scale_reeval.jsonl  results.jsonl with the l2o-scale rows' metrics replaced
                                    (+ ``eval_source: "unified-scale post-fix"``)
  <run>/summary_scale_reeval.json   the sweep's summarize() over those rows
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import pickle
import re
import shlex
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU: never compete with a running sweep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402,F401
import unified  # noqa: E402,F401
from core.fingerprint import METRIC_KEY, fingerprints_agree  # noqa: E402
from core.registry import get_method  # noqa: E402

_CELL_RE = re.compile(r"checkpoints[\\/]s(?P<seed>\d+)[\\/](?P<width>[^\\/]+)[\\/]l2o-scale$")
_EVAL_DIR_RE = re.compile(r"eval_p(?P<p>[0-9.]+)$")
_FLAG_KEYS = ("num_testing_itrs", "lasso_split", "lasso_batch_size", "lasso_lam", "lasso_x0_mode",
              "min_lr", "max_lr", "restore_model_name", "test_optimizer")


def _flags_from_eval_log(path):
    """The exact evaluate.py flags the cell was evaluated with (``# cmd:`` line)."""
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("# cmd:"):
                argv = shlex.split(line[len("# cmd:"):].strip(), posix=True)
                flags, i = {}, 0
                while i < len(argv):
                    tok = argv[i]
                    if tok.startswith("--"):
                        key = tok[2:]
                        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                            flags[key] = argv[i + 1]
                            i += 2
                        else:
                            flags[key] = True
                            i += 1
                    else:
                        i += 1
                return flags
    raise RuntimeError("no '# cmd:' line in {}".format(path))


def find_cells(run_dir):
    cells = []
    for cell in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "s*", "*", "l2o-scale"))):
        m = _CELL_RE.search(cell)
        if not m:
            continue
        evals = sorted(d for d in glob.glob(os.path.join(cell, "eval_p*")) if _EVAL_DIR_RE.search(d))
        if not evals:
            print("skip {} (no eval_p* dirs -- cell never evaluated)".format(cell))
            continue
        cells.append(dict(cell=cell, seed=int(m.group("seed")), width=m.group("width"), evals=evals))
    return cells


def _chunks_differing(x_stored, x_new, batch_size):
    rows = x_stored.shape[0]
    return [i for i, start in enumerate(range(0, rows, batch_size))
            if not np.array_equal(x_stored[start:start + batch_size], x_new[start:start + batch_size])]


def reevaluate_cell(run_dir, cell, cfg, force=False):
    prov_path = os.path.join(cell["cell"], "reeval.provenance.json")
    if os.path.exists(prov_path) and not force:
        with open(prov_path) as f:
            return json.load(f)["rows"], "cached"
    rows, report = {}, {}
    for eval_dir in cell["evals"]:
        p = float(_EVAL_DIR_RE.search(eval_dir).group("p"))
        flags = _flags_from_eval_log(os.path.join(eval_dir, "eval.log"))
        if flags.get("test_optimizer", "L2o") != "L2o":
            raise RuntimeError("{}: only L2o evaluations are re-generated (got {})".format(eval_dir, flags.get("test_optimizer")))
        ckpt = os.path.join(cell["cell"], "checkpoints", "HierarchicalRNN_GRUCell_20_2",
                            flags.get("restore_model_name", "model-final.l2o"))
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        data_dir = os.path.join(run_dir, "data", "seed{}".format(cell["seed"]), cell["width"])
        ev = {"test_optimizer": "L2o", "num_testing_itrs": int(flags["num_testing_itrs"]),
              "lasso_data_dir": data_dir, "lasso_split": flags["lasso_split"],
              "lasso_batch_size": int(flags.get("lasso_batch_size", 128)),
              "lasso_lam": float(flags.get("lasso_lam", 0.005)),
              "lasso_x0_mode": flags.get("lasso_x0_mode", "aligned"),
              "min_lr": float(flags.get("min_lr", 1e-6)), "max_lr": float(flags.get("max_lr", 1e-2)),
              "num_fista_iters": int(cfg.get("num_fista_iters", 50_000)),
              "xstar_cache_dir": os.path.join(run_dir, "xstar_cache")}
        config = {"method": "unified-scale", "problem": "lasso_dataset", "seed": cell["seed"], "eval": ev}
        out_dir = eval_dir + "_reeval"
        t0 = time.time()
        out = get_method("unified-scale").evaluate(config, ckpt, out_dir)
        eval_sec = time.time() - t0

        # cross-checks against the stored pre-fix artifacts
        stored_dir = os.path.join(eval_dir, "eval")
        stored_npz = np.load(os.path.join(stored_dir, "L2o_recovery-lasso_dataset.npz"))
        new_npz = np.load(out.artifacts["recovery"])
        if not (np.array_equal(stored_npz["b"], new_npz["b"]) and np.array_equal(stored_npz["x_true"], new_npz["x_true"])):
            raise RuntimeError("{}: re-evaluation walked different instances than the stored one".format(eval_dir))
        with open(os.path.join(stored_dir, "seed{}_eval_loss_record.pickle-benchmark".format(cell["seed"])), "rb") as f:
            stored_traj = np.asarray(pickle.load(f), np.float64)
        differing = _chunks_differing(stored_npz["x_pred"], new_npz["x_pred"], ev["lasso_batch_size"])
        report[str(p)] = {
            "eval_sec": eval_sec, "chunks_differing_from_prefix_eval": differing,
            "num_chunks": -(-stored_npz["b"].shape[0] // ev["lasso_batch_size"]),
            "stored_final_loss": float(stored_traj[-1]), "reeval_final_loss": float(out.trajectory[-1]),
            "stored_vs_reeval_trajectory_max_abs_diff": float(np.abs(stored_traj - np.asarray(out.trajectory)).max()),
        }
        rows[str(p)] = {"eval_sec": eval_sec, **out.metrics}
        print("  p={:.4f}: {:.0f}s, {}/{} chunks differed from the pre-fix eval, final_loss {:.5g} -> {:.5g}".format(
            p, eval_sec, len(differing), report[str(p)]["num_chunks"], stored_traj[-1], out.trajectory[-1]), flush=True)

    with open(prov_path, "w") as f:
        json.dump({"written_at": _dt.datetime.now().isoformat(timespec="seconds"), "method": "unified-scale",
                   "device": "cpu", "checkpoint": "restore_model_name from each eval_p*/eval.log",
                   "note": "pre-fix L2O-Scale/evaluate.py evaluations; see RESULTS_VALIDITY.md 2026-09-18",
                   "report": report, "rows": rows}, f, indent=2)
    return rows, "regenerated"


def patch_results(run_dir, cell_rows):
    src = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(src):
        print("no results.jsonl in {} -- per-cell provenance written only".format(run_dir))
        return
    dst = os.path.join(run_dir, "results_scale_reeval.jsonl")
    patched, missing = 0, 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            row = json.loads(line)
            if row.get("method") == "l2o-scale":
                key = (int(row["seed"]), row["width"])
                new = cell_rows.get(key, {}).get(str(float(row["test_sparsity"])))
                if new is None:
                    missing += 1
                else:
                    if METRIC_KEY in row and not fingerprints_agree(row[METRIC_KEY], new[METRIC_KEY]):
                        raise RuntimeError("{}: fingerprint mismatch for seed {} {} p={}".format(
                            src, row["seed"], row["width"], row["test_sparsity"]))
                    row.update(new)
                    row["eval_source"] = "unified-scale post-fix"
                    patched += 1
            g.write(json.dumps(row) + "\n")
    print("wrote {} ({} l2o-scale rows re-evaluated, {} without a re-evaluation)".format(dst, patched, missing))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
    from sweep import summarize  # noqa: E402
    summary_dst = os.path.join(run_dir, "summary_scale_reeval.json")
    summarize(dst, summary_dst, group_by=["method", "width", "test_sparsity"],
              non_metric_keys=["seed", "method", "width", "test_sparsity", "train_sec", "eval_sec",
                               "cell_sec", "trajectory_source", "eval_source"])
    print("wrote {}".format(summary_dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", action="store_true", help="re-evaluate cells that already have reeval.provenance.json")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run)
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        cfg = c.get("config", c)
    cells = find_cells(run_dir)
    print("{}: {} l2o-scale cells with evaluations".format(run_dir, len(cells)))
    cell_rows = {}
    for i, cell in enumerate(cells, start=1):
        print("[{}/{}] seed {} {} ({} evaluations)".format(i, len(cells), cell["seed"], cell["width"], len(cell["evals"])), flush=True)
        rows, how = reevaluate_cell(run_dir, cell, cfg, force=args.force)
        print("   {}".format(how))
        cell_rows[(cell["seed"], cell["width"])] = rows
    patch_results(run_dir, cell_rows)


if __name__ == "__main__":
    main()
