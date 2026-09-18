"""Recompute the per-layer LASSO trajectories of every LISTA / ALISTA cell of a
sweep from its checkpoint, replacing the values that ``Model_Base_L2O/train.py``
stored before its 2026-09-17 fix.

The defect (see ``RESULTS_VALIDITY.md``): the script created its
``lasso_layer{i}`` Keras ``Mean`` metrics once and reused them across the
``compile``/``evaluate`` of every test file, so the value stored for the k-th
test file is the running mean over files 1..k. The first file is correct; the
rest are not. ``<file>_final_output.npy`` (``predict()``) and therefore the
four Table-4 recovery metrics were never affected.

No retraining: the checkpoints are intact, and ``unified-lista`` /
``unified-alista`` (``Benchmarking/unified``) load them and evaluate layer by
layer. Every regenerated cell is cross-checked against what IS trustworthy in
the stored files -- ``x_pred`` must match ``<file>_final_output.npy`` and the
stored values must de-accumulate onto the regenerated ones
(``f_k = k*s_k - (k-1)*s_{k-1}``) -- so a wrong checkpoint or config cannot
pass silently.

    python regenerate_modelbased_trajectories.py --run runs/exp2_try3_210x300

Writes, never touching the originals:
  <model_dir>/all_test_metrics_perfile.json     same layout as all_test_metrics.json
  <model_dir>/all_test_metrics_perfile.provenance.json
  <run>/results_modelbased_trajectories_fixed.jsonl
        results.jsonl with final_loss/min_loss/mean_loss/num_points replaced
        for lista/alista rows (+ ``trajectory_source: "regenerated_perfile"``)
  <run>/summary_modelbased_trajectories_fixed.json
        the sweep's summarize() over those rows (summary.json keeps the old values)

Cells evaluated after the train.py fix already hold per-file values; the
script recognises that (``stored_form`` in the provenance json) and simply
re-derives them, so it can be re-run after a sweep resumes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU: never compete with a running sweep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified  # noqa: E402,F401
from core.registry import get_method  # noqa: E402
from core.lasso_metrics import lasso_objective  # noqa: E402
from core.result import summarize_trajectory  # noqa: E402

_METHODS = ("lista", "alista")
_CELL_RE = re.compile(r"checkpoints[\\/]s(?P<seed>\d+)[\\/](?P<width>[^\\/]+)[\\/](?P<method>lista|alista)[\\/]checkpoints$")
# The stored files came from the container's GPU; this replay runs on CPU.
# Float32 op-order differences occasionally flip a soft-threshold / support
# decision for one coordinate of one instance (observed: 1 row of 1280 with
# 32 entries off by up to 3e-2, objective identical to 1e-6). So x_pred is
# compared per ROW, and the objective the two x_pred achieve must agree.
_XPRED_ROW_ATOL = 1e-3      # a row "differs" if any entry moves more than this
_XPRED_MAX_ROW_FRAC = 0.01  # at most 1 % of rows may differ
_OBJ_RTOL = 1e-4


def _sweep_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        c = json.load(f)
    return c.get("config", c)


def _lam(run_dir, seed, cfg):
    if "lam" in cfg:
        return float(cfg["lam"])
    meta = os.path.join(run_dir, "data", "seed{}".format(seed), "metadata.json")
    with open(meta) as f:
        return float(json.load(f)["lam"])


def find_cells(run_dir):
    cells = []
    for base in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "s*", "*", "*", "checkpoints"))):
        m = _CELL_RE.search(base)
        if not m:
            continue
        reps = glob.glob(os.path.join(base, "models", "*", "replicate_1"))
        if len(reps) != 1:
            print("skip {} ({} replicate dirs)".format(base, len(reps)))
            continue
        stored = os.path.join(reps[0], "all_test_metrics.json")
        if not os.path.exists(stored):
            print("skip {} (no all_test_metrics.json -- cell never evaluated)".format(base))
            continue
        cells.append(dict(base=base, model_dir=reps[0], run_name=os.path.basename(os.path.dirname(reps[0])),
                          seed=int(m.group("seed")), width=m.group("width"), method=m.group("method")))
    return cells


def regenerate_cell(run_dir, cell, cfg, force=False):
    out_json = os.path.join(cell["model_dir"], "all_test_metrics_perfile.json")
    if os.path.exists(out_json) and not force:
        with open(out_json) as f:
            return json.load(f), "cached"
    with open(os.path.join(cell["model_dir"], "all_test_metrics.json")) as f:
        stored = json.load(f)
    files = list(stored.keys())
    num_layers = len(glob.glob(os.path.join(cell["model_dir"], "layer_*")))
    n_stored = sum(1 for k in stored[files[0]] if k.startswith("lasso_layer"))
    if num_layers != n_stored:
        raise RuntimeError("{}: {} layer dirs but {} stored layer metrics".format(
            cell["model_dir"], num_layers, n_stored))
    data_dir = os.path.join(run_dir, "data", "seed{}".format(cell["seed"]), cell["width"])
    lam = _lam(run_dir, cell["seed"], cfg)
    config = {
        "method": "unified-" + cell["method"], "problem": "lasso", "seed": cell["seed"],
        "run_name": cell["run_name"],
        "train": {"data_dir": data_dir, "num_layers": num_layers, "lasso_lam": lam, "model_lam": lam,
                  "ss_q_per_layer": float(cfg.get("model_based_ss_q_per_layer", 1.2)),
                  "ss_maxq": float(cfg.get("model_based_ss_maxq", 13.0))},
        "eval": {"data_dir": data_dir, "test_files": files},
    }
    # Evaluate into a temporary directory: nothing is written under runs/
    # until the cross-checks below have passed.
    scratch = tempfile.mkdtemp(prefix="regen_" + cell["run_name"] + "_")
    out = get_method("unified-" + cell["method"]).evaluate(config, cell["base"], scratch)
    with open(out.artifacts["all_test_metrics"]) as f:
        perfile = json.load(f)

    # Cross-checks against the trustworthy parts of the stored files.
    a = np.load(os.path.join(data_dir, "A.npy")).astype(np.float32)
    prev, xpred_report, stored_form = None, {}, {}
    for k, split in enumerate(files, start=1):
        x_stored = np.load(os.path.join(cell["model_dir"], split[:-4] + "_final_output.npy"))
        x_new = np.load(out.artifacts["{}:final_output".format(split)])
        b = np.load(os.path.join(data_dir, split))[:, :a.shape[0]]
        rows_off = int((np.abs(x_stored - x_new).max(axis=1) > _XPRED_ROW_ATOL).sum())
        f_stored = float(lasso_objective(a, x_stored.astype(np.float64), b, lam).mean())
        f_new = float(lasso_objective(a, x_new.astype(np.float64), b, lam).mean())
        if rows_off > _XPRED_MAX_ROW_FRAC * x_stored.shape[0] or abs(f_new - f_stored) > _OBJ_RTOL * abs(f_stored):
            raise RuntimeError("{} {}: x_pred differs from stored final_output in {} of {} rows, mean "
                               "objective {:.6g} vs {:.6g} -- wrong checkpoint/config, refusing to write"
                               .format(cell["run_name"], split, rows_off, x_stored.shape[0], f_new, f_stored))
        xpred_report[split] = {"rows_differing": rows_off, "max_abs_diff": float(np.abs(x_stored - x_new).max()),
                               "mean_objective_stored": f_stored, "mean_objective_regenerated": f_new}
        s_k = np.array([stored[split]["lasso_layer{}".format(i)] for i in range(num_layers)])
        new = np.array([perfile[split]["lasso_layer{}".format(i)] for i in range(num_layers)])
        # Cells evaluated BEFORE the train.py fix stored running means (they
        # de-accumulate onto the regenerated values); cells evaluated after it
        # stored per-file values (they match directly). Anything else means a
        # wrong checkpoint/config.
        deacc = s_k if prev is None else k * s_k - (k - 1) * prev
        if np.allclose(new, s_k, rtol=2e-4, atol=1e-5):
            stored_form[split] = "per_file"
        elif np.allclose(new, deacc, rtol=2e-4 * k, atol=1e-5 * k):
            stored_form[split] = "cumulative"
        else:
            raise RuntimeError("{} {}: stored values neither match (max rel {:.2e}) nor de-accumulate "
                               "onto (max rel {:.2e}) the regenerated trajectory -- wrong checkpoint/config, "
                               "refusing to write".format(
                                   cell["run_name"], split,
                                   float(np.max(np.abs(new - s_k) / np.abs(s_k))),
                                   float(np.max(np.abs(new - deacc) / np.abs(deacc)))))
        prev = s_k

    with open(out_json, "w") as f:
        json.dump(perfile, f, indent=2)
    with open(os.path.join(cell["model_dir"], "all_test_metrics_perfile.provenance.json"), "w") as f:
        json.dump({"written_at": _dt.datetime.now().isoformat(timespec="seconds"),
                   "method": "unified-" + cell["method"], "config": config, "device": "cpu",
                   "checks": "x_pred vs <file>_final_output.npy (per-row, <= {:.0%} rows may differ, objective "
                             "rtol {}), stored de-accumulates onto regenerated (rtol 2e-4*k)".format(
                                 _XPRED_MAX_ROW_FRAC, _OBJ_RTOL),
                   "xpred_check": xpred_report,
                   "stored_form": stored_form,
                   "source_files": glob.glob(os.path.join(cell["model_dir"], "layer_{}".format(num_layers), "*"))},
                  f, indent=2)
    shutil.rmtree(scratch, ignore_errors=True)
    return perfile, "regenerated"


def patch_results(run_dir, cells_perfile):
    src = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(src):
        print("no results.jsonl in {} -- per-cell json written only".format(run_dir))
        return
    dst = os.path.join(run_dir, "results_modelbased_trajectories_fixed.jsonl")
    patched, missing = 0, 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            row = json.loads(line)
            if row.get("method") in _METHODS:
                key = (row["method"], int(row["seed"]), row["width"])
                perfile = cells_perfile.get(key)
                split = "test_sparsity_{:.2f}.npy".format(float(row["test_sparsity"]))
                if perfile is None or split not in perfile:
                    missing += 1
                else:
                    n_layers = sum(1 for k in perfile[split] if k.startswith("lasso_layer"))
                    traj = [perfile[split]["lasso_layer{}".format(i)] for i in range(n_layers)]
                    row.update(summarize_trajectory(traj))
                    row["trajectory_source"] = "regenerated_perfile"
                    patched += 1
            g.write(json.dumps(row) + "\n")
    print("wrote {} ({} model-based rows patched, {} without a regenerated cell)".format(dst, patched, missing))
    # summary.json aggregates the wrong model-based values too; write the
    # corrected aggregate beside it with the sweep's own summarize().
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
    from sweep import summarize  # noqa: E402
    summary_dst = os.path.join(run_dir, "summary_modelbased_trajectories_fixed.json")
    summarize(dst, summary_dst, group_by=["method", "width", "test_sparsity"],
              non_metric_keys=["seed", "method", "width", "test_sparsity", "train_sec", "eval_sec",
                               "cell_sec", "trajectory_source"])
    print("wrote {}".format(summary_dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", action="store_true", help="recompute cells that already have a perfile json")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run)
    cfg = _sweep_config(run_dir)
    cells = find_cells(run_dir)
    print("{}: {} model-based cells".format(run_dir, len(cells)))
    cells_perfile = {}
    for i, cell in enumerate(cells, start=1):
        perfile, how = regenerate_cell(run_dir, cell, cfg, force=args.force)
        cells_perfile[(cell["method"], cell["seed"], cell["width"])] = perfile
        with open(os.path.join(cell["model_dir"], "all_test_metrics.json")) as f:
            stored = json.load(f)
        last = list(perfile.keys())[-1]
        n = sum(1 for k in perfile[last] if k.startswith("lasso_layer")) - 1
        print("[{}/{}] {} {}: last file {} final layer stored {:.4g} -> per-file {:.4g}".format(
            i, len(cells), how, cell["run_name"], last,
            stored[last]["lasso_layer{}".format(n)], perfile[last]["lasso_layer{}".format(n)]), flush=True)
    patch_results(run_dir, cells_perfile)


if __name__ == "__main__":
    main()
