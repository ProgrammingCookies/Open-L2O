"""
Classical ISTA / FISTA baselines for Experiment 2, scored with the exact
Table-4 metrics on the exact per-seed test files the L2O methods were scored on.

Why this exists: Table 3 lists only the four L2O methods, but the Primer's
own LASSO experiment (Chen et al. 2022, S4.1.2) reports every learned method
against ISTA and FISTA at a matched iteration count, and "is LISTA at 16
layers better than FISTA at 16 iterations?" is the first question a reader
asks of any Experiment-2 table. Both solvers are CPU-only and take seconds
per test file, so this is an analysis-time addition, not a sweep.

Protocol (chosen to match the L2O evaluations exactly):
  * starts from the seeded, shared x0 sibling file (`<test>_x0.npy`), the
    same starting point every L2O method's trajectory begins from (see
    README's "x0 alignment" decision) -- not from zero;
  * fixed step 1/L with L = lambda_max(A^T A), threshold lam/L, i.e. the
    textbook ISTA of pre-study Eq. 3 and the textbook (Beck-Teboulle) FISTA
    with the t_k momentum sequence -- deliberately NOT the adaptive-restart /
    certified-gap FISTA in `core/lasso_metrics.fista_solve`, which is a
    reference solver for x*, not a fair fixed-budget baseline;
  * iteration k means k proximal-gradient steps applied to x0, the same
    convention as the model-free eval's `loss_record[k]` (= f(x_k), where
    x_k is the iterate after k learned steps) and as a k-layer LISTA/ALISTA;
  * every metric goes through `core.lasso_metrics.compute_lasso_metrics`
    with the run's own `xstar_cache`, so f* / x* are byte-identical to what
    the L2O rows used.

Outputs, under `<run_dir>/`:
  classical_baselines.jsonl   one row per (method, iters, seed, test_sparsity),
                              same metric columns as results.jsonl, with
                              width="all" (the baselines do not depend on the
                              training width -- test files are shared across
                              widths within a seed, asserted below).
  classical_trajectories/     `<method>_seed<N>_p<p>.npy`: the mean-over-
                              instances objective f(x_k) for k = 0..max_iters,
                              so the compute-matched analysis can overlay the
                              classical convergence curves.

Usage:
  python analysis/classical_baselines.py --run runs/exp2_try2
  python analysis/classical_baselines.py --run runs/exp2_real --seeds 0 1 2
  python analysis/classical_baselines.py --run runs/exp2_try3_210x300 --iters 16 100 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.lasso_metrics import compute_lasso_metrics, lasso_objective, soft_threshold  # noqa: E402
from data.lasso import _test_filename, _x0_filename  # noqa: E402


def _seeds_in_run(run_dir: str) -> List[int]:
    data_dir = os.path.join(run_dir, "data")
    seeds = []
    for name in os.listdir(data_dir):
        if name.startswith("seed") and os.path.isdir(os.path.join(data_dir, name)):
            seeds.append(int(name[len("seed"):]))
    return sorted(seeds)


def _load_seed(run_dir: str, seed: int) -> Tuple[np.ndarray, dict, str]:
    """Returns (A, metadata, width_dir). Test files are shared across the
    three widths (same A, same instances, same x0); read them from the
    first width dir and assert the others agree."""
    seed_dir = os.path.join(run_dir, "data", "seed{}".format(seed))
    with open(os.path.join(seed_dir, "metadata.json")) as f:
        meta = json.load(f)
    widths = sorted(w for w in os.listdir(seed_dir)
                    if os.path.isdir(os.path.join(seed_dir, w)))
    ref = os.path.join(seed_dir, widths[0])
    a = np.load(os.path.join(ref, "A.npy"))
    for w in widths[1:]:
        other = os.path.join(seed_dir, w)
        assert np.array_equal(a, np.load(os.path.join(other, "A.npy"))), \
            "A.npy differs between widths for seed {}".format(seed)
        for p in meta["test_sparsities"]:
            fn = _test_filename(p)
            assert np.array_equal(np.load(os.path.join(ref, fn)),
                                  np.load(os.path.join(other, fn))), \
                "{} differs between widths for seed {}".format(fn, seed)
    return a, meta, ref


def run_classical(a: np.ndarray, b: np.ndarray, x0: np.ndarray, lam: float,
                  max_iters: int, method: str,
                  snapshot_at: List[int]) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Plain ISTA / FISTA from x0 for max_iters steps.

    Returns (mean_objective[k] for k = 0..max_iters, {k: x_k for k in snapshot_at}).
    """
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    l = float(np.linalg.eigvalsh(a64.T @ a64)[-1])
    eta = 1.0 / l
    thresh = lam * eta

    x = x0.astype(np.float64).copy()
    y = x.copy()
    t = 1.0
    traj = np.empty(max_iters + 1, dtype=np.float64)
    traj[0] = float(np.mean(lasso_objective(a64, x, b64, lam)))
    snaps: Dict[int, np.ndarray] = {}
    if 0 in snapshot_at:
        snaps[0] = x.copy()

    for k in range(1, max_iters + 1):
        if method == "ista":
            x = soft_threshold(x - eta * ((x @ a64.T - b64) @ a64), thresh)
        elif method == "fista":
            x_new = soft_threshold(y - eta * ((y @ a64.T - b64) @ a64), thresh)
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new
        else:
            raise ValueError(method)
        traj[k] = float(np.mean(lasso_objective(a64, x, b64, lam)))
        if k in snapshot_at:
            snaps[k] = x.copy()
    return traj, snaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir, e.g. runs/exp2_try2")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="default: every seed with a data/seed<N> dir")
    ap.add_argument("--iters", type=int, nargs="+", default=[16, 100, 1000],
                    help="iteration counts to score with the full Table-4 metrics; "
                         "the trajectory is always recorded up to max(iters)")
    ap.add_argument("--methods", nargs="+", default=["ista", "fista"])
    ap.add_argument("--num_fista_iters", type=int, default=50_000,
                    help="x* reference solve budget; must match the run's "
                         "xstar_cache keys (50,000 for every Experiment-2 run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute rows that already exist in classical_baselines.jsonl")
    args = ap.parse_args()

    run_dir = args.run
    seeds = args.seeds if args.seeds else _seeds_in_run(run_dir)
    out_path = os.path.join(run_dir, "classical_baselines.jsonl")
    traj_dir = os.path.join(run_dir, "classical_trajectories")
    cache_dir = os.path.join(run_dir, "xstar_cache")
    os.makedirs(traj_dir, exist_ok=True)
    max_iters = max(args.iters)

    done = set()
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["iters"], r["seed"], r["test_sparsity"]))

    t0 = time.time()
    n_rows = 0
    with open(out_path, "a") as out:
        for seed in seeds:
            a, meta, width_dir = _load_seed(run_dir, seed)
            lam = float(meta["lam"])
            m = int(meta["m"])
            for p in meta["test_sparsities"]:
                fn = _test_filename(p)
                data = np.load(os.path.join(width_dir, fn))
                x0 = np.load(os.path.join(width_dir, _x0_filename(fn)))
                b, x_true = data[:, :m], data[:, m:]
                for method in args.methods:
                    todo = [k for k in args.iters if (method, k, seed, p) not in done]
                    if not todo:
                        continue
                    traj, snaps = run_classical(a, b, x0, lam, max_iters, method, todo)
                    np.save(os.path.join(traj_dir, "{}_seed{}_p{:.4f}.npy".format(method, seed, p)), traj)
                    for k in todo:
                        met = compute_lasso_metrics(a, b, x_true, snaps[k], lam,
                                                    num_fista_iters=args.num_fista_iters,
                                                    xstar_cache_dir=cache_dir)
                        row = {"method": method, "iters": k, "seed": seed, "width": "all",
                               "test_sparsity": p, "final_loss": float(traj[k]),
                               "min_loss": float(np.min(traj[:k + 1])),
                               "num_points": int(k + 1)}
                        row.update(met.to_dict())
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        n_rows += 1
                print("[seed {} p={:.4f}] {} done, {:.0f}s elapsed".format(
                    seed, p, "/".join(args.methods), time.time() - t0), flush=True)
    print("wrote {} rows to {}".format(n_rows, out_path))


if __name__ == "__main__":
    main()
