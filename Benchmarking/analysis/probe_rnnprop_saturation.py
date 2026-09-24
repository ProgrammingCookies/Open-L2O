"""
For each seed's trained checkpoint (one width, one test file, the first
`--num_instances` test instances, the shared x0 -- all taken from ONE dataset,
`--data_seed`, so the five networks see identical instances; in the sweep each
seed runs on its own dataset, which is why the sweep's across-seed agreement
is on statistically equivalent problems, not identical ones), it runs the SAME evaluation
loop as evaluate_rnnprop.py (meta_rnnprop_eval.MetaOptimizer.step) and logs,
per learned step k:
  * fraction of coordinates whose normalised inputs |m_tilde|, |g_tilde| > 0.9
  * fraction of coordinates whose pre-tanh activation |z| > 2 (tanh(2)=0.96)
  * fraction of coordinates whose step |dx| > 0.9 * scale (= 0.009)
  * mean |dx| / scale
  * R_f(k)
plus the full per-step dx and x_k arrays, so the ACROSS-SEED agreement of the
steps themselves (not just of f(x_k)) can be computed afterwards:
  * per k, the mean over seed pairs of corr(dx_seed_a, dx_seed_b) over all
    coordinates and instances, and the relative distance ||x_a - x_b|| / ||x_a||.

Nothing in the libraries is modified: the network's `call` is wrapped at
runtime to capture its inputs and pre-tanh output.

Outputs: <out_dir>/probe_<run>_<width>_p<p>.npz (per-seed arrays) and a
markdown summary on stdout. Runs on CPU in a few minutes per seed at
(210,300) x 256 instances x 300 steps. Must run inside the Docker image
(needs TF + dill for the checkpoints); see Benchmarking/README.

Usage (from Benchmarking/, inside the image):
  python analysis/probe_rnnprop_saturation.py --run runs/exp2_try3_210x300 --width narrow \
      --test_sparsity 0.175 --seeds 21 22 23 24 25 --num_steps 300 --num_instances 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Model_Free_L2O", "L2O-DM and L2O-RNNProp")
sys.path.insert(0, os.path.abspath(LIB))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf  # noqa: E402
import meta_rnnprop_eval as meta  # noqa: E402
import util  # noqa: E402
from networks import load as net_load  # noqa: E402
from core.lasso_metrics import _GAP_TOL, _xstar_cache_path, lasso_objective  # noqa: E402
from data.lasso import _test_filename  # noqa: E402


def run_one(run_dir, width, seed, p, num_steps, num_instances, lam, data_seed):
    # The checkpoint is seed's; the DATA is data_seed's. In the sweep every seed
    # is evaluated on its own dataset (different A, b, x0), so an across-seed
    # comparison of the *steps* is only meaningful when all networks run on the
    # same instances -- hence the separate data_seed.
    ckpt_dir = os.path.join(run_dir, "checkpoints", "s{}".format(seed), width, "l2o-rnnprop", "checkpoint")
    data_dir = os.path.join(run_dir, "data", "seed{}".format(data_seed), width)
    split = _test_filename(p)
    problem, net_config, net_assignments = util.get_config(
        "lasso_dataset", None, mode="test", net_name="RNNprop", lasso_data_dir=data_dir,
        lasso_split=split, lasso_batch_size=num_instances, lasso_lam=lam, lasso_x0_mode="aligned")
    x_vars, const_vars, loss_fn = problem()   # first batch = first num_instances rows, shared x0

    optimizer = meta.MetaOptimizer(0.95, 0.95, **net_config)
    optimizer._setup(x_vars, net_assignments)
    # Build + load, exactly as evaluate_rnnprop.py does.
    for k, net in optimizer._nets.items():
        filename = os.path.join(ckpt_dir, "{}.l2l-0".format(k))
        dummy_state = optimizer.initial_state(x_vars)
        x_tmp = [tf.identity(v) for v in x_vars]
        with tf.GradientTape() as tape:
            tape.watch(x_tmp)
            _ = loss_fn(x_tmp)
        grads_tmp = tape.gradient(_, x_tmp)
        optimizer._apply_step(grads_tmp, x_tmp, dummy_state, *optimizer.initial_mt_state(x_vars), 1)
        net_load(net, filename)

    # Instrument: wrap each net's call to capture (m_tilde, g_tilde) and the
    # pre-tanh activation. The wrapped call re-does the tanh*scale itself.
    captured = {}
    for k, net in optimizer._nets.items():
        orig_call = net.call
        scale = net._scale

        def wrapped(m, g, state, _orig=orig_call, _net=net, _scale=scale):
            out, new_state = _orig(m, g, state)
            # recover pre-tanh z from the output: out = tanh(z)*scale
            z = tf.atanh(tf.clip_by_value(out / _scale, -0.999999, 0.999999))
            captured["m"] = m.numpy(); captured["g"] = g.numpy(); captured["z"] = z.numpy()
            return out, new_state
        net.call = wrapped

    state = optimizer.initial_state(x_vars)
    mt, vt = optimizer.initial_mt_state(x_vars)
    x = [tf.identity(v) for v in x_vars]
    t = 0
    a = np.load(os.path.join(data_dir, "A.npy")).astype(np.float64)
    data = np.load(os.path.join(data_dir, split))
    b = data[:num_instances, :a.shape[0]].astype(np.float64)
    x_star = np.load(_xstar_cache_path(os.path.join(run_dir, "xstar_cache"),
                                       a.astype(np.float32), data[:, :a.shape[0]], lam, 50_000, _GAP_TOL))[:num_instances]
    f_star = float(np.mean(lasso_objective(a, x_star, b, lam)))

    rec = {k: [] for k in ("frac_m_sat", "frac_g_sat", "frac_z_sat", "frac_dx_cap", "mean_abs_dx_over_scale", "rf")}
    dxs, xs = [], []
    scale = list(optimizer._nets.values())[0]._scale
    for k in range(num_steps):
        x_prev = x[0].numpy().copy()
        fx, x, state, mt, vt, t = optimizer.step(loss_fn, x, state, mt, vt, t)
        x_now = x[0].numpy()
        dx = x_now - x_prev
        rec["frac_m_sat"].append(float(np.mean(np.abs(captured["m"]) > 0.9)))
        rec["frac_g_sat"].append(float(np.mean(np.abs(captured["g"]) > 0.9)))
        rec["frac_z_sat"].append(float(np.mean(np.abs(captured["z"]) > 2.0)))
        rec["frac_dx_cap"].append(float(np.mean(np.abs(dx) > 0.9 * scale)))
        rec["mean_abs_dx_over_scale"].append(float(np.mean(np.abs(dx)) / scale))
        rec["rf"].append(float(np.mean(lasso_objective(a, x_now.astype(np.float64), b, lam))) / f_star - 1.0)
        dxs.append(dx.astype(np.float32)); xs.append(x_now.astype(np.float32))
    return {k: np.array(v) for k, v in rec.items()}, np.stack(dxs), np.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--width", default="narrow")
    ap.add_argument("--test_sparsity", type=float, default=0.175)
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23, 24, 25])
    ap.add_argument("--num_steps", type=int, default=300)
    ap.add_argument("--num_instances", type=int, default=256)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_seed", type=int, default=None,
                    help="evaluate EVERY seed's checkpoint on this seed's dataset (default: first of --seeds), "
                         "so that step/iterate agreement across seeds is measured on identical instances")
    args = ap.parse_args()
    with open(os.path.join(args.run, "data", "seed{}".format(args.seeds[0]), "metadata.json")) as f:
        lam = float(json.load(f)["lam"])
    out_dir = args.out_dir or os.path.join(args.run, "probe_saturation")
    os.makedirs(out_dir, exist_ok=True)

    per_seed = {}
    for seed in args.seeds:
        print("seed", seed, flush=True)
        rec, dxs, xs = run_one(args.run, args.width, seed, args.test_sparsity, args.num_steps, args.num_instances, lam,
                               args.data_seed if args.data_seed is not None else args.seeds[0])
        per_seed[seed] = (rec, dxs, xs)
    seeds = list(per_seed)

    # across-seed agreement of the STEPS and ITERATES, per k
    n_k = args.num_steps
    corr = np.zeros(n_k); xdist = np.zeros(n_k); npairs = 0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            da, db = per_seed[seeds[i]][1], per_seed[seeds[j]][1]
            xa, xb = per_seed[seeds[i]][2], per_seed[seeds[j]][2]
            for k in range(n_k):
                u, v = da[k].ravel(), db[k].ravel()
                corr[k] += float(np.corrcoef(u, v)[0, 1])
                xdist[k] += float(np.linalg.norm(xa[k] - xb[k]) / np.linalg.norm(xa[k]))
            npairs += 1
    corr /= npairs; xdist /= npairs

    stem = os.path.join(out_dir, "probe_{}_{}_p{:.4f}_data{}".format(os.path.basename(args.run.rstrip("/\\")), args.width, args.test_sparsity,
                                                                  args.data_seed if args.data_seed is not None else args.seeds[0]))
    np.savez(stem + ".npz", seeds=np.array(seeds), step_corr=corr, x_reldist=xdist,
             **{"{}_s{}".format(k, s): per_seed[s][0][k] for s in seeds for k in per_seed[s][0]})
    print("wrote", stem + ".npz")

    ks = [k for k in (1, 2, 5, 10, 16, 25, 50, 75, 100, 150, 200, 250, 300) if k <= n_k]
    print("\nPer-step saturation, mean over seeds (fraction of coordinates x instances):")
    print("| k | \\|m~\\|>0.9 | \\|g~\\|>0.9 | \\|z\\|>2 (tanh sat.) | \\|dx\\|>0.9*scale | mean\\|dx\\|/scale | R_f (median seed) | step corr across seeds | x rel. dist |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in ks:
        i = k - 1
        m = np.mean([per_seed[s][0]["frac_m_sat"][i] for s in seeds])
        g = np.mean([per_seed[s][0]["frac_g_sat"][i] for s in seeds])
        z = np.mean([per_seed[s][0]["frac_z_sat"][i] for s in seeds])
        c = np.mean([per_seed[s][0]["frac_dx_cap"][i] for s in seeds])
        d = np.mean([per_seed[s][0]["mean_abs_dx_over_scale"][i] for s in seeds])
        rf = np.median([per_seed[s][0]["rf"][i] for s in seeds])
        print("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.3g} | {:.3f} | {:.3f} |".format(
            k, m, g, z, c, d, rf, corr[i], xdist[i]))


if __name__ == "__main__":
    main()
