"""
Experiment 2, take 2. (It's a standalone file so if you're trying to understand the benchmarking platform's generality, you can skip it.)

Why this file exists, separate from experiment2.py: the original Experiment 2
config runs model-free L2O (L2O-DM/L2O-RNNProp) at (m,n)=(210,300), sourced
from the DA233X pre-study's Table 3. That scale was never actually the primer
paper's own tested model-free configuration -- Chen et al.'s "Learning to
Optimize: A Primer and a Benchmark" (JMLR 2022), section 4.1.2, states they
"did not go larger [than (25,50)] due to the high memory cost of LSTM-based
model-free L2O methods," and ran the whole LASSO experiment noiseless (not at
40dB SNR). At their own tested (25,50) scale, the paper singles out
L2O-RNNprop specifically as the one model-free method that "can converge
faster than ISTA and comparable to FISTA" -- unlike L2O-DM/L2O-enhanced,
which "perform poorly" there too.

This file re-runs Experiment 2's protocol at that scale/noise setting instead,
defaulting to L2O-RNNprop only (not the full 4-method sweep) so a first pass
can check whether results trend differently before committing to the full
matrix again. Seeds default to 21-30 (not 0-9) -- a fresh, non-overlapping
range from the original exp2_real sweep's seeds.

Usage:
    python experiment2_try2.py --results_dir runs/exp2_try2 --seeds 21 --smoke
    python experiment2_try2.py --results_dir runs/exp2_try2 --seeds 21
    python experiment2_try2.py --results_dir runs/exp2_try2 --seeds 21 22 23 24 25 26 27 28 29 30
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKING_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKING_ROOT)

import adapters
from core.registry import get_method
from data.lasso import DEFAULT_TEST_SPARSITIES, Experiment2Config, generate_experiment2_dataset

MODEL_FREE_METHODS = ("l2o-dm", "l2o-rnnprop")
MODEL_BASED_METHODS = ("lista", "alista")
ALL_METHODS = MODEL_FREE_METHODS + MODEL_BASED_METHODS
WIDTHS = ("narrow", "medium", "wide")

_NON_METRIC_ROW_KEYS = ("method", "width", "seed", "test_sparsity", "train_sec", "eval_sec")


def method_kind(method_name: str) -> str:
    if method_name in MODEL_FREE_METHODS:
        return "model_free"
    if method_name in MODEL_BASED_METHODS:
        return "model_based"
    raise ValueError("unknown method {!r} (expected one of {})".format(method_name, ALL_METHODS))


def _test_filename(p: float) -> str:
    return "test_sparsity_{:.2f}.npy".format(p)


@dataclass
class OrchestratorConfig:
    """Every value below is a deliberate decision -- see experiment2.py's
    OrchestratorConfig and Benchmarking/README.md's "Experiment 2
    hyperparameter decisions" table for the source/confidence behind values
    inherited unchanged from there. Only m/n/snr_db/methods differ from that
    file's defaults; see the module docstring above for why.
    """
    results_dir: str
    seeds: List[int]
    methods: List[str] = field(default_factory=lambda: ["l2o-rnnprop"])
    widths: List[str] = field(default_factory=lambda: list(WIDTHS))

    # Primer & Benchmark section 4.1.2's own tested scale (not Table 3's
    # (210,300), which the paper never actually validated for model-free
    # methods -- see module docstring).
    m: int = 25
    n: int = 50
    lam: float = 0.005
    train_size: int = 32_000
    val_size: int = 1_024
    test_size: int = 1_280
    test_sparsities: List[float] = field(default_factory=lambda: list(DEFAULT_TEST_SPARSITIES))
    # Section 4.1.2: "The samples are noiseless." float('inf') SNR makes
    # data/lasso.py's _add_noise_for_snr add exactly zero noise
    # (noise_power = signal_power / 10**(inf/10) == 0) -- no new code needed,
    # just the right value.
    snr_db: float = float("inf")

    # Model-free (L2O-DM / L2O-RNNProp) training budget. unroll_length=20 is
    # stated directly in both original papers' own (non-LASSO) experiments.
    # num_epochs/num_steps are the Primer & Benchmark's own §4.1.2 LASSO
    # recipe specifically ("trained... for 100 epochs and 1,000 iterations
    # per epoch") -- this is the exact experiment Experiment 2 replicates, so
    # it overrides the generic DM/RNNProp-paper convention (100 total steps)
    # used for other, non-LASSO problems. See README table.
    model_free_num_epochs: int = 100
    model_free_num_steps: int = 1_000
    model_free_unroll_length: int = 20
    # LASSO instances per meta-training step. Not paper-sourced -- LASSO isn't
    # one of either paper's own benchmarked problems, so neither states a
    # batch size for it. Kept equal to model-based's batch size (which *is*
    # paper-grounded) rather than picked independently, so this isn't an
    # arbitrary cross-family asymmetry on a parameter neither paper actually
    # constrains -- see README table, judgment call not a derived value.
    model_free_batch_size: int = 128
    # Primer & Benchmark section 4.1.2: 1000 iterations for model-free L2O at
    # test time (vs model-based's 16 layers -- deliberately not comparable,
    # see README table).
    model_free_eval_num_steps: int = 1000
    # Implementation convention (both papers' training-time LR is unspecified
    # / "chosen by random search"); 1e-3 matches train_dm.py/train_rnnprop.py's
    # own argparse defaults.
    model_free_lr: float = 1e-3
    # Periodic best-checkpoint-selection cadence during meta-training. Forced
    # identical for DM and RNNProp on purpose: the scripts' own defaults
    # disagree (train_dm.py=100, train_rnnprop.py=10), which would hand one
    # method far more chances to catch a lucky validation draw in a benchmark
    # that's explicitly comparing the two. The scripts' own defaults are
    # calibrated for their native (non-LASSO) experiments' much larger epoch
    # counts (e.g. Open-L2O's own recommended train_dm.py --num_epochs=10000);
    # rescaled down for num_epochs=100 so validation still happens more than
    # once -- not paper-stated, a judgment call like the "identical for both"
    # principle itself.
    model_free_evaluation_period: int = 5
    model_free_evaluation_epochs: int = 20
    # RNNProp-only: decay rates for its internal Adam-style input
    # normalization (m-tilde, g-tilde features fed to the LSTM, Lv et al.
    # section 4). Code default, carried forward explicitly rather than left
    # implicit -- NOT independently verified against the RNNProp paper's own
    # stated value (if any) in this audit. Flagged as a real gap, not a
    # confirmed decision. Only applies to l2o-rnnprop; l2o-dm has no such flag.
    model_free_rnnprop_beta1: float = 0.95
    model_free_rnnprop_beta2: float = 0.95
    # RNNProp-only: train against the final unroll step's loss only (w_T=1,
    # w_t=0 otherwise), matching Lv, Jiang & Li 2017's own stated RNNprop
    # training objective ("Learning Gradient Descent: Better Generalization
    # and Longer Horizons", arXiv:1703.03633, Section 5) -- explicitly
    # contrasted there against L2O-DM's own convention (Andrychowicz et al.
    # 2016) of summing the loss over every step. train_rnnprop.py used DM's
    # convention unconditionally for RNNprop too until this flag was added;
    # off by default so prior results (exp2_real, this file's own earlier
    # try2 runs) stay comparable/reproducible under the old behavior.
    model_free_rnnprop_last_step_loss: bool = False
    # Curriculum learning (--if_cl) and imitation learning (--if_mt), Chen et
    # al. 2020 "Training Stronger Baselines for Learning to Optimize"'s two
    # fixes for the truncation-bias instability of a fixed short unroll
    # length -- both already implemented in train_dm.py/train_rnnprop.py
    # (same VITA-Group lineage). Off by default here too -- this file's whole
    # point is testing scale/noise as the variable, not these on top of it.
    model_free_curriculum: bool = False
    model_free_imitation: bool = False
    # Writes a per-epoch (loss, meta-gradient-norm) trajectory to
    # <output_dir>/profile.jsonl (train_dm.py/train_rnnprop.py's own
    # profiling.RunProfiler, already used by experiment1.py).
    model_free_profile: bool = False

    # Model-based (LISTA/ALISTA). 16 layers is stated in both the Primer and
    # the ALISTA repo's own default. base_lr/train_batch_size are the Primer's
    # stated values (5e-4 / 128) made explicit rather than left to
    # train.py's coincidentally-identical flag defaults.
    model_based_num_layers: int = 16
    model_based_base_lr: float = 5e-4
    model_based_train_batch_size: int = 128
    # Per-layer threshold/weight init: theta_init = model_lam / (1.001*||A||_2^2),
    # W_init = A^T / (1.001*||A||_2^2) -- see models/lista.py:66-69 and
    # models/alista.py:62-63. Setting model_lam = lasso_lam makes layer 0
    # exactly one ISTA step on the actual objective (untrained net = ISTA),
    # which is what step_lista's `assert model_lam == lasso_lam` already
    # assumes for that one variant. The train.py default of 0.4 is a leftover
    # from the sparse-coding task (no LASSO lambda in that objective at all),
    # not a tuned value for this experiment -- see model_lam(cfg) below.
    #
    # epochs is a per-stage safety CAP, not the real budget: train.py already
    # runs 3 progressive stages per layer (new-layer-only, then two joint
    # fine-tune stages at 0.2x/0.02x base_lr), each with
    # EarlyStopping(patience=8 epochs) actually deciding stage length
    # (Model_Base_L2O/train.py:317-369). At train_batch_size=128 and 32,000
    # training instances that's 250 steps/epoch, patience=8 epochs=2000
    # steps. Set generously above what patience should ever need so the cap
    # doesn't silently become the real (and method-asymmetric) budget --
    # verified via the training log which mechanism actually fires (see
    # README's model_based_epochs row).
    model_based_epochs: int = 200
    # EarlyStopping patience (epochs) for each of the 3 progressive per-layer
    # stages -- this, not model_based_epochs above, is what actually decides
    # stage length. Raising it is "train to convergence" (see
    # project_exp2_followup_experiments memory, idea #2): if the paper's own
    # patience=8 default is compute-limited rather than a real convergence
    # point, a higher value should show it. model_based_epochs's cap was only
    # verified non-binding *at patience=8* -- bump it alongside any patience
    # increase (see the CLI override below) and re-verify via the training
    # log which mechanism actually fires.
    model_based_patience: int = 8
    # ALISTA support-selection schedule (models/alista.py: q[t] = clip((t+1)*
    # ss_q_per_layer, 0, ss_maxq), used by shrink_ss). Code default, carried
    # forward explicitly -- NOT independently verified against the
    # ALISTA/LISTA-CPSS papers' own stated schedule in this audit.
    model_based_ss_q_per_layer: float = 1.2
    model_based_ss_maxq: float = 13.0
    # Training-curve instrumentation (see project_exp2_followup_experiments
    # memory): if set, evaluate against a fixed test file after every layer
    # finishes training, not just once on the final model, so the trajectory
    # of that fixed-sparsity performance over training is visible. Off by
    # default -- opt-in per the same convention as model_free_profile.
    # training_curve_sparsity must be one of test_sparsities (narrow's own
    # fixed training center, 0.175, by default here).
    model_based_track_training_curve: bool = False
    model_based_training_curve_sparsity: float = 0.175

    # Reference x* solve for the recovery metrics (core/lasso_metrics.py).
    # Plain FISTA at a fixed 2000 iterations (the Primer & Benchmark's own
    # convention) is measurably under-converged at this problem's denser test
    # sparsities -- fista_solve now runs FISTA with adaptive restart and stops
    # on a certified duality-gap bound (1e-10 relative) instead, so this is a
    # safety iteration CAP that should not bind, not the real iteration count.
    num_fista_iters: int = 50_000

    def __post_init__(self):
        for m in self.methods:
            method_kind(m)  # raises on an unknown name
        for w in self.widths:
            if w not in WIDTHS:
                raise ValueError("unknown width {!r} (expected one of {})".format(w, WIDTHS))
        if self.model_based_track_training_curve and not any(
                abs(self.model_based_training_curve_sparsity - p) < 1e-9 for p in self.test_sparsities):
            raise ValueError(
                "model_based_training_curve_sparsity={!r} must be one of test_sparsities "
                "({!r}) -- the training-curve eval reuses the shared test_sparsity_*.npy "
                "file for that value, generated by data/lasso.py alongside the others."
                .format(self.model_based_training_curve_sparsity, self.test_sparsities))


def _build_train_config(method_name: str, cfg: OrchestratorConfig, width_dir: str,
                        seed: int, run_name: str, output_dir: str) -> Dict[str, Any]:
    if method_kind(method_name) == "model_free":
        train_flags: Dict[str, Any] = {
            "lasso_data_dir": width_dir,
            "lasso_lam": cfg.lam,
            "lasso_batch_size": cfg.model_free_batch_size,
            "num_epochs": cfg.model_free_num_epochs,
            "num_steps": cfg.model_free_num_steps,
            "unroll_length": cfg.model_free_unroll_length,
            "learning_rate": cfg.model_free_lr,
            "evaluation_period": cfg.model_free_evaluation_period,
            "evaluation_epochs": cfg.model_free_evaluation_epochs,
        }
        if method_name == "l2o-rnnprop":
            # train_dm.py has no --beta1/--beta2 flags at all (argparse would
            # error on an unrecognized one), so this must stay conditional --
            # unlike model-based's ss_q_per_layer/ss_maxq (absl flags, always
            # defined, harmless no-op for the model that doesn't use them).
            train_flags["beta1"] = cfg.model_free_rnnprop_beta1
            train_flags["beta2"] = cfg.model_free_rnnprop_beta2
            if cfg.model_free_rnnprop_last_step_loss:
                train_flags["last_step_loss"] = True
        if cfg.model_free_curriculum:
            train_flags["if_cl"] = True
        if cfg.model_free_imitation:
            train_flags["if_mt"] = True
        if cfg.model_free_profile:
            train_flags["profile_path"] = os.path.join(output_dir, "profile.jsonl")
        return {
            "method": method_name,
            "problem": "lasso_dataset",
            "seed": seed,
            "run_name": run_name,
            "output_dir": output_dir,
            "train": train_flags,
        }
    return {
        "method": method_name,
        "problem": "lasso",
        "seed": seed,
        "run_name": run_name,
        "output_dir": output_dir,
        "train": {
            "data_dir": width_dir,
            "num_layers": cfg.model_based_num_layers,
            "lasso_lam": cfg.lam,
            "epochs": cfg.model_based_epochs,
            "patience": cfg.model_based_patience,
            "base_lr": cfg.model_based_base_lr,
            "train_batch_size": cfg.model_based_train_batch_size,
            # theta_init = model_lam / L, W_init = A^T / L -- see models/lista.py:66-69,
            # models/alista.py:62-63. model_lam = lasso_lam makes layer 0 exactly
            # one ISTA step on the actual objective.
            "model_lam": cfg.lam,
            # ALISTA-only support-selection schedule (models/alista.py's `q`,
            # used by shrink_ss); harmless no-op for LISTA (absl flags are
            # globally defined, an unused one doesn't error). Code default,
            # carried forward explicitly -- NOT independently verified against
            # the ALISTA/LISTA-CPSS papers' own stated schedule in this audit.
            "ss_q_per_layer": cfg.model_based_ss_q_per_layer,
            "ss_maxq": cfg.model_based_ss_maxq,
            **({
                "track_training_curve": True,
                "training_curve_test_file": _test_filename(cfg.model_based_training_curve_sparsity),
            } if cfg.model_based_track_training_curve else {}),
        },
    }


def _make_row(method_name: str, width: str, seed: int, sparsity: float,
             metrics: Dict[str, Any], train_sec: float, eval_sec: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method_name, "width": width, "seed": seed, "test_sparsity": sparsity,
        "train_sec": train_sec, "eval_sec": eval_sec,
    }
    row.update(metrics)
    return row


def run_one(method_name: str, config: OrchestratorConfig, width_dir: str, seed: int,
           width: str, results_root: str) -> List[Dict[str, Any]]:
    """Train once, evaluate on every test sparsity, return one row per sparsity."""
    method = get_method(method_name)
    run_name = "{}__{}__s{}".format(method_name, width, seed)
    output_dir = os.path.join(results_root, "s{}".format(seed), width, method_name)
    train_config = _build_train_config(method_name, config, width_dir, seed, run_name, output_dir)

    t0 = time.time()
    train_out = method.train(train_config, output_dir)
    train_sec = time.time() - t0

    # x* (calculated with FISTA like done in primer and benchmark too) depends only on (A, b, lam,
    # num_fista_iters) so we can cache it across all test sparsities and all methods.
    xstar_cache_dir = os.path.join(config.results_dir, "xstar_cache")

    rows: List[Dict[str, Any]] = []
    if method_kind(method_name) == "model_free":
        for p in config.test_sparsities:
            eval_config = dict(train_config)
            eval_config["eval"] = {
                "optimizer": "L2L",
                "lasso_data_dir": width_dir,
                "lasso_split": _test_filename(p),
                "lasso_batch_size": config.model_free_batch_size,
                "num_steps": config.model_free_eval_num_steps,
                # Must match train()'s lasso_lam, one for
                # all three, so the metrics x*/f* stay comparable to all.
                "lasso_lam": config.lam,
                "num_fista_iters": config.num_fista_iters,
                "xstar_cache_dir": xstar_cache_dir,
            }
            if method_name == "l2o-rnnprop":
                # Must match train()'s beta1/beta2 -- evaluate_rnnprop.py uses
                # them to reconstruct the optimizer that the saved weights were
                # trained under; a mismatch here would apply the checkpoint
                # through a differently-configured normalization scheme.
                eval_config["eval"]["beta1"] = config.model_free_rnnprop_beta1
                eval_config["eval"]["beta2"] = config.model_free_rnnprop_beta2
            eval_output_dir = os.path.join(output_dir, "eval_p{:.2f}".format(p))
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint, eval_output_dir)
            eval_sec = time.time() - t1
            rows.append(_make_row(method_name, width, seed, p, eval_out.metrics, train_sec, eval_sec))
    else:
        eval_config = dict(train_config)
        eval_config["eval"] = {
            "data_dir": width_dir,
            "test_files": [_test_filename(p) for p in config.test_sparsities],
            "num_fista_iters": config.num_fista_iters,
            "xstar_cache_dir": xstar_cache_dir,
        }
        t1 = time.time()
        eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
        eval_sec = time.time() - t1
        for p in config.test_sparsities:
            metrics = eval_out.per_seed.get(_test_filename(p), {})
            rows.append(_make_row(method_name, width, seed, p, metrics, train_sec, eval_sec))
    return rows


def _load_completed(results_path: str) -> set:
    completed = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                completed.add((d["method"], d["width"], d["seed"]))
    return completed


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _write_config_snapshot(cfg: OrchestratorConfig) -> None:
    """Write config.json once per results_dir, so a summary.json produced
    today is still traceable to the exact OrchestratorConfig/commit that
    produced it after lasso_metrics.py or the config defaults change later.
    Left untouched on a resumed run -- same append-only-results, don't-clobber
    resumability as results.jsonl -- rather than overwritten with whatever
    flags happen to be passed on a later resume.
    """
    path = os.path.join(cfg.results_dir, "config.json")
    if os.path.exists(path):
        return
    snapshot = {
        "config": dataclasses.asdict(cfg),
        "git_commit": _git_commit(),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print("Wrote config snapshot to {}".format(path))


def run_sweep(cfg: OrchestratorConfig) -> None:
    cfg.results_dir = os.path.abspath(cfg.results_dir)
    os.makedirs(cfg.results_dir, exist_ok=True)
    _write_config_snapshot(cfg)
    results_path = os.path.join(cfg.results_dir, "results.jsonl")
    errors_path = os.path.join(cfg.results_dir, "errors.jsonl")
    completed = _load_completed(results_path)

    data_root = os.path.join(cfg.results_dir, "data")
    checkpoints_root = os.path.join(cfg.results_dir, "checkpoints")
    needs_alista_w = "alista" in cfg.methods

    for seed in cfg.seeds:
        data_cfg = Experiment2Config(
            seed=seed, m=cfg.m, n=cfg.n, lam=cfg.lam,
            train_size=cfg.train_size, val_size=cfg.val_size, test_size=cfg.test_size,
            snr_db=cfg.snr_db, test_sparsities=cfg.test_sparsities,
            compute_alista_w=needs_alista_w,
        )
        seed_dir = generate_experiment2_dataset(data_cfg, data_root)
        print("[seed {}] dataset ready: {}".format(seed, seed_dir))

        for width in cfg.widths:
            width_dir = os.path.join(seed_dir, width)
            for method_name in cfg.methods:
                key = (method_name, width, seed)
                if key in completed:
                    print("[skip, already done] {}".format(key))
                    continue
                print("[running] {}".format(key))
                t0 = time.time()
                try:
                    rows = run_one(method_name, cfg, width_dir, seed, width, checkpoints_root)
                except Exception as exc:  # keeps the sweep alive across a bad config/OOM/etc as the run is so long we won't actively supervise the run so
                    # its better to go back afterwards and fix the few runs that had errors.
                    with open(errors_path, "a") as f:
                        f.write(json.dumps({
                            "method": method_name, "width": width, "seed": seed,
                            "error": str(exc), "traceback": traceback.format_exc(),
                        }) + "\n")
                    print("[FAILED] {} : {} (see {})".format(key, exc, errors_path))
                    continue
                with open(results_path, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                completed.add(key)
                print("[done] {} in {:.1f}s".format(key, time.time() - t0))

    summarize(results_path, os.path.join(cfg.results_dir, "summary.json"))


def summarize(results_path: str, summary_path: str) -> None:
    """Aggregates results.jsonl into mean/std per (method, width, test_sparsity)
    across seeds -- Table 4: "Statistics: Mean, Std Dev across 10 runs"."""
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

    metric_keys = sorted({k for row in rows for k in row if k not in _NON_METRIC_ROW_KEYS})
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["width"], row["test_sparsity"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (method_name, width, sparsity), group_rows in sorted(groups.items()):
        entry: Dict[str, Any] = {
            "method": method_name, "width": width, "test_sparsity": sparsity,
            "num_seeds": len(group_rows),
        }
        for k in metric_keys:
            vals = [r[k] for r in group_rows if isinstance(r.get(k), (int, float))]
            if vals:
                entry[k + "_seed_mean"] = float(np.mean(vals))
                entry[k + "_seed_std"] = float(np.std(vals))
        summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary ({} groups) to {}".format(len(summary), summary_path))


def _smoke_config(results_dir: str, seeds: List[int], methods: List[str],
                  widths: List[str]) -> OrchestratorConfig:
    return OrchestratorConfig(
        results_dir=results_dir, seeds=seeds, methods=methods, widths=widths,
        m=20, n=30, train_size=200, val_size=64, test_size=32,
        test_sparsities=[0.1, 0.3], snr_db=float("inf"),
        model_free_num_epochs=3, model_free_num_steps=10, model_free_unroll_length=5,
        model_free_batch_size=16, model_free_eval_num_steps=10,
        model_based_num_layers=2, model_based_epochs=2,
        num_fista_iters=50,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(21, 31)))
    p.add_argument("--methods", nargs="+", default=["l2o-rnnprop"], choices=list(ALL_METHODS))
    p.add_argument("--widths", nargs="+", default=list(WIDTHS), choices=list(WIDTHS))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budgets to sanity-check the whole "
                        "pipeline in well under a minute. NOT a real Experiment-2 run.")
    # Individual overrides (ignored if --smoke; --smoke defines its own tiny config).
    p.add_argument("--model_free_num_epochs", type=int, default=None)
    p.add_argument("--model_free_num_steps", type=int, default=None)
    p.add_argument("--model_free_lr", type=float, default=None)
    p.add_argument("--model_based_num_layers", type=int, default=None)
    p.add_argument("--model_based_epochs", type=int, default=None)
    p.add_argument("--model_based_patience", type=int, default=None)
    p.add_argument("--num_fista_iters", type=int, default=None)
    p.add_argument("--train_size", type=int, default=None,
                   help="Override the number of training instances (default "
                        "32,000, same for all 3 widths). Idea #3 in "
                        "project_exp2_followup_experiments memory: tests "
                        "whether wide's worse own-center performance is a "
                        "sample-density artifact rather than a real "
                        "distribution-width tradeoff.")
    p.add_argument("--model_free_curriculum", action="store_true",
                   help="Enable train_dm.py/train_rnnprop.py's --if_cl "
                        "(curriculum learning on unroll length).")
    p.add_argument("--model_free_imitation", action="store_true",
                   help="Enable train_dm.py/train_rnnprop.py's --if_mt "
                        "(imitation learning from analytical optimizers).")
    p.add_argument("--model_free_profile", action="store_true",
                   help="Write a per-epoch (loss, meta-grad-norm) trajectory "
                        "to <output_dir>/profile.jsonl.")
    p.add_argument("--model_free_rnnprop_last_step_loss", action="store_true",
                   help="RNNProp only: train against the final unroll step's "
                        "loss only (w_T=1, w_t=0 otherwise), matching Lv, "
                        "Jiang & Li 2017's own stated training objective, "
                        "instead of this script's prior default of summing "
                        "the loss over every step (DM's convention).")
    p.add_argument("--model_based_track_training_curve", action="store_true",
                   help="Model-based (LISTA/ALISTA) only: evaluate against "
                        "--model_based_training_curve_sparsity's test file "
                        "after every layer finishes training, not just once "
                        "on the final model, and log the trajectory to "
                        "<model_dir>/training_curve/training_curve.jsonl.")
    p.add_argument("--model_based_training_curve_sparsity", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        cfg = _smoke_config(args.results_dir, args.seeds, args.methods, args.widths)
    else:
        cfg = OrchestratorConfig(results_dir=args.results_dir, seeds=args.seeds,
                                 methods=args.methods, widths=args.widths)
        for field_name in ("model_free_num_epochs", "model_free_num_steps", "model_free_lr",
                           "model_based_num_layers", "model_based_epochs", "model_based_patience",
                           "num_fista_iters", "model_based_training_curve_sparsity", "train_size"):
            value = getattr(args, field_name)
            if value is not None:
                setattr(cfg, field_name, value)
        if args.model_free_curriculum:
            cfg.model_free_curriculum = True
        if args.model_free_imitation:
            cfg.model_free_imitation = True
        if args.model_free_profile:
            cfg.model_free_profile = True
        if args.model_free_rnnprop_last_step_loss:
            cfg.model_free_rnnprop_last_step_loss = True
        if args.model_based_track_training_curve:
            cfg.model_based_track_training_curve = True
    run_sweep(cfg)


if __name__ == "__main__":
    main()
