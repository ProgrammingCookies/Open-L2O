"""
Experiment 2, take 2, ported onto ``experiments/sweep.py``.

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

The re-centered width scheme this file introduced (narrow p=0.175 fixed, medium
[0.1125, 0.2375], wide [0.05, 0.30] -- all three sharing centre 0.175, unlike
experiment2.py's, whose centres sit 0.075 apart and so confound width with
centre) is worth running at the pre-study's own Table-3 dimensions too. Pass
--m 210 --n 300 for that; see --n's help text for why n=50 is a poor host for a
width experiment. That combination -- real dims, noiseless, centred widths --
is the deconfounded version of Experiment 2.

PORTING NOTE. This is a 1:1 port of the pre-sweep original ``experiments/experiment2_try2.py``: same
hyperparameters, same flag translation, same metrics, and a ``--smoke`` run
produces the same result rows. Only the plumbing moved to ``sweep.py``.
The shared *vocabulary* (method names, widths, ``method_kind``,
``test_filename``) is imported from the ported ``experiment2.py`` rather than
re-declared, since it is bookkeeping rather than an experimental decision. The
config dataclass, ``_build_train_config`` and ``run_cell`` stay local and
complete: they genuinely differ from experiment2.py's (patience, the
training-curve hooks, the re-centred widths' justifications), and merging them
would mean either changing experiment2.py's behaviour or carrying dead
conditionals. Two experiments holding the same value for different reasons are
not duplication -- the reasons are the part worth keeping.

Usage:
    python experiments/thesis/experiment2_try2.py --results_dir runs/exp2_try2 --seeds 21 --smoke
    python experiments/thesis/experiment2_try2.py --results_dir runs/exp2_try2 --seeds 21
    python experiments/thesis/experiment2_try2.py --results_dir runs/exp2_try2 \
        --seeds 21 22 23 24 25 26 27 28 29 30
    python experiments/thesis/experiment2_try2.py --results_dir runs/exp2_try3_210x300 \
        --m 210 --n 300 --methods l2o-dm l2o-rnnprop lista alista \
        --seeds 41 42 43 44 45 46 47 48 49 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

_THESIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_THESIS_DIR)
_BENCHMARKING_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
# Order matters and must be enforced, not assumed. Python puts the script's own
# directory (thesis/) on sys.path automatically, but a plain insert(0, ...) of
# the others pushes it back, and `import experiment2` would then resolve
# against experiments/ or Benchmarking/ rather than this directory. Removing
# each path before re-inserting makes the final order exactly thesis/ >
# experiments/ > Benchmarking/, whatever was there before. (Until 2026-09-24
# this also guarded against same-named pre-port originals one level up; those
# are gone, but the ordering guarantee is still the thing being asserted.)
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _THESIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import adapters  # noqa: F401,E402  (import side effect: registers all methods)
from core.registry import get_method  # noqa: E402
from data.lasso import (  # noqa: E402
    DEFAULT_TEST_SPARSITIES, Experiment2Config, generate_experiment2_dataset)

from sweep import Axis, SweepSpec, add_overrides, apply_overrides, run_sweep  # noqa: E402
from experiment2 import (  # noqa: E402
    ALL_METHODS, WIDTHS, make_dataset, method_kind, test_filename)


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

    # L2O-Scale (Wichrowska et al. 2017) -- the third model-free arm, added
    # 2026-09-17. Its train.py/evaluate.py CLI differs from DM/RNNProp's, so
    # the shared model_free_* budget above is translated in
    # _build_train_config: num_epochs -> scale_num_problems (each sampled
    # problem draws a fresh training batch, exactly one per epoch like
    # train_dm.py's per-epoch problem() refresh), num_steps/unroll_length ->
    # --fix_num_steps/--fix_unroll_length, eval_num_steps -> --num_testing_itrs.
    # Architecture = experiment1.py's (the adapter has no default of its own).
    scale_optimizer: str = "HierarchicalRNN"
    scale_cell_cls: str = "GRUCell"
    scale_cell_size: int = 20
    scale_num_cells: int = 2
    # Meta-iterations per sampled problem. 1 => every meta-update sees a
    # fresh batch (DM/RNNProp parity: 100 epochs x 50 unrolls over 12,800
    # distinct instances); Experiment 1 used num_problems=1 x 50 iterations,
    # i.e. ONE fixed batch throughout, which together with the lr prior below
    # is why L2O-Scale never solved LASSO there.
    scale_num_meta_iterations: int = 1
    # Internal-validation cadence, in meta-iterations, for model-best.l2o
    # selection: with 1 meta-iteration per problem it must be 1 (evaluation
    # fires on (k+1) % period == 0 inside the per-problem loop), i.e. once per
    # problem with a single 1000-step restart -- 100 x 1 checks vs DM's
    # 20 x 20; same selection rule (lowest validation cost), coarser sample.
    scale_evaluation_period: int = 1
    scale_evaluation_epochs: int = 1
    # Initial per-coordinate learning-rate prior: log-uniform on
    # [min_lr, max_lr] (hierarchical_rnn.py's init_lr_range). The library
    # default (1e-6, 1e-2) has log-centre 1e-4, ~1000x below LASSO's 1/L ~ 0.2
    # at (25,50); the optimizer must then learn 3 decades of lr growth and
    # never does within budget (Exp 1; and the 2026-09-16 pilot arm A: R_f
    # 4.2, cost bouncing 0.2-4 vs f* 0.036). Pilot-derived, NOT paper-sourced:
    # (1e-3, 1e-1) reached R_f 0.075 / cost 0.038 with stable meta-gradients;
    # (1e-2, 1.0) was noisier (R_f 0.39, 10x larger meta-gradient norms).
    # Must be identical at train and eval (state is re-initialised at eval).
    scale_min_lr: float = 1e-3
    scale_max_lr: float = 1e-1
    # "best" = model-best.l2o (lowest internal-validation cost), the analogue
    # of train_dm.py's best-internal-eval `.l2l-0`. Same which-problem-was-best
    # caveat as DM: read `evaluation k, cost=` in train.log before comparing.
    scale_select_checkpoint: str = "best"
    # RMSProp meta-optimizer learning rate (train.py --meta_learning_rate).
    # Until 2026-09-18 this was never set here, so train.py's library default
    # 1e-6 applied silently (DM/RNNProp use Adam at 1e-3, see model_free_lr).
    # At this budget (100 problems x 50 unrolls = 5,000 meta-updates) 1e-6
    # moves each weight by at most ~5e-3: the s21 narrow/medium checkpoints
    # ended within 0.3-0.4 % (relative L2) of their initialisation and their
    # held-out numbers within 0-4 % of the UNTRAINED optimizer's, i.e. those
    # rows measure the HierarchicalRNN prior, not a meta-trained L2O-Scale
    # (additional_context/experiments_analysis.md, 2026-09-18 Scale entries).
    # Pilot 2026-09-18 (seed 21, narrow data, same batch sequence, paired
    # against 1e-6): 1e-5 -3 % in-sample cost, stable; 1e-4 -10 % by problem
    # 15 then drifts to +40 % with collapsing meta-gradients; 1e-3 thrashes
    # (+19 %, stopped at problem 51). The default below is the library value
    # so existing rows stay reproducible; override per the pilot's outcome.
    scale_meta_learning_rate: float = 1e-6

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
    if method_name == "l2o-scale":
        train_flags: Dict[str, Any] = {
            "optimizer": cfg.scale_optimizer,
            "cell_cls": cfg.scale_cell_cls,
            "cell_size": cfg.scale_cell_size,
            "num_cells": cfg.scale_num_cells,
            "lasso_data_dir": width_dir,
            "lasso_split": "train_data.npy",
            "lasso_batch_size": cfg.model_free_batch_size,
            "lasso_lam": cfg.lam,
            "num_problems": cfg.model_free_num_epochs,
            "num_meta_iterations": cfg.scale_num_meta_iterations,
            "fix_unroll": True,
            "fix_unroll_length": cfg.model_free_unroll_length,
            "fix_num_steps": cfg.model_free_num_steps,
            "fix_num_steps_eval": cfg.model_free_num_steps,
            "evaluation_period": cfg.scale_evaluation_period,
            "evaluation_epochs": cfg.scale_evaluation_epochs,
            # Only model-best/model-final are wanted; the per-iteration
            # model-iter{k}.l2o files would be 100 x ~1 MB of dead weight.
            "save_period": 10 ** 9,
            "min_lr": cfg.scale_min_lr,
            "max_lr": cfg.scale_max_lr,
            "select_checkpoint": cfg.scale_select_checkpoint,
            "meta_learning_rate": cfg.scale_meta_learning_rate,
        }
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
                "training_curve_test_file": test_filename(cfg.model_based_training_curve_sparsity),
            } if cfg.model_based_track_training_curve else {}),
        },
    }


def run_cell(cfg: OrchestratorConfig, cell: Dict[str, Any],
             context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Train once, evaluate on every test sparsity, return one row per sparsity.

    method/width/seed are stamped into the rows by the sweep driver, so they are
    deliberately absent here.
    """
    method_name, width, seed = cell["method"], cell["width"], cell["seed"]
    width_dir = os.path.join(context["seed_dir"], width)
    method = get_method(method_name)
    run_name = "{}__{}__s{}".format(method_name, width, seed)
    output_dir = os.path.join(context["checkpoints_dir"], "s{}".format(seed), width, method_name)
    train_config = _build_train_config(method_name, cfg, width_dir, seed, run_name, output_dir)

    t0 = time.time()
    train_out = method.train(train_config, output_dir)
    train_sec = time.time() - t0

    # x* (calculated with FISTA like done in primer and benchmark too) depends only on (A, b, lam,
    # num_fista_iters) so we can cache it across all test sparsities and all methods.
    xstar_cache_dir = os.path.join(context["results_dir"], "xstar_cache")

    rows: List[Dict[str, Any]] = []
    if method_kind(method_name) == "model_free":
        for p in cfg.test_sparsities:
            eval_config = dict(train_config)
            if method_name == "l2o-scale":
                eval_config["eval"] = {
                    "test_optimizer": "L2o",
                    "num_testing_itrs": cfg.model_free_eval_num_steps,
                    "lasso_data_dir": width_dir,
                    "lasso_split": test_filename(p),
                    # Same chunk size as training on purpose: L2O-Scale's
                    # tensor-level RNN and output scaling pool over the whole
                    # (batch, n) tensor, so this is part of the optimizer's
                    # input distribution, not just an efficiency knob.
                    "lasso_batch_size": cfg.model_free_batch_size,
                    "lasso_lam": cfg.lam,
                    "lasso_x0_mode": "aligned",
                    # Must match train() -- the per-coordinate lr state is
                    # re-initialised from this prior at eval time.
                    "min_lr": cfg.scale_min_lr,
                    "max_lr": cfg.scale_max_lr,
                    "num_fista_iters": cfg.num_fista_iters,
                    "xstar_cache_dir": xstar_cache_dir,
                }
            else:
                eval_config["eval"] = {
                    "optimizer": "L2L",
                    "lasso_data_dir": width_dir,
                    "lasso_split": test_filename(p),
                    "lasso_batch_size": cfg.model_free_batch_size,
                    "num_steps": cfg.model_free_eval_num_steps,
                    # Must match train()'s lasso_lam, one for
                    # all three, so the metrics x*/f* stay comparable to all.
                    "lasso_lam": cfg.lam,
                    "num_fista_iters": cfg.num_fista_iters,
                    "xstar_cache_dir": xstar_cache_dir,
                }
            if method_name == "l2o-rnnprop":
                # Must match train()'s beta1/beta2 -- evaluate_rnnprop.py uses
                # them to reconstruct the optimizer that the saved weights were
                # trained under; a mismatch here would apply the checkpoint
                # through a differently-configured normalization scheme.
                eval_config["eval"]["beta1"] = cfg.model_free_rnnprop_beta1
                eval_config["eval"]["beta2"] = cfg.model_free_rnnprop_beta2
            eval_output_dir = os.path.join(output_dir, "eval_p{:.2f}".format(p))
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint, eval_output_dir)
            eval_sec = time.time() - t1
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec,
                         **eval_out.metrics})
    else:
        eval_config = dict(train_config)
        eval_config["eval"] = {
            "data_dir": width_dir,
            "test_files": [test_filename(p) for p in cfg.test_sparsities],
            "num_fista_iters": cfg.num_fista_iters,
            "xstar_cache_dir": xstar_cache_dir,
        }
        t1 = time.time()
        eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
        eval_sec = time.time() - t1
        for p in cfg.test_sparsities:
            metrics = eval_out.per_seed.get(test_filename(p), {})
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec,
                         **metrics})
    return rows


def run_experiment(cfg: OrchestratorConfig, stop_on_error: bool = False) -> None:
    spec = SweepSpec(
        results_dir=cfg.results_dir,
        axes=[Axis("seed", cfg.seeds), Axis("width", cfg.widths),
              Axis("method", cfg.methods)],
        fanout_keys=("test_sparsity",),
        # experiment2.py's hook, reused unchanged: it reads whatever config is
        # in context and only touches fields both configs share.
        prepare=[(("seed",), make_dataset)],
        context={"experiment_config": cfg},
        stop_on_error=stop_on_error,
    )
    run_sweep(spec, lambda cell, ctx: run_cell(cfg, cell, ctx), config_snapshot=cfg,
              group_by=["method", "width", "test_sparsity"])


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


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(21, 31)))
    p.add_argument("--methods", nargs="+", default=["l2o-rnnprop"], choices=list(ALL_METHODS))
    p.add_argument("--widths", nargs="+", default=list(WIDTHS), choices=list(WIDTHS))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budgets to sanity-check the whole "
                        "pipeline in well under a minute. NOT a real Experiment-2 run.")
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise a failing cell instead of recording it in "
                        "errors.jsonl and continuing. For development.")
    # Individual overrides (ignored if --smoke; --smoke defines its own tiny
    # config), derived from OrchestratorConfig's scalar fields.
    scalar_fields = add_overrides(p, OrchestratorConfig, help_texts={
        "train_size": "Override the number of training instances (default "
                      "32,000, same for all 3 widths). Idea #3 in "
                      "project_exp2_followup_experiments memory: tests "
                      "whether wide's worse own-center performance is a "
                      "sample-density artifact rather than a real "
                      "distribution-width tradeoff.",
        "m": "Override the number of measurements (default 25). See --n.",
        "n": "Override the signal dimension (default 50). Use "
             "--m 210 --n 300 to run this file's re-centered width "
             "scheme at the pre-study's own Table-3 dimensions: at "
             "n=50 the training widths quantize hard -- 'medium' "
             "[0.1125, 0.2375] covers only 7 distinct support sizes "
             "and 'wide' [0.05, 0.30] only 14 (round(p*n) nonzeros), "
             "so the width contrast that is this experiment's whole "
             "independent variable is lumpy. At n=300 the same "
             "ranges give 39 and 76. (The 11 TEST sparsities do stay "
             "distinct at n=50 -- 1,2,5,6,8,9,10,12,15,18,20 "
             "nonzeros -- so it is the training widths, not the test "
             "grid, that motivate the larger dims.)",
        "model_free_curriculum": "Enable train_dm.py/train_rnnprop.py's --if_cl "
                                 "(curriculum learning on unroll length).",
        "model_free_imitation": "Enable train_dm.py/train_rnnprop.py's --if_mt "
                                "(imitation learning from analytical optimizers).",
        "model_free_profile": "Write a per-epoch (loss, meta-grad-norm) trajectory "
                              "to <output_dir>/profile.jsonl.",
        "model_free_rnnprop_last_step_loss":
            "RNNProp only: train against the final unroll step's loss only "
            "(w_T=1, w_t=0 otherwise), matching Lv, Jiang & Li 2017's own stated "
            "training objective, instead of this script's prior default of summing "
            "the loss over every step (DM's convention).",
        "model_based_track_training_curve":
            "Model-based (LISTA/ALISTA) only: evaluate against "
            "--model_based_training_curve_sparsity's test file after every layer "
            "finishes training, not just once on the final model, and log the "
            "trajectory to <model_dir>/training_curve/training_curve.jsonl.",
    })
    return p.parse_args(), scalar_fields


def main() -> None:
    args, scalar_fields = _parse_args()
    if args.smoke:
        cfg = _smoke_config(args.results_dir, args.seeds, args.methods, args.widths)
    else:
        cfg = OrchestratorConfig(results_dir=args.results_dir, seeds=args.seeds,
                                 methods=args.methods, widths=args.widths)
        apply_overrides(cfg, args, scalar_fields)
        # __post_init__ already validated the constructor's values; re-run it
        # so the training-curve/test_sparsities consistency check also covers
        # anything the CLI overrides just changed.
        cfg.__post_init__()
    run_experiment(cfg, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
