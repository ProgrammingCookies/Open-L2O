"""Adapter for the L2O-Scale family L2O-Scale, L2O-Entropy and L2O-Jacobian.

All three share an identical CLI as follows:

  train.py    --train_dir <dir> --optimizer <cls> --cell_cls/--cell_size/--num_cells
              --num_meta_iterations N --include_<problem>_problems
              => checkpoints at <train_dir>/<opt>_<cell_cls>_<cell_size>_<num_cells>/
                 model-{best,iter<k>,final}.l2o

  evaluate.py --train_dir <dir> --save_dir <dir> --test_optimizer {L2o,SGD,Adam,Adagrad}
              --num_testing_itrs N --model_name <tag> --include_<problem>_problems
              => per-seed pickles seed{6,12,18,24,30}_eval_loss_record.pickle-<tag>,
                 each a flat list of objective values over num_testing_itrs.

For problem="lasso_dataset" (L2O-Scale only -- Entropy/Jacobian's evaluate.py
have no LASSO flags) evaluate.py instead walks --lasso_split deterministically
from the split's seeded x0 and also writes L2o_recovery-lasso_dataset.npz
(x_pred/x_true/b/x0); evaluate() then scores it with the shared scoring code
(core/metrics.py) exactly as adapters/dm_rnnprop.py does, so every method's
x_pred is scored with the identical metric code and the identical FISTA x*.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import require
from core.method import L2OMethod
from core.metrics import compute_metrics, scoring_config_keys
from core.result import EvalOutput, TrainOutput, summarize_trajectory
from core.shell import REPO_ROOT, run_script

_EVAL_SEEDS = [6, 12, 18, 24, 30]
_EVAL_MODEL_TAG = "benchmark"

# Problem name -> {train flags, eval flags}. Train-time and eval-time
# problem sets differ in these libraries, so the two are specified separately.
PROBLEM_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "mnist_mlp": {
        "train": {"include_mnist_mlp_problems": True},
        "eval": {"include_mnist_mlp_problems": True},
    },
    "mnist_conv": {
        "train": {"include_mnist_mlp_problems": True},
        "eval": {"include_mnist_conv_problems": True},
    },
    "quadratic": {
        "train": {"include_quadratic_problems": True},
        "eval": {"include_mnist_mlp_problems": True},
    },
    # Pre-generated LASSO dataset (Benchmarking/data/lasso.py). Both phases
    # also need lasso_data_dir / lasso_split / lasso_batch_size / lasso_lam
    # from the config, passed through like any other train/eval flag.
    "lasso_dataset": {
        "train": {"include_lasso_problems": True},
        "eval": {"include_lasso_problems": True},
    },
}

# evaluate.py's final-iterate dump for a scored problem (core/problems.py);
# the name is fixed whatever --test_optimizer is.
_RECOVERY_FILE = "L2o_recovery-{}.npz"


class ScaleFamilyMethod(L2OMethod):
    def __init__(self, name: str, lib_subdir: str):
        self.name = name
        self.lib_dir = os.path.join(str(REPO_ROOT), "Model_Free_L2O", lib_subdir)

    def _optimizer_flags(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Flags that must match between train and eval so the checkpoint directory matches."""
        t = config.get("train", {})
        return {
            "optimizer": require(t, "optimizer",
                "{} requires config['train']['optimizer']".format(self.name)),
            "cell_cls": require(t, "cell_cls",
                "{} requires config['train']['cell_cls']".format(self.name)),
            "cell_size": require(t, "cell_size",
                "{} requires config['train']['cell_size']".format(self.name)),
            "num_cells": require(t, "num_cells",
                "{} requires config['train']['num_cells']".format(self.name)),
        }

    def _train_dir(self, output_dir: str) -> str:
        return os.path.join(os.path.abspath(output_dir), "checkpoints")

    def _logdir(self, output_dir: str, config: Dict[str, Any]) -> str:
        of = self._optimizer_flags(config)
        return os.path.join(self._train_dir(output_dir), "{optimizer}_{cell_cls}_{cell_size}_{num_cells}".format(**of))

    def _problem_flags(self, config: Dict[str, Any], phase: str) -> Dict[str, Any]:
        problem = config.get("problem")
        if problem in PROBLEM_MAP:
            return dict(PROBLEM_MAP[problem][phase])
        return {}

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        flags: Dict[str, Any] = {
            "train_dir": self._train_dir(output_dir),
            "num_meta_iterations": require(config.get("train", {}), "num_meta_iterations",
                "{} requires config['train']['num_meta_iterations']".format(self.name)),
        }
        flags.update(self._optimizer_flags(config))
        flags.update(self._problem_flags(config, "train"))
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        flags.update(config.get("train", {}))

        # select_checkpoint is consumed here, not a train.py flag: "final" (the
        # default, model-final.l2o) or "best" (model-best.l2o, the lowest
        # internal-validation cost -- the analogue of train_dm.py's
        # best-internal-eval `.l2l-0`, for like-for-like checkpoint selection).
        select = flags.pop("select_checkpoint", "final")
        if select not in ("final", "best"):
            raise ValueError("{}: config['train']['select_checkpoint'] must be 'final' "
                             "or 'best', got {!r}".format(self.name, select))

        cmd = run_script(self.lib_dir, "train.py", flags,
                         log_path=os.path.join(output_dir, "train.log"))
        logdir = self._logdir(output_dir, config)
        ckpt = os.path.join(logdir, "model-{}.l2o".format(select))
        if select == "best" and not os.path.exists(ckpt):
            raise FileNotFoundError(
                "{}: select_checkpoint='best' but {} was not written -- train.py only "
                "saves it when an evaluation ran (evaluation_period must divide "
                "num_meta_iterations)".format(self.name, ckpt))
        return TrainOutput(
            checkpoint=ckpt,
            metrics={"logdir": logdir},
            artifacts={"train_log": os.path.join(output_dir, "train.log")},
            command=cmd,
        )

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        # config["eval"]["seed"] narrows evaluate.py's sweep to that one seed instead
        # of the default _EVAL_SEEDS which makes num_seeds=1, so a
        # seeded run aren't directly comparable to an unseeded one.
        save_dir = os.path.join(os.path.abspath(output_dir), "eval")
        os.makedirs(save_dir, exist_ok=True)

        eval_cfg = config.get("eval", {})
        test_optimizer = require(eval_cfg, "test_optimizer",
            "{} requires config['eval']['test_optimizer']".format(self.name))
        # evaluate.py locates the checkpoint as <train_dir>/<optimizer tag>/
        # <restore_model_name>. When the caller hands us the checkpoint path
        # (train()'s TrainOutput.checkpoint) derive both from it, so evaluating
        # into a different output_dir than train() used (e.g. Experiment 2's
        # per-sparsity eval_p*/ subdirs) still finds the trained weights.
        if checkpoint is not None:
            train_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
            restore_model_name = os.path.basename(checkpoint)
        else:
            train_dir = self._train_dir(output_dir)
            restore_model_name = None
        flags: Dict[str, Any] = {
            "train_dir": train_dir,
            "save_dir": save_dir,
            "test_optimizer": test_optimizer,
            "num_testing_itrs": require(eval_cfg, "num_testing_itrs",
                "{} requires config['eval']['num_testing_itrs']".format(self.name)),
            "model_name": _EVAL_MODEL_TAG,
        }
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        if test_optimizer == "L2o":
            # optimizer/cell_cls/cell_size/num_cells select which trained L2O
            # checkpoint to restore, which must match train.
            flags.update(self._optimizer_flags(config))
            if restore_model_name is not None:
                flags["restore_model_name"] = restore_model_name
        flags.update(self._problem_flags(config, "eval"))
        flags.update({k: v for k, v in eval_cfg.items()
                      if k not in scoring_config_keys()})

        cmd = run_script(self.lib_dir, "evaluate.py", flags,
                         log_path=os.path.join(output_dir, "eval.log"))
        out = _read_scale_eval(save_dir, cmd)

        # Fingerprint + metrics, computed from the saved final iterate with the
        # shared scoring code (see module docstring).
        problem = config.get("problem")
        recovery_path = os.path.join(save_dir, _RECOVERY_FILE.format(problem))
        if problem is not None and os.path.exists(recovery_path):
            rec = np.load(recovery_path)
            out.metrics.update(compute_metrics(
                problem, {k: rec[k] for k in rec.files}, config, who=self.name))
            out.artifacts["recovery"] = recovery_path
        return out

    def run_classical_optimizer_eval(self, config: Dict[str, Any], classical: str,
                          output_dir: str) -> EvalOutput:
        """ Eval classical Optimizer """
        cfg = dict(config)
        cfg.setdefault("eval", {})
        cfg["eval"] = {**cfg["eval"], "test_optimizer": classical}
        return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)


def _read_scale_eval(save_dir: str, command: str) -> EvalOutput:
    """Aggregate the per-seed results into one EvalOutput."""
    files = sorted(glob.glob(os.path.join(
        save_dir, "seed*_eval_loss_record.pickle-*")))
    per_seed: Dict[str, Any] = {}
    trajectories: List[List[float]] = []
    for path in files:
        with open(path, "rb") as f:
            traj = pickle.load(f)
        traj = [float(v) for v in traj]
        seed = os.path.basename(path).split("_")[0]
        per_seed[seed] = summarize_trajectory(traj)
        trajectories.append(traj)

    mean_traj: List[float] = []
    if trajectories:
        n = min(len(t) for t in trajectories)
        mean_traj = [sum(t[i] for t in trajectories) / len(trajectories)
                     for i in range(n)]

    metrics = summarize_trajectory(mean_traj)
    metrics["num_seeds"] = len(trajectories)
    return EvalOutput(metrics=metrics, trajectory=mean_traj, per_seed=per_seed,
                      artifacts={"eval_dir": save_dir}, command=command)
