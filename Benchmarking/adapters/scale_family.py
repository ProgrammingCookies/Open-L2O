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
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Dict, List, Optional

from core.config import require
from core.method import L2OMethod
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
}


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

        cmd = run_script(self.lib_dir, "train.py", flags,
                         log_path=os.path.join(output_dir, "train.log"))
        logdir = self._logdir(output_dir, config)
        ckpt = os.path.join(logdir, "model-final.l2o")
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
        flags: Dict[str, Any] = {
            "train_dir": self._train_dir(output_dir),
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
        flags.update(self._problem_flags(config, "eval"))
        flags.update({k: v for k, v in eval_cfg.items()})

        cmd = run_script(self.lib_dir, "evaluate.py", flags,
                         log_path=os.path.join(output_dir, "eval.log"))
        return _read_scale_eval(save_dir, cmd)

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
