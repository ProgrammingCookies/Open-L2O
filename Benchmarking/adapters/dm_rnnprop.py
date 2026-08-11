"""Adapter for L2O-DM and L2O-RNNProp.

  train_{dm,rnnprop}.py     --save_path <dir> --problem <name> --num_steps N
                            --unroll_length L [--beta1/--beta2 (rnnprop)]
                            => saves per-net "<key>.l2l" weights into <save_path>
  evaluate_{dm,rnnprop}.py  --optimizer {L2L,Adam} --problem <name> --path <dir>
                            --output_path <dir> --num_steps N
                            => <optimizer>_eval_loss_record.pickle-<problem>
                               (a flat list of costs over the run)

The optimizee problem is selected by name, passed straight
through: for example "mnist", "simple", "cifar", or "lasso_dataset".
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np

from core.config import require
from core.lasso_metrics import evaluate_lasso_recovery
from core.method import L2OMethod
from core.result import EvalOutput, TrainOutput, summarize_trajectory
from core.shell import REPO_ROOT, run_script

_LIB_SUBDIR = os.path.join("Model_Free_L2O", "L2O-DM and L2O-RNNProp")


class DMRNNPropMethod(L2OMethod):
    def __init__(self, name: str, variant: str):
        assert variant in ("dm", "rnnprop")
        self.name = name
        self.variant = variant
        self.lib_dir = os.path.join(str(REPO_ROOT), _LIB_SUBDIR)
        self.train_script = "train_{}.py".format(variant)
        self.eval_script = "evaluate_{}.py".format(variant)

    def _save_path(self, output_dir: str) -> str:
        return os.path.join(os.path.abspath(output_dir), "checkpoint")

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        save_path = self._save_path(output_dir)
        flags: Dict[str, Any] = {
            "save_path": save_path,
            "problem": require(config, "problem",
                "{} requires config['problem']".format(self.name)),
            "num_epochs": require(config.get("train", {}), "num_epochs",
                "{} requires config['train']['num_epochs']".format(self.name)),
        }

        # Set seed for reproducibility. 
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        flags.update(config.get("train", {}))

        #runs training using run_script helper function
        cmd = run_script(self.lib_dir, self.train_script, flags, log_path=os.path.join(output_dir, "train.log"))
        
        return TrainOutput(
            checkpoint=save_path,
            metrics={"save_path": save_path},
            artifacts={"train_log": os.path.join(output_dir, "train.log")},
            command=cmd,
        )

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        output_path = os.path.join(os.path.abspath(output_dir), "eval")
        os.makedirs(output_path, exist_ok=True)
        opt = require(config.get("eval", {}), "optimizer",
            "{} requires config['eval']['optimizer']".format(self.name))
        problem = require(config, "problem",
            "{} requires config['problem']".format(self.name))

        flags: Dict[str, Any] = {
            "optimizer": opt,
            "problem": problem,
            "path": checkpoint if opt == "L2L" else None,
            "output_path": output_path,
            "num_steps": require(config.get("eval", {}), "num_steps",
                "{} requires config['eval']['num_steps']".format(self.name)),
        }
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]

        # Add any additional eval flags from the config, except for those that are handled separately.
        flags.update({k: v for k, v in config.get("eval", {}).items()
                     if k not in ("num_fista_iters", "xstar_cache_dir")})
        
        # runs evaluation using run_script helper function
        cmd = run_script(self.lib_dir, self.eval_script, flags,
                         log_path=os.path.join(output_dir, "eval.log"))

        pickle_path = os.path.join(
            output_path, "{}_eval_loss_record.pickle-{}".format(opt, problem))
        traj = [] #Saving the per-iteration loss trajectory
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                traj = [float(v) for v in pickle.load(f)]
        metrics = summarize_trajectory(traj)
        # Saving path to log files and data so that they can be used for further analysis or debugging.
        artifacts = {"eval_pickle": pickle_path} 

        # NOTE: For problem="lasso_dataset" the eval script also saves the recovered
        # signal + ground truth + b. Recompute the Experiment-2 recovery metrics
        # (core/lasso_metrics.py) from that plus the shared A.npy, so every method's
        # x_pred is scored with the identical metric formula and the identical
        # FISTA-computed x* reference gives fair scoring.
        recovery_path = os.path.join(output_path, "{}_recovery-{}.npz".format(opt, problem))
        if os.path.exists(recovery_path):
            lasso_data_dir = config.get("eval", {}).get("lasso_data_dir")
            if lasso_data_dir is not None:
                artifacts["recovery"] = recovery_path
                rec = np.load(recovery_path)
                a = np.load(os.path.join(lasso_data_dir, "A.npy")).astype(np.float32)
                # Required only inside this lasso-recovery branch (unlike
                # model_based.py's evaluate(), which always runs task="lasso" so
                # requires it unconditionally) -- this adapter also serves
                # non-lasso problems (mnist, simple, cifar, ...) where lasso_lam
                # is meaningless.
                lam = require(config.get("eval", {}), "lasso_lam",
                    "{} requires config['eval']['lasso_lam'] when problem="
                    "'lasso_dataset' -- must match the lambda used to generate "
                    "the dataset (see Benchmarking/data/lasso.py)".format(self.name))
                num_fista_iters = require(config.get("eval", {}), "num_fista_iters",
                    "{} requires config['eval']['num_fista_iters'] when "
                    "problem='lasso_dataset'".format(self.name))
                xstar_cache_dir = config.get("eval", {}).get("xstar_cache_dir")
                recovery_metrics = evaluate_lasso_recovery(
                    a, rec["b"], rec["x_true"], rec["x_pred"], lam,
                    num_fista_iters=num_fista_iters,
                    xstar_cache_dir=xstar_cache_dir).to_dict()
                metrics.update(recovery_metrics)

        return EvalOutput(metrics=metrics, trajectory=traj,
                          artifacts=artifacts, command=cmd)

    #TODO: Add support for other classical optimizers if needed
    def run_classical_optimizer_eval(self, config: Dict[str, Any], classical: str,
                          output_dir: str) -> EvalOutput:
        if classical != "Adam":
            raise ValueError(
                "{} evaluation harness only supports the 'Adam' classical "
                "optimizer (got {!r})".format(self.name, classical))
        cfg = dict(config)
        cfg["eval"] = {**cfg.get("eval", {}), "optimizer": "Adam"}
        return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)
