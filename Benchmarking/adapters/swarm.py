"""Adapter for L2O-Swarm (population-based learned optimizer).

  src/train.py     --save_path <dir> --problem <name> --num_steps N
                   --unroll_length L  => saves net weights + loss_record.pickle
  src/evaluate.py  --optimizer {L2L,Adam} --problem <name> --path <dir>
                   --num_steps N      => writes <path>/evaluate_record.pickle
                     = {"all_time_loss_record": [[...per step...] per epoch],
                        "min_loss_record":      [min per epoch]}
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

from core.config import require
from core.method import L2OMethod
from core.result import EvalOutput, TrainOutput, summarize_trajectory
from core.shell import REPO_ROOT, run_script

_LIB_DIR = os.path.join("Model_Free_L2O", "L2O-Swarm", "src")


class SwarmMethod(L2OMethod):
    name = "l2o-swarm"

    def __init__(self):
        self.lib_dir = os.path.join(str(REPO_ROOT), _LIB_DIR)

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
        # train.py now accepts --seed (seeds random/numpy/TF before problem/optimizer
        # construction). Top-level config["seed"] is the default; an explicit
        # config["train"]["seed"] below still wins.
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        flags.update(config.get("train", {}))
        cmd = run_script(self.lib_dir, "train.py", flags,
                         log_path=os.path.join(output_dir, "train.log"))
        return TrainOutput(
            checkpoint=save_path,
            metrics={"save_path": save_path},
            artifacts={"train_log": os.path.join(output_dir, "train.log")},
            command=cmd,
        )

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        # Fallback for callers that set config["seed"] but not config["eval"]["seed"]
        # (runner.py already threads it through; explicit eval.seed still wins).
        opt = require(config.get("eval", {}), "optimizer",
            "{} requires config['eval']['optimizer']".format(self.name))
        # evaluate.py writes evaluate_record.pickle *inside* --path, so for the
        # L2L case that is the checkpoint dir; for Adam there is no checkpoint,
        # so give it a writable eval dir to drop the record into.
        abs_path = os.path.abspath(checkpoint) if opt == "L2L" else os.path.join(
            os.path.abspath(output_dir), "eval")
        os.makedirs(abs_path, exist_ok=True)

        flags: Dict[str, Any] = {
            "optimizer": opt,
            "problem": require(config, "problem",
                "{} requires config['problem']".format(self.name)),
            "path": abs_path,
            "num_steps": require(config.get("eval", {}), "num_steps",
                "{} requires config['eval']['num_steps']".format(self.name)),
        }
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        flags.update(config.get("eval", {}))
        cmd = run_script(self.lib_dir, "evaluate.py", flags,
                         log_path=os.path.join(output_dir, "eval.log"))

        record_path = os.path.join(abs_path, "evaluate_record.pickle")
        traj: List[float] = []
        per_epoch: Dict[str, Any] = {}
        if os.path.exists(record_path):
            with open(record_path, "rb") as f:
                record = pickle.load(f)
            all_time = record.get("all_time_loss_record", [])
            # Representative trajectory (traj) = mean across epochs at each step.
            if all_time:
                n = min(len(e) for e in all_time)
                traj = [sum(float(e[i]) for e in all_time) / len(all_time)
                        for i in range(n)]
            per_epoch = {"min_loss_record": [float(v) for v in record.get("min_loss_record", [])]}
        metrics = summarize_trajectory(traj)
        return EvalOutput(metrics=metrics, trajectory=traj, per_seed=per_epoch,
                          artifacts={"eval_record": record_path}, command=cmd)

    def run_classical_optimizer_eval(self, config: Dict[str, Any], classical: str,
                          output_dir: str) -> EvalOutput:
        if classical != "Adam":
            raise ValueError(
                "l2o-swarm evaluation harness only supports the 'Adam' classical "
                "optimizer (got {!r})".format(classical))
        cfg = dict(config)
        cfg["eval"] = {**cfg.get("eval", {}), "optimizer": "Adam"}
        return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)
