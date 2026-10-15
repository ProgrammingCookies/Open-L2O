"""
Adapter for the model-based L2O family: LISTA, ALISTA LFISTA, LAMP, StepLista, ListaCp, ListaCpss, Glista, Tista.

Unlike model-free methods, ``Model_Base_L2O/train.py`` is a layer-wise has layer wise training, not a end-to-end trained
like model-free methods.
This is resolved here by treating each unrolled layer as one iteration.

  train.py --data_dir <dir> --base_dir <dir> --task=lasso --model_name=<name>
           --exp_name <name> --replicate <n> --num_layers N
           => layer-by-layer Keras fit(), checkpoints under
              <base_dir>/models/<exp_name>/replicate_<n>/
  train.py --test=True --test_files <f1> --test_files <f2> ...
           => <model_dir>/<file>_metrics.json (per-layer LassoObjective) +
              <model_dir>/all_test_metrics.json + <model_dir>/<file>_final_output.npy
              (the model's recovered sparse signal)

Each test file's final_output.npy, plus the split file's own b, x_true rows,
is scored per test file by the shared scoring code (core/metrics.py: the
problem "lasso"'s fingerprint and metrics, by default the four Table-4
metrics) and folded into EvalOutput.metrics/per_seed.

NOTE: Currently only handles LASSO problems.
TODO: Add support for other problems?
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import require
from core.method import L2OMethod
from core.metrics import compute_metrics, scoring_config_keys
from core.result import EvalOutput, TrainOutput, summarize_trajectory
from core.shell import REPO_ROOT, run_script

_LIB_DIR = "Model_Base_L2O"
_SCORED_PROBLEM = "lasso"

# supported model's names (must match Model_Base_L2O/train.py's --model_name choices)
#
# Only "lista" and "alista" (the two Experiment 2 actually uses) start their
# layer 0 from the seeded x0 that data_preprocessing.py gives every
# model's input for task="lasso",
# the others still use start from x0=0 internally.
# Not fixed as of yet cause they're not used in my experiments.
SUPPORTED_MODELS = (
    "lista", "lfista", "lamp", "step_lista", "lista_cp", "lista_cpss",
    "alista", "glista", "tista",
)


class ModelBasedMethod(L2OMethod):
    def __init__(self, name: str, model_name: str):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError("unsupported Model_Base_L2O model_name {!r} (expected one "
                             "of {})".format(model_name, SUPPORTED_MODELS))
        self.name = name
        self.model_name = model_name
        self.lib_dir = os.path.join(str(REPO_ROOT), _LIB_DIR)

    # identity helpers 
    # (train() and evaluate() must have the same values on these or we're not evaluating on the same optimizer we trained)
    def _exp_name(self, config: Dict[str, Any]) -> str:
        # "problem" is only actually needed to build a name here, so it's only
        # required when run_name is absent, if run_name is given, "problem"
        # is never read, not even to fall back to something.
        run_name = config.get("run_name")
        if run_name:
            return run_name
        problem = require(config, "problem",
            "{} requires config['problem']".format(self.name))
        return "{}__{}".format(self.name, problem)

    def _num_layers(self, config: Dict[str, Any]) -> int:
        return require(config.get("train", {}), "num_layers",
            "{} requires config['train']['num_layers'] and must match between "
            "train and evaluate ".format(self.name))

    def _lasso_lam(self, config: Dict[str, Any]) -> float:
        return require(config.get("train", {}), "lasso_lam",
            "{} requires config['train']['lasso_lam'] and must match between "
            "train and evaluate ".format(self.name))

    def _base_dir(self, output_dir: str) -> str:
        return os.path.join(os.path.abspath(output_dir), "checkpoints")

    def _model_dir(self, base_dir: str, config: Dict[str, Any]) -> str:
        return os.path.join(base_dir, "models", self._exp_name(config), "replicate_1")

    def _data_dir(self, config: Dict[str, Any], phase: str) -> str:
        data_dir = config.get(phase, {}).get("data_dir") or config.get("data_dir")
        if data_dir is None:
            raise ValueError(
                "ModelBasedMethod requires config['{}']['data_dir'], see Benchmarking/data/lasso.py".format(phase))
        return data_dir

    def _dataset_sizes(self, data_dir: str):
        """Checks train/val sample counts from the generated .npy files, so a
        mismatched number of samples can't silently fail."""
        train_n = int(np.load(os.path.join(data_dir, "train_data.npy"), mmap_mode="r").shape[0])
        val_path = os.path.join(data_dir, "val_data.npy")
        val_n = int(np.load(val_path, mmap_mode="r").shape[0]) if os.path.exists(val_path) else None
        return train_n, val_n

    def _warn_ignored(self, path: str, value: Any, reason: str) -> None:
        """A config value the caller explicitly set is about to be silently
        discarded, this function warns about it.
        """
        warnings.warn(
            "{} is ignoring {}={!r} -- {}".format(self.name, path, value, reason),
            stacklevel=3)

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        data_dir = self._data_dir(config, "train")
        train_n, val_n = self._dataset_sizes(data_dir)
        base_dir = self._base_dir(output_dir)

        flags: Dict[str, Any] = {
            "model_name": self.model_name,
            "task": "lasso",
            "exp_name": self._exp_name(config),
            "replicate": 1,
            "num_layers": self._num_layers(config),
            "lasso_lam": self._lasso_lam(config),
            "num_train_images": train_n,
        }
        if val_n is not None:
            flags["num_val_images"] = val_n
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        # User decided flags overrides presets except data_dir/base_dir are always set to the right values 
        # for variables that need to be consistent across train/evaluate.
        train_section = config.get("train", {})
        if "base_dir" in train_section:
            self._warn_ignored(
                "config['train']['base_dir']", train_section["base_dir"],
                "base_dir is always derived from output_dir (here: {!r}), not "
                "configurable through config['train'], pass a different "
                "output_dir instead.".format(base_dir))
        if "data_dir" in train_section and train_section["data_dir"] != data_dir:
            self._warn_ignored(
                "config['train']['data_dir']", train_section["data_dir"],
                "using {!r} instead (resolved via _data_dir(), the same value "
                "num_train_images={} was already computed from).".format(data_dir, train_n))
        train_overrides = dict(train_section)
        train_overrides.pop("data_dir", None)
        flags.update(train_overrides)
        flags["data_dir"] = data_dir
        flags["base_dir"] = base_dir

        model_dir = self._model_dir(base_dir, config)
        artifacts = {"train_log": os.path.join(output_dir, "train.log")}
        if train_overrides.get("track_training_curve") and "training_curve_dir" not in train_overrides:
            # Same default train.py itself would pick (<model_dir>/training_curve),
            # made explicit here so the artifact path below is guaranteed correct
            # rather than using train.py's default.
            training_curve_dir = os.path.join(model_dir, "training_curve")
            flags["training_curve_dir"] = training_curve_dir
        if train_overrides.get("track_training_curve"):
            artifacts["training_curve"] = os.path.join(
                flags["training_curve_dir"], "training_curve.jsonl")

        cmd = run_script(self.lib_dir, "train.py", flags,
                         log_path=os.path.join(output_dir, "train.log"))
        return TrainOutput(
            checkpoint=base_dir,
            metrics={"model_dir": model_dir},
            artifacts=artifacts,
            command=cmd,
        )

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        data_dir = self._data_dir(config, "eval")
        eval_cfg = config.get("eval", {})
        test_files: List[str] = eval_cfg.get("test_files") or []
        if not test_files:
            raise ValueError(
                "ModelBasedMethod.evaluate requires config['eval']['test_files'] (a list "
                "of split filenames under data_dir, e.g. Benchmarking/data/lasso.py's "
                "test_sparsity_*.npy)")

        base_dir = checkpoint or self._base_dir(output_dir)
        flags: Dict[str, Any] = {
            "model_name": self.model_name,
            "task": "lasso",
            "exp_name": self._exp_name(config),
            "replicate": 1,
            "num_layers": self._num_layers(config),
            "lasso_lam": self._lasso_lam(config),
            "test": True,
            "test_files": test_files,
        }
        if config.get("seed") is not None:
            flags["seed"] = config["seed"]
        # The subprocess call below and the recovery
        # metric recomputation below both derive it from self._lasso_lam(config)
        # (i.e. config["train"]["lasso_lam"]) and letting config["eval"] silently
        # override it here would desync those two, scoring an already-trained
        # checkpoint against a lambda it was never trained for without erroring.
        if "lasso_lam" in eval_cfg:
            self._warn_ignored(
                "config['eval']['lasso_lam']", eval_cfg["lasso_lam"],
                "lasso_lam is an identity field only config['train']['lasso_lam'] "
                "(currently {!r}) is used, so evaluation matches what was trained."
                .format(self._lasso_lam(config)))
        if "base_dir" in eval_cfg:
            self._warn_ignored(
                "config['eval']['base_dir']", eval_cfg["base_dir"],
                "base_dir is always `checkpoint` if given, else derived from "
                "output_dir (here: {!r}), not configurable through "
                "config['eval'].".format(base_dir))
        if "data_dir" in eval_cfg and eval_cfg["data_dir"] != data_dir:
            self._warn_ignored(
                "config['eval']['data_dir']", eval_cfg["data_dir"],
                "using {!r} instead (resolved via _data_dir()).".format(data_dir))
        eval_overrides = {k: v for k, v in eval_cfg.items()
                          if k not in ("data_dir", "test_files", "lasso_lam")
                          and k not in scoring_config_keys()}
        flags.update(eval_overrides)
        flags["data_dir"] = data_dir
        flags["base_dir"] = base_dir

        cmd = run_script(self.lib_dir, "train.py", flags,
                         log_path=os.path.join(output_dir, "eval.log"))
        model_dir = self._model_dir(base_dir, config)
        return _read_model_based_eval(model_dir, data_dir, test_files, config, cmd, self.name)


def _read_model_based_eval(model_dir: str, data_dir: str, test_files: List[str],
                           config: Dict[str, Any], command: str, who: str) -> EvalOutput:
    all_results_path = os.path.join(model_dir, "all_test_metrics.json")
    with open(all_results_path) as f:
        res_dict = json.load(f)
    m = int(np.load(os.path.join(data_dir, "A.npy"), mmap_mode="r").shape[0])

    per_test_file: Dict[str, Any] = {}
    artifacts: Dict[str, str] = {"all_test_metrics": all_results_path}
    representative_traj: List[float] = []
    representative_recovery: Dict[str, float] = {}
    for i, fname in enumerate(test_files):
        metrics_dict = res_dict.get(fname, {})
        # Per-layer keys are "lasso_layer{i}" for i in range(num_layers), in order.
        layer_keys = sorted(
            (k for k in metrics_dict if k.startswith("lasso_layer")),
            key=lambda k: int(k[len("lasso_layer"):]))
        traj = [float(metrics_dict[k]) for k in layer_keys]
        summary = summarize_trajectory(traj)

        basename = os.path.basename(fname).replace(".npy", "")
        output_path = os.path.join(model_dir, basename + "_final_output.npy")
        if os.path.exists(output_path):
            artifacts["final_output__{}".format(basename)] = output_path
            split_data = np.load(os.path.join(data_dir, fname)).astype(np.float32)
            b = split_data[:, :m]
            x_true = split_data[:, m:]
            x_pred = np.load(output_path).astype(np.float32)
            if x_pred.shape[0] != split_data.shape[0]:
                raise ValueError(
                    "{} has {} rows but {} has {}: the test pass did not cover the "
                    "split exactly once".format(output_path, x_pred.shape[0], fname,
                                               split_data.shape[0]))
            # Instances fingerprint. Model_Base_L2O/train.py's test mode writes
            # only x_pred, so b/x_true/x0 are taken from the split files in file
            # order -- the order data_preprocessing.input_fn(is_training=False)
            # walks them (no shuffle; covered by
            # tests/test_model_based_input_order.py). The row-count check above
            # is the part observed from the script itself.
            x0_path = os.path.join(data_dir, fname[:-len(".npy")] + "_x0.npy")
            x0 = np.load(x0_path).astype(np.float32) if os.path.exists(x0_path) else None
            recovery = compute_metrics(
                _SCORED_PROBLEM, {"x_pred": x_pred, "b": b, "x_true": x_true, "x0": x0},
                config, who=who)
            summary.update(recovery)
            if i == 0:
                representative_recovery = recovery

        per_test_file[fname] = summary
        if i == 0:
            representative_traj = traj

    metrics = summarize_trajectory(representative_traj)
    metrics.update(representative_recovery)
    metrics["num_test_files"] = len(test_files)
    return EvalOutput(metrics=metrics, trajectory=representative_traj,
                      per_seed=per_test_file, artifacts=artifacts, command=command)