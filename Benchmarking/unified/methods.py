"""``L2OMethod`` implementations backed by the unified in-process code.

Config vocabulary is the adapters' (``adapters/dm_rnnprop.py``), i.e. the
keys ``experiments/v2/experiment2.py::_build_train_config`` / ``run_cell``
emit, so an experiment can swap ``l2o-rnnprop`` for ``unified-rnnprop``
without touching its config builder. Unknown keys raise rather than being
silently ignored (the subprocess adapters would have failed on an unknown
CLI flag too).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import require
from core.method import L2OMethod
from core.result import EvalOutput, TrainOutput

_MODEL_FREE_TRAIN_KEYS = {
    # train_{dm,rnnprop}.py flags
    "save_path", "num_epochs", "evaluation_period", "evaluation_epochs", "num_steps",
    "unroll_length", "fix_num_unrolls", "profile_path", "seed", "learning_rate",
    "second_derivatives", "last_step_loss", "beta1", "beta2", "problem", "if_scale",
    "rd_scale_bound", "if_cl", "min_num_eval", "if_mt", "num_mt", "optimizers", "mt_ratio",
    "mt_ratios", "k", "lasso_data_dir", "lasso_split", "lasso_batch_size", "lasso_lam",
    "lasso_x0_mode",
}
# Accepted by the scripts but not ported in the unified slice (Experiment 2 never sets them).
_MODEL_FREE_UNPORTED = ("if_cl", "if_mt", "if_scale", "fix_num_unrolls")

_MODEL_FREE_EVAL_KEYS = {
    # evaluate_{dm,rnnprop}.py flags
    "optimizer", "problem", "path", "output_path", "num_epochs", "num_steps",
    "learning_rate", "seed", "lasso_data_dir", "lasso_split", "lasso_batch_size",
    "lasso_lam", "beta1", "beta2", "lasso_x0_mode",
    # harness-level keys consumed here, not by the scripts
    "num_fista_iters", "xstar_cache_dir", "test_files",
}


def _seed_everything(seed: Optional[int]) -> None:
    # The library scripts seed only tf (evaluate) / all three (train); seeding
    # all three here is a superset and changes nothing in a deterministic walk.
    if seed is None:
        return
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _check_keys(section: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError("{}: unknown config keys {} (allowed: {})".format(
            where, unknown, sorted(allowed)))


class UnifiedModelFreeMethod(L2OMethod):
    """L2O-DM / L2O-RNNProp (and, later, L2O-Scale) on the LASSO dataset problem."""

    def __init__(self, name: str, variant: str):
        assert variant in ("dm", "rnnprop")
        self.name = name
        self.variant = variant

    # ------------------------------------------------------------------ helpers

    def _policy(self, section: Dict[str, Any]):
        from unified.policies import DMPolicy, RNNPropPolicy
        if self.variant == "dm":
            if "beta1" in section or "beta2" in section:
                # train_dm.py / evaluate_dm.py have no such flags: argparse would reject them.
                raise ValueError("{}: beta1/beta2 are RNNProp-only".format(self.name))
            return DMPolicy()
        return RNNPropPolicy(beta1=float(section.get("beta1", 0.95)),
                             beta2=float(section.get("beta2", 0.95)))

    def _require_lasso(self, config: Dict[str, Any]) -> None:
        problem = require(config, "problem", "{} requires config['problem']".format(self.name))
        if problem != "lasso_dataset":
            raise NotImplementedError(
                "{} implements only problem='lasso_dataset' (got {!r}); use the "
                "adapter-backed l2o-{} for other problems".format(self.name, problem, self.variant))

    # ------------------------------------------------------------------- train

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        from unified.instances import LassoInstances
        from unified.meta_train import MetaTrainSpec, meta_train
        from unified.objective import LassoObjective

        self._require_lasso(config)
        tr = dict(config.get("train", {}))
        _check_keys(tr, _MODEL_FREE_TRAIN_KEYS, "{}.train config['train']".format(self.name))
        for flag in _MODEL_FREE_UNPORTED:
            if tr.get(flag):
                raise NotImplementedError("{}: --{} is not ported in the unified slice; use the "
                                          "adapter-backed l2o-{}".format(self.name, flag, self.variant))
        if tr.get("profile_path"):
            logging.warning("%s: profile_path is ignored by the unified slice (no per-epoch JSONL profile)", self.name)
        if self.variant == "dm" and ("last_step_loss" in tr or "lasso_x0_mode" in tr):
            raise ValueError("{}: last_step_loss/lasso_x0_mode are RNNProp-only flags".format(self.name))
        data_dir = require(tr, "lasso_data_dir", "{} requires config['train']['lasso_data_dir']".format(self.name))
        num_epochs = int(require(tr, "num_epochs", "{} requires config['train']['num_epochs']".format(self.name)))

        spec = MetaTrainSpec(
            num_epochs=num_epochs,
            num_steps=int(tr.get("num_steps", 100)),
            unroll_length=int(tr.get("unroll_length", 20)),
            meta_loss="last" if tr.get("last_step_loss") else "sum",
            learning_rate=float(tr.get("learning_rate", 0.001)),
            gradient_clip=None,
            second_derivatives=bool(tr.get("second_derivatives", False)),
            # train_dm.py defaults evaluation_period to 100, train_rnnprop.py to 10
            evaluation_period=int(tr.get("evaluation_period", 100 if self.variant == "dm" else 10)),
            evaluation_epochs=int(tr.get("evaluation_epochs", 20)),
            batch_size=int(tr.get("lasso_batch_size", 128)),
        )
        save_path = os.path.join(os.path.abspath(output_dir), "checkpoint")
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "train.log")

        _seed_everything(config.get("seed"))
        instances = LassoInstances(data_dir, tr.get("lasso_split", "train_data.npy"),
                                   x0_mode=tr.get("lasso_x0_mode", "aligned"))
        objective = LassoObjective(instances.a, float(tr.get("lasso_lam", 0.005)))
        policy = self._policy(tr)
        with open(log_path, "w") as log_file:
            def log(line):
                log_file.write(line + "\n")
                log_file.flush()
            spec.log = log
            log("# in-process {} train: {}".format(self.name, json.dumps(tr, sort_keys=True, default=str)))
            result = meta_train(policy, objective, instances, spec, save_path)
        return TrainOutput(
            checkpoint=save_path,
            metrics={"save_path": save_path, "best_evaluation": result.best_evaluation,
                     "saved_epochs": result.saved_epochs, "final_training_loss": result.epoch_losses[-1]},
            artifacts={"train_log": log_path, "recovery": os.path.join(save_path, "recovery.npz")},
            command="in-process {}".format(self.name),
        )

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        from unified.evalwalk import evaluate_split, to_eval_output, write_model_free_artifacts
        from unified.instances import LassoInstances
        from unified.objective import LassoObjective
        from unified.policies import ClassicalAdamPolicy
        from unified.solver import PolicySolver

        self._require_lasso(config)
        ev = dict(config.get("eval", {}))
        _check_keys(ev, _MODEL_FREE_EVAL_KEYS, "{}.evaluate config['eval']".format(self.name))
        opt = require(ev, "optimizer", "{} requires config['eval']['optimizer']".format(self.name))
        num_steps = int(require(ev, "num_steps", "{} requires config['eval']['num_steps']".format(self.name)))
        data_dir = require(ev, "lasso_data_dir", "{} requires config['eval']['lasso_data_dir']".format(self.name))
        batch_size = int(ev.get("lasso_batch_size", 128))
        lam = float(ev.get("lasso_lam", 0.005))
        x0_mode = ev.get("lasso_x0_mode", "aligned")
        if self.variant == "dm" and "lasso_x0_mode" in ev:
            raise ValueError("{}: evaluate_dm.py has no --lasso_x0_mode flag".format(self.name))
        if ev.get("num_epochs", 1) != 1:
            logging.warning("--num_epochs is ignored for lasso_dataset: one deterministic pass over the split")
        splits: List[str] = list(ev["test_files"]) if "test_files" in ev else [ev.get("lasso_split", "val_data.npy")]

        _seed_everything(config.get("seed"))
        output_path = os.path.join(os.path.abspath(output_dir), "eval")

        if opt == "Adam":
            def make_policy():
                return ClassicalAdamPolicy(float(ev.get("learning_rate", 0.001)))
            fixed_policy = None
        elif opt == "L2L":
            # One network for the whole walk (the scripts build it once, before
            # the chunk loop); RNN state is re-initialised per chunk by the solver.
            fixed_policy = self._policy(ev)
            if checkpoint is None:
                logging.warning("Evaluating untrained L2L optimizer test")
            def make_policy():
                return fixed_policy
        else:
            raise ValueError("{} is not a valid optimizer".format(opt))

        import tensorflow as tf
        results, artifacts, a = [], {}, None
        for split in splits:
            instances = LassoInstances(data_dir, split, x0_mode=x0_mode)
            a = instances.a if a is None else a
            objective = LassoObjective(instances.a, lam)
            if fixed_policy is not None and not fixed_policy.trainable_variables:
                # Build with a throw-away step, then load -- the order the
                # scripts use (Keras creates weights on the first forward pass).
                first = next(instances.walk(batch_size))
                fixed_policy.build(tf.constant(first.x0))
                if checkpoint is not None:
                    fixed_policy.restore(checkpoint, 0)
            result = evaluate_split(
                lambda: PolicySolver(make_policy(), objective),
                instances, objective, batch_size, num_steps, reduction="batch_mean")
            results.append(result)
            tag_dir = output_path if len(splits) == 1 else os.path.join(output_path, split[:-len(".npy")])
            for k, v in write_model_free_artifacts(result, tag_dir, opt).items():
                artifacts[k if len(splits) == 1 else "{}:{}".format(split, k)] = v

        return to_eval_output(results, a, lam, ev.get("num_fista_iters"), ev.get("xstar_cache_dir"),
                              artifacts, command="in-process {}".format(self.name))

    def run_classical_optimizer_eval(self, config: Dict[str, Any], classical: str,
                                     output_dir: str) -> EvalOutput:
        if classical != "Adam":
            raise ValueError("{} evaluation harness only supports the 'Adam' classical "
                             "optimizer (got {!r})".format(self.name, classical))
        cfg = dict(config)
        cfg["eval"] = {**cfg.get("eval", {}), "optimizer": "Adam"}
        return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)


_UNROLLED_TRAIN_KEYS = {
    # Model_Base_L2O/train.py flags the adapter forwards
    "data_dir", "num_layers", "lasso_lam", "model_lam", "epochs", "patience", "base_lr",
    "train_batch_size", "val_batch_size", "ss_q_per_layer", "ss_maxq", "share_W",
    "track_training_curve", "training_curve_dir", "training_curve_test_file", "seed",
    # unified-only, explicit non-parity option (see curriculum.py)
    "keras_train_loss",
}
_UNROLLED_EVAL_KEYS = {"data_dir", "test_files", "num_fista_iters", "xstar_cache_dir",
                       "val_batch_size", "seed", "lasso_lam"}


class UnifiedUnrolledMethod(L2OMethod):
    """LISTA / ALISTA: an unrolled network whose layer k is iteration k.

    Identity fields (``num_layers``, ``lasso_lam``, ``model_lam``, ``ss_*``,
    ``share_W``) are read from ``config["train"]`` in both phases, as
    ``adapters/model_based.py`` does, so evaluation cannot silently use a
    different architecture than was trained.
    """

    def __init__(self, name: str, model_name: str):
        assert model_name in ("lista", "alista")
        self.name = name
        self.model_name = model_name

    # ------------------------------------------------------------------ helpers

    def _exp_name(self, config: Dict[str, Any]) -> str:
        run_name = config.get("run_name")
        if run_name:
            return run_name
        return "{}__{}".format(self.name, require(config, "problem",
                                                  "{} requires config['problem']".format(self.name)))

    def _model_dir(self, base_dir: str, config: Dict[str, Any]) -> str:
        # adapters/model_based.py convention, so adapter checkpoints resolve as-is.
        if os.path.isdir(os.path.join(base_dir, "layer_1")):
            return base_dir
        return os.path.join(base_dir, "models", self._exp_name(config), "replicate_1")

    def _identity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        tr = config.get("train", {})
        _check_keys(tr, _UNROLLED_TRAIN_KEYS, "{} config['train']".format(self.name))
        return {
            "num_layers": int(require(tr, "num_layers", "{} requires config['train']['num_layers']".format(self.name))),
            "lasso_lam": float(require(tr, "lasso_lam", "{} requires config['train']['lasso_lam']".format(self.name))),
            # train.py flag defaults
            "model_lam": float(tr.get("model_lam", 0.4)),
            "ss_q_per_layer": float(tr.get("ss_q_per_layer", 1.2)),
            "ss_maxq": float(tr.get("ss_maxq", 13.0)),
            "share_W": bool(tr.get("share_W", False)),
        }

    def _build(self, ident: Dict[str, Any], instances, create_cells: bool = True):
        from unified.unrolled import build_model
        return build_model(self.model_name, instances.a, ident["num_layers"], ident["model_lam"],
                           w_alista=instances.w_alista, ss_q_per_layer=ident["ss_q_per_layer"],
                           ss_maxq=ident["ss_maxq"], share_W=ident["share_W"], create_cells=create_cells)

    # ------------------------------------------------------------------- train

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        from unified.curriculum import train_curriculum
        from unified.instances import LassoInstances

        tr = dict(config.get("train", {}))
        ident = self._identity(config)
        if tr.get("track_training_curve"):
            raise NotImplementedError("{}: track_training_curve is not ported in the unified slice; "
                                      "use the adapter-backed {}".format(self.name, self.model_name))
        data_dir = require(tr, "data_dir", "{} requires config['train']['data_dir']".format(self.name))
        base_dir = os.path.join(os.path.abspath(output_dir), "checkpoints")
        model_dir = os.path.join(base_dir, "models", self._exp_name(config), "replicate_1")
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "train.log")

        _seed_everything(config.get("seed"))
        train_instances = LassoInstances(data_dir, "train_data.npy")
        val_instances = LassoInstances(data_dir, "val_data.npy")
        model = self._build(ident, train_instances, create_cells=False)
        with open(log_path, "w") as log_file:
            def log(line):
                log_file.write(line + "\n")
                log_file.flush()
            log("# in-process {} train: {}".format(self.name, json.dumps(tr, sort_keys=True, default=str)))
            histories = train_curriculum(
                model, train_instances, val_instances, model_dir,
                lasso_lam=ident["lasso_lam"],
                # train.py flag defaults: epochs 100000 (a cap), patience 8, base_lr 5e-4, batch 128
                epochs=int(tr.get("epochs", 100000)), patience=int(tr.get("patience", 8)),
                base_lr=float(tr.get("base_lr", 0.0005)),
                train_batch_size=int(tr.get("train_batch_size", 128)),
                val_batch_size=int(tr.get("val_batch_size", 1024)),
                keras_train_loss=tr.get("keras_train_loss", "paper"), log=log)
        return TrainOutput(
            checkpoint=base_dir,
            metrics={"model_dir": model_dir, "layers": histories,
                     "final_val_lasso": histories[-1]["val_lasso"]},
            artifacts={"train_log": log_path},
            command="in-process {}".format(self.name),
        )

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        from unified.evalwalk import evaluate_split, to_eval_output
        from unified.instances import LassoInstances
        from unified.objective import LassoObjective
        from unified.unrolled import UnrolledSolver, load_layerwise_checkpoint

        ev = dict(config.get("eval", {}))
        _check_keys(ev, _UNROLLED_EVAL_KEYS, "{}.evaluate config['eval']".format(self.name))
        ident = self._identity(config)
        data_dir = ev.get("data_dir") or config.get("train", {}).get("data_dir")
        if data_dir is None:
            raise ValueError("{} requires config['eval']['data_dir']".format(self.name))
        test_files: List[str] = list(ev.get("test_files") or [])
        if not test_files:
            raise ValueError("{} requires config['eval']['test_files']".format(self.name))
        if "lasso_lam" in ev and float(ev["lasso_lam"]) != ident["lasso_lam"]:
            logging.warning("%s: config['eval']['lasso_lam'] ignored; the identity field "
                            "config['train']['lasso_lam']=%r is used", self.name, ident["lasso_lam"])
        batch_size = int(ev.get("val_batch_size", 1024))
        if checkpoint is None:
            raise ValueError("{} has no classical path: evaluate needs a trained checkpoint".format(self.name))
        model_dir = self._model_dir(checkpoint, config)

        _seed_everything(config.get("seed"))
        eval_dir = os.path.join(os.path.abspath(output_dir), "eval")
        os.makedirs(eval_dir, exist_ok=True)

        results, artifacts, model, a = [], {}, None, None
        all_metrics = {}
        for split in test_files:
            instances = LassoInstances(data_dir, split)
            a = instances.a if a is None else a
            objective = LassoObjective(instances.a, ident["lasso_lam"])
            if model is None:
                model = self._build(ident, instances)
                artifacts["checkpoint_keys"] = json.dumps(
                    load_layerwise_checkpoint(model, model_dir, ident["num_layers"]))
            solver = UnrolledSolver(model)
            result = evaluate_split(lambda: solver, instances, objective, batch_size,
                                    ident["num_layers"], reduction="sample_mean")
            results.append(result)
            base = split[:-len(".npy")]
            np.save(os.path.join(eval_dir, base + "_final_output.npy"), result.x_pred)
            all_metrics[split] = {"lasso_layer{}".format(i): v for i, v in enumerate(result.trajectory)}
            artifacts["{}:final_output".format(split)] = os.path.join(eval_dir, base + "_final_output.npy")
        with open(os.path.join(eval_dir, "all_test_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        artifacts["all_test_metrics"] = os.path.join(eval_dir, "all_test_metrics.json")

        return to_eval_output(results, a, ident["lasso_lam"], ev.get("num_fista_iters"),
                              ev.get("xstar_cache_dir"), artifacts, command="in-process {}".format(self.name))
