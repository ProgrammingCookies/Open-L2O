"""``L2OMethod`` implementations backed by the unified in-process code.

Two generic classes, no method or problem names inside:

* ``UnifiedModelFreeMethod(PolicySpec)`` -- a learned optimizer on any
  registered ``"model_free"`` problem dialect: ``meta_train`` for ``train()``,
  the evaluation walk with the learned policy or a classical baseline
  (``eval.optimizer``) for ``evaluate()``.
* ``UnifiedUnrolledMethod(ModelSpec)`` -- an unrolled network on its
  ``"unrolled"`` problem dialect: the spec's training procedure, then the
  walk with the spec's solver.

Config vocabulary is the adapters' (``adapters/dm_rnnprop.py``,
``adapters/model_based.py``): the keys ``experiments/v2/experiment2.py``
emits work unchanged. Each section is validated against the method spec's
schema merged with the problem dialect's, so unknown keys raise rather than
being silently ignored (the subprocess adapters would have failed on an
unknown CLI flag too).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np

from core.config import require
from core.method import L2OMethod
from core.result import EvalOutput, TrainOutput
from unified.model import ModelSpec
from unified.policy import PolicySpec
from unified.problem import get_dialect


def _seed_everything(seed: Optional[int]) -> None:
    # The library scripts seed only tf (evaluate) / all three (train); seeding
    # all three here is a superset and changes nothing in a deterministic walk.
    if seed is None:
        return
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class _LogFile:
    def __init__(self, path: str, header: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._f = open(path, "w")
        self(header)

    def __call__(self, line: str) -> None:
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------- model-free

class UnifiedModelFreeMethod(L2OMethod):
    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.name = spec.name

    def _dialect(self, config: Dict[str, Any]):
        problem = require(config, "problem", "{} requires config['problem']".format(self.name))
        return get_dialect(problem, "model_free", self.name)

    # ------------------------------------------------------------------- train

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        from unified.meta_train import MetaTrainSpec, meta_train

        dialect = self._dialect(config)
        schema = self.spec.train_schema.merge(dialect.train_schema)
        tr = schema.validate(config.get("train", {}), "{}.train config['train']".format(self.name))
        for k, v in self.spec.train_defaults.items():        # the script's own defaults
            if k not in config.get("train", {}):
                tr[k] = v
        cfg = {**config, "train": tr}
        if self.spec.train_loop == "scale":
            return self._train_scale(cfg, tr, dialect, output_dir)
        if self.spec.train_loop != "dm":
            raise ValueError("{}: unknown train_loop {!r}".format(self.name, self.spec.train_loop))

        spec = MetaTrainSpec(
            num_epochs=int(tr["num_epochs"]), num_steps=int(tr["num_steps"]),
            unroll_length=int(tr["unroll_length"]),
            meta_loss="last" if tr.get("last_step_loss") else "sum",
            learning_rate=float(tr["learning_rate"]), gradient_clip=None,
            second_derivatives=bool(tr["second_derivatives"]),
            evaluation_period=int(tr["evaluation_period"]), evaluation_epochs=int(tr["evaluation_epochs"]),
        )
        save_path = os.path.join(os.path.abspath(output_dir), "checkpoint")
        log = _LogFile(os.path.join(output_dir, "train.log"),
                       "# in-process {} train: {}".format(self.name, json.dumps(tr, sort_keys=True, default=str)))
        try:
            _seed_everything(config.get("seed"))
            problem = dialect.train_problem(cfg)
            policy = self.spec.make_policy(tr)
            spec.log = log
            result = meta_train(policy, problem, spec, save_path)
        finally:
            log.close()
        artifacts = {"train_log": log.path}
        if os.path.exists(os.path.join(save_path, "recovery.npz")):
            artifacts["recovery"] = os.path.join(save_path, "recovery.npz")
        return TrainOutput(
            checkpoint=save_path,
            metrics={"save_path": save_path, "best_evaluation": result.best_evaluation,
                     "saved_epochs": result.saved_epochs, "final_training_loss": result.epoch_losses[-1]},
            artifacts=artifacts, command="in-process {}".format(self.name))

    def _train_scale(self, cfg, tr, dialect, output_dir: str) -> TrainOutput:
        """L2O-Scale's loop shape (metaopt.train_optimizer); checkpoint layout of adapters/scale_family.py."""
        from unified.learned.scale import checkpoint_dir
        from unified.meta_train import ScaleTrainSpec, scale_train

        select = tr["select_checkpoint"]
        if select not in ("final", "best"):
            raise ValueError("{}: config['train']['select_checkpoint'] must be 'final' or 'best', "
                             "got {!r}".format(self.name, select))
        spec = ScaleTrainSpec(
            num_problems=int(tr["num_problems"]), num_meta_iterations=int(tr["num_meta_iterations"]),
            fix_num_steps=int(tr["fix_num_steps"]), fix_unroll_length=int(tr["fix_unroll_length"]),
            fix_num_steps_eval=int(tr["fix_num_steps_eval"]),
            evaluation_period=int(tr["evaluation_period"]), evaluation_epochs=int(tr["evaluation_epochs"]),
            save_period=int(tr["save_period"]), learning_rate=float(tr["meta_learning_rate"]),
            gradient_clip=float(tr["gradient_clip_level"]))
        if not tr.get("fix_unroll", True):
            raise NotImplementedError("{}: only --fix_unroll training is ported (random unroll "
                                      "lengths are not)".format(self.name))
        logdir = checkpoint_dir(output_dir)
        log = _LogFile(os.path.join(output_dir, "train.log"),
                       "# in-process {} train: {}".format(self.name, json.dumps(tr, sort_keys=True, default=str)))
        try:
            _seed_everything(cfg.get("seed"))
            problem = dialect.train_problem(cfg)
            policy = self.spec.make_policy(tr)
            spec.log = log
            result = scale_train(policy, problem, spec, logdir)
        finally:
            log.close()
        ckpt = os.path.join(logdir, "model-{}.l2o".format(select))
        if not os.path.exists(ckpt):
            raise FileNotFoundError("{}: select_checkpoint={!r} but {} was not written (model-best.l2o "
                                    "needs an evaluation: evaluation_period must divide "
                                    "num_meta_iterations)".format(self.name, select, ckpt))
        artifacts = {"train_log": log.path}
        if os.path.exists(os.path.join(logdir, "recovery.npz")):
            artifacts["recovery"] = os.path.join(logdir, "recovery.npz")
        return TrainOutput(checkpoint=ckpt,
                           metrics={"logdir": logdir, "best_evaluation": result.best_evaluation},
                           artifacts=artifacts, command="in-process {}".format(self.name))

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        import tensorflow as tf
        from unified.evalwalk import evaluate_split, to_eval_output
        from unified.solver import PolicySolver

        spec = self.spec
        dialect = self._dialect(config)
        schema = spec.eval_schema.merge(dialect.eval_schema)
        ev = schema.validate(config.get("eval", {}), "{}.evaluate config['eval']".format(self.name))
        cfg = {**config, "eval": ev}
        opt = ev[spec.eval_optimizer_key]
        num_steps = int(ev[spec.eval_steps_key])
        output_path = os.path.join(os.path.abspath(output_dir), "eval")
        seed = config.get("seed")

        _seed_everything(seed)
        labelled = dialect.eval_problems(cfg)

        on_first_batch = None
        if opt == spec.learned_tag:
            if checkpoint is None:
                logging.warning("Evaluating untrained %s optimizer", spec.learned_tag)
            if spec.fresh_policy_per_chunk:
                # The library script rebuilds and reloads the optimizer for
                # every chunk (L2O-Scale); the policy does the preamble itself.
                def make_policy():
                    return spec.make_policy(ev, checkpoint)
            else:
                # One network for the whole walk (the scripts build it once, before
                # the chunk loop); the solver re-initialises its state per chunk.
                policy = spec.make_policy(ev)

                def make_policy():
                    return policy

                def on_first_batch(batch):
                    # Build with a throw-away step, then load -- the scripts' order
                    # (Keras creates weights on the first forward pass).
                    if not policy.trainable_variables:
                        policy.build(tf.constant(batch.x0))
                        if checkpoint is not None:
                            policy.restore(checkpoint, 0)
            tag = spec.learned_tag
        elif opt in spec.classical:
            def make_policy():
                return spec.classical[opt](ev)
            tag = opt
        else:
            raise ValueError("{} is not a valid {} ({} or one of {})".format(
                opt, spec.eval_optimizer_key, spec.learned_tag, sorted(spec.classical)))

        results, problems, artifacts = [], [], {}
        for label, problem in labelled:
            result = evaluate_split(lambda: PolicySolver(make_policy(), problem), problem, num_steps,
                                    problem.eval_reduction, label=label, on_first_batch=on_first_batch)
            results.append(result)
            problems.append(problem)
            tag_dir = output_path if len(labelled) == 1 else os.path.join(output_path, os.path.splitext(label)[0])
            written = problem.write_eval_artifacts(result, tag_dir, tag, layout=spec.artifact_layout,
                                                   seed=seed if seed is not None else 0)
            for k, v in written.items():
                artifacts[k if len(labelled) == 1 else "{}:{}".format(label, k)] = v

        return to_eval_output(results, problems, ev, artifacts, command="in-process {}".format(self.name))

    def run_classical_optimizer_eval(self, config: Dict[str, Any], classical: str,
                                     output_dir: str) -> EvalOutput:
        if classical not in self.spec.classical:
            raise ValueError("{} evaluation harness only supports the classical optimizers {} "
                             "(got {!r})".format(self.name, sorted(self.spec.classical), classical))
        cfg = dict(config)
        cfg["eval"] = {**cfg.get("eval", {}), self.spec.eval_optimizer_key: classical}
        return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)


# ------------------------------------------------------------------ unrolled

class UnifiedUnrolledMethod(L2OMethod):
    """An unrolled network whose layer k is iteration k.

    Identity fields (``spec.identity_keys``) are read from ``config["train"]``
    in both phases, as ``adapters/model_based.py`` does, so evaluation cannot
    silently use a different architecture than was trained.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.name = spec.name

    def _dialect(self, config: Dict[str, Any]):
        problem = require(config, "problem", "{} requires config['problem']".format(self.name))
        if problem != self.spec.problem:
            raise NotImplementedError("{} solves problem {!r} only (got {!r})".format(
                self.name, self.spec.problem, problem))
        return get_dialect(problem, "unrolled", self.name)

    def _exp_name(self, config: Dict[str, Any]) -> str:
        return config.get("run_name") or "{}__{}".format(self.name, config["problem"])

    def _model_dir(self, base_dir: str, config: Dict[str, Any]) -> str:
        # adapters/model_based.py convention, so adapter checkpoints resolve as-is.
        if os.path.isdir(os.path.join(base_dir, "layer_1")):
            return base_dir
        return os.path.join(base_dir, "models", self._exp_name(config), "replicate_1")

    def _train_section(self, config: Dict[str, Any], dialect) -> Dict[str, Any]:
        schema = self.spec.train_schema.merge(dialect.train_schema)
        return schema.validate(config.get("train", {}), "{} config['train']".format(self.name))

    # ------------------------------------------------------------------- train

    def train(self, config: Dict[str, Any], output_dir: str) -> TrainOutput:
        dialect = self._dialect(config)
        tr = self._train_section(config, dialect)
        cfg = {**config, "train": tr}
        ident = self.spec.identity(tr)
        base_dir = os.path.join(os.path.abspath(output_dir), "checkpoints")
        model_dir = os.path.join(base_dir, "models", self._exp_name(config), "replicate_1")
        os.makedirs(model_dir, exist_ok=True)
        log = _LogFile(os.path.join(output_dir, "train.log"),
                       "# in-process {} train: {}".format(self.name, json.dumps(tr, sort_keys=True, default=str)))
        try:
            _seed_everything(config.get("seed"))
            train_problem = dialect.train_problem(cfg)
            val_problem = dialect.val_problem(cfg)
            model = self.spec.build(train_problem, ident, False)
            histories = self.spec.train(model, train_problem, val_problem, model_dir, tr, ident, log)
        finally:
            log.close()
        return TrainOutput(
            checkpoint=base_dir,
            metrics={"model_dir": model_dir, "layers": histories, "final_val_lasso": histories[-1]["val_lasso"]},
            artifacts={"train_log": log.path}, command="in-process {}".format(self.name))

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, config: Dict[str, Any], checkpoint: Optional[str],
                 output_dir: str) -> EvalOutput:
        from unified.evalwalk import evaluate_split, to_eval_output

        dialect = self._dialect(config)
        tr = self._train_section(config, dialect)
        ident = self.spec.identity(tr)
        schema = self.spec.eval_schema.merge(dialect.eval_schema)
        ev = schema.validate(config.get("eval", {}), "{}.evaluate config['eval']".format(self.name))
        cfg = {**config, "train": tr, "eval": ev}
        if checkpoint is None:
            raise ValueError("{} has no classical path: evaluate needs a trained checkpoint".format(self.name))
        model_dir = self._model_dir(checkpoint, config)
        num_layers = int(ident["num_layers"])

        _seed_everything(config.get("seed"))
        labelled = dialect.eval_problems(cfg)
        eval_dir = os.path.join(os.path.abspath(output_dir), "eval")
        os.makedirs(eval_dir, exist_ok=True)

        model = self.spec.build(labelled[0][1], ident, True)
        artifacts = {"checkpoint_keys": json.dumps(self.spec.load_checkpoint(model, model_dir, num_layers))}
        solver = self.spec.make_solver(model)
        results, problems, all_metrics = [], [], {}
        for label, problem in labelled:
            result = evaluate_split(lambda: solver, problem, num_layers, "sample_mean", label=label)
            results.append(result)
            problems.append(problem)
            base = os.path.splitext(label)[0]
            np.save(os.path.join(eval_dir, base + "_final_output.npy"), result.x_pred)
            all_metrics[label] = {"lasso_layer{}".format(i): v for i, v in enumerate(result.trajectory)}
            artifacts["{}:final_output".format(label)] = os.path.join(eval_dir, base + "_final_output.npy")
        with open(os.path.join(eval_dir, "all_test_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        artifacts["all_test_metrics"] = os.path.join(eval_dir, "all_test_metrics.json")

        return to_eval_output(results, problems, ev, artifacts, command="in-process {}".format(self.name))
