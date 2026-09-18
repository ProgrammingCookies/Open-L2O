"""The one meta-training loop for learned optimizers.

Port of ``train_dm.py`` / ``train_rnnprop.py`` (driver) and the ``unroll`` of
``meta_dm_train.py:121-162`` / ``meta_rnnprop_train.py:162-202``, with the
per-method differences pulled out into ``MetaTrainSpec`` and everything
problem-specific behind the ``Problem`` protocol:

  * meta-loss:  ``"sum"`` = sum of the T+1 objective values of the window
                (the T pre-step values plus the post-step one, as in the
                originals); ``"last"`` = the post-step value only
                (RNNProp ``--last_step_loss``).
  * meta-optimizer, gradient clipping (none for DM/RNNProp), second derivatives.
  * the policy owns its state layout; the global step ``t`` for bias
    correction is threaded by the loop (``step_offset`` in the originals).

Per epoch a fresh training batch is drawn (``problem()`` in the scripts),
``num_steps // unroll_length`` windows are run back to back with the policy
state carried across windows and ``x`` re-assigned from the window's last
iterate, and every ``evaluation_period`` epochs the optimizer is scored on
``evaluation_epochs`` fresh batches; on improvement the weights are saved as
``<key>.l2l-<epoch+1>`` and ``<key>.l2l-0`` ("best", what evaluation loads).

Random-number consumption follows the scripts exactly (one batch draw at
setup that is only used to size things, one per epoch, one per evaluation
batch; network weights created on the first forward pass), so a same-seed run
on the same device reproduces ``train_{dm,rnnprop}.py`` bit for bit -- for
any problem whose ``sample()`` draws the way the script's ``build()`` does.

Not ported (Experiment 2 never sets them; the schema raises): ``--if_cl``
curriculum, ``--if_mt`` imitation learning, ``--if_scale`` random rescaling,
``--fix_num_unrolls``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from unified.policy import OptimizerPolicy
from unified.problem import Problem


@dataclass
class MetaTrainSpec:
    num_epochs: int
    num_steps: int
    unroll_length: int
    meta_loss: str = "sum"                 # "sum" | "last"
    learning_rate: float = 1e-3            # tf.keras.optimizers.Adam(learning_rate)
    gradient_clip: Optional[float] = None  # per-element clip_by_value; None = no clipping
    second_derivatives: bool = False
    evaluation_period: int = 10
    evaluation_epochs: int = 20
    log: Callable[[str], None] = field(default=print)


@dataclass
class MetaTrainResult:
    best_evaluation: float
    epoch_losses: List[float]
    eval_history: List[Dict[str, float]]
    saved_epochs: List[int]


def unroll_window(policy: OptimizerPolicy, problem: Problem, x, batch, state,
                  step_offset: int, len_unroll: int, meta_loss: str, second_derivatives: bool):
    """One truncated-BPTT window; returns (loss, meta_grads, x_final, state_final)."""
    import tensorflow as tf
    with tf.GradientTape() as meta_tape:
        fx_list = []
        for t in range(len_unroll):
            fx, grad = problem.value_and_grad(x, batch, second_derivatives)
            fx_list.append(fx)
            x, state = policy.step(x, grad, state, step_offset + t + 1)
        fx_final = problem.objective(x, batch)
        fx_list.append(fx_final)
        if meta_loss == "last":
            total_loss = fx_final
        elif meta_loss == "sum":
            total_loss = tf.add_n(fx_list)
        else:
            raise ValueError("meta_loss must be 'sum' or 'last', got {!r}".format(meta_loss))
    net_vars = policy.trainable_variables
    meta_grads = meta_tape.gradient(total_loss, net_vars)
    return total_loss, meta_grads, x, state


def _run_windows(policy, problem, batch, num_unrolls, spec, meta_opt=None):
    """Run ``num_unrolls`` windows on one batch; trains when ``meta_opt`` is given.
    Returns (last window loss, sum of window losses, x_final, last grad norm)."""
    import tensorflow as tf
    x = tf.identity(tf.constant(batch.x0))
    state = policy.initial_state(x)
    step_offset, last_loss, loss_sum, grad_norm = 0, 0.0, 0.0, None
    for _ in range(num_unrolls):
        loss, meta_grads, x_final, state = unroll_window(
            policy, problem, x, batch, state, step_offset, spec.unroll_length,
            spec.meta_loss, spec.second_derivatives)
        if meta_opt is not None:
            if spec.gradient_clip is not None:
                meta_grads = [tf.clip_by_value(g, -spec.gradient_clip, spec.gradient_clip) for g in meta_grads]
            grad_norm = float(tf.linalg.global_norm(meta_grads))
            meta_opt.apply_gradients(zip(meta_grads, policy.trainable_variables))
        # train_*.py: x_vars.assign(x_final) between windows, i.e. the next
        # window starts from the value but not the graph of the previous one.
        x = tf.identity(tf.stop_gradient(x_final))
        step_offset += spec.unroll_length
        last_loss = float(loss)
        loss_sum += last_loss
    return last_loss, loss_sum, x, grad_norm


def meta_train(policy: OptimizerPolicy, problem: Problem, spec: MetaTrainSpec,
               save_path: Optional[str]) -> MetaTrainResult:
    import tensorflow as tf
    num_unrolls = spec.num_steps // spec.unroll_length
    if num_unrolls == 0:
        raise ValueError("num_steps={} < unroll_length={} runs zero windows; the network would "
                         "never be built and nothing could be saved".format(spec.num_steps, spec.unroll_length))
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    meta_opt = tf.keras.optimizers.Adam(spec.learning_rate)
    # The scripts build a problem once before the epoch loop ("initial setup")
    # purely to create the networks; it consumes one batch draw.
    problem.sample()

    best_evaluation = float("inf")
    result = MetaTrainResult(best_evaluation, [], [], [])
    for e in range(spec.num_epochs):
        batch = problem.sample()
        epoch_loss, _, _, grad_norm = _run_windows(policy, problem, batch, num_unrolls, spec, meta_opt)
        result.epoch_losses.append(epoch_loss)
        spec.log("training_loss={}".format(epoch_loss))

        if (e + 1) % spec.evaluation_period == 0 or e + 1 == spec.num_epochs:
            eval_cost = 0.0
            ev_batch, ev_x = None, None
            for _ in range(spec.evaluation_epochs):
                ev_batch = problem.sample()
                _, loss_sum, ev_x, _ = _run_windows(policy, problem, ev_batch, num_unrolls, spec)
                eval_cost += loss_sum
            if save_path is not None:
                # Overwritten every evaluation: what is on disk when training
                # ends is the final evaluation's recovery (train_*.py) -- for
                # problems whose script writes one.
                recovery = problem.training_recovery(ev_batch, np.asarray(ev_x))
                if recovery is not None:
                    np.savez(os.path.join(save_path, "recovery.npz"), **recovery)
            spec.log("epoch={}, num_steps={}, eval_loss={}".format(e, spec.num_steps, eval_cost / spec.evaluation_epochs))
            result.eval_history.append({"epoch": e, "eval_cost": eval_cost, "grad_norm": grad_norm})
            if eval_cost < best_evaluation:
                best_evaluation = eval_cost
                if save_path is not None:
                    policy.save(save_path, e + 1)
                    policy.save(save_path, 0)
                    result.saved_epochs.append(e + 1)
                    spec.log("Saving optimizer of epoch {}...".format(e + 1))
    result.best_evaluation = best_evaluation
    return result
