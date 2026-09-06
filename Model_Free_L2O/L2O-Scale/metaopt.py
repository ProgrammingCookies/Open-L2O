# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper utilities for training and testing optimizers (eager / TF2).

`train_optimizer`'s outer structure -- a fresh problem per `problem_itr`, a
fixed number of "partial unroll" segments per meta-iteration `k`, gradient
clipping, periodic evaluation/checkpointing -- is unchanged from the original.
What's gone is everything that existed only to satisfy TF1 graph/session
semantics: the per-problem tf.Graph(), the parameter-server device_setter
(single-machine eager execution has no ps/worker placement concept),
tf.train.Supervisor's managed_session/checkpointing (replaced by
TrainableOptimizer.save(), see optimizer/trainable_optimizer.py), and the
tf.summary/profiling scaffolding (dropped -- incidental monitoring
infrastructure, not core algorithm behavior).

The one piece of original behavior worth calling out explicitly because it's
easy to get wrong: `first_unroll` is true for the first partial-unroll segment
of *every* meta-iteration `k` (so a fresh starting point is drawn every `k`),
but `reset_state` (which reinitializes the optimizer's own RNN/momentum/decay
state) is true only for the very first segment of the very first meta-iteration
of a given problem instance (when reset_rnn_params, the default). This is
preserved via TrainableOptimizer.new_trajectory() (called every k) and
TrainableOptimizer.reset_state() (called only when reset_state is True below).
"""

import os
import random
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

import profiling
from optimizer import utils
from problems import datasets
from problems import problem_generator


def sigmoid_weights(n, slope=0.1, offset=5):
    """Generates a sigmoid, scaled to sum to 1.

    This function is used to generate weights that serve to mask out
    the early objective values of an optimization problem such that
    initial variation in the objective is phased out (hence the sigmoid
    starts at zero and ramps up to the maximum value, and the total
    weight is normalized to sum to one)

    Args:
      n: the number of samples
      slope: slope of the sigmoid (Default: 0.1)
      offset: threshold of the sigmoid (Default: 5)

    Returns:
      A length-n array of weights summing to 1.
    """
    x = np.arange(n)
    y = 1. / (1. + np.exp(-slope * (x - offset)))
    y_normalized = y / np.sum(y)
    return y_normalized


def sample_numiter(scale, min_steps=50):
    """Samples a number of iterations from an exponential distribution.

    Args:
      scale: parameter for the exponential distribution
      min_steps: minimum number of steps to run (additive)

    Returns:
      num_steps: An integer equal to a rounded sample from the exponential
                 distribution + the value of min_steps.
    """
    return int(np.round(np.random.exponential(scale=scale)) + min_steps)


def _batches_per_unroll(dataset, batch_size, partial_unroll_iters):
    """Splits dataset.batch_indices(...) into one list of batches per unroll segment."""
    total_num_iter = sum(partial_unroll_iters)
    db = dataset.batch_indices(total_num_iter, batch_size)
    dataset_batches = []
    last_index = 0
    for num in partial_unroll_iters:
        dataset_batches.append(db[last_index:last_index + num])
        last_index += num
    return dataset_batches


def _validate(opt, k, num_unrolls, partial_unroll_iters, batch_size,
             objective_weights, dataset, reset_rnn_params):
    """Runs a held-out sequence of unrolls (no gradient update) and returns the
    final subproblem objective -- the eager equivalent of the original's
    validate() helper. Note this mutates opt's state exactly like a training
    segment would (new_trajectory()/reset_state()/partial_unroll()), matching
    the original's tf.while_loop-graph reuse (evaluation and training share
    the same persistent optimizer state, so an evaluation run does perturb
    what the *next* meta-iteration's optimizer state starts from -- preserved
    here rather than "fixed", since it's the original's actual behavior).
    """
    dataset_batches = _batches_per_unroll(dataset, batch_size, partial_unroll_iters)
    sub_obj = None
    for unroll_itr in range(num_unrolls):
        first_unroll = unroll_itr == 0
        reset_state = (first_unroll and k == 0) if reset_rnn_params else first_unroll

        if first_unroll:
            opt.new_trajectory()
        if reset_state:
            opt.reset_state()

        _, sub_obj = opt.partial_unroll(
            partial_unroll_iters[unroll_itr], objective_weights[unroll_itr],
            dataset_batches[unroll_itr])
    return float(sub_obj[-1])


def train_optimizer(logdir,
                    optimizer_spec,
                    problems_and_data,
                    num_problems,
                    num_meta_iterations,
                    num_unroll_func,
                    num_partial_unroll_itrs_func,
                    learning_rate=1e-4,
                    gradient_clip=5.,
                    select_random_problems=True,
                    obj_train_max_multiplier=-1,
                    fix_unroll=False,
                    fix_unroll_length=20,
                    fix_num_steps=100,
                    fix_num_steps_eval=100,
                    evaluation_period=1,
                    evaluation_epochs=20,
                    save_period=1,
                    l2_reg=0.,
                    rms_decay=0.9,
                    rms_epsilon=1e-20,
                    reset_rnn_params=True,
                    profile_path=None):
    """Trains the meta-parameters of this optimizer.

    Args:
      logdir: a directory filepath for storing model checkpoints, or None to
        disable checkpointing.
      optimizer_spec: specification for an Optimizer (see problem_spec.Spec)
      problems_and_data: a list of tuples containing three elements: a problem
        specification (see problem_spec.Spec), a dataset (see
        datasets.Dataset), and a batch_size (int) for generating a problem and
        corresponding dataset. If the problem doesn't have data, set dataset
        to None.
      num_problems: the number of problems to sample during meta-training
      num_meta_iterations: the number of iterations (steps) to run the
        meta-optimizer for on each subproblem.
      num_unroll_func: called once per meta iteration and returns the number of
        unrolls to do for that meta iteration.
      num_partial_unroll_itrs_func: called once per unroll and returns the number
        of iterations to do for that unroll.
      learning_rate: learning rate of the RMSProp meta-optimizer (Default: 1e-4)
      gradient_clip: value to clip gradients at (Default: 5.0)
      select_random_problems: whether to select training problems randomly
          (Default: True)
      obj_train_max_multiplier: the maximum increase in the objective value over
          a single training run. Ignored if < 0.

    Raises:
      ValueError: If one of the subproblems has a negative objective value.
    """
    best_evaluation = float("inf")
    if select_random_problems:
        sampler = [random.choice(problems_and_data) for _ in range(num_problems)]
    else:
        num_repeats = int(num_problems / len(problems_and_data)) + 1
        shuffled = list(problems_and_data)
        random.shuffle(shuffled)
        sampler = (shuffled * num_repeats)[:num_problems]

    # Created once and reused across every problem in the sampler, so the
    # optimizer's learned weights (and meta_opt's own RMSprop momentum state)
    # keep improving across problems -- this is the eager equivalent of the
    # original's checkpoint save/restore-through-a-shared-logdir dance (each
    # problem there got a fresh tf.Graph() with fresh variables, but
    # tf.train.Supervisor(logdir=logdir).managed_session() transparently
    # restored them from the previous problem's checkpoint on the same
    # logdir). Keeping the same Python object achieves the same continuity
    # directly. It also sidesteps a Keras 3 constraint: once an optimizer's
    # apply_gradients has been called, it can only be called again with the
    # same variables it was first built with.
    opt = optimizer_spec.build()
    meta_opt = tf.keras.optimizers.RMSprop(
        learning_rate, rho=rms_decay, epsilon=rms_epsilon)

    profiler = profiling.RunProfiler(profile_path)
    profiler.start()
    global_step = 0

    for problem_itr, (problem_spec, dataset, batch_size) in enumerate(sampler):
        print("problem {}".format(problem_itr))
        problem_start_time = time.time()

        if dataset is None:
            dataset = datasets.EMPTY_DATASET
            batch_size = dataset.size

        problem = problem_spec.build()
        opt.setup(problem, (dataset.data, dataset.labels))

        print("Took {:.2f}s to initialize problem {}.".format(
            time.time() - problem_start_time, problem_itr))

        for k in range(num_meta_iterations):
            print("meta iteration {}".format(k), flush=True)

            if not fix_unroll:
                num_unrolls = num_unroll_func()
                partial_unroll_iters = [
                    num_partial_unroll_itrs_func() for _ in range(num_unrolls)]
            else:
                num_unrolls = fix_num_steps // fix_unroll_length
                partial_unroll_iters = [fix_unroll_length] * num_unrolls
            total_num_iter = sum(partial_unroll_iters)

            objective_weights = [np.ones(num) / float(num)
                                 for num in partial_unroll_iters]
            dataset_batches = _batches_per_unroll(dataset, batch_size, partial_unroll_iters)

            train_start_time = time.time()
            additional_log_info = ""

            for unroll_itr in range(num_unrolls):
                first_unroll = unroll_itr == 0
                reset_state = (first_unroll and k == 0) if reset_rnn_params else first_unroll

                with tf.GradientTape() as meta_tape:
                    if first_unroll:
                        opt.new_trajectory()
                    if reset_state:
                        opt.reset_state()

                    meta_obj, sub_obj = opt.partial_unroll(
                        partial_unroll_iters[unroll_itr], objective_weights[unroll_itr],
                        dataset_batches[unroll_itr])

                    meta_params = opt.trainable_variables
                    reg_l2 = l2_reg * sum(tf.reduce_sum(p ** 2) for p in meta_params)
                    total_loss = meta_obj + reg_l2

                grads = meta_tape.gradient(total_loss, meta_params)
                raw_grad_norm = float(tf.linalg.global_norm(
                    [g for g in grads if g is not None]))
                clipped_grads_and_vars = [
                    (tf.clip_by_value(utils.make_finite(g, tf.zeros_like(v)),
                                      -gradient_clip, gradient_clip), v)
                    for g, v in zip(grads, meta_params) if g is not None]
                # clip_by_value clips each element independently, so this post-clip norm tells you
                # about per-element magnitude saturation against gradient_clip.
                post_clip_grad_norm = float(tf.linalg.global_norm(
                    [g for g, _ in clipped_grads_and_vars]))
                meta_opt.apply_gradients(clipped_grads_and_vars)
                profiler.log_epoch(
                    global_step, float(total_loss), raw_grad_norm, post_clip_grad_norm,
                    num_optimizee_steps=partial_unroll_iters[unroll_itr])
                global_step += 1

                sub_obj_np = np.array([float(v) for v in sub_obj])
                if np.any(sub_obj_np < 0):
                    raise ValueError("Training problem objectives must be nonnegative.")

                # If the objective has increased more than we want, exit this
                # training run and start over on another meta iteration.
                init_obj = float(opt._initial_obj)
                if obj_train_max_multiplier > 0 and (
                        sub_obj_np[-1] > (init_obj +
                                          abs(init_obj) * (obj_train_max_multiplier - 1))):
                    additional_log_info += " Broke early at {} out of {} unrolls.".format(
                        unroll_itr + 1, num_unrolls)
                    break

            if (k + 1) % evaluation_period == 0:
                cost_total = 0.0
                for _ in range(evaluation_epochs):
                    num_unrolls_val = fix_num_steps_eval // fix_unroll_length
                    partial_unroll_iters_val = [fix_unroll_length] * num_unrolls_val
                    objective_weights_val = [np.ones(num) / float(num)
                                             for num in partial_unroll_iters_val]
                    cost_total += _validate(
                        opt, k, num_unrolls_val, partial_unroll_iters_val, batch_size,
                        objective_weights_val, dataset, reset_rnn_params)
                cost = cost_total / evaluation_epochs
                print("evaluation {}, cost={}".format(
                    (k + 1) // evaluation_period, cost), flush=True)

                # _validate's last call left opt._params at that restart's
                # final point -- for a Lasso problem (fixed A/b/x_true, see
                # problems/problem_generator.py's pg.Lasso) this is exactly
                # the x_pred Experiment 1's modified relative loss (Eq 10)
                # needs. Overwritten every evaluation, so what's on disk
                # when training ends is the final evaluation's recovery.
                if logdir is not None and hasattr(problem, "x_true"):
                    np.savez(
                        os.path.join(logdir, "recovery.npz"),
                        x_pred=opt._params[0].numpy(),
                        x_true=problem.x_true,
                        b=problem.b.numpy())

                if logdir is not None and cost < best_evaluation:
                    best_evaluation = cost
                    opt.save(os.path.join(logdir, "model-best.l2o"))

            if logdir is not None and (k + 1) % save_period == 0:
                opt.save(os.path.join(logdir, "model-iter{}.l2o".format(k + 1)))

            optimization_time = time.time() - train_start_time
            print("  [{:02}] {}s, {} iters (unrolled {}){}".format(
                k, optimization_time, total_num_iter,
                ", ".join(str(s) for s in partial_unroll_iters), additional_log_info))

        if logdir is not None:
            opt.save(os.path.join(logdir, "model-final.l2o"))

    profiler.finish()


def test_optimizer(optimizer,
                   problem,
                   num_iter,
                   dataset=None,
                   batch_size=None,
                   seed=None,
                   record_every=None):
    """Tests an optimization algorithm on a given problem.

    Args:
      optimizer: A TrainableOptimizer instance (or anything implementing
                 apply_gradients(grads_and_vars)).
      problem: A Problem instance that defines an optimization problem to solve
      num_iter: The number of iterations of the optimizer to run
      dataset: The dataset to train the problem against
      batch_size: The number of samples per batch. If None (default), the
        batch size is set to the full batch (dataset.size)
      seed: A random seed used for drawing the initial parameters, or a list of
        numpy arrays used to explicitly initialize the parameters.
      record_every: if an integer, stores the parameters, objective, and gradient
                    every record_every iterations. If None, nothing is stored

    Returns:
      objective_values: A list of the objective values during optimization
      parameters: The parameters obtained after training
      records: A dictionary containing lists of the parameters and gradients
               during optimization saved every record_every iterations (empty if
               record_every is set to None)
    """
    if dataset is None:
        dataset = datasets.EMPTY_DATASET
        batch_size = dataset.size
    else:
        batch_size = dataset.size if batch_size is None else batch_size

    if isinstance(seed, (list, tuple)):
        params = problem_generator.init_fixed_variables(seed)
    else:
        params = problem.init_variables(seed)

    batch_inds = dataset.batch_indices(num_iter, batch_size)

    records = defaultdict(list)
    objective_values = []

    for itr, batch in enumerate(batch_inds):
        if itr % 1000 == 0:
            print("iteration {}".format(itr + 1), flush=True)

        batch_data = dataset.data[batch]
        batch_labels = dataset.labels[batch]

        with tf.GradientTape() as tape:
            obj = problem.objective(params, batch_data, batch_labels)
        gradients = problem.gradients(obj, params, tape)

        if record_every is not None and (itr % record_every) == 0:
            def grad_value(g):
                return g.values if isinstance(g, tf.IndexedSlices) else g

            gav = [(grad_value(g), v) for g, v in zip(gradients, params)]
            full_obj = problem.objective(params, dataset.data, dataset.labels)

            records["objective"].append(float(full_obj))
            records["grad_norm"].append([float(tf.norm(g)) for g, _ in gav])
            records["param_norm"].append([float(tf.norm(v)) for _, v in gav])
            records["grad"].append([g.numpy() for g, _ in gav])
            records["param"].append([v.numpy() for _, v in gav])
            records["iter"].append(itr)

        optimizer.apply_gradients(zip(gradients, params))
        objective_values.append(float(obj))

    parameters = [p.numpy() for p in params]
    return objective_values, parameters, records


def run_wall_clock_test(optimizer,
                        problem,
                        num_steps,
                        dataset=None,
                        seed=None,
                        batch_size=None):
    """Runs optimization with the given parameters and return average iter time.

    Args:
      optimizer: A TrainableOptimizer instance (or anything implementing
                 apply_gradients(grads_and_vars)).
      problem: The problem to optimize (a problem_generator.Problem)
      num_steps: The number of steps to run optimization for
      dataset: The dataset to train the problem against
      seed: The seed used for drawing the initial parameters, or a list of
        numpy arrays used to explicitly initialize the parameters
      batch_size: The number of samples per batch.

    Returns:
      The average time in seconds for a single optimization iteration.
    """
    if dataset is None:
        dataset = datasets.EMPTY_DATASET
        batch_size = dataset.size
    else:
        batch_size = dataset.size if batch_size is None else batch_size

    if isinstance(seed, (list, tuple)):
        params = problem_generator.init_fixed_variables(seed)
    else:
        params = problem.init_variables(seed)

    batch_inds = dataset.batch_indices(num_steps, batch_size)

    avg_iter_time = []
    for batch in batch_inds:
        batch_data = dataset.data[batch]
        batch_labels = dataset.labels[batch]

        start = time.time()
        with tf.GradientTape() as tape:
            obj = problem.objective(params, batch_data, batch_labels)
        gradients = problem.gradients(obj, params, tape)
        optimizer.apply_gradients(zip(gradients, params))
        avg_iter_time.append(time.time() - start)

    return np.median(np.array(avg_iter_time))
