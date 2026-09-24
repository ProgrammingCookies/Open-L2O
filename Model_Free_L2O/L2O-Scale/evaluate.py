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

"""Scripts for meta-optimization (evaluation)."""

import argparse
import os
import pickle
import random

import numpy as np
import tensorflow as tf

import metaopt
from optimizer import coordinatewise_rnn
from optimizer import global_learning_rate
from optimizer import hierarchical_rnn
from optimizer import learning_rate_schedule
from optimizer import rnn_cells
from optimizer import trainable_adam
from problems import problem_sets as ps
from problems import problem_spec

HRNN_CELL_SIZES = [10, 20, 20]

CELL_CLASSES = {
    "GRUCell": tf.keras.layers.GRUCell,
    "LSTMCell": tf.keras.layers.LSTMCell,
    "BiasGRUCell": rnn_cells.BiasGRUCell,
}

BASELINE_OPTIMIZERS = {
    "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=0.001),
    "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=0.0001),
    "Adagrad": lambda: tf.keras.optimizers.Adagrad(learning_rate=0.00001),
}


def register_optimizers():
    return {
        "CoordinatewiseRNN": coordinatewise_rnn.CoordinatewiseRNN,
        "GlobalLearningRate": global_learning_rate.GlobalLearningRate,
        "HierarchicalRNN": hierarchical_rnn.HierarchicalRNN,
        "LearningRateSchedule": learning_rate_schedule.LearningRateSchedule,
        "TrainableAdam": trainable_adam.TrainableAdam,
    }


def _bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value (true/false)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="opt/", help="Directory holding the trained checkpoint.")
    p.add_argument("--save_dir", default="opt/", help="Directory to write evaluation results.")
    p.add_argument("--test_optimizer", default="L2o", choices=["L2o", "SGD", "Adam", "Adagrad"])
    p.add_argument("--num_testing_itrs", type=int, default=100)
    p.add_argument("--optimizer", default="HierarchicalRNN", choices=list(register_optimizers()))
    p.add_argument("--cell_size", type=int, default=20)
    p.add_argument("--num_cells", type=int, default=2)
    p.add_argument("--cell_cls", default="GRUCell", choices=list(CELL_CLASSES))

    p.add_argument("--include_mnist_mlp_problems", action="store_true")
    p.add_argument("--include_mnist_mlp_relu_problems", action="store_true")
    p.add_argument("--include_mnist_mlp_deeper_problems", action="store_true")
    p.add_argument("--include_mnist_conv_problems", action="store_true")
    p.add_argument("--include_mnist_conv_orig_problems", action="store_true")
    p.add_argument("--include_cifar10_conv_problems", action="store_true")
    p.add_argument("--include_lasso_problems", action="store_true",
                   help="Evaluate on a pre-generated LASSO dataset split (see "
                        "Benchmarking/data/lasso.py): one deterministic pass over "
                        "every row of --lasso_split, --lasso_batch_size rows at a "
                        "time, from the split's seeded x0 (--lasso_x0_mode aligned). "
                        "Replaces the [6, 12, 18, 24, 30] seed sweep.")
    p.add_argument("--lasso_data_dir", default=None,
                   help="Directory holding A.npy + split .npy files. Required with "
                        "--include_lasso_problems.")
    p.add_argument("--lasso_split", default="val_data.npy")
    p.add_argument("--lasso_batch_size", type=int, default=128,
                   help="Rows per chunk. Use the TRAINING batch size: the "
                        "tensor-level RNN input and the output scaling pool over "
                        "the whole (batch, n) tensor, so the chunk size changes the "
                        "statistics the optimizer sees.")
    p.add_argument("--lasso_lam", type=float, default=0.005)
    p.add_argument("--lasso_x0_mode", default="aligned", choices=["aligned", "random"])

    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_lr", type=float, default=1e-2)
    p.add_argument("--zero_init_lr_weights", type=_bool, default=True)
    p.add_argument("--use_relative_lr", type=_bool, default=True)
    p.add_argument("--use_extreme_indicator", type=_bool, default=False)
    p.add_argument("--use_log_means_squared", type=_bool, default=True)
    p.add_argument("--use_problem_lr_mean", type=_bool, default=True)
    p.add_argument("--learnable_decay", type=_bool, default=True)
    p.add_argument("--dynamic_output_scale", type=_bool, default=True)
    p.add_argument("--use_log_objective", type=_bool, default=True)
    p.add_argument("--use_attention", type=_bool, default=False)
    p.add_argument("--use_second_derivatives", type=_bool, default=True)
    p.add_argument("--num_gradient_scales", type=int, default=4)
    p.add_argument("--max_log_lr", type=float, default=33)
    p.add_argument("--objective_training_max_multiplier", type=float, default=-1)
    p.add_argument("--use_gradient_shortcut", type=_bool, default=True)
    p.add_argument("--use_lr_shortcut", type=_bool, default=False)
    p.add_argument("--use_grad_products", type=_bool, default=True)
    p.add_argument("--use_multiple_scale_decays", type=_bool, default=False)
    p.add_argument("--use_numerator_epsilon", type=_bool, default=False)
    p.add_argument("--learnable_inp_decay", type=_bool, default=True)
    p.add_argument("--learnable_rnn_init", type=_bool, default=True)

    p.add_argument("--model_name", default="mt")
    p.add_argument("--restore_model_name", default="model-final.l2o",
                   help="Checkpoint filename (relative to --train_dir/<optimizer dir>) to load.")
    p.add_argument("--seed", type=int, default=None,
                   help="If given, seeds random/numpy/TF (reproducible lr-prior "
                        "init and starting points) and evaluates on just this "
                        "seed instead of the default sweep [6, 12, 18, 24, 30].")
    return p.parse_args()


def _build_l2o_optimizer(FLAGS):
    optimizer_cls = register_optimizers()[FLAGS.optimizer]
    optimizer_kwargs = {
        "init_lr_range": (FLAGS.min_lr, FLAGS.max_lr),
        "learnable_decay": FLAGS.learnable_decay,
        "dynamic_output_scale": FLAGS.dynamic_output_scale,
        "use_attention": FLAGS.use_attention,
        "use_log_objective": FLAGS.use_log_objective,
        "num_gradient_scales": FLAGS.num_gradient_scales,
        "zero_init_lr_weights": FLAGS.zero_init_lr_weights,
        "use_log_means_squared": FLAGS.use_log_means_squared,
        "use_relative_lr": FLAGS.use_relative_lr,
        "use_extreme_indicator": FLAGS.use_extreme_indicator,
        "max_log_lr": FLAGS.max_log_lr,
        "obj_train_max_multiplier": FLAGS.objective_training_max_multiplier,
        "use_problem_lr_mean": FLAGS.use_problem_lr_mean,
        "use_gradient_shortcut": FLAGS.use_gradient_shortcut,
        "use_second_derivatives": FLAGS.use_second_derivatives,
        "use_lr_shortcut": FLAGS.use_lr_shortcut,
        "use_grad_products": FLAGS.use_grad_products,
        "use_multiple_scale_decays": FLAGS.use_multiple_scale_decays,
        "use_numerator_epsilon": FLAGS.use_numerator_epsilon,
        "learnable_inp_decay": FLAGS.learnable_inp_decay,
        "learnable_rnn_init": FLAGS.learnable_rnn_init,
    }
    if FLAGS.optimizer in ("HierarchicalRNN", "CoordinatewiseRNN"):
        optimizer_args = (HRNN_CELL_SIZES,)
        optimizer_kwargs["cell_cls"] = CELL_CLASSES[FLAGS.cell_cls]
    else:
        optimizer_args = ()
    return problem_spec.Spec(optimizer_cls, optimizer_args, optimizer_kwargs).build()


def main():
    FLAGS = parse_args()

    # Same three RNGs train.py seeds. Without this the evaluation is not
    # reproducible: HierarchicalRNN draws every coordinate's initial log-lr
    # with tf.random.uniform(seed=None) each time an optimizer is built
    # (hierarchical_rnn.py's _initialize_state), so two runs of the same
    # checkpoint on the same data would start from different lr priors.
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        tf.random.set_seed(FLAGS.seed)

    problems_and_data = []
    if FLAGS.include_mnist_mlp_problems:
        problems_and_data.extend(ps.test_mnist_mlp_problems())
    if FLAGS.include_mnist_mlp_relu_problems:
        problems_and_data.extend(ps.test_mnist_mlp_relu_problems())
    if FLAGS.include_mnist_mlp_deeper_problems:
        problems_and_data.extend(ps.test_mnist_mlp_deeper_problems())
    if FLAGS.include_mnist_conv_problems:
        problems_and_data.extend(ps.test_mnist_conv_problems())
    if FLAGS.include_mnist_conv_orig_problems:
        problems_and_data.extend(ps.test_mnist_conv_orig_problems())
    if FLAGS.include_cifar10_conv_problems:
        problems_and_data.extend(ps.test_cifar10_conv_problems())

    if FLAGS.include_lasso_problems:
        if FLAGS.lasso_data_dir is None:
            raise ValueError("--include_lasso_problems requires --lasso_data_dir")
        if problems_and_data:
            raise ValueError("--include_lasso_problems cannot be combined with other "
                             "--include_*_problems flags (its evaluation loop differs)")

    if not problems_and_data and not FLAGS.include_lasso_problems:
        raise ValueError("No problems selected -- pass at least one --include_*_problems flag.")

    logdir = os.path.join(FLAGS.train_dir, "{}_{}_{}_{}".format(
        FLAGS.optimizer, FLAGS.cell_cls, FLAGS.cell_size, FLAGS.num_cells))

    if FLAGS.test_optimizer == "L2o":
        print("using optimizer L2o")
        checkpoint_path = os.path.join(logdir, FLAGS.restore_model_name)
    elif FLAGS.test_optimizer in BASELINE_OPTIMIZERS:
        print("using optimizer {}".format(FLAGS.test_optimizer))
        checkpoint_path = None
    else:
        raise ValueError("{} is not a valid test_optimizer".format(FLAGS.test_optimizer))

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    def make_test_optimizer(problem, dataset, seed):
        """A fresh optimizer per evaluation run: the L2O one rebuilt and
        reloaded from checkpoint_path (a dummy step first builds every
        sub-layer's weights -- BiasGRUCell's gate/candidate affines build
        lazily on first call), or a baseline optimizer."""
        if FLAGS.test_optimizer != "L2o":
            return BASELINE_OPTIMIZERS[FLAGS.test_optimizer]()
        opt = _build_l2o_optimizer(FLAGS)
        if os.path.exists(checkpoint_path):
            dummy_params = problem.init_variables(seed)
            if dataset is not None:
                dummy_data = dataset.data[:2]
                dummy_labels = dataset.labels[:2]
            else:
                dummy_data = dummy_labels = None
            with tf.GradientTape() as tape:
                dummy_obj = problem.objective(dummy_params, dummy_data, dummy_labels)
            dummy_grads = problem.gradients(dummy_obj, dummy_params, tape)
            opt.apply_gradients(zip(dummy_grads, dummy_params))
            opt.load(checkpoint_path)
            # Killing dummy variable so it is not silently inherited by the real run
            for v in dummy_params:
                opt._slots.pop(id(v), None)
        else:
            print("WARNING: no checkpoint found at {}, evaluating untrained optimizer".format(
                checkpoint_path))
        return opt

    if FLAGS.include_lasso_problems:
        _evaluate_lasso_split(FLAGS, make_test_optimizer)
        return

    for problem_itr, (problem_spec_, dataset, batch_size) in enumerate(problems_and_data):
        problem = problem_spec_.build()
        problem_name = FLAGS.train_dir.split("/")[0]

        for seed in ([FLAGS.seed] if FLAGS.seed is not None else [6, 12, 18, 24, 30]):
            print("testing problem {} ({}) using seed {}".format(
                problem_itr, problem_name, seed))

            opt = make_test_optimizer(problem, dataset, seed)

            objective_values, _, _ = metaopt.test_optimizer(
                opt, problem, num_iter=FLAGS.num_testing_itrs,
                dataset=dataset, batch_size=batch_size, seed=seed)

            out_path = os.path.join(
                FLAGS.save_dir,
                "seed{}_eval_loss_record.pickle-{}".format(seed, FLAGS.model_name))
            with open(out_path, "wb") as f:
                pickle.dump(objective_values, f)


def _evaluate_lasso_split(FLAGS, make_test_optimizer):
    """One deterministic pass over every row of --lasso_split, in file order,
    --lasso_batch_size rows at a time, each chunk optimized from its seeded x0
    with a freshly (re)loaded optimizer. Mirrors evaluate_dm.py's
    lasso_dataset loop in L2O-DM and L2O-RNNProp so the four methods are scored
    on identical instances and starting points.

    Writes, under --save_dir:
      seed{seed}_eval_loss_record.pickle-<model_name>: per-step objective,
        each step averaged over every test instance (same layout as the
        other problems' records; one file, since the split IS the sweep);
      L2o_recovery-lasso_dataset.npz: x_pred / x_true / b / x0 for the whole split,
        row-aligned with the split file (same keys as evaluate_dm.py's).
    """
    from problems import problem_generator as pg

    data_path = os.path.join(FLAGS.lasso_data_dir, FLAGS.lasso_split)
    num_samples = np.load(data_path, mmap_mode="r").shape[0]
    seed = FLAGS.seed if FLAGS.seed is not None else 0
    chunk = FLAGS.lasso_batch_size

    sum_costs = None
    x_pred_chunks, x_true_chunks, b_chunks, x0_chunks = [], [], [], []
    for start in range(0, num_samples, chunk):
        idx = np.arange(start, min(start + chunk, num_samples))
        problem = pg.Lasso(FLAGS.lasso_data_dir, split=FLAGS.lasso_split,
                           lam=FLAGS.lasso_lam, indices=idx, x0_mode=FLAGS.lasso_x0_mode)
        print("testing lasso rows {}-{} of {} using seed {}".format(
            idx[0], idx[-1], num_samples, seed), flush=True)
        opt = make_test_optimizer(problem, None, seed)
        objective_values, parameters, _ = metaopt.test_optimizer(
            opt, problem, num_iter=FLAGS.num_testing_itrs, seed=seed)
        costs = np.asarray(objective_values, dtype=np.float64)
        sum_costs = costs if sum_costs is None else sum_costs + costs
        x_pred_chunks.append(parameters[0])
        x_true_chunks.append(problem.x_true)
        b_chunks.append(problem.b.numpy())
        x0_chunks.append(problem.last_x0)

    num_chunks = len(x_pred_chunks)
    loss_record = (sum_costs / num_chunks).tolist()
    out_path = os.path.join(
        FLAGS.save_dir, "seed{}_eval_loss_record.pickle-{}".format(seed, FLAGS.model_name))
    with open(out_path, "wb") as f:
        pickle.dump(loss_record, f)
    print("Saving evaluate loss record {}".format(out_path))

    recovery_file = os.path.join(FLAGS.save_dir, "L2o_recovery-lasso_dataset.npz")
    np.savez(recovery_file,
             x_pred=np.concatenate(x_pred_chunks, axis=0),
             x_true=np.concatenate(x_true_chunks, axis=0),
             b=np.concatenate(b_chunks, axis=0),
             x0=np.concatenate(x0_chunks, axis=0))
    print("Saving recovered signal + ground truth {}".format(recovery_file))


if __name__ == "__main__":
    main()
