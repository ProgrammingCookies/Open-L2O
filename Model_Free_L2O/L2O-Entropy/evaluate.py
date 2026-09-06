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
                   help="If given, evaluate on just this seed instead of the default "
                        "sweep [6, 12, 18, 24, 30].")
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

    if not problems_and_data:
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

    for problem_itr, (problem_spec_, dataset, batch_size) in enumerate(problems_and_data):
        problem = problem_spec_.build()
        problem_name = FLAGS.train_dir.split("/")[0]

        for seed in ([FLAGS.seed] if FLAGS.seed is not None else [6, 12, 18, 24, 30]):
            print("testing problem {} ({}) using seed {}".format(
                problem_itr, problem_name, seed))

            if FLAGS.test_optimizer == "L2o":
                opt = _build_l2o_optimizer(FLAGS)
                if os.path.exists(checkpoint_path):
                    # Trigger a dummy step to build every sub-layer's weights
                    # (BiasGRUCell's gate/candidate affines build lazily on
                    # first call), then load the saved ones.
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
                else:
                    print("WARNING: no checkpoint found at {}, evaluating untrained optimizer".format(
                        checkpoint_path))
            else:
                opt = BASELINE_OPTIMIZERS[FLAGS.test_optimizer]()

            objective_values, _, _ = metaopt.test_optimizer(
                opt, problem, num_iter=FLAGS.num_testing_itrs,
                dataset=dataset, batch_size=batch_size, seed=seed)

            out_path = os.path.join(
                FLAGS.save_dir,
                "seed{}_eval_loss_record.pickle-{}".format(seed, FLAGS.model_name))
            with open(out_path, "wb") as f:
                pickle.dump(objective_values, f)


if __name__ == "__main__":
    main()
