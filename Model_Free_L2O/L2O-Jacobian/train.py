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

"""Scripts for meta-optimization (training)."""

import argparse
import os

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

# The size of the RNN hidden state in each layer: [PerParam, PerTensor, Global].
# The length of this list must be 1, 2, or 3. If less than 3, the Global
# and/or PerTensor RNNs will not be created. Used for HierarchicalRNN's
# level_sizes and CoordinatewiseRNN's cell_sizes.
HRNN_CELL_SIZES = [10, 20, 20]

CELL_CLASSES = {
    "GRUCell": tf.keras.layers.GRUCell,
    "LSTMCell": tf.keras.layers.LSTMCell,
    "BiasGRUCell": rnn_cells.BiasGRUCell,
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
    p.add_argument("--train_dir", default="/tmp/lol/")
    p.add_argument("--num_problems", type=int, default=1)
    p.add_argument("--num_meta_iterations", type=int, default=5)
    p.add_argument("--num_unroll_scale", type=int, default=40)
    p.add_argument("--min_num_unrolls", type=int, default=1)
    p.add_argument("--num_partial_unroll_itr_scale", type=int, default=200)
    p.add_argument("--min_num_itr_partial_unroll", type=int, default=50)
    p.add_argument("--optimizer", default="HierarchicalRNN", choices=list(register_optimizers()))

    # CoordinatewiseRNN-specific flags
    p.add_argument("--cell_size", type=int, default=20)
    p.add_argument("--num_cells", type=int, default=2)
    p.add_argument("--cell_cls", default="GRUCell", choices=list(CELL_CLASSES))

    # Metaoptimization parameters
    p.add_argument("--meta_learning_rate", type=float, default=1e-6)
    p.add_argument("--gradient_clip_level", type=float, default=1e4)

    # Training set selection
    p.add_argument("--include_mnist_mlp_problems", action="store_true")
    p.add_argument("--include_quadratic_problems", action="store_true")
    p.add_argument("--include_noisy_quadratic_problems", action="store_true")
    p.add_argument("--include_large_quadratic_problems", action="store_true")
    p.add_argument("--include_bowl_problems", action="store_true")
    p.add_argument("--include_softmax_2_class_problems", action="store_true")
    p.add_argument("--include_noisy_softmax_2_class_problems", action="store_true")
    p.add_argument("--include_optimization_test_problems", action="store_true")
    p.add_argument("--include_noisy_optimization_test_problems", action="store_true")
    p.add_argument("--include_fully_connected_random_2_class_problems", action="store_true")
    p.add_argument("--include_matmul_problems", action="store_true")
    p.add_argument("--include_log_objective_problems", action="store_true")
    p.add_argument("--include_rescale_problems", action="store_true")
    p.add_argument("--include_norm_problems", action="store_true")
    p.add_argument("--include_sum_problems", action="store_true")
    p.add_argument("--include_sparse_gradient_problems", action="store_true")
    p.add_argument("--include_sparse_softmax_problems", action="store_true")
    p.add_argument("--include_one_hot_sparse_softmax_problems", action="store_true")
    p.add_argument("--include_noisy_bowl_problems", action="store_true")
    p.add_argument("--include_noisy_norm_problems", action="store_true")
    p.add_argument("--include_noisy_sum_problems", action="store_true")
    p.add_argument("--include_sum_of_quadratics_problems", action="store_true")
    p.add_argument("--include_projection_quadratic_problems", action="store_true")
    p.add_argument("--include_outward_snake_problems", action="store_true")
    p.add_argument("--include_dependency_chain_problems", action="store_true")
    p.add_argument("--include_min_max_well_problems", action="store_true")

    # HALO adaptation problem sets
    p.add_argument("--adapt_mnist_conv_problems", action="store_true")
    p.add_argument("--adapt_mnist_conv_problems_wide", action="store_true")
    p.add_argument("--adapt_cifar10_conv_problems", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)

    # Optimizer parameters: initialization and scale values
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_lr", type=float, default=1e-2)

    # Optimizer parameters: small features
    p.add_argument("--zero_init_lr_weights", type=_bool, default=True)
    p.add_argument("--use_relative_lr", type=_bool, default=True)
    p.add_argument("--use_extreme_indicator", type=_bool, default=False)
    p.add_argument("--use_log_means_squared", type=_bool, default=True)
    p.add_argument("--use_problem_lr_mean", type=_bool, default=True)

    # Optimizer parameters: major features
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

    p.add_argument("--fix_unroll", action="store_true")
    p.add_argument("--fix_unroll_length", type=int, default=20)
    p.add_argument("--fix_num_steps", type=int, default=100)
    p.add_argument("--fix_num_steps_eval", type=int, default=100)
    p.add_argument("--evaluation_period", type=int, default=1)
    p.add_argument("--evaluation_epochs", type=int, default=20)
    p.add_argument("--save_period", type=int, default=1)

    # Hessian/Jacobian flatness regularization
    p.add_argument("--reg_option", default="hessian",
                   choices=["hessian", "hessian-ev", "hessian-esd", "jacob"])
    p.add_argument("--hessian_itrs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=5e-4)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--reg_optimizer", type=_bool, default=False)
    p.add_argument("--reg_optimizee", type=_bool, default=False)
    p.add_argument("--regularize_time", default="posterior",
                   choices=["posterior", "prior", "none"])
    p.add_argument("--reg_scale", type=float, default=0.5)

    # HALO: auxiliary convex regularization, pretrained-model warm-start,
    # hardware-aware sparse update
    p.add_argument("--use_aux_convex_l1", type=_bool, default=False)
    p.add_argument("--use_aux_convex_l2", type=_bool, default=False)
    p.add_argument("--aux_convex_dim", type=int, default=20)
    p.add_argument("--aux_convex_ratio", type=float, default=1.0)
    p.add_argument("--convex_l1_ratio", type=float, default=0.1)
    p.add_argument("--pretrained_model_path", default=None)
    p.add_argument("--random_sparse_method", default="layer_wise",
                   choices=["layer_wise", "params_wise", "fix_num"])
    p.add_argument("--random_sparse_prob", default="1.0",
                   help="Space-separated list of per-tensor keep-probabilities, e.g. \"0.1 0.3 0.5\".")
    return p.parse_args()


def main():
    FLAGS = parse_args()
    opts = register_optimizers()

    # Choose a set of problems to optimize. By default this includes quadratics,
    # 2-dimensional bowls, 2-class softmax problems, and non-noisy optimization
    # test problems (e.g. Rosenbrock, Beale)
    problems_and_data = []

    if FLAGS.include_mnist_mlp_problems:
        problems_and_data.extend(ps.mnist_mlp_problems())
    if FLAGS.include_sparse_softmax_problems:
        problems_and_data.extend(ps.sparse_softmax_2_class_sparse_problems())
    if FLAGS.include_one_hot_sparse_softmax_problems:
        problems_and_data.extend(ps.one_hot_sparse_softmax_2_class_sparse_problems())
    if FLAGS.include_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems())
    if FLAGS.include_noisy_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_noisy())
    if FLAGS.include_large_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_large())
    if FLAGS.include_bowl_problems:
        problems_and_data.extend(ps.bowl_problems())
    if FLAGS.include_noisy_bowl_problems:
        problems_and_data.extend(ps.bowl_problems_noisy())
    if FLAGS.include_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems())
    if FLAGS.include_noisy_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems_noisy())
    if FLAGS.include_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems())
    if FLAGS.include_noisy_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems_noisy())
    if FLAGS.include_fully_connected_random_2_class_problems:
        problems_and_data.extend(ps.fully_connected_random_2_class_problems())
    if FLAGS.include_matmul_problems:
        problems_and_data.extend(ps.matmul_problems())
    if FLAGS.include_log_objective_problems:
        problems_and_data.extend(ps.log_objective_problems())
    if FLAGS.include_rescale_problems:
        problems_and_data.extend(ps.rescale_problems())
    if FLAGS.include_norm_problems:
        problems_and_data.extend(ps.norm_problems())
    if FLAGS.include_noisy_norm_problems:
        problems_and_data.extend(ps.norm_problems_noisy())
    if FLAGS.include_sum_problems:
        problems_and_data.extend(ps.sum_problems())
    if FLAGS.include_noisy_sum_problems:
        problems_and_data.extend(ps.sum_problems_noisy())
    if FLAGS.include_sparse_gradient_problems:
        problems_and_data.extend(ps.sparse_gradient_problems())
        if FLAGS.include_fully_connected_random_2_class_problems:
            problems_and_data.extend(ps.sparse_gradient_problems_mlp())
    if FLAGS.include_min_max_well_problems:
        problems_and_data.extend(ps.min_max_well_problems())
    if FLAGS.include_sum_of_quadratics_problems:
        problems_and_data.extend(ps.sum_of_quadratics_problems())
    if FLAGS.include_projection_quadratic_problems:
        problems_and_data.extend(ps.projection_quadratic_problems())
    if FLAGS.include_outward_snake_problems:
        problems_and_data.extend(ps.outward_snake_problems())
    if FLAGS.include_dependency_chain_problems:
        problems_and_data.extend(ps.dependency_chain_problems())

    use_aux_convex = FLAGS.use_aux_convex_l1 or FLAGS.use_aux_convex_l2
    if FLAGS.adapt_mnist_conv_problems:
        problems_and_data.extend(ps.adapt_mnist_conv_problems(
            use_aux_convex=use_aux_convex, aux_convex_dim=FLAGS.aux_convex_dim,
            aux_convex_ratio=FLAGS.aux_convex_ratio, batch_size=FLAGS.batch_size))
    if FLAGS.adapt_mnist_conv_problems_wide:
        problems_and_data.extend(ps.adapt_mnist_conv_problems_wide(
            use_aux_convex=use_aux_convex, aux_convex_dim=FLAGS.aux_convex_dim,
            aux_convex_ratio=FLAGS.aux_convex_ratio, batch_size=FLAGS.batch_size))
    if FLAGS.adapt_cifar10_conv_problems:
        problems_and_data.extend(ps.adapt_cifar10_conv_problems(
            use_aux_convex=use_aux_convex, aux_convex_dim=FLAGS.aux_convex_dim,
            aux_convex_ratio=FLAGS.aux_convex_ratio, batch_size=FLAGS.batch_size))

    if not problems_and_data:
        raise ValueError("No problems selected -- pass at least one --include_*_problems flag.")

    logdir = os.path.join(FLAGS.train_dir, "{}_{}_{}_{}".format(
        FLAGS.optimizer, FLAGS.cell_cls, FLAGS.cell_size, FLAGS.num_cells))

    optimizer_cls = opts[FLAGS.optimizer]

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
        "alpha": FLAGS.alpha,
        "beta": FLAGS.beta,
        "reg_optimizer": FLAGS.reg_optimizer,
        "reg_optimizee": FLAGS.reg_optimizee,
        "reg_option": FLAGS.reg_option,
        "hessian_itrs": FLAGS.hessian_itrs,
        "use_aux_convex_l1": FLAGS.use_aux_convex_l1,
        "use_aux_convex_l2": FLAGS.use_aux_convex_l2,
        "aux_convex_dim": FLAGS.aux_convex_dim,
        "aux_convex_ratio": FLAGS.aux_convex_ratio,
        "convex_l1_ratio": FLAGS.convex_l1_ratio,
        "pretrained_model_path": FLAGS.pretrained_model_path,
    }

    if FLAGS.optimizer in ("HierarchicalRNN", "CoordinatewiseRNN"):
        # Both take a list of per-level/per-cell hidden-state sizes as their
        # first positional arg. The other optimizers (GlobalLearningRate,
        # LearningRateSchedule, TrainableAdam) don't take a cell-sizes list at
        # all -- the original script passed HRNN_CELL_SIZES positionally to
        # every optimizer class regardless, which only actually worked for
        # HierarchicalRNN/CoordinatewiseRNN (for the others it would bind to
        # an unrelated first parameter, e.g. GlobalLearningRate's
        # initial_rate). Fixed here rather than replicated.
        optimizer_args = (HRNN_CELL_SIZES,)
        optimizer_kwargs["cell_cls"] = CELL_CLASSES[FLAGS.cell_cls]
    else:
        optimizer_args = ()

    if FLAGS.optimizer == "HierarchicalRNN":
        # random_sparse_method/prob are only meaningful for HierarchicalRNN's
        # own sparse-update mechanism (see optimizer/hierarchical_rnn.py) --
        # not accepted by the other optimizer classes.
        optimizer_kwargs["random_sparse_method"] = FLAGS.random_sparse_method
        optimizer_kwargs["random_sparse_prob"] = [
            float(p) for p in FLAGS.random_sparse_prob.split(" ")]

    optimizer_spec = problem_spec.Spec(optimizer_cls, optimizer_args, optimizer_kwargs)

    os.makedirs(logdir, exist_ok=True)

    def num_unrolls():
        return metaopt.sample_numiter(FLAGS.num_unroll_scale, FLAGS.min_num_unrolls)

    def num_partial_unroll_itrs():
        return metaopt.sample_numiter(
            FLAGS.num_partial_unroll_itr_scale, FLAGS.min_num_itr_partial_unroll)

    metaopt.train_optimizer(
        logdir,
        optimizer_spec,
        problems_and_data,
        FLAGS.num_problems,
        FLAGS.num_meta_iterations,
        num_unrolls,
        num_partial_unroll_itrs,
        learning_rate=FLAGS.meta_learning_rate,
        gradient_clip=FLAGS.gradient_clip_level,
        select_random_problems=True,
        obj_train_max_multiplier=FLAGS.objective_training_max_multiplier,
        fix_unroll=FLAGS.fix_unroll,
        fix_unroll_length=FLAGS.fix_unroll_length,
        fix_num_steps=FLAGS.fix_num_steps,
        fix_num_steps_eval=FLAGS.fix_num_steps_eval,
        evaluation_period=FLAGS.evaluation_period,
        evaluation_epochs=FLAGS.evaluation_epochs,
        save_period=FLAGS.save_period,
        regularize_time=FLAGS.regularize_time,
        reg_scale=FLAGS.reg_scale)


if __name__ == "__main__":
    main()
