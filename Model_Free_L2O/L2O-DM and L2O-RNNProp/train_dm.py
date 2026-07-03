# Copyright 2016 Google Inc.
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
"""Learning 2 Learn training — DM variant."""

import argparse
import os
import random
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from data_generator import data_loader
import meta_dm_train as meta
import util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", default=None)
    p.add_argument("--num_epochs", type=int, default=10000)
    p.add_argument("--evaluation_period", type=int, default=100)
    p.add_argument("--evaluation_epochs", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--unroll_length", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--second_derivatives", action="store_true")
    p.add_argument("--problem", default="mnist")
    p.add_argument("--if_scale", action="store_true")
    p.add_argument("--rd_scale_bound", type=float, default=3.0)
    p.add_argument("--if_cl", action="store_true")
    p.add_argument("--min_num_eval", type=int, default=3)
    p.add_argument("--if_mt", action="store_true")
    p.add_argument("--num_mt", type=int, default=1)
    p.add_argument("--optimizers", default="adam")
    p.add_argument("--mt_ratio", type=float, default=0.3)
    p.add_argument("--mt_ratios", default="0.0 0.1 0.3 0.3 0.3 0.3 0.3 0.3")
    p.add_argument("--k", type=int, default=1)
    return p.parse_args()


def main():
    FLAGS = parse_args()

    if FLAGS.if_cl:
        num_steps_sched = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        num_unrolls_sched = [ns // FLAGS.unroll_length for ns in num_steps_sched]
        num_unrolls_eval = num_unrolls_sched[1:]
        curriculum_idx = 0
    else:
        num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

    if FLAGS.save_path is not None:
        os.makedirs(FLAGS.save_path, exist_ok=True)

    # Problem + optimizer
    problem, net_config, net_assignments = util.get_config(FLAGS.problem)
    optimizer = meta.MetaOptimizer(FLAGS.num_mt, **net_config)
    meta_opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    # Initial problem setup to create networks
    x_vars, const_vars, loss_fn = problem()
    optimizer._setup(x_vars, net_assignments)
    state = optimizer.initial_state(x_vars)

    # Multi-task data loader
    if FLAGS.if_mt:
        mt_loader = data_loader(problem, optimizer._subsets, FLAGS.optimizers, FLAGS.unroll_length)
        if FLAGS.if_cl:
            mt_ratios = [float(r) for r in FLAGS.mt_ratios.split()]

    best_evaluation = float("inf")
    num_eval = 0
    improved = False
    mti = -1
    start_time = timer()

    for e in range(FLAGS.num_epochs):
        # Refresh problem each epoch (fresh random W, y, etc.)
        x_vars, const_vars, loss_fn = problem()
        state = optimizer.initial_state(x_vars)

        if FLAGS.if_mt:
            if FLAGS.if_cl:
                mt_ratio = (mt_ratios[curriculum_idx]
                            if curriculum_idx < len(mt_ratios)
                            else mt_ratios[-1])
            else:
                mt_ratio = FLAGS.mt_ratio
            if random.random() < mt_ratio:
                mti = (mti + 1) % FLAGS.num_mt
                task_i = mti
            else:
                task_i = -1
        else:
            task_i = -1

        if FLAGS.if_cl:
            num_unrolls_cur = num_unrolls_sched[curriculum_idx]
        else:
            num_unrolls_cur = num_unrolls

        # Scale augmentation
        if FLAGS.if_scale:
            r_scale = [np.exp(np.random.uniform(
                -FLAGS.rd_scale_bound, FLAGS.rd_scale_bound, size=v.shape))
                for v in x_vars]
            for v, rs in zip(x_vars, r_scale):
                v.assign(v.numpy() / rs)
            scale = [tf.constant(rs, dtype=tf.float32) for rs in r_scale]
        else:
            scale = None

        epoch_loss = 0.0

        if task_i == -1:
            for _ in range(num_unrolls_cur):
                loss, meta_grads, net_vars, x_final, state, _ = optimizer.unroll(
                    loss_fn, x_vars, state, FLAGS.unroll_length,
                    scale=scale,
                    second_derivatives=FLAGS.second_derivatives)
                meta_opt.apply_gradients(zip(meta_grads, net_vars))
                for var, val in zip(x_vars, x_final):
                    var.assign(val)
                epoch_loss = float(loss)
        else:
            data_e = mt_loader.get_data(
                task_i, num_unrolls_cur,
                rd_scale_bound=FLAGS.rd_scale_bound,
                if_scale=FLAGS.if_scale, mt_k=FLAGS.k)
            # Multi-task: use per-window data
            mt_state = [optimizer.initial_state(x_vars)[si]
                        for si in range(len(optimizer._subsets))]
            for w in range(num_unrolls_cur):
                inputs_mt = data_e["inputs"][w]
                labels_mt = data_e["labels"][w]
                loss, meta_grads, net_vars, mt_state = optimizer.unroll_mt(
                    inputs_mt, labels_mt, mt_state, FLAGS.unroll_length)
                meta_opt.apply_gradients(zip(meta_grads, net_vars))
                epoch_loss = float(loss)

        print("training_loss={}".format(epoch_loss))

        # Evaluation
        if (e + 1) % FLAGS.evaluation_period == 0 or e + 1 == FLAGS.num_epochs:
            if FLAGS.if_cl:
                num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
            else:
                num_unrolls_eval_cur = num_unrolls
            num_eval += 1

            eval_cost = 0.0
            for _ in range(FLAGS.evaluation_epochs):
                ev_x_vars, _, ev_loss_fn = problem()
                ev_state = optimizer.initial_state(ev_x_vars)
                for _ in range(num_unrolls_eval_cur):
                    with tf.GradientTape() as tape:
                        tape.watch([tf.identity(v) for v in ev_x_vars])
                    loss_val, _, _, ev_x_final, ev_state, _ = optimizer.unroll(
                        ev_loss_fn, ev_x_vars, ev_state, FLAGS.unroll_length)
                    for var, val in zip(ev_x_vars, ev_x_final):
                        var.assign(val)
                    eval_cost += float(loss_val)

            if FLAGS.if_cl:
                num_steps_cur = num_steps_sched[curriculum_idx]
            else:
                num_steps_cur = FLAGS.num_steps
            print("epoch={}, num_steps={}, eval_loss={}".format(
                e, num_steps_cur, eval_cost / FLAGS.evaluation_epochs), flush=True)

            if not FLAGS.if_cl:
                if eval_cost < best_evaluation:
                    best_evaluation = eval_cost
                    optimizer.save(FLAGS.save_path, e + 1)
                    optimizer.save(FLAGS.save_path, 0)
                    print("Saving optimizer of epoch {}...".format(e + 1))
                continue

            # Curriculum learning update
            if eval_cost < best_evaluation:
                best_evaluation = eval_cost
                improved = True
                optimizer.save(FLAGS.save_path, curriculum_idx)
                optimizer.save(FLAGS.save_path, 0)
            elif num_eval >= FLAGS.min_num_eval and improved:
                optimizer.restore(FLAGS.save_path, curriculum_idx)
                num_eval = 0
                improved = False
                curriculum_idx += 1
                if curriculum_idx >= len(num_unrolls_sched):
                    curriculum_idx = len(num_unrolls_sched) - 1

                # Re-evaluate at new curriculum level
                eval_cost = 0.0
                for _ in range(FLAGS.evaluation_epochs):
                    ev_x_vars, _, ev_loss_fn = problem()
                    ev_state = optimizer.initial_state(ev_x_vars)
                    for _ in range(num_unrolls_eval[curriculum_idx]):
                        loss_val, _, _, ev_x_final, ev_state, _ = optimizer.unroll(
                            ev_loss_fn, ev_x_vars, ev_state, FLAGS.unroll_length)
                        for var, val in zip(ev_x_vars, ev_x_final):
                            var.assign(val)
                        eval_cost += float(loss_val)
                best_evaluation = eval_cost
                print("epoch={}, num_steps={}, eval loss={}".format(
                    e, num_steps_sched[curriculum_idx],
                    eval_cost / FLAGS.evaluation_epochs), flush=True)
            elif num_eval >= FLAGS.min_num_eval and not improved:
                print("no improve during curriculum {} --> stop".format(curriculum_idx))
                break

    print("total time = {}s...".format(timer() - start_time))


if __name__ == "__main__":
    main()
