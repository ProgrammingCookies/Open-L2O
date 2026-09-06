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
"""Learning 2 Learn training — RNNprop variant."""

import argparse
import os
import random
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from data_generator import data_loader
import meta_rnnprop_train as meta
import profiling
import util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", default=None)
    p.add_argument("--num_epochs", type=int, default=10000)
    p.add_argument("--evaluation_period", type=int, default=10)
    p.add_argument("--evaluation_epochs", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--unroll_length", type=int, default=20)
    p.add_argument("--fix_num_unrolls", type=int, default=None,
                   help="If set, overrides num_steps//unroll_length: total "
                        "horizon becomes unroll_length * fix_num_unrolls "
                        "instead of being truncated by integer division of "
                        "a fixed num_steps.")
    p.add_argument("--profile_path", default=None,
                   help="If set, write a per-epoch JSONL trajectory here "
                        "plus a summary '<stem>_summary.json'.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seeds random/numpy/TF RNGs before problem/optimizer "
                        "construction for reproducible training.")
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--second_derivatives", action="store_true")
    p.add_argument("--last_step_loss", action="store_true",
                   help="Train against the FINAL unroll step's loss only "
                        "(w_T=1, w_t=0 otherwise), matching Lv, Jiang & Li "
                        "2017's own stated RNNprop objective, instead of this script's prior default of "
                        "summing the loss over every step.")
    p.add_argument("--beta1", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--problem", default="mnist")
    p.add_argument("--if_scale", action="store_true")
    p.add_argument("--rd_scale_bound", type=float, default=3.0)
    p.add_argument("--if_cl", action="store_true")
    p.add_argument("--min_num_eval", type=int, default=3)
    p.add_argument("--if_mt", action="store_true")
    p.add_argument("--num_mt", type=int, default=1)
    p.add_argument("--optimizers", default="adam")
    p.add_argument("--mt_ratio", type=float, default=0.3)
    p.add_argument("--mt_ratios", default="0.3 0.3 0.3")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--lasso_data_dir", default=None,
                   help="Directory holding A.npy + split .npy files from "
                        "Benchmarking/data/lasso.py. Required when --problem=lasso_dataset.")
    p.add_argument("--lasso_split", default="train_data.npy")
    p.add_argument("--lasso_batch_size", type=int, default=128)
    p.add_argument("--lasso_lam", type=float, default=0.005)
    p.add_argument("--lasso_x0_mode", default="aligned", choices=["aligned", "random"],
                   help="'aligned' (default) requires the dataset's seeded x0 "
                        "sibling files. 'random' draws a fresh N(0,x0_stddev^2) "
                        "init each call instead so no x0 files required.")
    return p.parse_args()


def main():
    FLAGS = parse_args()
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        tf.random.set_seed(FLAGS.seed)

    if FLAGS.if_cl:
        num_steps_sched = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        num_unrolls_sched = [ns // FLAGS.unroll_length for ns in num_steps_sched]
        num_unrolls_eval = num_unrolls_sched[1:]
        curriculum_idx = 0
    elif FLAGS.fix_num_unrolls is not None:
        num_unrolls = FLAGS.fix_num_unrolls
    else:
        num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

    if FLAGS.save_path is not None:
        os.makedirs(FLAGS.save_path, exist_ok=True)

    problem, net_config, net_assignments = util.get_config(
        FLAGS.problem, net_name="RNNprop", lasso_data_dir=FLAGS.lasso_data_dir,
        lasso_split=FLAGS.lasso_split, lasso_batch_size=FLAGS.lasso_batch_size,
        lasso_lam=FLAGS.lasso_lam, lasso_x0_mode=FLAGS.lasso_x0_mode)
    optimizer = meta.MetaOptimizer(FLAGS.num_mt, FLAGS.beta1, FLAGS.beta2, **net_config)
    meta_opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    # Initial setup
    x_vars, const_vars, loss_fn = problem()
    optimizer._setup(x_vars, net_assignments)
    state = optimizer.initial_state(x_vars)
    mt, vt = optimizer.initial_mt_state(x_vars)

    if FLAGS.if_mt:
        mt_loader = data_loader(problem, optimizer._subsets, FLAGS.optimizers, FLAGS.unroll_length)
        if FLAGS.if_cl:
            mt_ratios = [float(r) for r in FLAGS.mt_ratios.split()]

    best_evaluation = float("inf")
    num_eval = 0
    improved = False
    mti = -1
    global_step = 0
    start_time = timer()
    profiler = profiling.RunProfiler(FLAGS.profile_path)
    profiler.start()

    for e in range(FLAGS.num_epochs):
        x_vars, const_vars, loss_fn = problem()
        state = optimizer.initial_state(x_vars)
        mt, vt = optimizer.initial_mt_state(x_vars)
        step_offset = 0

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
        grad_norm = None

        if task_i == -1:
            for _ in range(num_unrolls_cur):
                (loss, meta_grads, net_vars,
                 x_final, state, mt, vt, _) = optimizer.unroll(
                    loss_fn, x_vars, state, mt, vt, step_offset,
                    FLAGS.unroll_length, scale=scale,
                    second_derivatives=FLAGS.second_derivatives,
                    last_step_loss=FLAGS.last_step_loss)
                grad_norm = float(tf.linalg.global_norm(meta_grads))
                meta_opt.apply_gradients(zip(meta_grads, net_vars))
                for var, val in zip(x_vars, x_final):
                    var.assign(val)
                step_offset += FLAGS.unroll_length
                epoch_loss = float(loss)
            profiler.log_epoch(e, epoch_loss, grad_norm,
                               num_optimizee_steps=num_unrolls_cur * FLAGS.unroll_length)
        else:
            data_e = mt_loader.get_data(
                task_i, num_unrolls_cur,
                rd_scale_bound=FLAGS.rd_scale_bound,
                if_scale=FLAGS.if_scale, mt_k=FLAGS.k)
            mt_rnn_state = [optimizer.initial_state(x_vars)[si]
                            for si in range(len(optimizer._subsets))]
            mt_m = [tf.zeros_like(x_vars[j])
                    for subset in optimizer._subsets for j in subset]
            mt_v = [tf.zeros_like(x_vars[j])
                    for subset in optimizer._subsets for j in subset]
            # Flatten per-subset
            mt_m_s = [[tf.zeros_like(x_vars[j]) for j in subset]
                      for subset in optimizer._subsets]
            mt_v_s = [[tf.zeros_like(x_vars[j]) for j in subset]
                      for subset in optimizer._subsets]

            for w in range(num_unrolls_cur):
                inputs_mt = data_e["inputs"][w]
                labels_mt = data_e["labels"][w]
                (loss, meta_grads, net_vars,
                 mt_rnn_state, mt_m_s, mt_v_s) = optimizer.unroll_mt(
                    inputs_mt, labels_mt, mt_rnn_state,
                    mt_m_s, mt_v_s, step_offset, FLAGS.unroll_length)
                grad_norm = float(tf.linalg.global_norm(meta_grads))
                meta_opt.apply_gradients(zip(meta_grads, net_vars))
                step_offset += FLAGS.unroll_length
                epoch_loss = float(loss)
            profiler.log_epoch(e, epoch_loss, grad_norm,
                               num_optimizee_steps=num_unrolls_cur * FLAGS.unroll_length)

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
                ev_x, _, ev_loss_fn = problem()
                ev_state = optimizer.initial_state(ev_x)
                ev_mt, ev_vt = optimizer.initial_mt_state(ev_x)
                ev_step = 0
                for _ in range(num_unrolls_eval_cur):
                    (loss_val, _, _, ev_x_final, ev_state,
                     ev_mt, ev_vt, _) = optimizer.unroll(
                        ev_loss_fn, ev_x, ev_state, ev_mt, ev_vt, ev_step,
                        FLAGS.unroll_length, last_step_loss=FLAGS.last_step_loss)
                    for var, val in zip(ev_x, ev_x_final):
                        var.assign(val)
                    ev_step += FLAGS.unroll_length
                    eval_cost += float(loss_val)

            # Overwritten every evaluation so that what's on saved when training
            # ends is the final evaluation's recovery.
            if FLAGS.save_path is not None and getattr(problem, "last_x_true", None) is not None:
                np.savez(
                    os.path.join(FLAGS.save_path, "recovery.npz"),
                    x_pred=ev_x[0].numpy(), x_true=problem.last_x_true, b=problem.last_b)

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

                eval_cost = 0.0
                for _ in range(FLAGS.evaluation_epochs):
                    ev_x, _, ev_loss_fn = problem()
                    ev_state = optimizer.initial_state(ev_x)
                    ev_mt, ev_vt = optimizer.initial_mt_state(ev_x)
                    ev_step = 0
                    for _ in range(num_unrolls_eval[curriculum_idx]):
                        (loss_val, _, _, ev_x_final, ev_state,
                         ev_mt, ev_vt, _) = optimizer.unroll(
                            ev_loss_fn, ev_x, ev_state, ev_mt, ev_vt, ev_step,
                            FLAGS.unroll_length)
                        for var, val in zip(ev_x, ev_x_final):
                            var.assign(val)
                        ev_step += FLAGS.unroll_length
                        eval_cost += float(loss_val)
                best_evaluation = eval_cost
                print("epoch={}, num_steps={}, eval loss={}".format(
                    e, num_steps_sched[curriculum_idx],
                    eval_cost / FLAGS.evaluation_epochs), flush=True)
            elif num_eval >= FLAGS.min_num_eval and not improved:
                print("no improve during curriculum {} --> stop".format(curriculum_idx))
                break

    profiler.finish()
    print("total time = {}s...".format(timer() - start_time))


if __name__ == "__main__":
    main()
