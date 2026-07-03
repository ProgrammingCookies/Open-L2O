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
"""Learning 2 Learn evaluation — RNNprop variant."""

import argparse
import os
import pickle
from timeit import default_timer as timer

import tensorflow as tf

import meta_rnnprop_eval as meta
import util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="L2L", choices=["L2L", "Adam"])
    p.add_argument("--problem", default="simple")
    p.add_argument("--path", default=None)
    p.add_argument("--output_path", default=None)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--beta1", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.95)
    return p.parse_args()


def main():
    FLAGS = parse_args()

    if FLAGS.seed is not None:
        tf.random.set_seed(FLAGS.seed)

    problem, net_config, net_assignments = util.get_config(
        FLAGS.problem, FLAGS.path, net_name="RNNprop")

    total_time = 0.0
    total_cost = 0.0
    loss_record = []

    for e in range(FLAGS.num_epochs):
        start = timer()
        x_vars, const_vars, loss_fn = problem()

        if FLAGS.optimizer == "Adam":
            adam = tf.keras.optimizers.Adam(FLAGS.learning_rate)
            costs = []
            for _ in range(FLAGS.num_steps):
                with tf.GradientTape() as tape:
                    loss = loss_fn([tf.identity(v) for v in x_vars])
                grads = tape.gradient(loss, x_vars)
                adam.apply_gradients(zip(grads, x_vars))
                costs.append(float(loss))
            loss_record.extend(costs)
            total_cost += sum(costs) / FLAGS.num_steps

        elif FLAGS.optimizer == "L2L":
            optimizer = meta.MetaOptimizer(FLAGS.beta1, FLAGS.beta2, **net_config)
            optimizer._setup(x_vars, net_assignments)

            # Load weights if available
            if FLAGS.path is not None:
                for k, net in optimizer._nets.items():
                    filename = os.path.join(FLAGS.path, "{}.l2l".format(k))
                    if os.path.exists(filename):
                        from networks import load as net_load
                        # Need one forward pass to build the network first
                        dummy_state = optimizer.initial_state(x_vars)
                        x_tmp = [tf.identity(v) for v in x_vars]
                        with tf.GradientTape() as tape:
                            tape.watch(x_tmp)
                            _ = loss_fn(x_tmp)
                        grads_tmp = tape.gradient(_, x_tmp)
                        optimizer._apply_step(grads_tmp, x_tmp,
                                              dummy_state, optimizer.initial_mt_state(x_vars)[0],
                                              optimizer.initial_mt_state(x_vars)[1], 1)
                        net_load(net, filename)

            state = optimizer.initial_state(x_vars)
            mt, vt = optimizer.initial_mt_state(x_vars)
            x = [tf.identity(v) for v in x_vars]
            costs = []
            t = 0

            for _ in range(FLAGS.num_steps):
                fx, x, state, mt, vt, t = optimizer.step(loss_fn, x, state, mt, vt, t)
                costs.append(float(fx))

            loss_record.extend(costs)
            total_cost += sum(costs) / FLAGS.num_steps
        else:
            raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

        total_time += timer() - start

    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                     total_time, FLAGS.num_epochs)

    if FLAGS.output_path is not None:
        os.makedirs(FLAGS.output_path, exist_ok=True)
    if FLAGS.output_path:
        output_file = os.path.join(
            FLAGS.output_path,
            "{}_eval_loss_record.pickle-{}".format(FLAGS.optimizer, FLAGS.problem))
        with open(output_file, "wb") as f:
            pickle.dump(loss_record, f)
        print("Saving evaluate loss record {}".format(output_file))


if __name__ == "__main__":
    main()
