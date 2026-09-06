"""Learning to optimize in swarms — evaluation."""

import argparse
import logging
import os
import pickle
from timeit import default_timer as timer

import tensorflow as tf

import meta
import networks
import util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="L2L", choices=["L2L", "Adam"])
    p.add_argument("--path", default=None, help="Path to saved meta-optimizer network.")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--problem", default="simple")
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.001)
    return p.parse_args()


def main():
    FLAGS = parse_args()

    if FLAGS.seed is not None:
        tf.random.set_seed(FLAGS.seed)

    problem, net_config, net_assignments = util.get_config(FLAGS.problem)

    total_time = 0.0
    total_cost = 0.0
    min_loss_record = []
    all_time_loss_record = []

    if FLAGS.optimizer == "Adam":
        for _ in range(FLAGS.num_epochs):
            start = timer()
            x_vars, const_vars, loss_fn = problem()
            adam = tf.keras.optimizers.Adam(FLAGS.learning_rate)
            costs = []
            for _ in range(FLAGS.num_steps):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(loss_fn(x_vars))
                grads = tape.gradient(loss, x_vars)
                adam.apply_gradients(zip(grads, x_vars))
                costs.append(float(loss))
            all_time_loss_record.append(costs)
            min_loss_record.append(min(costs))
            total_time += timer() - start
            total_cost += min(costs)

    elif FLAGS.optimizer == "L2L":
        if FLAGS.path is None:
            logging.warning("Evaluating untrained L2L optimizer")

        optimizer = meta.MetaOptimizer(**net_config)
        optimizer.setup(problem, net_assignments=net_assignments, model_path=FLAGS.path)

        if FLAGS.path is not None:
            # Trigger a dummy step to build the network's weights, then load the
            # saved ones (Keras layers build lazily on first call).
            optimizer.unroll_step(1)
            for net in optimizer._nets.values():
                filename = os.path.join(FLAGS.path, "cw.l2l")
                if os.path.exists(filename):
                    networks.load(net, filename)

        for _ in range(FLAGS.num_epochs):
            start = timer()
            optimizer.reset_state()
            costs = []
            for _ in range(FLAGS.num_steps):
                _, fx_final = optimizer.unroll_step(1)
                costs.append(sum(float(v) for v in fx_final) / len(fx_final))
            all_time_loss_record.append(costs)
            min_loss_record.append(min(costs))
            total_time += timer() - start
            total_cost += min(costs)
    else:
        raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

    if FLAGS.path is not None:
        with open(os.path.join(FLAGS.path, "evaluate_record.pickle"), "wb") as l_record:
            record = {
                "all_time_loss_record": all_time_loss_record,
                "min_loss_record": min_loss_record,
            }
            pickle.dump(record, l_record)

    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost, total_time, FLAGS.num_epochs)


if __name__ == "__main__":
    main()
