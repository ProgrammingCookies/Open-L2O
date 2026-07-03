"""Learning to optimize in swarms — training."""

import argparse
import os
import pickle
from timeit import default_timer as timer

import meta
import util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", default=None)
    p.add_argument("--num_epochs", type=int, default=10000)
    p.add_argument("--log_period", type=int, default=100)
    p.add_argument("--evaluation_period", type=int, default=1000)
    p.add_argument("--evaluation_epochs", type=int, default=20)
    p.add_argument("--problem", default="simple")
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--unroll_length", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--second_derivatives", action="store_true")
    return p.parse_args()


def run_epoch(optimizer, problem, num_unrolls, unroll_length, second_derivatives):
    """Redraws random particle positions, then runs num_unrolls training segments.

    Returns (elapsed_time, fx_final) where fx_final is the list of per-particle
    final losses from the last segment (matching the original run_epoch, which only
    reported the last segment's cost).
    """
    start = timer()
    optimizer.reset_state()
    fx_final = None
    for _ in range(num_unrolls):
        _, fx_final, _ = optimizer.meta_minimize(
            problem, unroll_length, second_derivatives=second_derivatives)
    return timer() - start, fx_final


def main():
    FLAGS = parse_args()

    num_unrolls = FLAGS.num_steps // FLAGS.unroll_length
    problem, net_config, net_assignments = util.get_config(FLAGS.problem)
    optimizer = meta.MetaOptimizer(**net_config)

    path = None
    if FLAGS.save_path is not None:
        if not os.path.exists(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
        elif os.path.exists("{}/loss_record.pickle".format(FLAGS.save_path)):
            path = FLAGS.save_path

    optimizer.setup(problem, net_assignments=net_assignments, model_path=path)

    best_evaluation = float("inf")
    total_time = 0.0
    total_cost = 0.0
    loss_record = []

    for e in range(FLAGS.num_epochs):
        elapsed, fx_final = run_epoch(
            optimizer, problem, num_unrolls, FLAGS.unroll_length, FLAGS.second_derivatives)
        cost = sum(float(v) for v in fx_final) / len(fx_final)
        total_time += elapsed
        total_cost += cost
        loss_record.append(cost)

        if (e + 1) % FLAGS.log_period == 0:
            util.print_stats("Epoch {}".format(e + 1), total_cost, total_time, FLAGS.log_period)
            total_time = 0.0
            total_cost = 0.0

        if (e + 1) % FLAGS.evaluation_period == 0:
            eval_cost = 0.0
            eval_time = 0.0
            for _ in range(FLAGS.evaluation_epochs):
                elapsed, fx_final = run_epoch(
                    optimizer, problem, num_unrolls, FLAGS.unroll_length, FLAGS.second_derivatives)
                eval_time += elapsed
                eval_cost += sum(float(v) for v in fx_final) / len(fx_final)

            util.print_stats("EVALUATION", eval_cost, eval_time, FLAGS.evaluation_epochs)

            if FLAGS.save_path is not None and eval_cost < best_evaluation:
                print("Removing previously saved meta-optimizer")
                for f in os.listdir(FLAGS.save_path):
                    os.remove(os.path.join(FLAGS.save_path, f))
                print("Saving meta-optimizer to {}".format(FLAGS.save_path))
                optimizer.save(FLAGS.save_path)
                with open(FLAGS.save_path + "/loss_record.pickle", "wb") as l_record:
                    record = {
                        "loss_record": loss_record,
                        "fc_weights": [v.numpy() for v in optimizer.fc_kernel],
                        "fc_bias": [v.numpy() for v in optimizer.fc_bias],
                        "fc_va": [v.numpy() for v in optimizer.fc_va],
                    }
                    pickle.dump(record, l_record)
                best_evaluation = eval_cost


if __name__ == "__main__":
    main()
