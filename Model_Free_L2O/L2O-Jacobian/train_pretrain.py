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

"""HALO Step 1: train the pretrained model that adapt_*_problems warm-starts
from (see train.py's --pretrained_model_path). Baseline optimizers only
(SGD/Adam/Adagrad) -- no meta-learned optimizer is needed to produce a
pretrained baseline. Only the non-excluded problem families are wired in
(mnist_conv, cifar10_conv); see migration_changes.txt for the excluded ones
(PIE, TDD, preresnet, lstm/HAR).
"""

import argparse
import os
import pickle

import tensorflow as tf

import metaopt
from problems import problem_sets as ps

BASELINE_OPTIMIZERS = {
    "SGD": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr),
    "Adam": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
    "Adagrad": lambda lr: tf.keras.optimizers.Adagrad(learning_rate=lr),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="pretrain/")
    p.add_argument("--test_optimizer", default="Adam", choices=list(BASELINE_OPTIMIZERS))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_testing_itrs", type=int, default=100)
    p.add_argument("--custom_flag", default="")
    p.add_argument("--include_mnist_conv_problems", action="store_true")
    p.add_argument("--include_mnist_conv_problems_wide", action="store_true")
    p.add_argument("--include_cifar10_conv_problems", action="store_true")
    return p.parse_args()


def main():
    FLAGS = parse_args()

    problems_and_data = []
    if FLAGS.include_mnist_conv_problems:
        problems_and_data.extend(ps.pretrain_mnist_conv_problems())
    if FLAGS.include_mnist_conv_problems_wide:
        problems_and_data.extend(ps.pretrain_mnist_conv_problems_wide())
    if FLAGS.include_cifar10_conv_problems:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems())

    if not problems_and_data:
        raise ValueError("No problems selected -- pass at least one --include_*_problems flag.")

    for problem_itr, (problem_spec_, dataset, batch_size) in enumerate(problems_and_data):
        problem = problem_spec_.build()
        problem_name = os.path.basename(os.path.normpath(FLAGS.train_dir))
        save_dir = os.path.join("records", problem_name,
                                "{}_lr_{}_{}".format(FLAGS.test_optimizer, FLAGS.lr, FLAGS.custom_flag))
        os.makedirs(save_dir, exist_ok=True)

        for seed in range(0, 20, 4):
            print("pretraining problem {} ({}) using seed {}".format(problem_itr, problem_name, seed))
            opt = BASELINE_OPTIMIZERS[FLAGS.test_optimizer](FLAGS.lr)
            objective_values, parameters, _ = metaopt.test_optimizer(
                opt, problem, num_iter=FLAGS.num_testing_itrs,
                dataset=dataset, batch_size=batch_size, seed=seed)

            with open(os.path.join(save_dir, "seed{}_eval_loss_record.pickle".format(seed)), "wb") as f:
                pickle.dump(objective_values, f)
            with open(os.path.join(save_dir, "seed{}_model_params.pickle".format(seed)), "wb") as f:
                pickle.dump(parameters, f)
            print("saved pretrained params for seed {} to {}".format(seed, save_dir))


if __name__ == "__main__":
    main()
