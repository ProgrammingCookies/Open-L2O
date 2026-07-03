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
"""Data generator for multi-task learning (imitation of reference optimizers)."""

import numpy as np
import tensorflow as tf


def _flatten_and_concat(tensors):
    """Flatten each tensor and concatenate into a single 1-D array."""
    parts = [t.numpy().reshape(-1) for t in tensors]
    return np.concatenate(parts, axis=0)


class data_loader:
    """Generates (gradient, parameter-update) pairs by running a reference optimizer."""

    def __init__(self, make_loss, subsets, optimizers, unroll_len):
        self.unroll_len = unroll_len
        self.optimizers = optimizers.split(",")
        self.num_subsets = len(subsets)
        self.subsets = subsets
        self.make_loss = make_loss

    def _make_optimizer(self, name):
        if name == "adam":
            return tf.keras.optimizers.Adam(0.01)
        elif name == "rmsprop":
            return tf.keras.optimizers.RMSprop(0.01)
        elif name == "nag":
            return tf.keras.optimizers.SGD(0.01, momentum=0.9, nesterov=True)
        else:
            raise ValueError("Unknown optimizer: {}".format(name))

    def get_data(self, task_i, num_unrolls, assign_func=None,
                 rd_scale_bound=3.0, if_scale=False, mt_k=1):
        """Run reference optimizer and collect (grad, update) pairs.

        Returns:
            dict with keys "inputs" and "labels", each a list over num_unrolls
            of lists over num_subsets of numpy arrays [unroll_len, num_params].
        """
        opt_name = self.optimizers[task_i]
        opt = self._make_optimizer(opt_name)

        # Fresh problem instance
        x_vars, const_vars, loss_fn = self.make_loss()

        # Optionally apply random scale
        if if_scale:
            r_scale = [np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                                                size=v.shape))
                       for v in x_vars]
            for v, rs in zip(x_vars, r_scale):
                v.assign(v.numpy() / rs)
            scale = r_scale
        else:
            scale = None

        data = {"inputs": [], "labels": []}

        for _ in range(num_unrolls):
            inputs_window = []   # [unroll_len][num_subsets][num_params]
            labels_window = []

            x_prev_flat = [_flatten_and_concat([x_vars[j] for j in subset])
                           for subset in self.subsets]

            for _ in range(self.unroll_len):
                with tf.GradientTape() as tape:
                    x_tensors = [tf.identity(v) for v in x_vars]
                    if scale is not None:
                        x_tensors = [xi * tf.constant(rs, dtype=tf.float32)
                                     for xi, rs in zip(x_tensors, scale)]
                    loss = loss_fn(x_tensors)

                grads = tape.gradient(loss, x_vars)

                # Collect gradients per subset (flattened)
                step_inputs = [
                    _flatten_and_concat([grads[j] for j in subset])
                    for subset in self.subsets
                ]
                inputs_window.append(step_inputs)

                # Apply reference optimizer
                opt.apply_gradients(zip(grads, x_vars))
                for _ in range(mt_k - 1):
                    with tf.GradientTape() as tape2:
                        loss2 = loss_fn([tf.identity(v) for v in x_vars])
                    grads2 = tape2.gradient(loss2, x_vars)
                    opt.apply_gradients(zip(grads2, x_vars))

                x_cur_flat = [_flatten_and_concat([x_vars[j] for j in subset])
                              for subset in self.subsets]
                step_labels = [cur - prev for cur, prev in
                               zip(x_cur_flat, x_prev_flat)]
                labels_window.append(step_labels)
                x_prev_flat = x_cur_flat

            # Transpose from [time][subset] to [subset][time]
            input_subsets = [
                np.stack([inputs_window[t][si] for t in range(self.unroll_len)], axis=0)
                for si in range(self.num_subsets)
            ]
            label_subsets = [
                np.stack([labels_window[t][si] for t in range(self.unroll_len)], axis=0)
                for si in range(self.num_subsets)
            ]
            data["inputs"].append(input_subsets)
            data["labels"].append(label_subsets)

        return data
