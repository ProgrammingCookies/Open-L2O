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
"""Learning to learn (meta) optimizer — DM variant (used by evaluate_dm)."""

import collections
import os
import pickle

import tensorflow as tf

import networks


def _make_nets(variables, config, net_assignments):
    """Create optimizer networks and map them to variable subsets."""
    name_to_index = {v.name.split(":")[0]: i for i, v in enumerate(variables)}

    if net_assignments is None:
        if len(config) != 1:
            raise ValueError("Default net_assignments requires exactly one net config.")
        key = next(iter(config))
        net = networks.factory(**config[key])
        return {key: net}, [key], [list(range(len(variables)))]

    nets, keys, subsets = {}, [], []
    for key, names in net_assignments:
        if key in nets:
            raise ValueError("Repeated netid in net_assignments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))
    return nets, keys, subsets


class MetaOptimizer:
    """Learning to learn (meta) optimizer — DM / evaluate variant.

    The optimizer network receives gradients and outputs parameter updates.
    State (LSTM hidden state) is carried as Python tensors between steps.
    """

    def __init__(self, **kwargs):
        self._nets = None
        self._net_keys = None
        self._subsets = None

        self._config = kwargs if kwargs else {
            "coordinatewise": {
                "net": "CoordinateWiseDeepLSTM",
                "net_options": {
                    "layers": (20, 20),
                    "preprocess_name": "LogAndSign",
                    "preprocess_options": {"k": 5},
                    "scale": 0.01,
                },
            }
        }

    @property
    def trainable_variables(self):
        return [v for net in self._nets.values() for v in net.trainable_variables]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self, x_vars, net_assignments):
        nets, net_keys, subsets = _make_nets(x_vars, self._config, net_assignments)
        self._nets = nets
        self._net_keys = net_keys
        self._subsets = subsets

    def initial_state(self, x_vars):
        """Return initial RNN state (list of per-subset, per-variable states)."""
        state = []
        for subset, key in zip(self._subsets, self._net_keys):
            net = self._nets[key]
            state_i = [net.initial_state_for_inputs(x_vars[j]) for j in subset]
            state.append(state_i)
        return state

    # ------------------------------------------------------------------
    # Single optimizer step
    # ------------------------------------------------------------------

    def _apply_step(self, gradients, x, state, second_derivatives=False):
        """Apply one meta-optimizer step and return (x_next, state_next, deltas)."""
        if not second_derivatives:
            gradients = [tf.stop_gradient(g) for g in gradients]

        x_next = list(x)
        state_next = []
        for subset, key, s_i in zip(self._subsets, self._net_keys, state):
            net = self._nets[key]
            x_i = [x[j] for j in subset]
            g_i = [gradients[j] for j in subset]
            results = [net(g, s) for g, s in zip(g_i, s_i)]
            deltas = [r[0] for r in results]
            s_next = [r[1] for r in results]
            for idx, j in enumerate(subset):
                x_next[j] = x[j] + deltas[idx]
            state_next.append(s_next)

        return x_next, state_next

    # ------------------------------------------------------------------
    # Unroll
    # ------------------------------------------------------------------

    def meta_loss(self, make_loss, len_unroll, net_assignments=None,
                  second_derivatives=False):
        """Set up networks and run one unroll, returning loss, grads and state.

        Returns:
            (total_loss, meta_grads, x_final, state_final, fx_final, x_vars)
        """
        x_vars, const_vars, loss_fn = make_loss()
        self._setup(x_vars, net_assignments)
        state = self.initial_state(x_vars)

        with tf.GradientTape() as meta_tape:
            x = [tf.identity(v) for v in x_vars]
            fx_list = []

            for _ in range(len_unroll):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    fx = loss_fn(x)
                fx_list.append(fx)

                grads = inner_tape.gradient(fx, x)
                x, state = self._apply_step(grads, x, state, second_derivatives)

            fx_final = loss_fn(x)
            fx_list.append(fx_final)
            total_loss = tf.add_n(fx_list)

        net_vars = self.trainable_variables
        meta_grads = meta_tape.gradient(total_loss, net_vars)
        return total_loss, meta_grads, x, state, fx_final, x_vars

    def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
        """One-shot: run meta_loss and minimise it, returning the loss."""
        total_loss, meta_grads, x, state, fx_final, x_vars = self.meta_loss(
            make_loss, len_unroll, **kwargs)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer.apply_gradients(zip(meta_grads, self.trainable_variables))
        return total_loss, x, state, fx_final, x_vars

    # ------------------------------------------------------------------
    # Save / restore
    # ------------------------------------------------------------------

    def save(self, path=None, index=None):
        result = {}
        for k, net in self._nets.items():
            if path is None:
                filename = None
                key = k
            elif index is not None:
                filename = os.path.join(path, "{}.l2l-{}".format(k, index))
                key = filename
            else:
                filename = os.path.join(path, "{}.l2l".format(k))
                key = filename
            net_vars = networks.save(net, filename=filename)
            result[key] = net_vars
        return result

    def restore(self, path, index):
        for k, net in self._nets.items():
            filename = os.path.join(path, "{}.l2l-{}".format(k, index))
            networks.load(net, filename)
