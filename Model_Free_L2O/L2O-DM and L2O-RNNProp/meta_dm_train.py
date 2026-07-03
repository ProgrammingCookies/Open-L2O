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
"""Learning to learn (meta) optimizer — DM training variant."""

import collections
import os
import pickle

import numpy as np
import tensorflow as tf

import networks


def _make_nets(variables, config, net_assignments):
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
    """DM meta-optimizer with optional multi-task learning support."""

    def __init__(self, num_mt=0, **kwargs):
        self._nets = None
        self._net_keys = None
        self._subsets = None
        self.num_mt = num_mt

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
        print("Nets:", list(nets.keys()))
        print("Subsets:", subsets)

    def initial_state(self, x_vars):
        state = []
        for subset, key in zip(self._subsets, self._net_keys):
            net = self._nets[key]
            state_i = [net.initial_state_for_inputs(x_vars[j]) for j in subset]
            state.append(state_i)
        return state

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def _apply_step(self, gradients, x, state, second_derivatives=False):
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

    def unroll(self, loss_fn, x_vars, state, len_unroll,
               scale=None, second_derivatives=False):
        """Run one unroll window.

        Args:
            loss_fn: callable(x_tensors) -> scalar loss.
            x_vars:  list of tf.Variable (current parameter values).
            state:   RNN state from previous window.
            len_unroll: number of inner steps.
            scale:   optional list of scale factors (same length as x_vars).
            second_derivatives: whether to allow higher-order gradients.

        Returns:
            (total_loss, meta_grads, net_vars, x_final, state_final, fx_final)
        """
        with tf.GradientTape() as meta_tape:
            x = [tf.identity(v) for v in x_vars]
            if scale is not None:
                x = [xi * si for xi, si in zip(x, scale)]

            fx_list = []
            for _ in range(len_unroll):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    # loss is computed on un-scaled x when scale is provided
                    x_unscaled = ([xi / si for xi, si in zip(x, scale)]
                                  if scale is not None else x)
                    fx = loss_fn(x_unscaled)
                fx_list.append(fx)

                grads = inner_tape.gradient(fx, x)
                x, state = self._apply_step(grads, x, state, second_derivatives)

            x_unscaled = ([xi / si for xi, si in zip(x, scale)]
                          if scale is not None else x)
            fx_final = loss_fn(x_unscaled)
            fx_list.append(fx_final)
            total_loss = tf.add_n(fx_list)

        net_vars = self.trainable_variables
        meta_grads = meta_tape.gradient(total_loss, net_vars)
        return total_loss, meta_grads, net_vars, x, state, fx_final

    # ------------------------------------------------------------------
    # Multi-task unroll (imitation learning from standard optimizers)
    # ------------------------------------------------------------------

    def unroll_mt(self, inputs_mt, labels_mt, state_mt, len_unroll):
        """Multi-task unroll: imitate gradient steps from a reference optimizer.

        Args:
            inputs_mt: list over subsets of arrays [len_unroll, num_params].
            labels_mt: list over subsets of arrays [len_unroll, num_params].
            state_mt:  RNN state for multi-task head.
            len_unroll: number of steps.

        Returns:
            (loss, meta_grads, net_vars, state_final)
        """
        num_params_total = sum(inp.shape[1] for inp in inputs_mt)

        with tf.GradientTape() as meta_tape:
            loss_list = []
            state = list(state_mt)
            state_new = [None] * len(state)

            for t in range(len_unroll):
                for si, (key, s_list) in enumerate(zip(self._net_keys, state)):
                    net = self._nets[key]
                    g_all = tf.cast(inputs_mt[si][t], tf.float32)
                    g_label_all = tf.cast(labels_mt[si][t], tf.float32)

                    # Split concatenated gradient by per-variable sizes (from state shapes)
                    var_sizes = [s_list[vi][0][0].shape[0] for vi in range(len(s_list))]
                    g_per_var = tf.split(g_all, var_sizes)

                    delta_parts = []
                    s_new_list = []
                    for gv, sv in zip(g_per_var, s_list):
                        delta_v, sv_new = net(gv, sv)
                        delta_parts.append(delta_v)
                        s_new_list.append(sv_new)

                    delta = tf.concat(delta_parts, axis=0)
                    loss_t = tf.reduce_sum((g_label_all - delta) ** 2) * 0.5
                    loss_list.append(loss_t / num_params_total)
                    state_new[si] = s_new_list

                state = state_new
                state_new = [None] * len(state)

            total_loss = tf.add_n(loss_list)

        net_vars = self.trainable_variables
        meta_grads = meta_tape.gradient(total_loss, net_vars)
        return total_loss, meta_grads, net_vars, state

    # ------------------------------------------------------------------
    # High-level entry points (used by training scripts)
    # ------------------------------------------------------------------

    def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01,
                      net_assignments=None, second_derivatives=False):
        """Set up for training: build problem, create networks, return handles.

        Returns a dict with keys:
            loss_fn, x_vars, const_vars, state (initial),
            subsets, net_keys, nets
        """
        x_vars, const_vars, loss_fn = make_loss()
        self._setup(x_vars, net_assignments)
        state = self.initial_state(x_vars)
        return {
            "loss_fn": loss_fn,
            "x_vars": x_vars,
            "const_vars": const_vars,
            "state": state,
            "subsets": self._subsets,
            "net_keys": self._net_keys,
            "nets": self._nets,
        }

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
