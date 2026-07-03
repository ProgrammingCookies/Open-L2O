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
"""Learning to learn (meta) optimizer — RNNprop training variant."""

import os

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
    """RNNprop meta-optimizer.

    Inputs to the RNN are (m_tilde, g_tilde) — Adam-normalised gradient and
    momentum, as described in the RNNprop paper.
    """

    def __init__(self, num_mt=0, beta1=0.95, beta2=0.95, **kwargs):
        self._nets = None
        self._net_keys = None
        self._subsets = None
        self.num_mt = num_mt
        self.beta1 = beta1
        self.beta2 = beta2

        self._config = kwargs if kwargs else {
            "coordinatewise": {
                "net": "RNNprop",
                "net_options": {
                    "layers": (20, 20),
                    "preprocess_name": "fc",
                    "preprocess_options": {"dim": 20},
                    "scale": 0.01,
                    "tanh_output": True,
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
        state = []
        for subset, key in zip(self._subsets, self._net_keys):
            net = self._nets[key]
            state_i = [net.initial_state_for_inputs(x_vars[j]) for j in subset]
            state.append(state_i)
        return state

    def initial_mt_state(self, x_vars):
        """Initial Adam moment state (m, v) per variable."""
        mt = []
        vt = []
        for subset in self._subsets:
            mt_i = [tf.zeros_like(x_vars[j]) for j in subset]
            vt_i = [tf.zeros_like(x_vars[j]) for j in subset]
            mt.append(mt_i)
            vt.append(vt_i)
        return mt, vt

    # ------------------------------------------------------------------
    # Single step (with Adam pre-conditioning)
    # ------------------------------------------------------------------

    def _apply_step(self, gradients, x, state, mt, vt, t,
                    second_derivatives=False):
        """Apply one RNNprop step.

        Args:
            t: current global step (int or Python int, used for bias-correction).
        Returns:
            (x_next, state_next, mt_next, vt_next)
        """
        if not second_derivatives:
            gradients = [tf.stop_gradient(g) for g in gradients]

        b1 = tf.cast(self.beta1, tf.float32)
        b2 = tf.cast(self.beta2, tf.float32)
        t_f = tf.cast(t, tf.float32)

        x_next = list(x)
        state_next = []
        mt_next_all = []
        vt_next_all = []

        for subset, key, s_i, mt_i, vt_i in zip(
                self._subsets, self._net_keys, state, mt, vt):
            net = self._nets[key]
            x_i = [x[j] for j in subset]
            g_i = [gradients[j] for j in subset]

            mt_next = [b1 * m + (1.0 - b1) * g for m, g in zip(mt_i, g_i)]
            mt_hat = [m / (1.0 - tf.pow(b1, t_f)) for m in mt_next]
            vt_next = [b2 * v + (1.0 - b2) * g * g for v, g in zip(vt_i, g_i)]
            vt_hat = [v / (1.0 - tf.pow(b2, t_f)) for v in vt_next]
            mt_tilde = [m / (tf.sqrt(v) + 1e-8) for m, v in zip(mt_hat, vt_hat)]
            gt_tilde = [g / (tf.sqrt(v) + 1e-8) for g, v in zip(g_i, vt_hat)]

            results = [net(m, g, s) for m, g, s in zip(mt_tilde, gt_tilde, s_i)]
            deltas = [r[0] for r in results]
            s_next = [r[1] for r in results]

            for idx, j in enumerate(subset):
                x_next[j] = x[j] + deltas[idx]
            state_next.append(s_next)
            mt_next_all.append(mt_next)
            vt_next_all.append(vt_next)

        return x_next, state_next, mt_next_all, vt_next_all

    # ------------------------------------------------------------------
    # Unroll
    # ------------------------------------------------------------------

    def unroll(self, loss_fn, x_vars, state, mt, vt, step_offset,
               len_unroll, scale=None, second_derivatives=False):
        """Run one unroll window.

        Args:
            step_offset: global step at the start of this window (for bias correction).
        Returns:
            (total_loss, meta_grads, net_vars, x_final, state_final,
             mt_final, vt_final, fx_final)
        """
        with tf.GradientTape() as meta_tape:
            x = [tf.identity(v) for v in x_vars]
            if scale is not None:
                x = [xi * si for xi, si in zip(x, scale)]

            fx_list = []
            for t in range(len_unroll):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    x_unscaled = ([xi / si for xi, si in zip(x, scale)]
                                  if scale is not None else x)
                    fx = loss_fn(x_unscaled)
                fx_list.append(fx)

                grads = inner_tape.gradient(fx, x)
                global_t = step_offset + t + 1
                x, state, mt, vt = self._apply_step(
                    grads, x, state, mt, vt, global_t, second_derivatives)

            x_unscaled = ([xi / si for xi, si in zip(x, scale)]
                          if scale is not None else x)
            fx_final = loss_fn(x_unscaled)
            fx_list.append(fx_final)
            total_loss = tf.add_n(fx_list)

        net_vars = self.trainable_variables
        meta_grads = meta_tape.gradient(total_loss, net_vars)
        return total_loss, meta_grads, net_vars, x, state, mt, vt, fx_final

    # ------------------------------------------------------------------
    # Multi-task unroll
    # ------------------------------------------------------------------

    def unroll_mt(self, inputs_mt, labels_mt, state_mt, mt_state, vt_state,
                  step_offset, len_unroll):
        """Multi-task imitation unroll (RNNprop variant)."""
        b1 = tf.cast(self.beta1, tf.float32)
        b2 = tf.cast(self.beta2, tf.float32)
        num_params_total = sum(inp.shape[1] for inp in inputs_mt)

        with tf.GradientTape() as meta_tape:
            loss_list = []
            state = list(state_mt)
            mt_cur = list(mt_state)
            vt_cur = list(vt_state)

            for t in range(len_unroll):
                global_t = tf.cast(step_offset + t + 1, tf.float32)
                state_new = []
                mt_new = []
                vt_new = []

                for si, (key, s_list, m_list, v_list) in enumerate(
                        zip(self._net_keys, state, mt_cur, vt_cur)):
                    net = self._nets[key]
                    g_all = tf.cast(inputs_mt[si][t], tf.float32)
                    g_label_all = tf.cast(labels_mt[si][t], tf.float32)

                    # inputs_mt/labels_mt are flattened+concatenated per subset
                    # (see data_generator._flatten_and_concat); split back into
                    # per-variable chunks matching mt_state/vt_state/state shapes.
                    var_shapes = [m.shape for m in m_list]
                    var_sizes = [int(np.prod(shape)) for shape in var_shapes]
                    g_per_var = [tf.reshape(g, shape) for g, shape in
                                 zip(tf.split(g_all, var_sizes), var_shapes)]

                    mt_next_list = [b1 * m + (1.0 - b1) * g
                                    for m, g in zip(m_list, g_per_var)]
                    mt_hat_list = [m / (1.0 - tf.pow(b1, global_t))
                                   for m in mt_next_list]
                    vt_next_list = [b2 * v + (1.0 - b2) * g * g
                                    for v, g in zip(v_list, g_per_var)]
                    vt_hat_list = [v / (1.0 - tf.pow(b2, global_t))
                                   for v in vt_next_list]
                    mt_tilde_list = [m / (tf.sqrt(v) + 1e-8)
                                     for m, v in zip(mt_hat_list, vt_hat_list)]
                    gt_tilde_list = [g / (tf.sqrt(v) + 1e-8)
                                     for g, v in zip(g_per_var, vt_hat_list)]

                    results = [net(mt_tilde, gt_tilde, s) for mt_tilde, gt_tilde, s
                               in zip(mt_tilde_list, gt_tilde_list, s_list)]
                    delta_parts = [tf.reshape(r[0], [-1]) for r in results]
                    s_new_list = [r[1] for r in results]
                    delta = tf.concat(delta_parts, axis=0)

                    loss_t = tf.reduce_sum((g_label_all - delta) ** 2) * 0.5
                    loss_list.append(loss_t / num_params_total)
                    state_new.append(s_new_list)
                    mt_new.append(mt_next_list)
                    vt_new.append(vt_next_list)

                state = state_new
                mt_cur = mt_new
                vt_cur = vt_new

            total_loss = tf.add_n(loss_list)

        net_vars = self.trainable_variables
        meta_grads = meta_tape.gradient(total_loss, net_vars)
        return total_loss, meta_grads, net_vars, state, mt_cur, vt_cur

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01,
                      net_assignments=None, second_derivatives=False):
        x_vars, const_vars, loss_fn = make_loss()
        self._setup(x_vars, net_assignments)
        state = self.initial_state(x_vars)
        mt, vt = self.initial_mt_state(x_vars)
        return {
            "loss_fn": loss_fn,
            "x_vars": x_vars,
            "const_vars": const_vars,
            "state": state,
            "mt": mt,
            "vt": vt,
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
