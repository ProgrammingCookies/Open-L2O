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
"""RNNprop meta-optimizer — evaluation variant (no training, no multi-task)."""

import os

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
    return nets, keys, subsets


class MetaOptimizer:
    """RNNprop meta-optimizer for evaluation."""

    def __init__(self, beta1=0.95, beta2=0.95, **kwargs):
        self._nets = None
        self._net_keys = None
        self._subsets = None
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
        mt, vt = [], []
        for subset in self._subsets:
            mt.append([tf.zeros_like(x_vars[j]) for j in subset])
            vt.append([tf.zeros_like(x_vars[j]) for j in subset])
        return mt, vt

    def _apply_step(self, gradients, x, state, mt, vt, t):
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
            g_i = [tf.stop_gradient(gradients[j]) for j in subset]

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

    def meta_loss(self, make_loss, len_unroll, net_assignments=None):
        """Set up and return handles for evaluation.

        Returns:
            (total_loss, x_final, state_final, mt_final, vt_final, fx_final,
             x_vars, step_counter)
        where step_counter is a Python int that callers should increment.
        """
        x_vars, const_vars, loss_fn = make_loss()
        self._setup(x_vars, net_assignments)
        state = self.initial_state(x_vars)
        mt, vt = self.initial_mt_state(x_vars)

        x = [tf.identity(v) for v in x_vars]
        fx_list = []
        t = 0

        for _ in range(len_unroll):
            with tf.GradientTape() as tape:
                tape.watch(x)
                fx = loss_fn(x)
            fx_list.append(fx)
            grads = tape.gradient(fx, x)
            t += 1
            x, state, mt, vt = self._apply_step(grads, x, state, mt, vt, t)

        fx_final = loss_fn(x)
        fx_list.append(fx_final)
        total_loss = tf.add_n(fx_list)

        return total_loss, x, state, mt, vt, fx_final, x_vars, t

    def step(self, loss_fn, x, state, mt, vt, t):
        """Run a single evaluation step."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            fx = loss_fn(x)
        grads = tape.gradient(fx, x)
        t += 1
        x, state, mt, vt = self._apply_step(grads, x, state, mt, vt, t)
        return fx, x, state, mt, vt, t

    def restore(self, path, index=None):
        for k, net in self._nets.items():
            if index is not None:
                filename = os.path.join(path, "{}.l2l-{}".format(k, index))
            else:
                filename = os.path.join(path, "{}.l2l".format(k))
            networks.load(net, filename)
