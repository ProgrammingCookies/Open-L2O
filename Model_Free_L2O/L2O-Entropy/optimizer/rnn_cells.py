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

"""Custom RNN cells for hierarchical RNNs."""

import tensorflow as tf

from optimizer import utils


class BiasGRUCell(tf.keras.layers.Layer):
    """GRU cell (cf. http://arxiv.org/abs/1406.1078) with an additional bias.

    The "gates"/"candidate" affine weights are owned sub-layers (AffineLayer),
    created once and reused on every call — the eager-mode replacement for the
    TF1 version's tf.variable_scope(reuse=...) weight sharing. They build
    lazily on first call (their input width depends on the caller's RNN input
    size, not on num_units alone), matching the original's "create on first
    use, reuse thereafter" behavior.
    """

    def __init__(self, num_units, activation=tf.tanh, scale=0.1,
                 gate_bias_init=0., random_seed=None, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._activation = activation
        self._scale = scale
        self._gate_bias_init = gate_bias_init
        self._random_seed = random_seed

        self._gates = utils.AffineLayer(
            2 * num_units, scale=scale, bias_init=gate_bias_init,
            random_seed=random_seed, name="gates")
        self._candidate = utils.AffineLayer(
            num_units, scale=scale, random_seed=random_seed, name="candidate")

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        # Sub-layers (_gates/_candidate) own the actual weights and build
        # themselves lazily on first call; nothing to build here directly.
        self.built = True

    def call(self, inputs, state, bias=None):
        # Keras RNN-cell convention: state is always a list of tensors, even
        # for a single-tensor state (matches tf.keras.layers.GRUCell so this
        # cell composes correctly inside tf.keras.layers.StackedRNNCells).
        if isinstance(state, (list, tuple)):
            state = state[0]

        # Split the injected bias vector into a bias for the r, u, and c updates.
        if bias is None:
            bias = tf.zeros((1, 3))

        r_bias, u_bias, c_bias = tf.split(bias, 3, 1)

        proj = self._gates([inputs, state])
        r_lin, u_lin = tf.split(proj, 2, 1)
        r, u = tf.nn.sigmoid(r_lin + r_bias), tf.nn.sigmoid(u_lin + u_bias)

        proj = self._candidate([inputs, r * state])
        c = self._activation(proj + c_bias)

        new_h = u * state + (1 - u) * c

        return new_h, [new_h]
