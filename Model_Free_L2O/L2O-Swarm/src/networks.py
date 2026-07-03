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
"""Learning 2 Learn meta-optimizer networks."""

import abc
import sys

import dill as pickle
import numpy as np
import tensorflow as tf

import preprocess


def factory(net, net_options=(), net_path=None):
    """Network factory."""
    net_class = getattr(sys.modules[__name__], net)
    net_options = dict(net_options)
    if net_path:
        with open(net_path, "rb") as f:
            net_options["initializer"] = pickle.load(f)
    return net_class(**net_options)


def save(network, filename=None):
    """Save the variables contained by a network to disk."""
    to_save = {v.name: v.numpy() for v in network.trainable_variables}
    if filename:
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)
    return to_save


def load(network, filename):
    """Load saved weights into a network (must be called after first forward pass)."""
    with open(filename, "rb") as f:
        saved = pickle.load(f)
    var_by_name = {v.name: v for v in network.trainable_variables}
    for name, value in saved.items():
        if name in var_by_name:
            var_by_name[name].assign(value)


class Network(tf.keras.layers.Layer, abc.ABC):
    """Base class for meta-optimizer networks."""

    @abc.abstractmethod
    def initial_state_for_inputs(self, inputs, **kwargs):
        """Return initial RNN state given an example input tensor."""
        pass


class StandardDeepLSTM(Network):
    """LSTM layers with a Dense layer on top."""

    def __init__(self, output_size, layers, preprocess_name="identity",
                 preprocess_options=None, scale=1.0, initializer=None,
                 name="deep_lstm"):
        super().__init__(name=name)
        self._output_size = output_size
        self._scale = scale
        self._initializer = initializer  # saved weights dict, applied on first call

        if preprocess_options is None:
            preprocess_options = {}

        if hasattr(preprocess, preprocess_name):
            preprocess_class = getattr(preprocess, preprocess_name)
            self._preprocess = preprocess_class(**preprocess_options)
        else:
            self._preprocess = getattr(tf, preprocess_name)

        self._lstm_cells = [
            tf.keras.layers.LSTMCell(size, name="lstm_{}".format(i))
            for i, size in enumerate(layers, start=1)
        ]
        self._linear = tf.keras.layers.Dense(output_size, name="linear")
        self._built_once = False

    def _maybe_load_weights(self):
        if self._initializer is not None and not self._built_once and isinstance(self._initializer, dict):
            var_by_name = {v.name: v for v in self.trainable_variables}
            for name, value in self._initializer.items():
                if name in var_by_name:
                    var_by_name[name].assign(value)
            self._built_once = True

    def call(self, inputs, state):
        """
        Args:
            inputs: 2D tensor [batch_size, input_size].
            state: list of per-layer states, each state is [h, c].
        Returns:
            (output tensor, new_state list)
        """
        inputs = self._preprocess(tf.expand_dims(inputs, -1))
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])

        new_states = []
        x = inputs
        for cell, s in zip(self._lstm_cells, state):
            x, new_s = cell(x, s)
            new_states.append(new_s)

        output = self._linear(x)
        self._maybe_load_weights()

        return output * self._scale, new_states

    def initial_state_for_inputs(self, inputs, dtype=tf.float32):
        flat = tf.reshape(inputs, [-1, 1])
        batch_size = tf.shape(flat)[0]
        return [[tf.zeros([batch_size, cell.units], dtype=dtype),
                 tf.zeros([batch_size, cell.units], dtype=dtype)]
                for cell in self._lstm_cells]


class CoordinateWiseDeepLSTM(StandardDeepLSTM):
    """Coordinate-wise DeepLSTM: processes each parameter element independently."""

    def __init__(self, name="cw_deep_lstm", **kwargs):
        super().__init__(1, name=name, **kwargs)

    def _reshape_inputs(self, inputs):
        return tf.reshape(inputs, [-1, 1])

    def call(self, inputs, state):
        input_shape = tf.shape(inputs)
        reshaped = self._reshape_inputs(inputs)
        output, new_state = super().call(reshaped, state)
        return tf.reshape(output, input_shape), new_state

    def initial_state_for_inputs(self, inputs, dtype=tf.float32):
        reshaped = self._reshape_inputs(inputs)
        return super().initial_state_for_inputs(reshaped, dtype=dtype)


class KernelDeepLSTM(StandardDeepLSTM):
    """DeepLSTM for convolutional filters.

    The inputs are assumed to be shaped as convolutional filters with an extra
    preprocessing dimension ([kernel_w, kernel_h, n_input_channels,
    n_output_channels]).
    """

    def __init__(self, kernel_shape, name="kernel_deep_lstm", **kwargs):
        self._kernel_shape = kernel_shape
        output_size = int(np.prod(kernel_shape))
        super().__init__(output_size, name=name, **kwargs)

    def _reshape_inputs(self, inputs):
        transposed = tf.transpose(inputs, perm=[2, 3, 0, 1])
        return tf.reshape(transposed, [-1] + list(self._kernel_shape))

    def call(self, inputs, state):
        input_shape = tf.shape(inputs)
        reshaped = self._reshape_inputs(inputs)
        output, new_state = super().call(reshaped, state)
        transposed_output = tf.transpose(output, [1, 0])
        return tf.reshape(transposed_output, input_shape), new_state

    def initial_state_for_inputs(self, inputs, dtype=tf.float32):
        reshaped = self._reshape_inputs(inputs)
        return super().initial_state_for_inputs(reshaped, dtype=dtype)


class Sgd(Network):
    """Identity network which acts like SGD."""

    def __init__(self, learning_rate=0.001, name="sgd"):
        super().__init__(name=name)
        self._learning_rate = learning_rate

    def call(self, inputs, state):
        return -self._learning_rate * inputs, []

    def initial_state_for_inputs(self, inputs, **kwargs):
        return []


def _update_adam_estimate(estimate, value, b):
    return b * estimate + (1 - b) * value


def _debias_adam_estimate(estimate, b, t):
    return estimate / (1 - tf.pow(b, t))


class Adam(Network):
    """Adam algorithm (https://arxiv.org/pdf/1412.6980v8.pdf)."""

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, name="adam"):
        super().__init__(name=name)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def call(self, g, state):
        b1 = self._beta1
        b2 = self._beta2

        g_shape = tf.shape(g)
        g = tf.reshape(g, (-1, 1))

        t, m, v = state
        t_next = t + 1

        m_next = _update_adam_estimate(m, g, b1)
        m_hat = _debias_adam_estimate(m_next, b1, t_next)

        v_next = _update_adam_estimate(v, tf.square(g), b2)
        v_hat = _debias_adam_estimate(v_next, b2, t_next)

        update = -self._learning_rate * m_hat / (tf.sqrt(v_hat) + self._epsilon)
        return tf.reshape(update, g_shape), (t_next, m_next, v_next)

    def initial_state_for_inputs(self, inputs, dtype=tf.float32, **kwargs):
        batch_size = int(np.prod(inputs.shape))
        t = tf.zeros((), dtype=dtype)
        m = tf.zeros((batch_size, 1), dtype=dtype)
        v = tf.zeros((batch_size, 1), dtype=dtype)
        return (t, m, v)
