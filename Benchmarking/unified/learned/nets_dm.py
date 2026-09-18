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
"""Learning 2 Learn meta-optimizer networks (DM / RNNProp lineage).

Verbatim copy of ``Model_Free_L2O/L2O-DM and L2O-RNNProp/networks.py``
(``_weight_key`` .. ``RNNprop``) plus the two preprocessing layers from its
``preprocess.py``, under this package's namespace. Verbatim on purpose: Keras
creates variables in attribute-assignment order and the index-keyed
checkpoint format below is that order, so the copied classes load the
adapters' ``{cw,rp}.l2l-<idx>`` pickles by construction. Dropped: ``factory``
(name-based reflection), ``KernelDeepLSTM``, ``Sgd``, ``Adam`` (not used by
any LASSO configuration).

The only edit: ``StandardDeepLSTM`` resolved its preprocessor through the
``preprocess`` module with ``hasattr``; here that lookup goes through the
``_PREPROCESSORS`` table since no such module exists in this namespace.
"""

import abc
import re

import dill as pickle
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------- preprocess.py

class Clamp(tf.keras.layers.Layer):

    def __init__(self, min_value=None, max_value=None, **kwargs):
        super().__init__(**kwargs)
        self._min = min_value
        self._max = max_value

    def call(self, inputs):
        output = inputs
        if self._min is not None:
            output = tf.maximum(output, self._min)
        if self._max is not None:
            output = tf.minimum(output, self._max)
        return output


class LogAndSign(tf.keras.layers.Layer):
    """Log and sign preprocessing.

    As described in https://arxiv.org/pdf/1606.04474v1.pdf (Appendix A).
    """

    def __init__(self, initializer, k, **kwargs):
        super().__init__(**kwargs)
        self._k = k

    def call(self, gradients):
        eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
        ndims = gradients.shape.rank

        log = tf.math.log(tf.abs(gradients) + eps)
        clamped_log = Clamp(min_value=-1.0)(log / self._k)
        sign = Clamp(min_value=-1.0, max_value=1.0)(gradients * np.exp(self._k))

        return tf.concat([clamped_log, sign], ndims - 1)


_PREPROCESSORS = {"LogAndSign": LogAndSign, "Clamp": Clamp}


# ----------------------------------------------------------------- networks.py

# Weight dicts are keyed "<index>:<name>", NOT by name alone.
#
# Under Keras 3 `v.name` is the BARE variable name ("kernel", "bias",
# "recurrent_kernel") rather than TF1/Sonnet's unique scoped path
# ("deep_lstm/lstm_1/kernel:0"), so names are NOT unique within a network: a
# 2-layer LSTM + Dense head has four variables named "kernel" and four named
# "bias". The previous `{v.name: ...}` comprehension therefore collided on both
# save and load, silently collapsing 10 tensors to 3 -- a restored optimizer
# kept ~70% of its random initialisation with no error raised. Prefixing the
# index makes the key unique and order-stable, and the checks below make any
# future mismatch fail loudly instead of silently under-restoring.
_INDEXED_KEY_RE = re.compile(r"^\d+:")


def _weight_key(index, variable):
    return "{}:{}".format(index, variable.name)


def _check_shape(source, key, value, variable):
    if tuple(np.shape(value)) != tuple(variable.shape):
        raise RuntimeError(
            "{}: weight {!r} has shape {} but the network variable has shape "
            "{}.".format(source, key, np.shape(value), tuple(variable.shape)))


def _assign_saved_weights(variables, saved, source):
    """Assign a saved weight dict into `variables`, or raise.

    Accepts both the indexed format written by `save` and the legacy
    bare-name format -- but a legacy dict is only used when it is provably
    COMPLETE (unique names, one entry per variable). A legacy dict written for
    a network with duplicate variable names lost tensors at save time and is
    unrecoverable, so it is rejected loudly rather than silently restoring a
    fraction of the weights.
    """
    variables = list(variables)
    if saved and all(_INDEXED_KEY_RE.match(k) for k in saved):
        if len(saved) != len(variables):
            raise RuntimeError(
                "{} holds {} weight tensors but the network has {}.".format(
                    source, len(saved), len(variables)))
        for i, v in enumerate(variables):
            key = _weight_key(i, v)
            if key not in saved:
                raise RuntimeError(
                    "{} is missing weight {!r} (network variable {} of {}).".format(
                        source, key, i, len(variables)))
            _check_shape(source, key, saved[key], v)
            v.assign(saved[key])
        return

    names = [v.name for v in variables]
    duplicates = len(set(names)) != len(names)
    missing = [n for n in names if n not in saved]
    if duplicates or missing or len(saved) != len(variables):
        raise RuntimeError(
            "{} holds a legacy weight dict keyed by bare variable name, with {} "
            "entries for a network of {} tensors{}. Under Keras 3 `v.name` is "
            "not unique within a network, so such a dict silently dropped every "
            "colliding tensor when it was written -- the missing weights were "
            "never saved and cannot be restored. Retrain to regenerate this "
            "checkpoint.".format(
                source, len(saved), len(variables),
                " (duplicate variable names: {})".format(
                    sorted({n for n in names if names.count(n) > 1}))
                if duplicates else ""))
    for v in variables:
        _check_shape(source, v.name, saved[v.name], v)
        v.assign(saved[v.name])


def save(network, filename=None):
    """Save the variables contained by a network to disk."""
    variables = list(network.trainable_variables)
    if not variables:
        raise RuntimeError(
            "refusing to save an empty checkpoint: the network has no trainable "
            "variables. Under Keras 3 variables are created on the first forward "
            "pass, so this means the network was never called -- e.g. num_steps < "
            "unroll_length runs zero unrolls, leaving the net unbuilt. The count "
            "check below cannot catch this, since 0 keys for 0 variables compares "
            "equal; without this guard an empty pickle is written and only fails "
            "much later, at load time.")
    to_save = {_weight_key(i, v): v.numpy() for i, v in enumerate(variables)}
    if len(to_save) != len(variables):
        raise RuntimeError(
            "refusing to save an incomplete checkpoint: {} keys for {} "
            "trainable variables (duplicate keys would be silently dropped)."
            .format(len(to_save), len(variables)))
    if filename:
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)
    return to_save


def load(network, filename):
    """Load saved weights into a network (must be called after first forward pass)."""
    with open(filename, "rb") as f:
        saved = pickle.load(f)
    _assign_saved_weights(network.trainable_variables, saved, filename)


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
                 name="deep_lstm", tanh_output=False):
        super().__init__(name=name)
        self._output_size = output_size
        self._scale = scale
        self._preprocess_name = preprocess_name
        self.tanh_output = tanh_output
        self._initializer = initializer  # saved weights dict, applied on first call

        if preprocess_options is None:
            preprocess_options = {}

        if preprocess_name == "fc":
            self._preprocess = tf.keras.layers.Dense(
                preprocess_options["dim"], name="input_projection")
        elif preprocess_name in _PREPROCESSORS:
            preprocess_class = _PREPROCESSORS[preprocess_name]
            self._preprocess = preprocess_class(initializer, **preprocess_options)
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
            # Same indexed-key contract as networks.save/load -- see the comment
            # on _weight_key for why keying by v.name alone silently drops most
            # of the tensors under Keras 3.
            _assign_saved_weights(self.trainable_variables, self._initializer,
                                  "the initializer passed to {}".format(type(self).__name__))
            self._built_once = True

    def call(self, inputs, state):
        """
        Args:
            inputs: 2D tensor [batch_size, input_size].
            state: list of per-layer states, each state is [h, c].
        Returns:
            (output tensor, new_state list)
        """
        if self._preprocess_name == "fc":
            inputs = tf.nn.elu(self._preprocess(inputs))
        else:
            inputs = self._preprocess(tf.expand_dims(inputs, -1))

        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])

        new_states = []
        x = inputs
        for cell, s in zip(self._lstm_cells, state):
            x, new_s = cell(x, s)
            new_states.append(new_s)

        output = self._linear(x)
        self._maybe_load_weights()

        if self.tanh_output:
            return tf.nn.tanh(output) * self._scale, new_states
        else:
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


class RNNprop(StandardDeepLSTM):
    """RNNprop network: takes (m_tilde, g_tilde) as inputs."""

    def __init__(self, name="RNNprop", **kwargs):
        super().__init__(1, name=name, **kwargs)

    def _reshape_inputs(self, m, g):
        inputs = tf.concat([tf.expand_dims(m, -1), tf.expand_dims(g, -1)], axis=-1)
        return tf.reshape(inputs, [-1, 2])

    def call(self, m, g, state):
        output_shape = tf.shape(g)
        reshaped = self._reshape_inputs(m, g)
        output, new_state = super().call(reshaped, state)
        return tf.reshape(output, output_shape), new_state

    def initial_state_for_inputs(self, inputs, dtype=tf.float32):
        flat = tf.reshape(inputs, [-1, 1])
        return super().initial_state_for_inputs(flat, dtype=dtype)
