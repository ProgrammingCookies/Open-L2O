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

"""Collection of trainable optimizers for meta-optimization."""

import math

import numpy as np
import tensorflow as tf

from optimizer import utils
from optimizer import trainable_optimizer as opt


class CoordinatewiseRNN(opt.TrainableOptimizer):
    """RNN that operates on each coordinate of the problem independently."""

    def __init__(self,
                 cell_sizes,
                 cell_cls,
                 init_lr_range=(1., 1.),
                 dynamic_output_scale=True,
                 learnable_decay=True,
                 zero_init_lr_weights=False,
                 rnn_readout_scale=0.5,
                 default_decay_var_init=2.2,
                 **kwargs):
        """Initializes the RNN per-parameter optimizer.

        Args:
          cell_sizes: List of hidden state sizes for each RNN cell in the network
          cell_cls: A tf.keras.layers RNN cell class (e.g. GRUCell, LSTMCell) or
              rnn_cells.BiasGRUCell, used to build one cell per entry in
              cell_sizes.
          init_lr_range: the range in which to initialize the learning rates.
          dynamic_output_scale: whether to learn weights that dynamically modulate
              the output scale (default: True)
          learnable_decay: whether to learn weights that dynamically modulate the
              input scale via RMS style decay (default: True)
          zero_init_lr_weights: whether to initialize the lr weights to zero
          rnn_readout_scale: initialization scale for the RNN readout weights.
          default_decay_var_init: default initializer value for decay/momentum
              style variables and constants (sigmoid(2.2) ~ 0.9).
          **kwargs: args passed to TrainableOptimizer's constructor

        Raises:
          ValueError: If the init lr range is not of length 2.
          ValueError: If the init lr range is not a valid range (min > max).
        """
        if len(init_lr_range) != 2:
            raise ValueError(
                "Initial LR range must be len 2, was {}".format(len(init_lr_range)))
        if init_lr_range[0] > init_lr_range[1]:
            raise ValueError("Initial LR range min is greater than max.")
        self.init_lr_range = init_lr_range

        self.zero_init_lr_weights = zero_init_lr_weights
        self._default_decay_var_init = default_decay_var_init

        # create the RNN cell
        self.component_cells = [cell_cls(sz) for sz in cell_sizes]
        self.cell = tf.keras.layers.StackedRNNCells(self.component_cells)

        # random normal initialization scaled by the output size
        scale_factor = rnn_readout_scale / math.sqrt(cell_sizes[-1])
        scaled_init = tf.random_normal_initializer(0., scale_factor)

        # weights for projecting the hidden state to a parameter update
        self.update_weights = tf.Variable(
            scaled_init((cell_sizes[-1], 1)), name="update_weights")

        self._initialize_decay(learnable_decay, (cell_sizes[-1], 1), scaled_init)
        self._initialize_lr(dynamic_output_scale, (cell_sizes[-1], 1), scaled_init)

        total_state_size = sum(self._cell_state_sizes())
        self._init_vector = tf.Variable(
            tf.random.uniform([1, total_state_size], -1., 1.), name="init_vector")

        state_keys = ["rms", "rnn", "learning_rate", "decay"]
        super().__init__("cRNN", state_keys, **kwargs)

    @property
    def trainable_variables(self):
        variables = [self.update_weights, self._init_vector]
        for name in ("decay_weights", "decay_bias", "lr_weights", "lr_bias"):
            v = getattr(self, name)
            if isinstance(v, tf.Variable):
                variables.append(v)
        for cell in self.component_cells:
            variables.extend(cell.trainable_variables if hasattr(cell, "trainable_variables") else [])
        return variables

    def _initialize_decay(
            self, learnable_decay, weights_tensor_shape, scaled_init):
        """Initializes the decay weights and bias variables or tensors.

        Args:
          learnable_decay: Whether to use learnable decay.
          weights_tensor_shape: The shape the weight tensor should take.
          scaled_init: The scaled initialization for the weights tensor.
        """
        if learnable_decay:
            # weights for projecting the hidden state to the RMS decay term
            self.decay_weights = tf.Variable(
                scaled_init(weights_tensor_shape), name="decay_weights")
            self.decay_bias = tf.Variable(
                tf.fill((1,), self._default_decay_var_init), name="decay_bias")
        else:
            self.decay_weights = tf.zeros_like(self.update_weights)
            self.decay_bias = tf.constant(self._default_decay_var_init)

    def _initialize_lr(
            self, dynamic_output_scale, weights_tensor_shape, scaled_init):
        """Initializes the learning rate weights and bias variables or tensors.

        Args:
          dynamic_output_scale: Whether to use a dynamic output scale.
          weights_tensor_shape: The shape the weight tensor should take.
          scaled_init: The scaled initialization for the weights tensor.
        """
        if dynamic_output_scale:
            zero_init = tf.zeros_initializer()
            wt_init = zero_init if self.zero_init_lr_weights else scaled_init
            self.lr_weights = tf.Variable(
                wt_init(weights_tensor_shape), name="learning_rate_weights")
            self.lr_bias = tf.Variable(zero_init((1,)), name="learning_rate_bias")
        else:
            self.lr_weights = tf.zeros_like(self.update_weights)
            self.lr_bias = tf.zeros([1, 1])

    def _initialize_state(self, var):
        """Return a dictionary mapping names of state variables to their values."""
        vectorized_shape = [var.shape.num_elements(), 1]

        min_lr = self.init_lr_range[0]
        max_lr = self.init_lr_range[1]
        if min_lr == max_lr:
            init_lr = tf.constant(min_lr, shape=vectorized_shape, dtype=tf.float32)
        else:
            actual_vals = tf.random.uniform(vectorized_shape,
                                            np.log(min_lr),
                                            np.log(max_lr))
            init_lr = tf.exp(actual_vals)

        ones = tf.ones(vectorized_shape)
        rnn_init = ones * self._init_vector

        return {
            "rms": tf.ones(vectorized_shape),
            "learning_rate": init_lr,
            "rnn": rnn_init,
            "decay": tf.ones(vectorized_shape),
        }

    def _compute_update(self, param, grad, state):
        """Update parameters given the gradient and state.

        Args:
          param: tensor of parameters
          grad: tensor of gradients with the same shape as param
          state: a dictionary containing any state for the optimizer

        Returns:
          updated_param: updated parameters
          updated_state: updated state variables in a dictionary
        """
        param_shape = tf.shape(param)

        (grad_values, decay_state, rms_state, rnn_state, learning_rate_state,
         grad_indices) = self._extract_gradients_and_internal_state(
             grad, state, param_shape)

        # Vectorize and scale the gradients.
        grad_scaled, rms = utils.rms_scaling(grad_values, decay_state, rms_state)

        # Apply the RNN update.
        rnn_state_tuples = self._unpack_rnn_state(rnn_state)
        rnn_output, rnn_state_tuples = self.cell(grad_scaled, rnn_state_tuples)
        rnn_state = self._pack_rnn_state(rnn_state_tuples)

        # Compute the update direction (a linear projection of the RNN output).
        delta = utils.project(rnn_output, self.update_weights)

        # The updated decay is an affine projection of the hidden state
        decay = utils.project(rnn_output, self.decay_weights,
                              bias=self.decay_bias, activation=tf.nn.sigmoid)

        # Compute the change in learning rate (an affine projection of the RNN
        # state, passed through a 2x sigmoid, so the change is bounded).
        learning_rate_change = 2. * utils.project(rnn_output, self.lr_weights,
                                                  bias=self.lr_bias,
                                                  activation=tf.nn.sigmoid)

        # Update the learning rate.
        new_learning_rate = learning_rate_change * learning_rate_state

        # Apply the update to the parameters.
        update = tf.reshape(new_learning_rate * delta, tf.shape(grad_values))

        if isinstance(grad, tf.IndexedSlices):
            update = utils.stack_tensor(update, grad_indices, param,
                                        param_shape[:1])
            rms = utils.update_slices(rms, grad_indices, state["rms"], param_shape)
            new_learning_rate = utils.update_slices(new_learning_rate, grad_indices,
                                                     state["learning_rate"],
                                                     param_shape)
            rnn_state = utils.update_slices(rnn_state, grad_indices, state["rnn"],
                                            param_shape)
            decay = utils.update_slices(decay, grad_indices, state["decay"],
                                        param_shape)

        new_param = param - update

        # Collect the update and new state.
        new_state = {
            "rms": rms,
            "learning_rate": new_learning_rate,
            "rnn": rnn_state,
            "decay": decay,
        }

        return new_param, new_state

    def _extract_gradients_and_internal_state(self, grad, state, param_shape):
        """Extracts the gradients and relevant internal state.

        If the gradient is sparse, extracts the appropriate slices from the state.

        Args:
          grad: The current gradient.
          state: The current state.
          param_shape: The shape of the parameter (used if gradient is sparse).

        Returns:
          grad_values: The gradient value tensor.
          decay_state: The current decay state.
          rms_state: The current rms state.
          rnn_state: The current state of the internal rnns.
          learning_rate_state: The current learning rate state.
          grad_indices: The indices for the gradient tensor, if sparse.
              None otherwise.
        """
        if isinstance(grad, tf.IndexedSlices):
            grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)
            decay_state = utils.slice_tensor(state["decay"], grad_indices,
                                             param_shape)
            rms_state = utils.slice_tensor(state["rms"], grad_indices, param_shape)
            rnn_state = utils.slice_tensor(state["rnn"], grad_indices, param_shape)
            learning_rate_state = utils.slice_tensor(state["learning_rate"],
                                                      grad_indices, param_shape)
            decay_state.set_shape([None, 1])
            rms_state.set_shape([None, 1])
        else:
            grad_values = grad
            grad_indices = None

            decay_state = state["decay"]
            rms_state = state["rms"]
            rnn_state = state["rnn"]
            learning_rate_state = state["learning_rate"]
        return (grad_values, decay_state, rms_state, rnn_state, learning_rate_state,
                grad_indices)

    def _cell_state_sizes(self):
        """Flat list of individual state-tensor sizes, one entry per state
        tensor across all component cells (a cell with a tuple state_size,
        e.g. an LSTM's (c, h), contributes two entries)."""
        sizes = []
        for cell in self.component_cells:
            state_size = cell.state_size
            if isinstance(state_size, (list, tuple)):
                sizes.extend(state_size)
            else:
                sizes.append(state_size)
        return sizes

    def _unpack_rnn_state(self, rnn_state):
        """Splits the flat per-coordinate rnn state vector into the nested
        per-cell state structure `self.cell` (a StackedRNNCells) expects: a
        list with one entry per component cell, each entry itself a list of
        that cell's state tensor(s) (Keras cells always take/return state as
        a list, even single-tensor-state cells like GRUCell/BiasGRUCell).

        Generalizes the original's tf.split(..., num_or_size_splits=2, ...),
        which hardcoded a 2-tuple (LSTM-shaped) per-cell state and would raise
        for single-tensor-state cells like GRUCell/BiasGRUCell -- the actual
        default cell_cls.
        """
        parts = tf.split(rnn_state, self._cell_state_sizes(), axis=1)
        result = []
        i = 0
        for cell in self.component_cells:
            state_size = cell.state_size
            n = len(state_size) if isinstance(state_size, (list, tuple)) else 1
            result.append(list(parts[i:i + n]))
            i += n
        return result

    def _pack_rnn_state(self, nested_state):
        """Concatenates the nested per-cell state structure back into a single
        flat per-coordinate vector for storage (inverse of _unpack_rnn_state)."""
        flat = []
        for cell_state in nested_state:
            flat.extend(cell_state)
        return tf.concat(flat, axis=1)
