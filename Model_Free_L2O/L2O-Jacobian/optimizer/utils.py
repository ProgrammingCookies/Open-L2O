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

"""Utilities and helper functions."""

import numpy as np
import tensorflow as tf


def make_finite(t, replacement):
    """Replaces non-finite tensor values with the replacement value."""
    return tf.where(tf.math.is_finite(t), t, replacement)


def asinh(x):
    """Computes the inverse hyperbolic sine function (in tensorflow)."""
    return tf.math.log(x + tf.sqrt(1. + x ** 2))


class AffineLayer(tf.keras.layers.Layer):
    """Computes an affine function of the inputs.

    Owns "Matrix" and "Bias" weights, created once (in build()) and reused on
    every call — the eager-mode replacement for the TF1 version's
    tf.get_variable-under-variable_scope(reuse=...) weight sharing.

    If the inputs are a list of tensors, they are concatenated together.

    Initial weights for the matrix are drawn from a Gaussian with zero
    mean and standard deviation that is the given scale divided by the
    square root of the input dimension. Initial weights for the bias are
    set to zero.
    """

    def __init__(self, output_size, scale=0.1, vec_mean=0.,
                 include_bias=True, bias_init=0., random_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.scale = scale
        self.vec_mean = vec_mean
        self.include_bias = include_bias
        self.bias_init = bias_init
        self.random_seed = random_seed
        self.matrix = None
        self.bias = None

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and not isinstance(input_shape[0], int):
            input_size = sum(int(s[-1]) for s in input_shape)
        else:
            input_size = int(input_shape[-1])

        sigma = self.scale / np.sqrt(input_size)
        self.matrix = self.add_weight(
            name="Matrix", shape=[input_size, self.output_size], dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=self.vec_mean, stddev=sigma, seed=self.random_seed))

        if self.include_bias:
            self.bias = self.add_weight(
                name="Bias", shape=[self.output_size], dtype=tf.float32,
                initializer=tf.constant_initializer(self.bias_init))
        else:
            self.bias = None

    def call(self, inputs):
        x = tf.concat(inputs, 1) if isinstance(inputs, (list, tuple)) else inputs
        output = tf.matmul(x, self.matrix)
        if self.bias is not None:
            output = output + self.bias
        return output


def project(inputs, weights, bias=0., activation=tf.identity):
    """Computes an affine or linear projection of the inputs.

    Projects the inputs onto the given weight vector and (optionally)
    adds a bias and passes the result through an activation function.

    Args:
      inputs: matrix of inputs with shape [batch_size, dim]
      weights: weight matrix with shape [dim, output_dim]
      bias: bias vector with shape [output_dim] (default: 0)
      activation: nonlinear activation function (default: tf.identity)

    Returns:
      outputs: an op which computes activation(inputs @ weights + bias)
    """
    return activation(tf.matmul(inputs, weights) + bias)


def new_mean_squared(grad_vec, decay, ms):
    """Calculates the new accumulated mean squared of the gradient.

    Args:
      grad_vec: the vector for the current gradient
      decay: the decay term
      ms: the previous mean_squared value

    Returns:
      the new mean_squared value
    """
    decay_size = decay.shape.num_elements()
    tf.debugging.assert_less_equal(decay, 1., summarize=decay_size)
    tf.debugging.assert_greater_equal(decay, 0., summarize=decay_size)

    grad_squared = tf.square(grad_vec)

    # If the previous mean_squared is the 0 vector, don't use the decay and just
    # return the full grad_squared. This should only happen on the first timestep.
    if bool(tf.reduce_all(tf.equal(ms, 0.))):
        decay = tf.zeros_like(decay, dtype=tf.float32)

    # Update the running average of squared gradients.
    epsilon = 1e-12
    return (1. - decay) * (grad_squared + epsilon) + decay * ms


def rms_scaling(gradient, decay, ms, update_ms=True):
    """Vectorizes and scales a tensor of gradients.

    Args:
      gradient: the current gradient
      decay: the current decay value.
      ms: the previous mean squared value
      update_ms: Whether to update the mean squared value (default: True)

    Returns:
      The scaled gradient and the new ms value if update_ms is True,
      the old ms value otherwise.
    """

    # Vectorize the gradients and compute the squared gradients.
    grad_vec = tf.reshape(gradient, [-1, 1])

    if update_ms:
        ms = new_mean_squared(grad_vec, decay, ms)

    # Scale the current gradients by the RMS, squashed by the asinh function.
    scaled_gradient = asinh(grad_vec / tf.sqrt(ms + 1e-16))

    return scaled_gradient, ms


def accumulate_sparse_gradients(grad):
    """Accumulates repeated indices of a sparse gradient update.

    Args:
      grad: a tf.IndexedSlices gradient

    Returns:
      grad_indices: unique indices
      grad_values: gradient values corresponding to the indices
    """

    grad_indices, grad_segments = tf.unique(grad.indices)
    grad_values = tf.math.unsorted_segment_sum(grad.values, grad_segments,
                                                tf.shape(grad_indices)[0])
    return grad_indices, grad_values


def slice_tensor(dense_tensor, indices, head_dims):
    """Extracts slices from a partially flattened dense tensor.

    indices is assumed to index into the first dimension of head_dims.
    dense_tensor is assumed to have a shape [D_0, D_1, ...] such that
    prod(head_dims) == D_0. This function will extract slices along the
    first_dimension of head_dims.

    Example:

    Consider a tensor with shape head_dims = [100, 2] and a dense_tensor with
    shape [200, 3]. Note that the first dimension of dense_tensor equals the
    product of head_dims. This function will reshape dense_tensor such that
    its shape is now [100, 2, 3] (i.e. the first dimension became head-dims)
    and then slice it along the first dimension. After slicing, the slices will
    have their initial dimensions flattened just as they were in dense_tensor
    (e.g. if there are 4 indices, the return value will have a shape of [4, 3]).

    Args:
      dense_tensor: a N-D dense tensor. Shape: [D_0, D_1, ...]
      indices: a 1-D integer tensor. Shape: [K]
      head_dims: True dimensions of the dense_tensor's first dimension.

    Returns:
      Extracted slices. Shape [K, D_1, ...]
    """

    tail_dims = tf.shape(dense_tensor)[1:]
    dense_tensor = tf.reshape(dense_tensor,
                              tf.concat([head_dims, tail_dims], 0))

    slices = tf.gather(dense_tensor, indices)
    # NOTE(siege): This kills the shape annotation.
    return tf.reshape(slices, tf.concat([[-1], tail_dims], 0))


def stack_tensor(slices, indices, dense_tensor, head_dims):
    """Reconsititutes a tensor from slices and corresponding indices.

    This is an inverse operation to slice_tensor. Missing slices are set to 0.

    Args:
      slices: a tensor. Shape [K, D_1, ...]
      indices: a 1-D integer tensor. Shape: [K]
      dense_tensor: the original tensor the slices were taken
        from. Shape: [D_0, D_1, ...]
      head_dims: True dimensions of the dense_tensor's first dimension.

    Returns:
      Reconsituted tensor. Shape: [D_0, D_1, ...]
    """
    indices = tf.cast(indices, tf.int32)

    tail_dims = tf.shape(dense_tensor)[1:]
    dense_shape = tf.concat([head_dims, tail_dims], 0)

    slices = tf.reshape(slices, tf.concat([[-1], dense_shape[1:]], 0))
    indices = tf.expand_dims(indices, -1)

    return tf.reshape(tf.scatter_nd(indices, slices, dense_shape),
                      tf.shape(dense_tensor))


def update_slices(slices, indices, dense_tensor, head_dims):
    """Reconstitutes a tensor from slices and corresponding indices.

    Like _stack_tensor, but instead of setting missing slices to 0, sets them to
    what they were in the original tensor. The return value is reshaped to be
    the same as dense_tensor.

    Args:
      slices: a tensor. Shape [K, D_1, ...]
      indices: a 1-D integer tensor. Shape: [K]
      dense_tensor: the original tensor the slices were taken
        from. Shape: [D_0, D_1, ...]
      head_dims: True dimensions of the dense_tensor's first dimension.

    Returns:
      Reconsituted tensor. Shape: [D_0, D_1, ...]
    """
    indices = tf.cast(indices, tf.int32)

    tail_dims = tf.shape(dense_tensor)[1:]
    dense_shape = tf.concat([head_dims, tail_dims], 0)

    update_mask_vals = tf.fill(tf.shape(indices), 1)
    reshaped_indices = tf.expand_dims(indices, -1)
    update_mask = tf.equal(
        tf.scatter_nd(reshaped_indices, update_mask_vals, head_dims[:1]), 1)

    reshaped_dense_slices = tf.reshape(
        stack_tensor(slices, indices, dense_tensor, head_dims), dense_shape)
    reshaped_dense_tensor = tf.reshape(dense_tensor, dense_shape)

    # TF1's legacy tf.where, given a rank-1 condition and higher-rank x/y,
    # selected whole sub-tensors along axis 0 (rather than broadcasting
    # elementwise). TF2's tf.where always does numpy-style elementwise
    # broadcasting, which misaligns a rank-1 mask against a higher-rank
    # tensor (a [K] mask against a [K, D_1, ...] tensor does not broadcast
    # to select along axis 0). Reshape the mask to [K, 1, 1, ...] so it
    # broadcasts against the trailing dims the way the original intended.
    extra_dims = reshaped_dense_tensor.shape.rank - 1
    update_mask = tf.reshape(update_mask, [-1] + [1] * extra_dims)

    return tf.reshape(
        tf.where(update_mask, reshaped_dense_slices, reshaped_dense_tensor),
        tf.shape(dense_tensor))
