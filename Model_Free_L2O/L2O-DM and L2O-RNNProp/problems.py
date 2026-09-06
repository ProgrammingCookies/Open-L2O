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
"""Learning 2 Learn problems.

Each problem factory returns a `build` callable.  Calling `build()` creates
fresh tf.Variable objects and returns:
    (x_vars, const_vars, loss_fn)
where
    x_vars    — list of trainable tf.Variable (the optimizee parameters)
    const_vars — list of non-trainable tf.Variable (fixed per episode)
    loss_fn   — callable loss_fn(x_tensors) -> scalar loss
"""

import os
import sys
import tarfile

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz"
_CIFAR10_FILE = "cifar-10-python.tar.gz"
_CIFAR10_FOLDER = "cifar-10-batches-py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xent_loss(output, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)
    return tf.reduce_mean(loss)


def _load_mnist(mode="train"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if mode == "train":
        images, labels = x_train, y_train
    else:
        images, labels = x_test, y_test
    images = images.reshape(-1, 784).astype(np.float32) / 255.0
    labels = labels.astype(np.int64)
    return images, labels


def _load_cifar10(mode="train"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if mode == "train":
        images, labels = x_train, y_train
    else:
        images, labels = x_test, y_test
    images = images.astype(np.float32) / 255.0          # [N, 32, 32, 3]
    labels = labels.reshape(-1).astype(np.int64)        # [N]
    return images, labels


# ---------------------------------------------------------------------------
# Simple problems
# ---------------------------------------------------------------------------

def simple():
    """f(x) = x^2."""
    def build():
        x = tf.Variable(tf.ones([]), name="x", dtype=tf.float32)

        def loss_fn(x_tensors):
            return tf.square(x_tensors[0])

        return [x], [], loss_fn
    return build


def simple_multi_optimizer(num_dims=2):
    """Multi-dimensional f(x) = sum(x_i^2)."""
    def build():
        x_vars = [tf.Variable(tf.ones([]), name="x_{}".format(i), dtype=tf.float32)
                  for i in range(num_dims)]

        def loss_fn(x_tensors):
            x = tf.stack(x_tensors)
            return tf.reduce_sum(tf.square(x))

        return x_vars, [], loss_fn
    return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
    """f(x) = ||Wx - y||^2."""
    def build():
        x = tf.Variable(
            tf.random.normal([batch_size, num_dims], stddev=stddev, dtype=dtype), name="x")
        w = tf.Variable(
            tf.random.uniform([batch_size, num_dims, num_dims], dtype=dtype),
            trainable=False, name="w")
        y = tf.Variable(
            tf.random.uniform([batch_size, num_dims], dtype=dtype),
            trainable=False, name="y")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            product = tf.squeeze(tf.matmul(w, tf.expand_dims(x_t, -1)))
            return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

        return [x], [w, y], loss_fn
    return build


def lasso(batch_size=128, num_dims=10, stddev=0.01, l=0.005, dtype=tf.float32):
    """f(x) = 0.5*||Wx - y||^2 + lambda*||x||_1."""
    def build():
        x = tf.Variable(
            tf.random.normal([batch_size, num_dims], stddev=stddev, dtype=dtype), name="x")
        w = tf.Variable(
            tf.random.uniform([batch_size, num_dims, num_dims], dtype=dtype),
            trainable=False, name="w")
        y = tf.Variable(
            tf.random.uniform([batch_size, num_dims, 1], dtype=dtype),
            trainable=False, name="y")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            product = tf.matmul(w, tf.expand_dims(x_t, -1))
            left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
            other_term = l * tf.norm(x_t, ord=1, axis=1, keepdims=True)
            return tf.reduce_mean(left_term + other_term)

        return [x], [w, y], loss_fn
    return build


def lasso_fixed(data_A, data_b, stddev=0.01, l=0.005, dtype=tf.float32):
    """Lasso with fixed A and b matrices."""
    a = data_A
    b = data_b
    print("=" * 100)
    print("LASSO: A_size={} b_size={}".format(a.shape, b.shape))
    print("=" * 100)

    w_const = tf.constant(a, dtype=dtype)
    y_const = tf.constant(b, dtype=dtype)

    def build():
        x = tf.Variable(
            tf.random.normal([a.shape[0], a.shape[2]], stddev=stddev, dtype=dtype), name="x")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            product = tf.matmul(w_const, tf.expand_dims(x_t, -1))
            left_term = 0.5 * tf.reduce_sum((product - y_const) ** 2, 1)
            other_term = l * tf.norm(x_t, ord=1, axis=1, keepdims=True)
            return tf.reduce_mean(left_term + other_term)

        return [x], [], loss_fn
    return build


def lasso_from_dataset(data_dir, split="train_data.npy", batch_size=128, l=0.005,
                       dtype=tf.float32, deterministic=False, x0_mode="aligned",
                       x0_stddev=0.01):
    """Lasso problem sampled from a pre-generated dataset´.
    Loads a single shared dictionary A.npy (m, n) and a split file
    whose shape is [b (m,); x_true (n,)].

    x0_mode="aligned" (default): also loads the sibling
    <split>_x0.npy (shape (num_samples, n)) written by
    Benchmarking/data/lasso.py -- the seeded starting point every method
    (including LISTA/ALISTA) begins its recovery trajectory from for that
    same instance and seed. Requires the sibling file to exist.

    x0_mode="random": no x0 sibling file needed/used as each ``build()``
    call draws a fresh `N(0, x0_stddev^2)`` init instead (the pre-x0-
    alignment convention)

    deterministic=False (default, used for training): every call draws a
    fresh random batch, with replacement, from the whole split -- standard
    stochastic-minibatch behaviour.

    deterministic=True (used for evaluation): calls walk the split
    exactly once, in order, batch_size rows at a time.
    """
    if x0_mode not in ("aligned", "random"):
        raise ValueError("x0_mode must be 'aligned' or 'random', got {!r}".format(x0_mode))

    a = np.load(os.path.join(data_dir, "A.npy")).astype(np.float32)
    data = np.load(os.path.join(data_dir, split)).astype(np.float32)
    m, n = a.shape
    if data.shape[1] != m + n:
        raise ValueError(
            "{} has row width {}, expected {} (= m={} + n={} from {}/A.npy)".format(
                os.path.join(data_dir, split), data.shape[1], m + n, m, n, data_dir))

    x0_data = None
    if x0_mode == "aligned":
        x0_path = os.path.join(data_dir, split[:-len(".npy")] + "_x0.npy")
        if not os.path.exists(x0_path):
            raise FileNotFoundError(
                "{} not found -- x0_mode='aligned' requires a seeded x0 sibling "
                "file generated by Benchmarking/data/lasso.py alongside {}. Pass "
                "x0_mode='random' for datasets that don't have one.".format(
                    x0_path, split))
        x0_data = np.load(x0_path).astype(np.float32)
        if x0_data.shape != (data.shape[0], n):
            raise ValueError(
                "{} has shape {}, expected {} (= num_samples={}, n={})".format(
                    x0_path, x0_data.shape, (data.shape[0], n), data.shape[0], n))

    num_samples = data.shape[0]
    a_const = tf.constant(a, dtype=dtype)
    cursor = {"pos": 0}

    def build():
        if deterministic:
            start = cursor["pos"]
            end = min(start + batch_size, num_samples)
            idx = np.arange(start, end)
            cursor["pos"] = end % num_samples
        else:
            idx = np.random.randint(0, num_samples, size=batch_size)
        batch = data[idx]
        build.last_x_true = batch[:, m:]
        build.last_b = batch[:, :m]

        if x0_mode == "random":
            x0_init = tf.random.normal((len(idx), n), stddev=x0_stddev, dtype=dtype)
        else:
            x0_init = tf.constant(x0_data[idx], dtype=dtype)
        x = tf.Variable(x0_init, name="x")
        b_const = tf.constant(batch[:, :m], dtype=dtype)

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            residual = tf.matmul(x_t, a_const, transpose_b=True) - b_const
            left_term = 0.5 * tf.reduce_sum(residual ** 2, axis=1)
            other_term = l * tf.norm(x_t, ord=1, axis=1)
            return tf.reduce_mean(left_term + other_term)

        return [x], [], loss_fn

    build.last_x_true = None
    build.last_b = None
    build.num_samples = num_samples
    return build


def rastrigin(batch_size=128, num_dims=10, alpha=10, stddev=1, dtype=tf.float32):
    def build():
        x = tf.Variable(
            tf.random.normal([batch_size, num_dims, 1], stddev=stddev, dtype=dtype), name="x")
        A = tf.Variable(
            tf.random.normal([batch_size, num_dims, num_dims], stddev=stddev, dtype=dtype),
            trainable=False, name="A")
        B = tf.Variable(
            tf.random.normal([batch_size, num_dims, 1], stddev=stddev, dtype=dtype),
            trainable=False, name="B")
        C = tf.Variable(
            tf.random.normal([batch_size, num_dims, 1], stddev=stddev, dtype=dtype),
            trainable=False, name="C")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            product = tf.matmul(A, x_t)
            ras_norm = tf.norm(product - B, ord=2, axis=[-2, -1])
            cqTcos = tf.squeeze(
                tf.matmul(tf.transpose(C, perm=[0, 2, 1]), tf.cos(2 * np.pi * x_t)))
            return tf.reduce_mean(0.5 * (ras_norm ** 2) - alpha * cqTcos + alpha * num_dims)

        return [x], [A, B, C], loss_fn
    return build


def square_cos(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
    def build():
        x = tf.Variable(
            tf.random.normal([batch_size, num_dims], stddev=stddev, dtype=dtype), name="x")
        w = tf.Variable(
            tf.random.uniform([batch_size, num_dims, num_dims], dtype=dtype),
            trainable=False, name="w")
        y = tf.Variable(
            tf.random.uniform([batch_size, num_dims], dtype=dtype),
            trainable=False, name="y")
        wcos = tf.Variable(
            tf.random.uniform([batch_size, num_dims, num_dims], dtype=dtype),
            trainable=False, name="wcos")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]
            product = tf.squeeze(tf.matmul(w, tf.expand_dims(x_t, -1)))
            product2 = tf.squeeze(
                tf.matmul(wcos, tf.expand_dims(10 * tf.math.cos(2 * 3.1415926 * x_t), -1)))
            product3 = (tf.reduce_sum((product - y) ** 2, 1)
                        - tf.reduce_sum(product2, 1)
                        + 10 * num_dims)
            return tf.reduce_mean(product3)

        return [x], [w, y, wcos], loss_fn
    return build


def ensemble(problems, weights=None):
    """Weighted sum of multiple problems."""
    if weights and len(weights) != len(problems):
        raise ValueError("len(weights) != len(problems)")

    build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
                 for p in problems]

    def build():
        all_x, all_c = [], []
        loss_fns = []
        offsets = []
        offset = 0
        for i, bfn in enumerate(build_fns):
            x_i, c_i, loss_i = bfn()
            offsets.append((offset, offset + len(x_i)))
            offset += len(x_i)
            all_x.extend(x_i)
            all_c.extend(c_i)
            loss_fns.append((loss_i, len(x_i), weights[i] if weights else 1.0))

        def loss_fn(x_tensors):
            total = 0.0
            off = 0
            for loss_i, n_xi, w_i in loss_fns:
                total = total + w_i * loss_i(x_tensors[off: off + n_xi])
                off += n_xi
            return total

        return all_x, all_c, loss_fn
    return build


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def mnist(layers, activation="sigmoid", batch_size=128, mode="train"):
    """MNIST classification with a multi-layer perceptron."""
    if activation == "sigmoid":
        act_fn = tf.sigmoid
    elif activation == "relu":
        act_fn = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))

    images_np, labels_np = _load_mnist(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)

    def build():
        layer_sizes = [784] + list(layers) + [10]
        x_vars = []
        for i in range(len(layer_sizes) - 1):
            w = tf.Variable(
                tf.random.normal([layer_sizes[i], layer_sizes[i + 1]], stddev=0.01),
                name="w_{}".format(i))
            b = tf.Variable(
                tf.random.normal([layer_sizes[i + 1]], stddev=0.01),
                name="b_{}".format(i))
            x_vars.extend([w, b])

        def loss_fn(x_tensors):
            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_images = tf.gather(images_const, indices)
            batch_labels = tf.gather(labels_const, indices)

            h = batch_images
            for i in range(0, len(x_tensors) - 2, 2):
                w_t, b_t = x_tensors[i], x_tensors[i + 1]
                h = act_fn(tf.matmul(h, w_t) + b_t)
            w_out, b_out = x_tensors[-2], x_tensors[-1]
            logits = tf.matmul(h, w_out) + b_out
            return _xent_loss(logits, batch_labels)

        return x_vars, [], loss_fn
    return build


def mnist_conv(batch_norm=True, batch_size=128, mode="train"):
    """MNIST classification with a small convolutional network."""
    images_np, labels_np = _load_mnist(mode)
    num_examples = len(images_np)
    # Reshape to [N, 28, 28, 1]
    images_np_4d = images_np.reshape(-1, 28, 28, 1)
    images_const = tf.constant(images_np_4d)
    labels_const = tf.constant(labels_np)

    def build():
        # Conv layer 1: 3x3, 1->16
        k1 = tf.Variable(tf.random.normal([3, 3, 1, 16], stddev=0.01), name="k1")
        b1 = tf.Variable(tf.zeros([16]), name="b1")
        # Conv layer 2: 5x5, 16->32
        k2 = tf.Variable(tf.random.normal([5, 5, 16, 32], stddev=0.01), name="k2")
        b2 = tf.Variable(tf.zeros([32]), name="b2")

        # We need to figure out the FC input size; build once with dummy input
        dummy = np.zeros([1, 28, 28, 1], dtype=np.float32)
        h = tf.nn.conv2d(dummy, k1.numpy(), [1, 1, 1, 1], "VALID")
        h = tf.nn.bias_add(h, b1)
        h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")
        h = tf.nn.conv2d(h, k2.numpy(), [1, 1, 1, 1], "VALID")
        h = tf.nn.bias_add(h, b2)
        h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")
        fc_in = int(np.prod(h.shape[1:]))

        w_fc = tf.Variable(tf.random.normal([fc_in, 10], stddev=0.01), name="w_fc")
        b_fc = tf.Variable(tf.zeros([10]), name="b_fc")

        bn1 = tf.keras.layers.BatchNormalization(name="bn1") if batch_norm else None
        bn2 = tf.keras.layers.BatchNormalization(name="bn2") if batch_norm else None
        bn_vars = (bn1.trainable_variables + bn2.trainable_variables
                   if batch_norm else [])

        x_vars = [k1, b1, k2, b2, w_fc, b_fc] + bn_vars

        def loss_fn(x_tensors, training=True):
            k1_t, b1_t, k2_t, b2_t, w_fc_t, b_fc_t = x_tensors[:6]
            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)

            h = tf.nn.conv2d(batch_imgs, k1_t, [1, 1, 1, 1], "VALID")
            h = tf.nn.bias_add(h, b1_t)
            if batch_norm:
                h = bn1(h, training=training)
            h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")

            h = tf.nn.conv2d(h, k2_t, [1, 1, 1, 1], "VALID")
            h = tf.nn.bias_add(h, b2_t)
            if batch_norm:
                h = bn2(h, training=training)
            h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")

            h = tf.reshape(h, [batch_size, -1])
            logits = tf.nn.relu(tf.matmul(h, w_fc_t) + b_fc_t)
            return _xent_loss(logits, batch_lbls)

        return x_vars, [], loss_fn
    return build


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def cifar10(path=None, batch_norm=True, batch_size=128, mode="train"):
    """CIFAR-10 classification with a small convolutional network."""
    images_np, labels_np = _load_cifar10(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)

    def build():
        k1 = tf.Variable(tf.random.normal([3, 3, 3, 16], stddev=0.01), name="k1")
        b1 = tf.Variable(tf.zeros([16]), name="b1")
        k2 = tf.Variable(tf.random.normal([5, 5, 16, 32], stddev=0.01), name="k2")
        b2 = tf.Variable(tf.zeros([32]), name="b2")

        dummy = np.zeros([1, 32, 32, 3], dtype=np.float32)
        h = tf.nn.conv2d(dummy, k1.numpy(), [1, 2, 2, 1], "VALID")
        h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")
        h = tf.nn.conv2d(h, k2.numpy(), [1, 2, 2, 1], "VALID")
        h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")
        fc_in = int(np.prod(h.shape[1:]))

        w_fc = tf.Variable(tf.random.normal([fc_in, 10], stddev=0.01), name="w_fc")
        b_fc = tf.Variable(tf.zeros([10]), name="b_fc")

        bn1 = tf.keras.layers.BatchNormalization(name="bn1") if batch_norm else None
        bn2 = tf.keras.layers.BatchNormalization(name="bn2") if batch_norm else None
        bn_vars = (bn1.trainable_variables + bn2.trainable_variables
                   if batch_norm else [])

        x_vars = [k1, b1, k2, b2, w_fc, b_fc] + bn_vars

        def loss_fn(x_tensors, training=True):
            k1_t, b1_t, k2_t, b2_t, w_fc_t, b_fc_t = x_tensors[:6]
            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)

            h = tf.nn.conv2d(batch_imgs, k1_t, [1, 2, 2, 1], "VALID")
            h = tf.nn.bias_add(h, b1_t)
            if batch_norm:
                h = bn1(h, training=training)
            h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")

            h = tf.nn.conv2d(h, k2_t, [1, 2, 2, 1], "VALID")
            h = tf.nn.bias_add(h, b2_t)
            if batch_norm:
                h = bn2(h, training=training)
            h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="VALID")

            h = tf.reshape(h, [batch_size, -1])
            logits = tf.nn.relu(tf.matmul(h, w_fc_t) + b_fc_t)
            return _xent_loss(logits, batch_lbls)

        return x_vars, [], loss_fn
    return build


def LeNet(path=None, conv_channels=None, linear_layers=None,
          batch_norm=True, batch_size=128, mode="train"):
    """LeNet-style network on CIFAR-10."""
    images_np, labels_np = _load_cifar10(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)

    if conv_channels is None:
        conv_channels = (6, 16)
    if linear_layers is None:
        linear_layers = (120, 84)

    def build():
        # Build conv layers
        conv_kernels = []
        conv_biases = []
        in_ch = 3
        for i, out_ch in enumerate(conv_channels):
            k = tf.Variable(tf.random.normal([5, 5, in_ch, out_ch], stddev=0.01), name="ck_{}".format(i))
            b = tf.Variable(tf.zeros([out_ch]), name="cb_{}".format(i))
            conv_kernels.append(k)
            conv_biases.append(b)
            in_ch = out_ch

        # Determine FC input size with dummy forward
        dummy = np.zeros([1, 32, 32, 3], dtype=np.float32)
        h = dummy
        for k, b in zip(conv_kernels, conv_biases):
            h = tf.sigmoid(tf.nn.bias_add(
                tf.nn.conv2d(h, k.numpy(), [1, 1, 1, 1], "VALID"), b))
            h = tf.nn.max_pool2d(h, ksize=2, strides=2, padding="VALID")
        fc_in = int(np.prod(h.shape[1:]))

        # Build FC layers
        fc_weights = []
        fc_biases = []
        prev_size = fc_in
        for i, size in enumerate(list(linear_layers) + [10]):
            w = tf.Variable(tf.random.normal([prev_size, size], stddev=0.01), name="fw_{}".format(i))
            b = tf.Variable(tf.zeros([size]), name="fb_{}".format(i))
            fc_weights.append(w)
            fc_biases.append(b)
            prev_size = size

        x_vars = conv_kernels + conv_biases + fc_weights + fc_biases
        n_conv = len(conv_channels)

        def loss_fn(x_tensors):
            ks = x_tensors[:n_conv]
            bs_c = x_tensors[n_conv:2 * n_conv]
            ws = x_tensors[2 * n_conv:2 * n_conv + len(fc_weights)]
            bs_f = x_tensors[2 * n_conv + len(fc_weights):]

            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)

            h = batch_imgs
            for k, b in zip(ks, bs_c):
                h = tf.sigmoid(tf.nn.bias_add(
                    tf.nn.conv2d(h, k, [1, 1, 1, 1], "VALID"), b))
                h = tf.nn.max_pool2d(h, ksize=2, strides=2, padding="VALID")
            h = tf.reshape(h, [batch_size, -1])
            for i, (w, b) in enumerate(zip(ws, bs_f)):
                h = tf.sigmoid(tf.matmul(h, w) + b) if i < len(ws) - 1 else tf.matmul(h, w) + b
            return _xent_loss(h, batch_lbls)

        return x_vars, [], loss_fn
    return build


def NAS(path=None, batch_norm=True, batch_size=128, mode="train"):
    """NAS-like cell network on CIFAR-10."""
    images_np, labels_np = _load_cifar10(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)

    def build():
        def _make_conv(in_ch, out_ch, name):
            k = tf.Variable(tf.random.normal([3, 3, in_ch, out_ch], stddev=0.01), name=name + "_k")
            b = tf.Variable(tf.zeros([out_ch]), name=name + "_b")
            return k, b

        n0k, n0b = _make_conv(3, 16, "node0")
        n0_2k, n0_2b = _make_conv(16, 16, "node0_2")
        n1k, n1b = _make_conv(16, 16, "node1")
        n1_3k, n1_3b = _make_conv(16, 16, "node1_3")
        w_fc = tf.Variable(tf.random.normal([16, 10], stddev=0.01), name="fc_w")
        b_fc = tf.Variable(tf.zeros([10]), name="fc_b")

        x_vars = [n0k, n0b, n0_2k, n0_2b, n1k, n1b, n1_3k, n1_3b, w_fc, b_fc]

        def conv(x, k, b):
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, k, [1, 1, 1, 1], "SAME"), b))

        def loss_fn(x_tensors):
            n0k_t, n0b_t, n0_2k_t, n0_2b_t = x_tensors[0], x_tensors[1], x_tensors[2], x_tensors[3]
            n1k_t, n1b_t, n1_3k_t, n1_3b_t = x_tensors[4], x_tensors[5], x_tensors[6], x_tensors[7]
            w_fc_t, b_fc_t = x_tensors[8], x_tensors[9]

            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)

            node0 = conv(batch_imgs, n0k_t, n0b_t)
            node0_onto_node2 = conv(node0, n0_2k_t, n0_2b_t)
            node1 = conv(node0, n1k_t, n1b_t)
            node1_onto_node3 = conv(node1, n1_3k_t, n1_3b_t)
            node2 = tf.nn.avg_pool2d(node1, ksize=3, strides=1, padding="SAME") + node0_onto_node2
            node3 = node2 + node1_onto_node3 + node0
            node_final = tf.reduce_mean(tf.reshape(node3, [batch_size, -1, 16]), axis=1)
            logits = tf.nn.relu(tf.matmul(node_final, w_fc_t) + b_fc_t)
            return _xent_loss(logits, batch_lbls)

        return x_vars, [], loss_fn
    return build


def vgg16_cifar10(path=None, batch_norm=False, batch_size=128, mode="train"):
    """CIFAR-10 with VGG16."""
    from vgg16 import VGG16
    images_np, labels_np = _load_cifar10(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)
    vgg = VGG16(0.5, 10)

    def build():
        # Build VGG by doing a dummy forward pass to create variables
        dummy = np.zeros([1, 32, 32, 3], dtype=np.float32)
        vgg._build_model(tf.constant(dummy))
        x_vars = vgg.trainable_variables

        def loss_fn(x_tensors):
            # VGG16 is now a Keras model; x_tensors are its weights
            # Apply weights temporarily
            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)
            output = vgg._build_model(batch_imgs)
            return _xent_loss(output, batch_lbls)

        return list(x_vars), [], loss_fn
    return build


# ---------------------------------------------------------------------------
# Confocal microscopy (training / simulation mode only)
# ---------------------------------------------------------------------------

def confocal_microscopy_3d(batch_size=128, num_points=5, ROI=None,
                           stddev=0.01, dtype=tf.float32, inference=False):
    if ROI is None:
        ROI = [28, 28, 28]

    if inference:
        raise NotImplementedError(
            "inference=True requires an external image array; "
            "pass it via loss_fn(x_tensors, img) after migration.")

    def _psf(theta, roi):
        priors = [
            tfd.Uniform(low=0.5, high=2.0),
            tfd.Uniform(low=0.5, high=float(roi[0] - 1)),
            tfd.Uniform(low=0.5, high=float(roi[1] - 1)),
            tfd.Uniform(low=0.5, high=float(roi[2] - 1)),
            tfd.Uniform(low=2.0, high=4.0),
            tfd.Uniform(low=2.0, high=4.0),
        ]
        xs = tf.linspace(0.0, float(roi[0] - 1), roi[0])
        ys = tf.linspace(0.0, float(roi[1] - 1), roi[1])
        zs = tf.linspace(0.0, float(roi[2] - 1), roi[2])
        X, Y, Z = tf.meshgrid(xs, ys, zs)

        I0 = priors[0].quantile(tf.reshape(theta[0], [theta[0].shape[0], 1]))
        x0 = priors[1].quantile(tf.reshape(theta[1], [theta[1].shape[0], 1]))
        y0 = priors[2].quantile(tf.reshape(theta[2], [theta[2].shape[0], 1]))
        z0 = priors[3].quantile(tf.reshape(theta[3], [theta[3].shape[0], 1]))
        sxy = priors[4].quantile(tf.reshape(theta[4], [theta[4].shape[0], 1]))
        sz = priors[5].quantile(tf.reshape(theta[5], [theta[5].shape[0], 1]))

        xk = tf.reshape(X, [1, -1])
        yk = tf.reshape(Y, [1, -1])
        zk = tf.reshape(Z, [1, -1])
        sq2 = tf.math.sqrt(2.0)

        I = I0 * (
            (-tf.math.erf((-0.5 - x0 + xk) / (sq2 * sxy))
             + tf.math.erf((0.5 - x0 + xk) / (sq2 * sxy)))
            * (-tf.math.erf((-0.5 - y0 + yk) / (sq2 * sxy))
               + tf.math.erf((0.5 - y0 + yk) / (sq2 * sxy)))
            * (-tf.math.erf((-0.5 - z0 + zk) / (sq2 * sz))
               + tf.math.erf((0.5 - z0 + zk) / (sq2 * sz)))
        ) / 8.0
        return I

    def build():
        def _make_vars(suffix, trainable):
            init = tf.random.uniform if trainable else tf.random.uniform
            return [
                tf.Variable(init([batch_size, 1], dtype=dtype),
                            trainable=trainable, name="{}_{}".format(n, suffix))
                for n in ["I", "x", "y", "z", "sxy", "sz"]
            ]

        opt_vars_list = [_make_vars(str(i), True) for i in range(num_points)]
        sim_vars_list = [_make_vars("sim_{}".format(i), False) for i in range(num_points)]

        bg_var = tf.Variable(tf.random.normal([batch_size, 1], stddev=stddev, dtype=dtype),
                             name="bg_var")
        bg_sim = tf.Variable(tf.random.uniform([batch_size, 1], dtype=dtype),
                             trainable=False, name="bg_sim")

        x_vars = [v for group in opt_vars_list for v in group] + [bg_var]
        const_vars = [v for group in sim_vars_list for v in group] + [bg_sim]

        def loss_fn(x_tensors):
            n_opt = len(x_tensors) - 1
            vpt = 6
            n_pts = n_opt // vpt
            bg_v = x_tensors[-1]

            y_pred = tf.add_n([
                _psf([x_tensors[i * vpt + j] for j in range(vpt)], ROI)
                for i in range(n_pts)
            ])
            y_sim = tf.add_n([
                _psf([sim_vars_list[i][j] for j in range(vpt)], ROI)
                for i in range(num_points)
            ])
            target = tf.math.l2_normalize(y_sim + bg_sim, axis=1)
            return tf.reduce_mean(
                tf.math.reduce_sum((y_pred + bg_v - target) ** 2, axis=1))

        return x_vars, const_vars, loss_fn
    return build
