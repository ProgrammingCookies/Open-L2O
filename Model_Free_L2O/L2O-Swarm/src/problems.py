"""Learning 2 Learn problems.

Each problem factory returns a `build` callable.  Calling `build()` creates
fresh tf.Variable objects and returns:
    (x_vars, const_vars, loss_fn)
where
    x_vars    — list of trainable tf.Variable (the optimizee parameters)
    const_vars — list of non-trainable tf.Variable (fixed per episode)
    loss_fn   — callable loss_fn(x_tensors) -> scalar loss
"""

import sys

import numpy as np
import tensorflow as tf

from dataloader import data_loader


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
    images = images.astype(np.float32) / 255.0
    labels = labels.reshape(-1).astype(np.int64)
    return images, labels


def simple():
    """f(x) = x^2."""
    def build():
        x = tf.Variable(tf.ones([]), name="x", dtype=tf.float32)

        def loss_fn(x_tensors):
            return tf.square(x_tensors[0])

        return [x], [], loss_fn
    return build


def simple_multi_optimizer(num_dims=2):
    """Multidimensional simple problem."""
    def build():
        x_vars = [tf.Variable(tf.ones([]), name="x_{}".format(i), dtype=tf.float32)
                  for i in range(num_dims)]

        def loss_fn(x_tensors):
            x = tf.stack(x_tensors)
            return tf.reduce_sum(tf.square(x))

        return x_vars, [], loss_fn
    return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
    """Quadratic problem: f(x) = ||Wx - y||."""
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
            return tf.reduce_sum((product - y) ** 2, 1)

        return [x], [w, y], loss_fn
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
            return product3

        return [x], [w, y, wcos], loss_fn
    return build


def protein_dock(batch_size=128, num_dims=12, stddev=0.5, dtype=tf.float32):
    scoor_init, sq, se, sr, sbasis, seval = data_loader()
    batch_size = 125
    num_dims = 12
    natoms = 100

    def build():
        x = tf.Variable(
            tf.random.normal([batch_size, num_dims], stddev=stddev, dtype=dtype), name="x")

        coor_init = tf.Variable(tf.constant(scoor_init, dtype=dtype),
                                trainable=False, name="coor_init")
        q = tf.Variable(tf.constant(sq, dtype=dtype), trainable=False, name="q")
        e = tf.Variable(tf.constant(se, dtype=dtype), trainable=False, name="e")
        r = tf.Variable(tf.constant(sr, dtype=dtype), trainable=False, name="r")
        basis = tf.Variable(tf.constant(sbasis, dtype=dtype), trainable=False, name="basis")
        eigval = tf.Variable(tf.constant(seval, dtype=dtype), trainable=False, name="eval")

        def loss_fn(x_tensors):
            x_t = x_tensors[0]

            eigval_sqrt = 1.0 / tf.sqrt(eigval)
            product = tf.squeeze(tf.matmul(tf.expand_dims(x_t * eigval_sqrt, 1), basis))
            new_coor = tf.reshape(product, coor_init.shape) + coor_init

            p2 = tf.reduce_sum(new_coor * new_coor, 2)
            p3 = tf.matmul(new_coor, tf.transpose(new_coor, perm=[0, 2, 1]))
            p2 = tf.expand_dims(p2, -1)
            pair_dis = tf.sqrt(p2 - 2 * p3 + tf.transpose(p2, perm=[0, 2, 1]) + 0.01)

            c7_small = tf.cast(tf.math.less(pair_dis, 7), dtype)
            c7 = tf.cast(tf.math.greater(pair_dis, 7), dtype)
            c0 = tf.cast(tf.math.greater(pair_dis, 0.1), dtype)
            c9 = tf.cast(tf.math.less(pair_dis, 9), dtype)

            c79 = c7 * c9 * c0
            c7_small = c7_small * c0

            pair_dis = pair_dis + tf.eye(natoms, num_columns=natoms, batch_shape=[batch_size])

            coeff = q / (4. * pair_dis) + tf.sqrt(e) * ((r / pair_dis) ** 12 - (r / pair_dis) ** 6)

            energy = tf.reduce_mean(tf.reduce_sum(
                c7_small * coeff * 10
                + 10 * c79 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8), 1), -1) - 7000

            return energy

        return [x], [coor_init, q, e, r, basis, eigval], loss_fn
    return build


def ensemble(problems, weights=None):
    """Ensemble of problems."""
    if weights and len(weights) != len(problems):
        raise ValueError("len(weights) != len(problems)")

    build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
                 for p in problems]

    def build():
        all_x, all_c = [], []
        loss_fns = []
        for i, bfn in enumerate(build_fns):
            x_i, c_i, loss_i = bfn()
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


def mnist(layers, activation="sigmoid", batch_size=128, mode="train"):
    """Mnist classification with a multi-layer perceptron."""
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


def cifar10(path=None, conv_channels=None, linear_layers=None,
            batch_norm=True, batch_size=128, mode="train"):
    """Cifar10 classification with a convolutional network."""
    images_np, labels_np = _load_cifar10(mode)
    num_examples = len(images_np)
    images_const = tf.constant(images_np)
    labels_const = tf.constant(labels_np)

    conv_channels = list(conv_channels) if conv_channels else [16, 16, 16]
    linear_layers = list(linear_layers) if linear_layers else [32]

    def build():
        conv_kernels, conv_biases = [], []
        in_ch = 3
        for i, out_ch in enumerate(conv_channels):
            k = tf.Variable(tf.random.normal([5, 5, in_ch, out_ch], stddev=0.01), name="ck_{}".format(i))
            b = tf.Variable(tf.zeros([out_ch]), name="cb_{}".format(i))
            conv_kernels.append(k)
            conv_biases.append(b)
            in_ch = out_ch

        bn_layers = ([tf.keras.layers.BatchNormalization(name="bn_{}".format(i))
                     for i in range(len(conv_channels))] if batch_norm else [])

        # Determine FC input size with a dummy forward pass.
        dummy = np.zeros([1, 32, 32, 3], dtype=np.float32)
        h = dummy
        for i, (k, b) in enumerate(zip(conv_kernels, conv_biases)):
            h = tf.nn.conv2d(h, k.numpy(), [1, 1, 1, 1], "SAME")
            h = tf.nn.bias_add(h, b.numpy())
            if batch_norm:
                h = bn_layers[i](h, training=False)
            h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="SAME")
        fc_in = int(np.prod(h.shape[1:]))

        fc_weights, fc_biases = [], []
        prev_size = fc_in
        for i, size in enumerate(linear_layers + [10]):
            w = tf.Variable(tf.random.normal([prev_size, size], stddev=0.01), name="fw_{}".format(i))
            b = tf.Variable(tf.zeros([size]), name="fb_{}".format(i))
            fc_weights.append(w)
            fc_biases.append(b)
            prev_size = size

        bn_vars = [v for bn in bn_layers for v in bn.trainable_variables] if batch_norm else []
        x_vars = conv_kernels + conv_biases + fc_weights + fc_biases + bn_vars
        n_conv = len(conv_channels)
        n_fc = len(fc_weights)

        def loss_fn(x_tensors):
            ks = x_tensors[:n_conv]
            bs = x_tensors[n_conv:2 * n_conv]
            ws = x_tensors[2 * n_conv:2 * n_conv + n_fc]
            bsf = x_tensors[2 * n_conv + n_fc:2 * n_conv + 2 * n_fc]

            indices = tf.random.uniform([batch_size], 0, num_examples, tf.int64)
            batch_imgs = tf.gather(images_const, indices)
            batch_lbls = tf.gather(labels_const, indices)

            h = batch_imgs
            for i, (k, b) in enumerate(zip(ks, bs)):
                h = tf.nn.conv2d(h, k, [1, 1, 1, 1], "SAME")
                h = tf.nn.bias_add(h, b)
                if batch_norm:
                    h = bn_layers[i](h, training=True)
                h = tf.nn.max_pool2d(tf.nn.relu(h), ksize=2, strides=2, padding="SAME")

            h = tf.reshape(h, [batch_size, -1])
            for i, (w, b) in enumerate(zip(ws, bsf)):
                h = tf.matmul(h, w) + b
                if i < n_fc - 1:
                    h = tf.nn.relu(h)
            return _xent_loss(h, batch_lbls)

        return x_vars, [], loss_fn
    return build
