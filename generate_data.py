import os
import numpy as np
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('m', 25, 'Number of measurements (rows of A).')
flags.DEFINE_integer('n', 50, 'Signal dimension (columns of A).')
flags.DEFINE_integer('num_train', 12800, 'Number of training samples.')
flags.DEFINE_integer('num_val', 1280, 'Number of validation samples.')
flags.DEFINE_integer('num_test', 1280, 'Number of test samples.')
flags.DEFINE_float('sparsity', 0.1, 'Fraction of non-zero entries in the sparse signal.')
flags.DEFINE_float('noise_std', 0.0, 'Standard deviation of Gaussian measurement noise.')
flags.DEFINE_integer('seed', 42, 'RNG seed.')
flags.DEFINE_string('output_dir', None, 'Directory to save generated data files.')


def generate_A(M, N, rng):
    A = rng.randn(M, N).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def generate_samples(A, num_samples, sparsity, noise_std, rng):
    M, N = A.shape
    k = max(1, int(round(sparsity * N)))
    data = np.zeros((num_samples, M + N), dtype=np.float32)
    for i in range(num_samples):
        x = np.zeros(N, dtype=np.float32)
        support = rng.choice(N, k, replace=False)
        x[support] = rng.randn(k).astype(np.float32)
        y = A @ x
        if noise_std > 0:
            y += (noise_std * rng.randn(M)).astype(np.float32)
        data[i, :M] = y
        data[i, M:] = x
    return data


def main(_):
    rng = np.random.RandomState(FLAGS.seed)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    A = generate_A(FLAGS.m, FLAGS.n, rng)
    np.save(os.path.join(FLAGS.output_dir, 'A.npy'), A)
    logging.info('Saved A.npy with shape %s', A.shape)

    for split, count in [('train', FLAGS.num_train), ('val', FLAGS.num_val), ('test', FLAGS.num_test)]:
        data = generate_samples(A, count, FLAGS.sparsity, FLAGS.noise_std, rng)
        path = os.path.join(FLAGS.output_dir, f'{split}_data.npy')
        np.save(path, data)
        logging.info('Saved %s with shape %s', path, data.shape)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
