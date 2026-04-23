"""Evaluate a trained L2O-DM or L2O-RNNProp on fixed LASSO test data.

Records per-step mean LASSO objective and saves as a pickle file (same format
as evaluate_dm.py / evaluate_rnnprop.py) so plot_figure6.py can consume it.

Usage:
  # From the L2O-DM and L2O-RNNProp directory:
  python evaluate_lasso_fixed.py \
    --model_type dm \
    --path /path/to/saved/dm/model \
    --data_dir /path/to/data/25_50 \
    --lam 0.005 \
    --num_steps 16 \
    --output_path /path/to/output
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf

import problems

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_type", "dm",
                    "Which meta-optimizer to load: 'dm' or 'rnnprop'.")
flags.DEFINE_string("path", None,
                    "Directory of the saved meta-optimizer checkpoint.")
flags.DEFINE_string("data_dir", None,
                    "Directory containing A.npy and test_data.npy.")
flags.DEFINE_float("lam", 0.005, "LASSO regularisation lambda.")
flags.DEFINE_integer("num_steps", 16,
                     "Number of optimizer steps to run (= x-axis length).")
flags.DEFINE_string("output_path", None,
                    "Directory to write the pickle file into.")
flags.DEFINE_float("beta1", 0.95, "RNNProp beta1 (ignored for DM).")
flags.DEFINE_float("beta2", 0.95, "RNNProp beta2 (ignored for DM).")


def main(_):
    # ------------------------------------------------------------------
    # Load fixed test data
    # ------------------------------------------------------------------
    A = np.load(os.path.join(FLAGS.data_dir, 'A.npy')).astype(np.float32)
    test_data = np.load(os.path.join(FLAGS.data_dir, 'test_data.npy')).astype(np.float32)
    M, N = A.shape
    Y = test_data[:, :M]           # measurements  (B, M)
    B = Y.shape[0]

    # Build (B, M, N) and (B, M, 1) arrays for lasso_fixed
    data_A = np.tile(A[np.newaxis], (B, 1, 1))   # (B, M, N)
    data_b = Y[:, :, np.newaxis]                   # (B, M, 1)

    print("=" * 70)
    print("Evaluating L2O-{} on fixed LASSO: A={}, B={}, lam={}".format(
        FLAGS.model_type.upper(), A.shape, B, FLAGS.lam))
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build problem and meta-optimizer graph
    # ------------------------------------------------------------------
    problem_fn = problems.lasso_fixed(data_A, data_b, l=FLAGS.lam)

    if FLAGS.model_type == 'dm':
        import meta as meta_module
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": FLAGS.path,
        }}
        optimizer = meta_module.MetaOptimizer(**net_config)
        meta_result = optimizer.meta_loss(problem_fn, 1, net_assignments=None)
        _, update, reset, cost_op, _ = meta_result
        step_pl = None

    elif FLAGS.model_type == 'rnnprop':
        import meta_rnnprop_eval as meta_module
        net_config = {"rp": {
            "net": "RNNprop",
            "net_options": {
                "layers": (20, 20),
                "preprocess_name": "fc",
                "preprocess_options": {"dim": 20},
                "scale": 0.01,
                "tanh_output": True,
            },
            "net_path": FLAGS.path,
        }}
        optimizer = meta_module.MetaOptimizer(FLAGS.beta1, FLAGS.beta2,
                                              **net_config)
        meta_result, _, _, step_pl = optimizer.meta_loss(
            problem_fn, 1, net_assignments=None)
        _, update, reset, cost_op, _ = meta_result

    else:
        raise ValueError("model_type must be 'dm' or 'rnnprop', "
                         "got '{}'".format(FLAGS.model_type))

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with ms.MonitoredSession(session_creator=ms.ChiefSessionCreator(
            config=config)) as sess:
        sess.run(reset)
        tf.get_default_graph().finalize()

        loss_record = []
        feed_dict = {}
        for i in xrange(FLAGS.num_steps):
            if step_pl is not None:
                feed_dict[step_pl] = i + 1
            cost = sess.run([cost_op, update], feed_dict=feed_dict)[0]
            loss_record.append(float(cost))
            print("  step {:3d}: cost = {:.6e}".format(i + 1, cost))

    # ------------------------------------------------------------------
    # Save pickle
    # ------------------------------------------------------------------
    out_dir = FLAGS.output_path or '.'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    suffix = 'lasso_fixed_{}'.format(FLAGS.model_type)
    out_file = os.path.join(out_dir,
                            'L2L_eval_loss_record.pickle-{}'.format(suffix))
    with open(out_file, 'wb') as fh:
        pickle.dump(loss_record, fh)
    print("Saved pickle to: {}".format(out_file))


if __name__ == "__main__":
    tf.app.run()
