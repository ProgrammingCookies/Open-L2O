"""Figure-6-style log-log plot of modified relative LASSO loss vs iterations.

Each model is opt-in via a flag (all off by default).
--max_iter controls how many layers are shown on the x-axis.

Modified relative loss (paper definition):
    R(k) = E_q[f_q(x_k) - f_q*] / E_q[f_q*]
f_q* is obtained by running FISTA for --fista_iters steps on each test sample.

Example:
    python plot_figure6.py --lista --alista --max_iter 16
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as _tf_v1
_tf_v1.enable_eager_execution()
import tensorflow.compat.v2 as tf

sys.path.insert(0, os.path.dirname(__file__))
import models

# ---------------------------------------------------------------------------
# Model registry
# Each entry: (label, exp_name_prefix, output_interval_fn, builder_fn)
# output_interval_fn(M, N) -> int   — elements each layer appends to output
# builder_fn(A, T, lam, data_dir)  -> keras model (not yet built)
# ---------------------------------------------------------------------------

def _lista_builder(A, T, lam, data_dir):
    return models.Lista(A, T, lam, share_W=False, name='Lista')

def _lfista_builder(A, T, lam, data_dir):
    return models.Lfista(A, T, lam, share_W=False, name='Lfista')

def _lista_cp_builder(A, T, lam, data_dir):
    return models.ListaCp(A, T, lam, share_W=False, name='ListaCp')

def _lista_cpss_builder(A, T, lam, data_dir):
    return models.ListaCpss(A, T, lam, 1.2, 13.0, share_W=False, name='ListaCpss')

def _step_lista_builder(A, T, lam, data_dir):
    return models.StepLista(A, T, lam, name='StepLista')

def _alista_builder(A, T, lam, data_dir):
    W = np.load(os.path.join(data_dir, 'W.npy')).astype(np.float32)
    return models.Alista(A, W, T, lam, 1.2, 13.0, name='Alista')

def _glista_builder(A, T, lam, data_dir):
    return models.Glista(A, T, lam, 1.2, 13.0, share_W=False, name='Glista')

def _lamp_builder(A, T, lam, data_dir):
    return models.Lamp(A, T, lam, share_W=False, name='Lamp')

def _tista_builder(A, T, lam, data_dir):
    return models.Tista(A, T, lam, 1.0, share_W=False, name='Tista')


MODEL_REGISTRY = {
    #  key          label         exp_prefix    interval(M,N)    builder
    'lista':     ('LISTA',      'Lista',      lambda M, N: N,     _lista_builder),
    'lfista':    ('LFISTA',     'Lfista',     lambda M, N: N,     _lfista_builder),
    'lista_cp':  ('LISTA-CP',   'ListaCp',    lambda M, N: N,     _lista_cp_builder),
    'lista_cpss':('LISTA-CPSS', 'ListaCpss',  lambda M, N: N,     _lista_cpss_builder),
    'step_lista':('StepLISTA',  'StepLista',  lambda M, N: N,     _step_lista_builder),
    'alista':    ('ALISTA',     'Alista',     lambda M, N: N,     _alista_builder),
    'glista':    ('GLISTA',     'Glista',     lambda M, N: N * 2, _glista_builder),
    'lamp':      ('LAMP',       'Lamp',       lambda M, N: M + N, _lamp_builder),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lasso_obj_batch(x, y, A, lam):
    """Per-sample LASSO objective. x: (B,N), y: (B,M) -> (B,)."""
    residual = x @ A.T - y
    return 0.5 * np.sum(residual ** 2, axis=1) + lam * np.sum(np.abs(x), axis=1)


def run_fista(y, A, lam, n_iters):
    """FISTA from x=0. Returns x of shape (B, N)."""
    step = 1.0 / np.linalg.norm(A, ord=2) ** 2
    x = np.zeros((y.shape[0], A.shape[1]), dtype=np.float32)
    z = x.copy()
    t = 1.0
    for _ in range(n_iters):
        x_prev = x.copy()
        grad = (z @ A.T - y) @ A
        z_step = z - step * grad
        x = np.sign(z_step) * np.maximum(np.abs(z_step) - step * lam, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        z = x + (t - 1.0) / t_new * (x - x_prev)
        t = t_new
    return x


def load_and_evaluate(key, args, A, Y, M, N):
    """Build model, load checkpoint, return list of mean LASSO obj per layer."""
    label, exp_prefix, interval_fn, builder = MODEL_REGISTRY[key]
    interval = interval_fn(M, N)

    num_layers = args.num_layers
    exp_name   = f'{exp_prefix}_lasso-{args.lam}_m{M}_n{N}'
    model_dir  = os.path.join(args.base_dir, 'models', exp_name,
                               f'replicate_{args.replicate}')

    ckpt_dir  = os.path.join(model_dir, f'layer_{num_layers}')
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f'[{key}] No checkpoint found in {ckpt_dir}\n'
            f'  Expected exp_name: {exp_name}'
        )

    model = builder(A, num_layers, args.lam, args.data_dir)
    for layer_id in range(num_layers):
        model.create_cell(layer_id)
    _ = model(tf.zeros((1, M), dtype=tf.float32))

    tf.train.Checkpoint(model=model).restore(ckpt_path).expect_partial()
    print(f'[{key}] Loaded: {ckpt_path}')

    output = model(tf.constant(Y)).numpy()
    # output shape: (B, M + num_layers * interval)

    n_show = min(args.max_iter, num_layers) if args.max_iter else num_layers
    layer_mean_obj = []
    for k in range(n_show):
        idx_end = M + (k + 1) * interval
        x_k = output[:, idx_end - N: idx_end]
        layer_mean_obj.append(lasso_obj_batch(x_k, Y, A, args.lam).mean())

    return layer_mean_obj


# ---------------------------------------------------------------------------
# Pickle helpers (for L2O-DM / L2O-RNNProp)
# ---------------------------------------------------------------------------

def _plot_pickle(ax, pickle_path, mean_f_star, label, max_iter):
    """Load a loss_record pickle and plot modified relative loss."""
    with open(pickle_path, 'rb') as fh:
        loss_record = pickle.load(fh)

    n_show = min(max_iter, len(loss_record)) if max_iter else len(loss_record)
    vals = loss_record[:n_show]
    rel_loss   = [(v - mean_f_star) / mean_f_star for v in vals]
    iterations = list(range(1, len(rel_loss) + 1))
    ax.loglog(iterations, rel_loss, marker='s', linewidth=2,
              markersize=5, label=label)
    for k, rl in enumerate(rel_loss, 1):
        print('  [{}] iter {:2d}: rel_loss = {:.6e}'.format(label, k, rl))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    enabled = [key for key in MODEL_REGISTRY if getattr(args, key, False)]
    l2o_dm_pickle          = getattr(args, 'l2o_dm_pickle', None)
    l2o_dm_enhanced_pickle = getattr(args, 'l2o_dm_enhanced_pickle', None)
    l2o_rnnprop_pickle     = getattr(args, 'l2o_rnnprop_pickle', None)

    if not enabled and not l2o_dm_pickle and not l2o_dm_enhanced_pickle \
            and not l2o_rnnprop_pickle:
        print('No models selected. Use flags such as --lista --alista '
              '--l2o_dm_pickle PATH to choose models.')
        return

    A = np.load(os.path.join(args.data_dir, 'A.npy')).astype(np.float32)
    test_data = np.load(os.path.join(args.data_dir, 'test_data.npy')).astype(np.float32)
    M, N = A.shape
    Y = test_data[:, :M]

    # FISTA reference
    print(f'Running FISTA for {args.fista_iters} iterations ...')
    x_star      = run_fista(Y, A, args.lam, args.fista_iters)
    f_star_q    = lasso_obj_batch(x_star, Y, A, args.lam)
    mean_f_star = f_star_q.mean()
    print(f'mean f* = {mean_f_star:.6f}')

    fig, ax = plt.subplots(figsize=(7, 4))

    for key in enabled:
        label = MODEL_REGISTRY[key][0]
        try:
            layer_mean_obj = load_and_evaluate(key, args, A, Y, M, N)
        except FileNotFoundError as e:
            print('WARNING: skipping {} — {}'.format(key, e))
            continue

        rel_loss   = [(fk - mean_f_star) / mean_f_star for fk in layer_mean_obj]
        iterations = list(range(1, len(rel_loss) + 1))
        ax.loglog(iterations, rel_loss, marker='o', linewidth=2,
                  markersize=5, label=label)
        for k, rl in enumerate(rel_loss, 1):
            print('  [{}] iter {:2d}: rel_loss = {:.6e}'.format(key, k, rl))

    # --- L2O-DM (pickle) ---
    if l2o_dm_pickle:
        if not os.path.isfile(l2o_dm_pickle):
            print('WARNING: --l2o_dm_pickle not found: {}'.format(l2o_dm_pickle))
        else:
            _plot_pickle(ax, l2o_dm_pickle, mean_f_star, 'L2O-DM', args.max_iter)

    # --- L2O-DM Enhanced (pickle) ---
    if l2o_dm_enhanced_pickle:
        if not os.path.isfile(l2o_dm_enhanced_pickle):
            print('WARNING: --l2o_dm_enhanced_pickle not found: {}'.format(
                l2o_dm_enhanced_pickle))
        else:
            _plot_pickle(ax, l2o_dm_enhanced_pickle, mean_f_star,
                         'L2O-DM+', args.max_iter)

    # --- L2O-RNNProp (pickle) ---
    if l2o_rnnprop_pickle:
        if not os.path.isfile(l2o_rnnprop_pickle):
            print('WARNING: --l2o_rnnprop_pickle not found: {}'.format(
                l2o_rnnprop_pickle))
        else:
            _plot_pickle(ax, l2o_rnnprop_pickle, mean_f_star,
                         'L2O-RNNProp', args.max_iter)

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel(r'Modified Relative Loss $R_{f,Q}$')
    ax.set_title(f'Figure 6 – LASSO ($m={M}$, $n={N}$, $\\lambda={args.lam}$)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.fig_out)), exist_ok=True)
    fig.savefig(args.fig_out, dpi=200)
    plt.close(fig)
    print(f'\nSaved figure to: {args.fig_out}')


if __name__ == '__main__':
    _BASE_DIR = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Model flags (all off by default) ---
    model_group = parser.add_argument_group('models (all off by default)')
    for key in MODEL_REGISTRY:
        model_group.add_argument(f'--{key}', action='store_true', default=False,
                                 help=f'Plot {MODEL_REGISTRY[key][0]}')

    # --- X-axis / iteration control ---
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Max iterations to show on x-axis (default: all layers)')
    parser.add_argument('--num_layers', type=int, default=16,
                        help='Number of layers the models were trained with (default: 16)')

    # --- Problem / experiment settings ---
    parser.add_argument('--lam', type=float, default=0.005,
                        help='LASSO lambda used during training (default: 0.005)')
    parser.add_argument('--data_dir', default=os.path.join(_BASE_DIR, 'data', '25_50'),
                        help='Directory containing A.npy and test_data.npy')
    parser.add_argument('--base_dir', default=_BASE_DIR,
                        help='Base directory containing models/')
    parser.add_argument('--replicate', type=int, default=1,
                        help='Replicate id used during training (default: 1)')
    parser.add_argument('--fista_iters', type=int, default=2000,
                        help='FISTA iterations for f* reference (default: 2000)')

    # --- L2O model-free methods (pickle paths from evaluate_lasso_fixed.py) ---
    l2o_group = parser.add_argument_group('model-free L2O (provide pickle path to enable)')
    l2o_group.add_argument('--l2o_dm_pickle', default=None, metavar='PATH',
                           help='Pickle from evaluate_lasso_fixed.py --model_type dm')
    l2o_group.add_argument('--l2o_dm_enhanced_pickle', default=None, metavar='PATH',
                           help='Pickle from evaluate_lasso_fixed.py for enhanced DM (CL+MT)')
    l2o_group.add_argument('--l2o_rnnprop_pickle', default=None, metavar='PATH',
                           help='Pickle from evaluate_lasso_fixed.py --model_type rnnprop')

    # --- Output ---
    parser.add_argument('--fig_out',
                        default=os.path.join(os.path.dirname(_BASE_DIR), 'Figs', 'figure6.png'),
                        help='Output PNG path')

    main(parser.parse_args())
