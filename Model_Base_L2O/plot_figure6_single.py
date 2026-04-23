"""Plot Figure-6-style modified relative loss for a single LISTA-LASSO model.

Modified relative loss (paper definition):
    R(k) = E_q[f_q(x_k) - f_q*] / E_q[f_q*]
where f_q* is obtained by running FISTA for 2000 iterations on each test sample.
Both axes are on a log scale (log-log), matching Figure 6.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as _tf_v1
_tf_v1.enable_eager_execution()
import tensorflow.compat.v2 as tf

sys.path.insert(0, os.path.dirname(__file__))
import models
import utils as U

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_BASE_DIR  = os.path.dirname(__file__)
_DATA_DIR  = os.path.join(_BASE_DIR, 'data', '25_50')
_MODEL_DIR = os.path.join(
    _BASE_DIR, 'models', 'Lista_lasso-0.005_m25_n50', 'replicate_1'
)
_FIG_OUT   = os.path.join(os.path.dirname(_BASE_DIR), 'Figs', 'figure6_single.png')

LAM         = 0.005
NUM_LAYERS  = 16
FISTA_ITERS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lasso_obj_batch(x: np.ndarray, y: np.ndarray, A: np.ndarray, lam: float) -> np.ndarray:
    """Per-sample LASSO objective. x: (B,N), y: (B,M) → (B,)."""
    residual = x @ A.T - y
    return 0.5 * np.sum(residual ** 2, axis=1) + lam * np.sum(np.abs(x), axis=1)


def run_fista(y: np.ndarray, A: np.ndarray, lam: float, n_iters: int) -> np.ndarray:
    """FISTA from x=0 for n_iters steps. Returns x of shape (B, N)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str, model_dir: str, fig_out: str) -> None:
    # --- Load data ---
    A = np.load(os.path.join(data_dir, 'A.npy')).astype(np.float32)
    test_data = np.load(os.path.join(data_dir, 'test_data.npy')).astype(np.float32)
    M, N = A.shape

    # test_data rows: [y (M), x_true (N)]
    Y = test_data[:, :M]

    # --- Build model and load checkpoint ---
    model = models.Lista(A, NUM_LAYERS, LAM, share_W=False, name='Lista')
    for layer_id in range(NUM_LAYERS):
        model.create_cell(layer_id)
    _ = model(tf.zeros((1, M), dtype=tf.float32))

    ckpt_dir = os.path.join(model_dir, f'layer_{NUM_LAYERS}')
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path).expect_partial()
    print(f'Loaded checkpoint: {ckpt_path}')

    # --- Forward pass over full test set ---
    output = model(tf.constant(Y)).numpy()
    # shape: (num_samples, M + NUM_LAYERS * N)

    # --- LASSO objective at each layer ---
    layer_mean_obj = []
    for k in range(NUM_LAYERS):
        idx = M + k * N
        x_k = output[:, idx: idx + N]
        layer_mean_obj.append(lasso_obj_batch(x_k, Y, A, LAM).mean())

    # --- Reference: FISTA f* (2000 iters, per paper) ---
    print(f'Running FISTA for {FISTA_ITERS} iterations to estimate f* ...')
    x_star   = run_fista(Y, A, LAM, FISTA_ITERS)
    f_star_q = lasso_obj_batch(x_star, Y, A, LAM)   # per-sample, shape (B,)
    mean_f_star = f_star_q.mean()

    print(f'mean f* = {mean_f_star:.6f}')
    for k, fk in enumerate(layer_mean_obj, 1):
        print(f'  layer {k:2d}: mean f(x_k) = {fk:.6f}')

    # --- Modified relative loss (paper formula) ---
    # R(k) = E_q[f_q(x_k) - f_q*] / E_q[f_q*]
    rel_loss = [(fk - mean_f_star) / mean_f_star for fk in layer_mean_obj]

    # --- Log-log plot (matches Figure 6 style) ---
    iterations = list(range(1, NUM_LAYERS + 1))
    plt.figure(figsize=(7, 4))
    plt.loglog(iterations, rel_loss, marker='o', linewidth=2, markersize=5,
               label=r'LISTA ($\lambda=0.005$, $m=25$, $n=50$)')
    plt.xlabel('Number of Iterations')
    plt.ylabel(r'Modified Relative Loss $R_{f,Q}$')
    plt.title('Figure 6 – LASSO Convergence')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    plt.savefig(fig_out, dpi=200)
    plt.close()
    print(f'\nSaved figure to: {fig_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default=_DATA_DIR)
    parser.add_argument('--model_dir', default=_MODEL_DIR)
    parser.add_argument('--fig_out',   default=_FIG_OUT)
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.fig_out)
