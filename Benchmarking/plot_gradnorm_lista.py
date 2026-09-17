"""Computes and plots LISTA's per-layer gradient norm from EXISTING trained
checkpoints -- a read-only diagnostic. Does NOT modify Model_Base_L2O's
training code (utils.py/train.py); it only imports the same model/loss
classes those scripts already use and calls them exactly as train.py does.

For each of LISTA's 16 layers, restores that layer's saved checkpoint (the
model's state right after that layer's own progressive 3-stage training
finished -- Model_Base_L2O/train.py's per-layer loop already writes one
checkpoint per layer, unprompted by this script) and recomputes the
LASSO-loss gradient w.r.t. every trainable variable at that point, on one
real batch of training data. That's the direct LISTA analog of L2O-DM's raw
meta-gradient norm (plot_gradnorm_instability.py) -- but at per-LAYER
resolution rather than per-epoch, since LISTA has no single continuous
training loop the way L2O-DM/RNNProp do (it's ~48 separate model.fit() calls,
one per progressive stage per layer); a live per-step trace would require
instrumenting utils.Adam.apply_gradients, which this script deliberately
does not do.

Reads the (25,50)/narrow/seed-0 checkpoints from the scale_diagnostic.py
diagnostic sequence (runs/scale_diag/25x50/...), the same scale as
plot_gradnorm_instability.py's L2O-DM figure, for a same-scale comparison.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_MODEL_BASE_L2O = os.path.abspath(os.path.join("..", "Model_Base_L2O"))
if _MODEL_BASE_L2O not in sys.path:
    sys.path.insert(0, _MODEL_BASE_L2O)

import data_preprocessing  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402

DATA_DIR = os.path.join("runs", "scale_diag", "25x50", "data", "seed0", "narrow")
CHECKPOINT_ROOT = os.path.join(
    "runs", "scale_diag", "25x50", "checkpoints", "s0", "narrow", "lista",
    "checkpoints", "models", "lista__narrow__s0", "replicate_1")
NUM_LAYERS = 16
LASSO_LAM = 0.005
BATCH_SIZE = 256  # one fixed real batch, same batch for every layer

OUT_NAME = "gradnorm_lista_25x50"
THESIS_FIGS_DIR = os.path.join("..", "Figs")

LINE_COLOR = "#2a78d6"  # dataviz skill's categorical blue -- same tokens as
                        # plot_ood_degradation.py / plot_gradnorm_instability.py
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def load_batch():
    """One real, fixed batch of (model_input, label) rows, built exactly the
    way data_preprocessing.dataset_parser does for task='lasso': input is
    [b; x0] (the seeded starting point), label is [b; x_true]."""
    a = np.load(os.path.join(DATA_DIR, "A.npy")).astype(np.float32)
    m, n = a.shape
    train = np.load(os.path.join(DATA_DIR, "train_data.npy")).astype(np.float32)
    x0 = np.load(os.path.join(DATA_DIR, "train_data_x0.npy")).astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(train.shape[0], size=BATCH_SIZE, replace=False)
    b = train[idx, :m]
    x_true = train[idx, m:m + n]
    x0_batch = x0[idx]
    model_input = tf.constant(np.concatenate([b, x0_batch], axis=1))
    label = tf.constant(np.concatenate([b, x_true], axis=1))
    return a, m, n, model_input, label


def main():
    a, m, n, model_input, label = load_batch()
    a_const = tf.constant(a, name="A_lasso_const")
    loss_fn = utils.LassoLoss(a_const, LASSO_LAM, n, None)

    model = models.Lista(a, NUM_LAYERS, LASSO_LAM, False, None, name="Lista")

    grad_norms = []
    for layer_id in range(NUM_LAYERS):
        model.create_cell(layer_id)
        # Trigger a forward pass so every layer's weights are actually built
        # before checkpoint.restore -- same trick evaluate_*.py scripts use
        # for the model-free adapters (dummy step before loading weights).
        _ = model(model_input, training=False)

        layer_dir = os.path.join(CHECKPOINT_ROOT, "layer_{}".format(layer_id + 1))
        ckpt_path = tf.train.latest_checkpoint(layer_dir)
        if ckpt_path is None:
            print("no checkpoint at {}, stopping".format(layer_dir))
            break
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(ckpt_path).expect_partial()

        with tf.GradientTape() as tape:
            y_pred = model(model_input, training=False)
            loss = loss_fn(label, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [g for g in grads if g is not None]
        norm = float(tf.linalg.global_norm(grads))
        grad_norms.append(norm)
        print("layer {:2d}: loss={:.6f} grad_norm={:.6f}".format(
            layer_id + 1, float(loss), norm))

    layers = np.arange(1, len(grad_norms) + 1)
    grad_norms = np.array(grad_norms)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)

    ax.plot(layers, grad_norms, color=LINE_COLOR, linewidth=1.6, marker="o",
           markersize=4, zorder=3)

    ax.set_title("LISTA gradient norm across its 16 layers",
                fontsize=12, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Layer (checkpoint after that layer's training)", color=INK_PRIMARY)
    ax.set_ylabel(r"$\Vert\nabla_\theta \mathcal{L}\Vert$ (log scale)",
                 color=INK_PRIMARY, fontsize=10)
    ax.set_xticks(layers)
    fig.tight_layout()

    for out_dir in (".", THESIS_FIGS_DIR):
        out_stem = os.path.join(out_dir, OUT_NAME)
        fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", out_stem + ".png")
    plt.close(fig)


if __name__ == "__main__":
    main()
