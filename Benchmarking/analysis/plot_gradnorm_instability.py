"""Plots the L2O-DM meta-gradient-norm instability figure used in the
Experiment-2 model-free investigation writeup: raw (unclipped)
tf.linalg.global_norm(meta_grads) per training epoch, log scale, with the
curriculum training-horizon extension (epoch 84, 100-step rung -> 500-step
rung) marked.

Not part of the Benchmarking platform -- a one-off analysis script, same
pattern as plot_ood_degradation.py. Reads the profile.jsonl written by
train_dm.py's profiling.RunProfiler for the scale_diag_profile diagnostic
run (l2o-dm, (m,n)=(25,50), narrow width, seed 0, --if_cl --profile_path),
not from the exp2_real sweep.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

PROFILE_PATH = os.path.join(
    "runs", "scale_diag_profile", "25x50", "checkpoints", "s0", "narrow",
    "l2o-dm", "profile.jsonl")
OUT_NAME = "gradnorm_instability_l2o-dm_25x50"
# Thesis figures directory (repo root, sibling of Benchmarking/). Figures go
# here ONLY: these scripts used to write a second copy into Benchmarking/,
# which left tracked duplicates of every figure in the source tree.
THESIS_FIGS_DIR = os.path.join("..", "Figs")

# Curriculum advanced from the 100-step to the 500-step rung at this epoch
# (confirmed via train.log) -- the point where the previously-stabilized
# meta-gradient destabilizes again.
HORIZON_EXTENSION_EPOCH = 84

# dataviz skill's validated categorical color (orange slot) + light chart
# chrome -- same tokens as plot_ood_degradation.py, for a consistent look
# across every figure in the writeup.
LINE_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def main():
    with open(PROFILE_PATH) as f:
        rows = [json.loads(line) for line in f]
    epochs = np.array([r["epoch"] for r in rows])
    grad_norm = np.array([r["grad_norm"] for r in rows])

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

    ax.plot(epochs, grad_norm, color=LINE_COLOR, linewidth=1.4, zorder=3)

    ax.axvline(HORIZON_EXTENSION_EPOCH, color=INK_MUTED, linestyle="--",
              linewidth=1, alpha=0.7, zorder=2)
    ax.annotate(
        "curriculum extends\ntraining horizon",
        xy=(HORIZON_EXTENSION_EPOCH, ax.get_ylim()[1]),
        xytext=(HORIZON_EXTENSION_EPOCH + 2, 9000),
        fontsize=8, color=INK_MUTED, ha="left", va="top")

    ax.set_title("L2O-DM meta-gradient norm during training (unclipped)",
                fontsize=12, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Training epoch", color=INK_PRIMARY)
    ax.set_ylabel(r"$\Vert\nabla_\phi \mathcal{L}\Vert$ (log scale)",
                 color=INK_PRIMARY, fontsize=10)
    fig.tight_layout()

    for out_dir in (THESIS_FIGS_DIR,):
        out_stem = os.path.join(out_dir, OUT_NAME)
        fig.savefig(out_stem + ".png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
        print("wrote", out_stem + ".png")
    plt.close(fig)


if __name__ == "__main__":
    main()
