"""Opt-in per-run profiling for Experiment 1 (peak
memory, meta-gradient norm, timing). A no-op unless --profile_path is set,
so ordinary training runs are unaffected.

Cross-method comparability note: DM/RNNProp log one point per meta-training
EPOCH (which internally runs several unroll segments), while L2O-Scale
(problem_generator.py's separate profiling.py copy) logs one point per
UNROLL SEGMENT. So "how many logged points per second" is not the same
quantity across the two libraries, and `avg_time_per_epoch_s` /
`mean_loss` / `mean_grad_norm` are NOT directly comparable across methods
for that reason -- they average over different-sized populations. What IS
comparable is `avg_time_per_optimizee_step_s` (every logged point also
carries how many optimizee steps it covered, via `num_optimizee_steps`),
and the `final_*` fields (both libraries' last logged point is the true
end of that run's training horizon).
"""

import json
import time

import tensorflow as tf


def _gpu_device_name():
    return "GPU:0" if tf.config.list_physical_devices("GPU") else None


def reset_peak_memory():
    """Resets the BFC allocator's peak-usage counter. No-op without a GPU."""
    device = _gpu_device_name()
    if device is not None:
        tf.config.experimental.reset_memory_stats(device)


def peak_memory_mb():
    """BFC allocator peak usage in MB since the last reset, or None if no
    GPU device is registered with TF (this dev machine has none). TF's
    get_memory_info() exposes one 'peak' figure -- unlike PyTorch there is
    no separate allocated/reserved pair; callers report this single number
    for both Table 2 columns.
    """
    device = _gpu_device_name()
    if device is None:
        return None
    return tf.config.experimental.get_memory_info(device)["peak"] / 1e6


class RunProfiler:
    """Accumulates per-logged-point (loss, grad_norm, elapsed,
    num_optimizee_steps) and, on finish(), writes a JSONL trajectory plus a
    `<profile_path stem>_summary.json` aggregate that an orchestrator can
    read back without replaying the trajectory.
    """

    def __init__(self, profile_path):
        self.profile_path = profile_path
        self._rows = []
        self._start_time = None
        if profile_path is not None:
            reset_peak_memory()

    def start(self):
        self._start_time = time.time()

    def log_epoch(self, epoch, loss, grad_norm, grad_norm_post_clip=None,
                  num_optimizee_steps=None):
        if self.profile_path is None:
            return
        self._rows.append({
            "epoch": epoch,
            "loss": loss,
            "grad_norm": grad_norm,
            "grad_norm_post_clip": grad_norm_post_clip,
            "num_optimizee_steps": num_optimizee_steps,
            "elapsed_s": time.time() - self._start_time,
        })

    def finish(self):
        if self.profile_path is None:
            return None
        total_time = time.time() - self._start_time
        with open(self.profile_path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")

        losses = [r["loss"] for r in self._rows]
        grad_norms = [r["grad_norm"] for r in self._rows if r["grad_norm"] is not None]
        post_clip = [r["grad_norm_post_clip"] for r in self._rows
                     if r["grad_norm_post_clip"] is not None]
        step_counts = [r["num_optimizee_steps"] for r in self._rows
                      if r["num_optimizee_steps"] is not None]
        total_steps = sum(step_counts) if step_counts else None

        summary = {
            "num_logged_points": len(self._rows),
            "final_loss": losses[-1] if losses else None,
            "mean_loss": (sum(losses) / len(losses)) if losses else None,
            "final_grad_norm": grad_norms[-1] if grad_norms else None,
            "mean_grad_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
            "final_grad_norm_post_clip": post_clip[-1] if post_clip else None,
            "mean_grad_norm_post_clip": (sum(post_clip) / len(post_clip)) if post_clip else None,
            "peak_mem_mb": peak_memory_mb(),
            "total_time_s": total_time,
            "total_optimizee_steps": total_steps,
            # Not comparable across methods (see module docstring) -- kept
            # for within-method diagnostics.
            "avg_time_per_logged_point_s": (total_time / len(self._rows)) if self._rows else None,
            # The one comparable timing figure: the same "optimizee step"
            # unit for every method regardless of how often each logs.
            "avg_time_per_optimizee_step_s": (
                total_time / total_steps if total_steps else None),
        }
        summary_path = self.profile_path.rsplit(".", 1)[0] + "_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary
