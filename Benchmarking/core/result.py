"""Standardized result and I/O containers shared by every adapter.

These dataclasses are the *contract* between the generic runner and the
library-specific adapters. An adapter's ``train`` returns a :class:`TrainOutput`
and its ``evaluate`` returns an :class:`EvalOutput`; the runner folds both into a
:class:`BenchmarkResult`, which is what the user's ``train_evaluate`` ultimately
returns (a dict with paths to the model, training metrics, validation/eval
metrics, and artifacts).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def summarize_trajectory(trajectory: List[float]) -> Dict[str, float]:
    """Reduce a per-iteration loss/objective trajectory to scalar summaries.

    Every model-free L2O evaluation ultimately produces a sequence of objective
    values as the (optimizee) is optimized. This gives every method a comparable
    set of headline numbers regardless of which library produced the trajectory.
    """
    if not trajectory:
        return {"final_loss": float("nan"), "min_loss": float("nan"),
                "mean_loss": float("nan"), "num_points": 0}
    return {
        "final_loss": float(trajectory[-1]),
        "min_loss": float(min(trajectory)),
        "mean_loss": float(sum(trajectory) / len(trajectory)),
        "num_points": len(trajectory),
    }


@dataclass
class TrainOutput:
    """What an adapter's ``train`` returns."""
    checkpoint: Optional[str]           # path/dir the eval step can restore from
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    command: Optional[str] = None       # the shell command that was run (provenance)


@dataclass
class EvalOutput:
    """What an adapter's ``evaluate`` returns."""
    metrics: Dict[str, Any] = field(default_factory=dict)     # includes summarize_trajectory keys
    trajectory: List[float] = field(default_factory=list)     # representative loss trajectory
    per_seed: Dict[str, Any] = field(default_factory=dict)    # optional per-seed breakdown
    artifacts: Dict[str, str] = field(default_factory=dict)
    command: Optional[str] = None


@dataclass
class BenchmarkResult:
    """The single normalized result object for one (method, problem) run."""
    method: str
    problem: str
    config: Dict[str, Any]
    checkpoint: Optional[str] = None
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    eval_metrics: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[float] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, output_dir: str) -> str:
        """Write ``result.json`` (trajectory truncated for readability) + full copy."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "result.json")
        d = self.to_dict()
        # Keep result.json compact: store trajectory length + head/tail, full copy separately.
        traj = d.pop("trajectory")
        d["trajectory_summary"] = {
            "length": len(traj),
            "head": traj[:5],
            "tail": traj[-5:],
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)
        if traj:
            with open(os.path.join(output_dir, "trajectory.json"), "w") as f:
                json.dump(traj, f)
        return path

    @staticmethod
    def load(output_dir: str) -> "BenchmarkResult":
        with open(os.path.join(output_dir, "result.json")) as f:
            d = json.load(f)
        d.pop("trajectory_summary", None)
        traj_path = os.path.join(output_dir, "trajectory.json")
        if os.path.exists(traj_path):
            with open(traj_path) as f:
                d["trajectory"] = json.load(f)
        return BenchmarkResult(**d)
