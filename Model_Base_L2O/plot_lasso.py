import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
  from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:
  raise SystemExit(
      "Missing dependency: tensorboard. Install with `pip install tensorboard`."
  ) from exc


def _collect_event_files(root: str) -> List[str]:
  event_files = []
  for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
      if "tfevents" in filename:
        event_files.append(os.path.join(dirpath, filename))
  return sorted(event_files)


def _read_scalars_from_event_file(path: str, tag: str) -> List[Tuple[int, float]]:
  ea = event_accumulator.EventAccumulator(
      path,
      size_guidance={
          event_accumulator.SCALARS: 0,
          event_accumulator.TENSORS: 0,
      },
  )
  ea.Reload()

  points: List[Tuple[int, float]] = []

  if tag in ea.Tags().get("scalars", []):
    for scalar_event in ea.Scalars(tag):
      points.append((int(scalar_event.step), float(scalar_event.value)))
    return points

  if tag in ea.Tags().get("tensors", []):
    for tensor_event in ea.Tensors(tag):
      # tensor_proto typically stores one scalar float value.
      values = list(tensor_event.tensor_proto.float_val)
      if values:
        points.append((int(tensor_event.step), float(values[0])))
    return points

  return points


def _load_series_from_logs(logs_root: str, tag: str) -> Dict[str, List[Tuple[int, float]]]:
  series = {}
  if not os.path.isdir(logs_root):
    return series

  for exp_name in sorted(os.listdir(logs_root)):
    exp_path = os.path.join(logs_root, exp_name)
    if not os.path.isdir(exp_path):
      continue

    event_files = _collect_event_files(exp_path)
    points = []
    for event_file in event_files:
      points.extend(_read_scalars_from_event_file(event_file, tag))

    if not points:
      continue
    # Deduplicate by step, keep last value.
    step_to_value = {step: value for step, value in points}
    deduped = sorted(step_to_value.items(), key=lambda x: x[0])
    series[exp_name] = deduped

  return series


def _plot_series(series: Dict[str, List[Tuple[int, float]]], output_path: str, tag: str) -> None:
  plt.figure(figsize=(8, 5))
  for name, points in series.items():
    steps = [x[0] for x in points]
    values = [x[1] for x in points]
    plt.plot(steps, values, linewidth=2, label=name)

  plt.xlabel("Layer / Step")
  plt.ylabel("Lasso Objective")
  plt.title(f"Lasso Results ({tag})")
  plt.grid(alpha=0.3)
  plt.legend(loc="best", fontsize=8)
  plt.tight_layout()

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  plt.savefig(output_path, dpi=200)
  plt.close()


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Plot Lasso curves from TensorBoard logs and save as PNG."
  )
  parser.add_argument(
      "--base_dir",
      default=".",
      help="Base experiment directory used by train.py (contains logs/).",
  )
  parser.add_argument(
      "--tag",
      default="val_lasso",
      help="Scalar tag to plot from TensorBoard event files.",
  )
  parser.add_argument(
      "--output",
      default=os.path.join("..", "Figs", "lasso.png"),
      help="Output PNG path.",
  )
  args = parser.parse_args()

  logs_root = os.path.join(os.path.abspath(args.base_dir), "logs")
  series = _load_series_from_logs(logs_root, args.tag)
  if not series:
    raise SystemExit(
        f"No scalar data found for tag '{args.tag}' under '{logs_root}'. "
        "Run lasso experiments first, then retry."
    )

  output_path = os.path.abspath(args.output)
  _plot_series(series, output_path, args.tag)
  print(f"Saved plot to: {output_path}")
  print(f"Included runs: {', '.join(sorted(series.keys()))}")


if __name__ == "__main__":
  main()
