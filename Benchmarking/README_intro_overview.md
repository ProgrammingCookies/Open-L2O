# L2O Benchmarking Platform

This benchmarking platform lets you train and evaluate different "learning to optimize"
(L2O) methods, algorithms that learn how to optimize other problems (meta-learn), and
compare their results side by side, using one shared command instead of each
method's own separate setup, enabling easier implementation of fairness, reproducability 
and comparison between methods.

## The three things you choose

- **A method** — which optimizer to run, e.g. `l2o-scale`, `l2o-dm`, or a
  plain baseline like `adam`.
- **A problem** — what it's optimizing, e.g. `mnist_mlp` or a LASSO task.
- **A config** — a small JSON file saying which method, which problem, and
  any settings for training/evaluation.

## Basic usage

```bash
conda activate thesis_updated
cd Benchmarking

python run.py --list-methods                          # see what's available
python run.py --config configs/scale_hrnn_mnist.json   # train + evaluate
python run.py --suite  configs/suite_mnist.json        # run several & rank them
```

Each run writes its results (checkpoint, logs, scores) under `results/`, and
`--suite` produces a simple leaderboard so different methods can be compared
on the same problem.

## Want more detail?

- **`README.md`** — the full technical reference: how the platform is built,
  the config file format, seeding/reproducibility, and the specific
  decisions behind the LASSO benchmark experiments.
- **`CONTRIBUTING.md`** — a step-by-step guide for plugging in a new method
  of your own.
