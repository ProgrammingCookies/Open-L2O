"""Core abstractions for the generalized L2O benchmarking platform.

The platform is intentionally *thin*: it does not re-implement any training or
evaluation logic. Each migrated L2O library keeps its own ``train.py`` /
``evaluate.py`` entry points; an *adapter* (see ``Benchmarking/adapters``) knows
how to invoke those scripts and how to read the artifacts they produce, and
normalizes everything into a single :class:`~core.result.BenchmarkResult`.

Public surface:
    - :class:`core.method.L2OMethod`      -- the interface every adapter implements
    - :func:`core.registry.register_method` / :func:`core.registry.get_method`
    - :class:`core.result.BenchmarkResult`
    - :func:`core.runner.train_evaluate`  -- run one (method, problem) benchmark
"""
