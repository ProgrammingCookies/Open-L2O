"""Registry of evaluation metrics, and the one scoring entry point adapters call.

A metric is a function of an :class:`core.problems.EvalRecord` returning a
flat ``{key: value}`` dict that is merged into ``EvalOutput.metrics`` (and so
into ``results.jsonl``). It is registered as such (see ``scoring/`` for the
built-ins):

    from core.metrics import register_metric

    @register_metric("my_metric", config_keys=("my_option",))
    def my_metric(record):
        return {"my_metric_mean": float(...)}

``config_keys`` are the ``config["eval"]`` keys the metric reads; they are
consumed by the platform and never forwarded to the library scripts as flags.

Which metrics run is ``config["eval"]["metrics"]`` (a list of registered
names) if given, else the problem's ``ProblemSpec.default_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from core.fingerprint import METRIC_KEY as FINGERPRINT_KEY, fingerprint_arrays
from core.problems import EvalRecord, get_problem

#: config["eval"] key selecting the metrics to run.
METRICS_CONFIG_KEY = "metrics"

MetricFn = Callable[[EvalRecord], Dict[str, Any]]


@dataclass
class MetricSpec:
    name: str
    fn: MetricFn
    config_keys: Tuple[str, ...] = ()


_METRICS: Dict[str, MetricSpec] = {}


def register_metric(name: str, config_keys: Tuple[str, ...] = ()):
    """Decorator registering ``fn`` under ``name``; returns ``fn`` unchanged."""
    def decorator(fn: MetricFn) -> MetricFn:
        if name in _METRICS:
            raise ValueError("metric {!r} is already registered".format(name))
        _METRICS[name] = MetricSpec(name=name, fn=fn, config_keys=tuple(config_keys))
        return fn
    return decorator


def get_metric(name: str) -> MetricSpec:
    if name not in _METRICS:
        raise KeyError("unknown metric {!r}; registered metrics: {}".format(
            name, ", ".join(sorted(_METRICS)) or "<none>"))
    return _METRICS[name]


def list_metrics() -> List[str]:
    return sorted(_METRICS)


def scoring_config_keys() -> Set[str]:
    """``config["eval"]`` keys consumed by scoring, which adapters must not
    forward to their library scripts as command-line flags."""
    keys = {METRICS_CONFIG_KEY}
    for spec in _METRICS.values():
        keys.update(spec.config_keys)
    return keys


def selected_metrics(problem: str, eval_cfg: Mapping[str, Any]) -> List[str]:
    spec = get_problem(problem)
    names = eval_cfg.get(METRICS_CONFIG_KEY)
    if names is None:
        return list(spec.default_metrics) if spec is not None else []
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise TypeError("config['eval']['{}'] must be a list of metric names, got {!r}".format(
            METRICS_CONFIG_KEY, names))
    return list(names)


def compute_metrics(problem: Optional[str], arrays: Mapping[str, Optional[np.ndarray]],
                    config: Mapping[str, Any], who: str = "") -> Dict[str, Any]:
    """Instances fingerprint + the selected metrics of one evaluation.

    problem: the ``config["problem"]`` vocabulary the arrays belong to.
    arrays: what the evaluation produced (see ``EvalRecord.arrays``).
    Returns ``{}`` when ``problem`` is not registered.
    """
    spec = get_problem(problem)
    if spec is None:
        return {}
    eval_cfg = dict(config.get("eval", {}))
    missing = [k for k in spec.required_arrays if arrays.get(k) is None]
    if missing:
        raise ValueError("{}: evaluation of problem {!r} produced no {} (required by its "
                         "ProblemSpec)".format(who, problem, ", ".join(missing)))

    out: Dict[str, Any] = {}
    if spec.fingerprint_arrays:
        out[FINGERPRINT_KEY] = fingerprint_arrays(
            **{k: arrays.get(k) for k in spec.fingerprint_arrays})

    names = selected_metrics(problem, eval_cfg)
    if not names:
        return out
    metric_specs = [get_metric(n) for n in names]  # fail on an unknown name before any work
    record = EvalRecord(problem=problem, arrays=dict(arrays),
                        params=spec.load_params(dict(config), who),
                        eval_cfg=eval_cfg, who=who)
    for m in metric_specs:
        values = m.fn(record)
        clash = sorted(set(values) & set(out))
        if clash:
            raise ValueError("{}: metric {!r} writes key(s) {} already written by an earlier "
                             "metric or the fingerprint".format(who, m.name, clash))
        out.update(values)
    return out
