"""Registry of problem types the platform can score.

An adapter's job ends at reading what its library's evaluation script
produced into named arrays (the final iterate ``x_pred`` plus whatever
identifies the instances it was evaluated on, e.g. LASSO's ``b``, ``x_true``,
``x0``). What happens to those arrays -- which of them form the instances
fingerprint, which constants of the problem the metrics need, and which
metrics run -- is described once per problem type by a :class:`ProblemSpec`,
so no adapter contains problem-specific scoring code.

A problem type is registered as such (see ``scoring/`` for the built-ins):

    from core.problems import ProblemSpec, register_problem
    register_problem(ProblemSpec(name="my_problem", ...))

``name`` is the value of ``config["problem"]``. Problems that are not
registered (e.g. ``"mnist"``) are simply not scored: they still get their loss
trajectory, but no fingerprint and no metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvalRecord:
    """Everything a metric function gets to compute from.

    arrays: what the evaluation produced, read by the adapter (``x_pred`` and
        the instance arrays), all with the instances along axis 0.
    params: the problem's constants, from ``ProblemSpec.load_params`` (LASSO:
        the matrix ``a`` and ``lam``).
    eval_cfg: ``config["eval"]``, for metric-specific options.
    who: the method name, for error messages.
    """
    problem: str
    arrays: Dict[str, Optional[np.ndarray]]
    params: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    who: str = ""


@dataclass
class ProblemSpec:
    #: config["problem"] value this spec applies to.
    name: str
    #: Arrays every evaluation of this problem must provide (checked before scoring).
    required_arrays: Tuple[str, ...] = ("x_pred",)
    #: Arrays hashed into the instances fingerprint (core/fingerprint.py); a
    #: component the adapter could not provide (None/missing) is skipped.
    #: Empty = the problem has no fingerprint.
    fingerprint_arrays: Tuple[str, ...] = ()
    #: (config, who) -> constants of the problem that metrics need. Only called
    #: when at least one metric runs.
    load_params: Callable[[Dict[str, Any], str], Dict[str, Any]] = lambda config, who: {}
    #: Metrics run when config["eval"]["metrics"] is not given.
    default_metrics: Tuple[str, ...] = ()
    description: str = ""


_PROBLEMS: Dict[str, ProblemSpec] = {}


def register_problem(spec: ProblemSpec) -> ProblemSpec:
    if not isinstance(spec, ProblemSpec):
        raise TypeError("register_problem expects a ProblemSpec, got {!r}".format(type(spec)))
    if spec.name in _PROBLEMS:
        raise ValueError("problem {!r} is already registered".format(spec.name))
    _PROBLEMS[spec.name] = spec
    return spec


def get_problem(name: Optional[str]) -> Optional[ProblemSpec]:
    """The spec registered under ``name``, or None if that problem is not scored."""
    return _PROBLEMS.get(name) if name is not None else None


def list_problems() -> List[str]:
    return sorted(_PROBLEMS)
