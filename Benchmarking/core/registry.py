"""Registries for benchmarkable methods.

Adding a new L2O model to the benchmark is done as such:
    from core.registry import register_method
    register_method(NewL2OModel()) (of class L2O method)

After that --method model_name works in run.py with no other change.
"""

from __future__ import annotations
from typing import Dict, List
from core.method import L2OMethod

_METHODS: Dict[str, L2OMethod] = {}


def register_method(method: L2OMethod) -> L2OMethod:
    """Registers an L2O method in _METHODS (seen initialized above)."""
    if not isinstance(method, L2OMethod):
        raise TypeError(
            "register_method expects an L2OMethod instance, got {!r}".format(type(method)))
    name = method.name
    if name in _METHODS:
        raise ValueError("method {!r} is already registered".format(name))
    _METHODS[name] = method
    return method


def get_method(name: str) -> L2OMethod:
    if name not in _METHODS:
        raise KeyError(
            "unknown method {!r}; registered methods: {}".format(name, ", ".join(sorted(_METHODS)) or "<none>"))
    return _METHODS[name]


def list_methods() -> List[str]:
    return sorted(_METHODS)
