"""Declarative config validation.

Every method and every problem dialect declares the config keys it
understands; a section is validated against the union. This replaces
hand-maintained key sets and keeps the adapters' contract: an unknown key
raises (the subprocess adapters would have died on an unknown CLI flag),
a flag the unified code deliberately does not port raises
``NotImplementedError`` naming it, and a flag it cannot honour is dropped
with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Tuple


@dataclass(frozen=True)
class ConfigSchema:
    allowed: FrozenSet[str] = frozenset()
    required: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    #: truthy value -> NotImplementedError (accepted by the library script, not ported)
    unported: Tuple[str, ...] = ()
    #: accepted but ignored with a warning
    ignored: Tuple[str, ...] = ()
    #: keys this method rejects even though the problem's schema allows them
    #: (e.g. train_dm.py has no --beta1 flag: argparse would have rejected it)
    forbidden: Tuple[str, ...] = ()

    @staticmethod
    def of(allowed: Iterable[str] = (), required: Iterable[str] = (), defaults: Dict[str, Any] = None,
           unported: Iterable[str] = (), ignored: Iterable[str] = (), forbidden: Iterable[str] = ()) -> "ConfigSchema":
        keys = set(allowed) | set(required) | set(defaults or {}) | set(unported) | set(ignored)
        return ConfigSchema(frozenset(keys), tuple(required), dict(defaults or {}), tuple(unported),
                            tuple(ignored), tuple(forbidden))

    def merge(self, other: "ConfigSchema") -> "ConfigSchema":
        return ConfigSchema(self.allowed | other.allowed, self.required + other.required,
                            {**self.defaults, **other.defaults}, self.unported + other.unported,
                            self.ignored + other.ignored, self.forbidden + other.forbidden)

    def validate(self, section: Dict[str, Any], where: str) -> Dict[str, Any]:
        """Return ``section`` with defaults applied, or raise."""
        section = dict(section)
        forbidden = sorted(k for k in self.forbidden if k in section)
        if forbidden:
            raise ValueError("{}: keys {} are not accepted by this method".format(where, forbidden))
        unknown = sorted(set(section) - self.allowed)
        if unknown:
            raise ValueError("{}: unknown config keys {} (allowed: {})".format(
                where, unknown, sorted(self.allowed)))
        missing = [k for k in self.required if section.get(k) is None]
        if missing:
            raise ValueError("{}: missing required keys {}".format(where, missing))
        for k in self.unported:
            if section.get(k):
                raise NotImplementedError("{}: --{} is not ported in the unified implementation; "
                                          "use the adapter-backed method".format(where, k))
        for k in self.ignored:
            if section.get(k):
                logging.warning("%s: %s is ignored by the unified implementation", where, k)
        for k, v in self.defaults.items():
            section.setdefault(k, v)
        return section
