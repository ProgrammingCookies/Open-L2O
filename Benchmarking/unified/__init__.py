"""One in-process implementation of the Experiment-2 L2O methods.

The subprocess adapters in ``adapters/`` wrap five libraries that each ship
their own ``problems.py`` / ``networks.py`` / ``meta.py``. This package is the
implementation-layer counterpart: ONE LASSO instance loader, ONE gradient
oracle, ONE iterative-solver interface and ONE evaluation walk, behind the
same ``core.method.L2OMethod`` ABC, registered under new names so the
adapter-backed ``l2o-*`` / ``lista`` / ``alista`` methods are never shadowed.

Scope (deliberately minimal): the LASSO problem from ``data/lasso.py`` and the
methods Experiment 2 uses -- L2O-DM, L2O-RNNProp, L2O-Scale, LISTA, ALISTA.
Swarm / Entropy / Jacobian stay adapter-only.

Registration is opt-in: ``import unified`` (nothing in ``adapters/`` changes,
so an import error here cannot break the adapter registry). TensorFlow is
imported lazily inside ``train()``/``evaluate()`` so importing this package
is cheap and the fast test suite never pays for it.
"""

from core.registry import list_methods, register_method

from unified.methods import UnifiedModelFreeMethod, UnifiedUnrolledMethod

MODEL_FREE_NAMES = ("unified-dm", "unified-rnnprop")
UNROLLED_NAMES = ("unified-lista", "unified-alista")
METHOD_NAMES = MODEL_FREE_NAMES + UNROLLED_NAMES


def _register() -> None:
    registered = set(list_methods())
    for name in MODEL_FREE_NAMES:
        if name not in registered:  # skip if an earlier import in this process did it
            register_method(UnifiedModelFreeMethod(name, variant=name.split("-", 1)[1]))
    for name in UNROLLED_NAMES:
        if name not in registered:
            register_method(UnifiedUnrolledMethod(name, model_name=name.split("-", 1)[1]))


_register()
