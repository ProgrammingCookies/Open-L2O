"""One in-process implementation of L2O methods behind the platform's interface.

The subprocess adapters in ``adapters/`` wrap five libraries that each ship
their own ``problems.py`` / ``networks.py`` / ``meta.py``. This package is the
implementation-layer counterpart: ONE problem protocol, ONE learned-optimizer
protocol, ONE meta-training loop, ONE iterative-solver interface and ONE
evaluation walk, behind the same ``core.method.L2OMethod`` ABC, registered
under new names so the adapter-backed ``l2o-*`` / ``lista`` / ``alista``
methods are never shadowed.

Extension points (see README "Adding ..."):
  * a learned optimizer  -> ``learned/<name>.py``: an ``OptimizerPolicy`` + a ``PolicySpec``
  * an unrolled network  -> ``unrolled/``: a model + a ``ModelSpec``
  * a problem            -> ``tasks/<name>.py``: a ``Problem`` + one or more ``ProblemDialect``s
and one line in the lists below.

Registration is opt-in: ``import unified`` (nothing in ``adapters/`` changes,
so an import error here cannot break the adapter registry). TensorFlow is
imported lazily inside ``train()``/``evaluate()``; importing this package is
cheap and the fast test suite never pays for it.
"""

from core.registry import list_methods, register_method

from unified.learned.dm import DM_SPEC
from unified.learned.rnnprop import RNNPROP_SPEC
from unified.learned.scale import SCALE_SPEC
from unified.methods import UnifiedModelFreeMethod, UnifiedUnrolledMethod
from unified.problem import PROBLEMS, register_problem
from unified.tasks.lasso import LassoDatasetDialect, LassoUnrolledDialect
from unified.tasks.quadratic import QuadraticDialect
from unified.unrolled.specs import ALISTA_SPEC, LFISTA_SPEC, LISTA_SPEC

PROBLEM_DIALECTS = (LassoDatasetDialect(), LassoUnrolledDialect(), QuadraticDialect())
METHODS = (
    UnifiedModelFreeMethod(DM_SPEC),
    UnifiedModelFreeMethod(RNNPROP_SPEC),
    UnifiedModelFreeMethod(SCALE_SPEC),
    UnifiedUnrolledMethod(LISTA_SPEC),
    UnifiedUnrolledMethod(ALISTA_SPEC),
    UnifiedUnrolledMethod(LFISTA_SPEC),
)
METHOD_NAMES = tuple(m.name for m in METHODS)
MODEL_FREE_NAMES = tuple(m.name for m in METHODS if isinstance(m, UnifiedModelFreeMethod))
UNROLLED_NAMES = tuple(m.name for m in METHODS if isinstance(m, UnifiedUnrolledMethod))


def _register() -> None:
    for dialect in PROBLEM_DIALECTS:
        if dialect.name not in PROBLEMS:
            register_problem(dialect)
    registered = set(list_methods())
    for method in METHODS:
        if method.name not in registered:   # skip if an earlier import in this process did it
            register_method(method)


_register()
