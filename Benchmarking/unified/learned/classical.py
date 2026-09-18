"""Classical optimizers driven through the policy interface (the scripts'
``--optimizer Adam`` baseline branch, ``evaluate_dm.py:87-96``).

Keras optimizers update a ``tf.Variable`` in place, so the policy owns one and
copies the functional ``x`` through it each step -- bit-identical to the
script's Adam branch while fitting the ``IterativeSolver`` contract. Any
optimizer that mutates state in place (a future L2O-Scale policy, whose
``apply_gradients`` needs ``id()``-keyed slots on real Variables) follows the
same pattern.
"""

from __future__ import annotations

from unified.policy import OptimizerPolicy


class ClassicalPolicy(OptimizerPolicy):
    """A Keras optimizer behind the policy interface.

    ``make_optimizer`` builds a fresh optimizer per run (Keras 3 optimizers
    cannot be re-built on new variables, and every chunk of the eval walk is
    a new run). ``dead_draws`` reproduces L2O-Scale's ``Problem.gradients``,
    which draws ``tf.random.normal(grad.shape)`` on every step even at zero
    noise -- meaningless for the result, but it advances the global RNG the
    same way the script does.
    """

    net_key = "classical"

    def __init__(self, make_optimizer, dead_draws: bool = False):
        self._make_optimizer = make_optimizer
        self._dead_draws = dead_draws
        self._var = None
        self._opt = None

    def initial_state(self, x):
        import tensorflow as tf
        self._var = tf.Variable(x)
        self._opt = self._make_optimizer()
        return None

    def step(self, x, grad, state, t):
        import tensorflow as tf
        if self._dead_draws:
            grad = grad + 0.0 * tf.random.normal(grad.shape)
        self._var.assign(x)
        self._opt.apply_gradients([(grad, self._var)])
        return tf.identity(self._var), state

    def build(self, x_example) -> None:
        pass

    def restore(self, path, index=0):
        pass


class ClassicalAdamPolicy(ClassicalPolicy):
    """evaluate_dm.py:87-96 -- ``tf.keras.optimizers.Adam(--learning_rate)``."""

    def __init__(self, learning_rate=0.001):
        import tensorflow as tf
        super().__init__(lambda: tf.keras.optimizers.Adam(learning_rate))


#: evaluate_{dm,rnnprop}.py: ``--optimizer Adam`` with ``--learning_rate`` (default 1e-3)
DM_SCRIPT_CLASSICAL = {
    "Adam": lambda section: ClassicalAdamPolicy(float(section.get("learning_rate", 0.001))),
}


def _scale_baseline(name, learning_rate):
    def make(section):
        import tensorflow as tf
        cls = {"Adam": tf.keras.optimizers.Adam, "SGD": tf.keras.optimizers.SGD,
               "Adagrad": tf.keras.optimizers.Adagrad}[name]
        return ClassicalPolicy(lambda: cls(learning_rate=learning_rate), dead_draws=True)
    return make


#: L2O-Scale/evaluate.py:44-48 BASELINE_OPTIMIZERS (fixed learning rates, no flag)
SCALE_SCRIPT_CLASSICAL = {
    "SGD": _scale_baseline("SGD", 0.001),
    "Adam": _scale_baseline("Adam", 0.0001),
    "Adagrad": _scale_baseline("Adagrad", 0.00001),
}
