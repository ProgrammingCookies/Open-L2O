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


class ClassicalAdamPolicy(OptimizerPolicy):
    net_key = "adam"

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self._var = None
        self._opt = None

    def initial_state(self, x):
        import tensorflow as tf
        # Fresh optimizer per run: Keras 3 optimizers cannot be re-built on
        # new variables, and every chunk of the eval walk is a new run.
        self._var = tf.Variable(x)
        self._opt = tf.keras.optimizers.Adam(self.learning_rate)
        return None

    def step(self, x, grad, state, t):
        import tensorflow as tf
        self._var.assign(x)
        self._opt.apply_gradients([(grad, self._var)])
        return tf.identity(self._var), state

    def build(self, x_example) -> None:
        pass

    def restore(self, path, index=0):
        pass


#: ``eval.optimizer`` value -> factory(section). "L2L" is the learned policy itself.
CLASSICAL = {
    "Adam": lambda section: ClassicalAdamPolicy(float(section.get("learning_rate", 0.001))),
}
