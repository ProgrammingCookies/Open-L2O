"""L2O-Scale (Wichrowska et al. 2017, HierarchicalRNN) as a policy.

The optimizer itself is the library's own code, copied verbatim under
``learned/hrnn/``; this module only drives it the way
``Model_Free_L2O/L2O-Scale/evaluate.py`` and ``metaopt.py`` do, so that the
unified evaluation and training are bit-identical to the scripts on the same
machine. Three things about that code shape the policy:

1. **It updates ``tf.Variable``s in place** (``TrainableOptimizer.apply_gradients``
   keeps per-variable slots keyed by ``id(var)`` and a ``_standalone_global_state``
   that is created once and then carried). The policy therefore owns a
   ``tf.Variable``, assigns ``x`` into it, applies, and returns its value --
   the ``ClassicalPolicy`` pattern.

2. **It consumes random numbers in odd places.** The HRNN's initializers draw
   from the global TF RNG at construction (init vectors, decay weights, the
   affines) and lazily at the first update (the six cell affines); the
   per-coordinate initial log learning rate is drawn when a variable's slot
   is created; and ``Problem.gradients`` draws ``tf.random.normal(grad.shape)``
   on every step even at ``noise_stdev=0``. With ``rng_compat=True`` (default)
   the policy performs that dead draw before every ``apply_gradients``, so the
   stream stays aligned with the script's.

3. **Evaluation rebuilds the optimizer per chunk** (``evaluate.py:223-246``,
   ``273-304``): fresh HRNN -> a *dummy* ``apply_gradients`` with the chunk's
   real gradient on a throw-away ``tf.Variable(x0)`` (builds the cell weights,
   draws the dummy's log-lr slot, advances the global state by one untrained
   step) -> ``load(checkpoint)`` -> a new ``tf.Variable(x0)`` for the real
   run. Because that dummy step needs the real gradient, the whole preamble
   runs inside the policy's first ``step``. Whether the real variable then
   reuses the dummy's slot depends, in the script, on CPython handing the new
   variable the freed dummy's ``id()``; ``dummy_slot`` makes the choice explicit
   (see ``DUMMY_SLOT_DEFAULT`` for what the script does on this machine).

Construction kwargs are ``evaluate.py:139-169``'s (``HRNN_CELL_SIZES=[10,20,20]``
hard-coded; ``cell_cls``/``cell_size``/``num_cells`` only name the checkpoint
directory). Checkpoints are the library's dill files (``model-best.l2o`` etc.),
loaded/saved through the verbatim ``TrainableOptimizer.load/save``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from unified.learned.classical import SCALE_SCRIPT_CLASSICAL
from unified.policy import OptimizerPolicy, PolicySpec
from unified.schema import ConfigSchema

HRNN_CELL_SIZES = [10, 20, 20]
CHECKPOINT_DIRNAME = "HierarchicalRNN_GRUCell_20_2"    # {optimizer}_{cell_cls}_{cell_size}_{num_cells}

#: What the library script does with the dummy variable's slot, settled
#: empirically against the adapter on the tiny fixture (2026-09-18, CPU,
#: tests/test_unified_parity.py): "fresh" -- the real variable gets a NEW slot
#: (its own two log-lr uniforms) and reproduces the script bit for bit; with
#: "inherit" (the real variable taking over the freed dummy's id and post-step
#: slot) the trajectories diverge from step 2. In principle the script's own
#: behaviour is allocator-dependent; in practice CPython does not hand the
#: dummy's id to the next tf.Variable.
DUMMY_SLOT_DEFAULT = "fresh"

# evaluate.py:108-127 / train.py flag defaults
HRNN_FLAG_DEFAULTS = dict(
    learnable_decay=True, dynamic_output_scale=True, use_attention=False, use_log_objective=True,
    num_gradient_scales=4, zero_init_lr_weights=True, use_log_means_squared=True,
    use_relative_lr=True, use_extreme_indicator=False, max_log_lr=33,
    obj_train_max_multiplier=-1, use_problem_lr_mean=True, use_gradient_shortcut=True,
    use_second_derivatives=True, use_lr_shortcut=False, use_grad_products=True,
    use_multiple_scale_decays=False, use_numerator_epsilon=False, learnable_inp_decay=True,
    learnable_rnn_init=True,
)


def build_hrnn(min_lr: float, max_lr: float):
    import tensorflow as tf
    from unified.learned.hrnn import hierarchical_rnn
    return hierarchical_rnn.HierarchicalRNN(HRNN_CELL_SIZES, init_lr_range=(min_lr, max_lr),
                                            cell_cls=tf.keras.layers.GRUCell, **HRNN_FLAG_DEFAULTS)


class ScalePolicy(OptimizerPolicy):
    net_key = "hrnn"

    def __init__(self, min_lr: float = 1e-6, max_lr: float = 1e-2, checkpoint: str = None,
                 rng_compat: bool = True, dummy_slot: str = DUMMY_SLOT_DEFAULT, opt=None):
        if dummy_slot not in ("inherit", "fresh"):
            raise ValueError("dummy_slot must be 'inherit' or 'fresh'")
        self.min_lr, self.max_lr = float(min_lr), float(max_lr)
        self.checkpoint = checkpoint
        self.rng_compat = rng_compat
        self.dummy_slot = dummy_slot
        self.opt = opt if opt is not None else build_hrnn(self.min_lr, self.max_lr)
        self._var = None
        self._dummy = None
        self._pending_load = checkpoint is not None

    # --- OptimizerPolicy -------------------------------------------------
    @property
    def trainable_variables(self):
        return list(self.opt.trainable_variables)

    def initial_state(self, x):
        # Evaluation (the script's standalone apply_gradients path): a new
        # variable per chunk, whose slot the HRNN creates on its first apply.
        self._var = None
        return None

    # --- functional mode: the training path (metaopt.py / partial_unroll) ------
    def reset_state(self, x):
        """``TrainableOptimizer.reset_state``: fresh per-variable + global state
        (draws the two log-lr uniforms). Call inside the meta tape at k == 0 so
        the learned init vectors receive gradients, as the script does."""
        return ([self.opt._initialize_state(x)], self.opt._initialize_global_state())

    def _dead_draw(self, grad):
        import tensorflow as tf
        return grad + 0.0 * tf.random.normal(grad.shape) if self.rng_compat else grad

    def _preamble(self, x, grad):
        """evaluate.py's make_test_optimizer: dummy step on a throw-away variable, then load."""
        import tensorflow as tf
        if self.checkpoint is not None and not os.path.exists(self.checkpoint):
            raise FileNotFoundError("{}: no checkpoint {} -- refusing to evaluate an untrained "
                                    "optimizer".format(type(self).__name__, self.checkpoint))
        dummy = tf.Variable(x)
        self.opt.apply_gradients([(self._dead_draw(grad), dummy)])
        self.opt.load(self.checkpoint)
        return dummy

    def step(self, x, grad, state, t):
        import tensorflow as tf
        if state is not None:
            # Functional path (training): explicit state, tensors in, tensors out.
            states, global_state = state
            new_params, new_states, new_global, _ = self.opt._compute_updates(
                [x], [self._dead_draw(grad)], states, global_state)
            return new_params[0], (new_states, new_global)
        # Standalone path (evaluation): the HRNN's own apply_gradients on a Variable.
        if self._var is None:
            dummy = self._preamble(x, grad) if self._pending_load else None
            self._pending_load = False
            self._var = tf.Variable(x)
            if dummy is not None:
                if self.dummy_slot == "inherit":
                    self.opt._slots[id(self._var)] = self.opt._slots.pop(id(dummy))
                else:
                    self._dummy = dummy       # keep it alive so its id cannot be reused
        else:
            self._var.assign(x)
        self.opt.apply_gradients([(self._dead_draw(grad), self._var)])
        return tf.identity(self._var), state

    def build(self, x_example) -> None:
        pass   # the preamble/first step builds everything the script builds

    # --- checkpoints: the library's dill files, by full path ---------------
    def save(self, filename: str, index: int = None) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.opt.save(filename)

    def restore(self, filename: str, index: int = 0) -> None:
        self.opt.load(filename)


# ---------------------------------------------------------------- spec

def _make_policy(section: Dict[str, Any], checkpoint: str = None) -> ScalePolicy:
    return ScalePolicy(min_lr=float(section["min_lr"]), max_lr=float(section["max_lr"]),
                       checkpoint=checkpoint)


def checkpoint_dir(output_dir: str) -> str:
    """adapters/scale_family.py: ``<output_dir>/checkpoints/<optimizer>_<cell_cls>_<cell_size>_<num_cells>``."""
    return os.path.join(os.path.abspath(output_dir), "checkpoints", CHECKPOINT_DIRNAME)


# adapters/scale_family.py + L2O-Scale/train.py flags (problem keys live in the lasso dialect)
SCALE_TRAIN_SCHEMA = ConfigSchema.of(
    allowed=("seed", "problem", "train_dir", "optimizer", "cell_cls", "cell_size", "num_cells",
             "include_lasso_problems", "fix_unroll"),
    defaults={"num_problems": 1, "num_meta_iterations": 5, "fix_unroll_length": 20, "fix_num_steps": 100,
              "fix_num_steps_eval": 100, "evaluation_period": 1, "evaluation_epochs": 20, "save_period": 1,
              "min_lr": 1e-6, "max_lr": 1e-2, "select_checkpoint": "final",
              "meta_learning_rate": 1e-6, "gradient_clip_level": 1e4},
    ignored=("profile_path",),
)
SCALE_EVAL_SCHEMA = ConfigSchema.of(
    required=("test_optimizer", "num_testing_itrs"),
    allowed=("seed", "problem", "train_dir", "save_dir", "model_name", "restore_model_name",
             "optimizer", "cell_cls", "cell_size", "num_cells", "include_lasso_problems"),
    defaults={"min_lr": 1e-6, "max_lr": 1e-2},
)

SCALE_SPEC = PolicySpec(
    name="unified-scale",
    make_policy=_make_policy,
    train_schema=SCALE_TRAIN_SCHEMA,
    eval_schema=SCALE_EVAL_SCHEMA,
    learned_tag="L2o",
    eval_optimizer_key="test_optimizer",
    eval_steps_key="num_testing_itrs",
    classical=SCALE_SCRIPT_CLASSICAL,
    fresh_policy_per_chunk=True,
    artifact_layout="scale",
    train_loop="scale",
)
