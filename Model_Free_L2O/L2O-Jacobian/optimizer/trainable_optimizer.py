# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A base class definition for trainable optimizers (eager / TF2).

Two independent call surfaces, both delegating to the same
`_compute_update`/`_compute_updates` subclass override points:

  - Standalone "drop-in optimizer" surface (`apply_gradients`): applies the
    (already meta-trained) optimizer to a fresh problem, exactly like a plain
    tf.keras.optimizers.Optimizer would. Slot state is a plain Python dict
    keyed by variable identity, replacing the TF1 tf.train.Optimizer slot
    mechanism (which had no TF2 counterpart worth preserving here).

  - Meta-training surface (`setup`/`new_trajectory`/`reset_state`/
    `partial_unroll`): the eager replacement for the TF1 `train()` method's
    tf.while_loop-built graph. A TF1 graph is built once and then executed
    via many separate sess.run() calls ("partial unrolling", to bound
    memory/compute); anything that needed to survive between those calls had
    to be smuggled into the graph as a tf.get_local_variable(). In eager mode
    a Python `for` loop over unroll steps *is* the while_loop, and a plain
    instance attribute (self._params/self._state/...) *is* the local
    variable -- both the while_loop's shape_invariants/state-flattening
    machinery and the local-variable workaround disappear entirely; state is
    just carried forward in place between calls to `partial_unroll`.
"""

import dill
import tensorflow as tf

EPSILON = 1e-6


class TrainableOptimizer:
    """Base class for trainable optimizers.

    A trainable optimizer is an optimizer that has parameters that can
    themselves be learned (meta-optimized).

    Subclasses must implement:
        _compute_update(self, param, grad, state)
    """

    def __init__(self, name, state_keys, use_attention=False,
                 use_log_objective=False, obj_train_max_multiplier=-1,
                 use_second_derivatives=True, use_numerator_epsilon=False,
                 alpha=5e-4, beta=1e-4, reg_optimizer=False, reg_optimizee=False,
                 reg_option="hessian", hessian_itrs=10,
                 use_aux_convex_l1=False, use_aux_convex_l2=False,
                 aux_convex_dim=20, convex_l1_ratio=0.1, aux_convex_ratio=1.0,
                 pretrained_model_path=None, **kwargs):
        """Initializes the optimizer with the given name and settings.

        Args:
          name: The name string for this optimizer.
          state_keys: The names of any required state variables (list)
          use_attention: Whether this optimizer uses attention (Default: True)
          use_log_objective: Whether this optimizer uses the logarithm of the
              objective when computing the loss (Default: False)
          obj_train_max_multiplier: The maximum multiplier for the increase in the
              objective before meta-training is stopped. If <= 0, meta-training is
              not stopped early. (Default: -1)
          use_second_derivatives: Whether this optimizer uses second derivatives in
              meta-training. This should be set to False if some second derivatives
              in the meta-training problem set are not defined in Tensorflow.
              (Default: True)
          use_numerator_epsilon: Whether to use epsilon in the numerator when
              scaling the problem objective during meta-training. (Default: False)
          alpha: Scale for the optimizer (Hessian/Jacobian flatness) regularization
              term, see reg_optimizer. (Default: 5e-4)
          beta: Scale for the optimizee regularization term, see reg_optimizee.
              (Default: 1e-4)
          reg_optimizer: Whether to add a Hessian/Jacobian-based regularization
              term (problem.regularizer(), see problems/problem_generator.py) to
              the meta-objective, gated per-unroll-segment by partial_unroll's
              jacob_switch argument. (Default: False)
          reg_optimizee: Whether to add the same regularization term directly
              into the per-step objective the optimizee's gradients are computed
              from (independent of jacob_switch). (Default: False)
          reg_option: Which regularizer problem.regularizer() computes: one of
              'hessian', 'hessian-ev', 'hessian-esd', 'jacob'. Only meaningful
              if reg_optimizer or reg_optimizee is set. (Default: 'hessian')
          hessian_itrs: Number of power-iteration/Hutchinson/Lanczos steps used
              by problem.regularizer(). (Default: 10)
          use_aux_convex_l1, use_aux_convex_l2: HALO's auxiliary convex
              regularization -- whether to draw a fresh random target tensor
              each partial_unroll() segment and pass it to problem.objective()
              as aux_conv_labels_l1/l2 (see ConvNet's use_aux_convex, which
              must also be set on the problem for the extra parameter to
              exist at all). (Default: False)
          aux_convex_dim: shape of the random target tensor. (Default: 20)
          convex_l1_ratio: scale of the l1 term relative to the l2 one, passed
              straight through to problem.objective(). (Default: 0.1)
          aux_convex_ratio: overall scale of the aux-convex penalty terms,
              read by problem.objective() from the problem itself (see
              ConvNet's aux_convex_ratio) -- kept here too only so callers
              that only configure the optimizer still have a consistent
              default to inspect. (Default: 1.0)
          pretrained_model_path: path to a pickled dict of numpy arrays (see
              train_pretrain.py) to warm-start every new_trajectory() from,
              or None to initialize randomly as usual. (Default: None)
          **kwargs: Any additional keyword arguments (unused, kept for signature
              compatibility with callers that forward extra config).
        """
        self._name = name
        self.use_second_derivatives = use_second_derivatives
        self.state_keys = sorted(state_keys)
        self.use_attention = use_attention
        self.use_log_objective = use_log_objective
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.use_numerator_epsilon = use_numerator_epsilon
        self.alpha = alpha
        self.beta = beta
        self.reg_optimizer = reg_optimizer
        self.reg_optimizee = reg_optimizee
        self.reg_option = reg_option
        self.hessian_itrs = hessian_itrs
        self.use_aux_convex_l1 = use_aux_convex_l1
        self.use_aux_convex_l2 = use_aux_convex_l2
        self.aux_convex_dim = aux_convex_dim
        self.convex_l1_ratio = convex_l1_ratio
        self.aux_convex_ratio = aux_convex_ratio
        self.pretrained_model_path = pretrained_model_path

        # Standalone "drop-in optimizer" surface: id(var) -> state dict.
        self._slots = {}
        self._standalone_global_state = None

        # Meta-training surface state, populated by setup()/new_trajectory()/
        # reset_state() and carried forward across partial_unroll() calls.
        self._problem = None
        self._dataset = None
        self._params = None
        self._attend_params = None
        self._state = None
        self._global_state = None
        self._initial_obj = None

    def get_name(self):
        return self._name

    @property
    def trainable_variables(self):
        """The optimizer's own learned weights. Subclasses with weights override this."""
        return []

    # ------------------------------------------------------------------
    # Standalone "drop-in optimizer" surface
    # ------------------------------------------------------------------

    def _create_slots(self, var_list):
        """Creates all slots needed by the variables.

        Args:
          var_list: A list of `Variable` objects.
        """
        for var in var_list:
            if id(var) not in self._slots:
                self._slots[id(var)] = self._initialize_state(var)

    def _initialize_state(self, var):
        """Initializes any state required for this variable.

        Args:
          var: a tensor containing parameters to be optimized

        Returns:
          state: a dictionary mapping state keys to initial state values (tensors)
        """
        return {}

    def _initialize_global_state(self):
        """Initializes any global state values."""
        return []

    def apply_gradients(self, grads_and_vars):
        """Applies the optimizer updates to a list of (grad, tf.Variable) pairs.

        Routed through `_compute_updates` (plural), not `_compute_update`, so
        this works uniformly for subclasses that couple information across
        parameters (e.g. HierarchicalRNN's per-tensor/global state) as well as
        subclasses that only override the simple per-parameter
        `_compute_update` -- the base `_compute_updates` already reduces to
        that case. This also means HierarchicalRNN needs no override of this
        method at all (the original TF1 version's `apply_gradients` override
        existed mainly to route through `_compute_updates` and to handle its
        `_create_slots`/`get_slot` tf.train.Optimizer plumbing, both handled
        generically here).

        Args:
          grads_and_vars: iterable of (grad, tf.Variable) pairs, same shape.
        """
        grads_and_vars = list(grads_and_vars)
        variables = [var for _, var in grads_and_vars]
        grads = [grad for grad, _ in grads_and_vars]

        self._create_slots(variables)
        if self._standalone_global_state is None:
            self._standalone_global_state = self._initialize_global_state()
        states = [self._slots[id(var)] for var in variables]

        new_params, new_states, new_global_state, _ = self._compute_updates(
            variables, grads, states, self._standalone_global_state)
        self._standalone_global_state = new_global_state

        for var, new_var, new_state in zip(variables, new_params, new_states):
            var.assign(new_var)
            self._slots[id(var)] = new_state

    def _compute_update(self, param, grad, state):
        """Computes the update step for optimization.

        Args:
          param: A tensor of parameters to optimize.
          grad: The gradient tensor of the objective with respect to the parameters.
              (It has the same shape as param.)
          state: A dictionary containing any extra state required by the optimizer.

        Returns:
          updated_params: The updated parameters.
          updated_state: The dictionary of updated state variable(s).
        """
        raise NotImplementedError

    def _compute_updates(self, params, grads, states, global_state):
        """Maps the compute update functions for each parameter.

        This function can be overriden by a subclass if the subclass wants to
        combine information across the different parameters in the list.

        Args:
          params: A list of parameter tensors.
          grads: A list of gradients corresponding to each parameter.
          states: A list of state variables corresponding to each parameter.
          global_state: A list of global state variables for the problem.

        Returns:
          new_params: The updated parameters.
          new_states: The updated states.
          new_global_state: The updated global state.
          attention_params: A list of attention parameters. This is the same as
              new_params if the optimizer does not use attention.
        """
        new_params, new_states = zip(*[
            self._compute_update(p, g, s) for p, g, s in zip(params, grads, states)])
        # Global state is unused in the basic case, just pass it through.
        return list(new_params), list(new_states), global_state, list(new_params)

    # ------------------------------------------------------------------
    # Meta-training surface
    # ------------------------------------------------------------------

    def setup(self, problem, dataset=None):
        """Binds a fixed problem instance (+ optional dataset) to meta-train against."""
        self._problem = problem
        self._dataset = dataset

    def new_trajectory(self):
        """Draws a fresh starting point and recomputes the normalizing initial objective.

        Call once per meta-iteration (this is the eager replacement for the
        `first_unroll` placeholder, which was true at the start of every
        meta-iteration in the original).
        """
        problem = self._problem
        data, labels = self._dataset if self._dataset is not None else (None, None)

        # Only pass pretrained_model_path when actually configured -- avoids
        # requiring every Problem subclass's init_tensors to accept this kwarg
        # (only ConvNet does; matches the original's HALO warm-start, which
        # is likewise only ever exercised against ConvNet-based problems).
        if self.pretrained_model_path is not None:
            initial_tensors = problem.init_tensors(pretrained_model_path=self.pretrained_model_path)
        else:
            initial_tensors = problem.init_tensors()
        self._params = list(initial_tensors)
        self._attend_params = list(initial_tensors)
        self._initial_obj = problem.objective(self._params, data, labels)

    def reset_state(self):
        """(Re)initializes the optimizer's own per-parameter/global state.

        Call only when the caller's policy says the RNN state should be reset
        (in the original, this was gated by `reset_state`, which defaulted to
        true only at the very first meta-iteration of a problem instance and
        false thereafter -- that policy decision belongs to the caller, not
        this class; see metaopt.py). During meta-training this must be called
        with the outer tf.GradientTape already active, so that any learned
        initial-state weights (e.g. CoordinatewiseRNN's init_vector) receive
        gradients on the meta-iterations where a reset actually happens --
        calling it outside the tape silently drops those gradients (they'll
        just come back None), matching how the original TF1 graph included
        the state-initialization ops in the same differentiable graph as the
        rest of the unroll.
        """
        self._state = [self._initialize_state(p) for p in self._params]
        self._global_state = self._initialize_global_state()

    def partial_unroll(self, num_iter, obj_weights, batch_indices=None,
                       jacob_switch=False):
        """Runs num_iter optimization steps, continuing the current trajectory.

        The eager replacement for one execution of the TF1 tf.while_loop
        (previously one sess.run call in a "partial unroll"); state
        (self._params/self._attend_params/self._state/self._global_state) is
        updated in place, so calling this repeatedly continues the same
        trajectory -- no separate state-propagation step is needed.

        Must be called with an active outer tf.GradientTape when used for
        meta-training, so gradients can flow from the returned objective back
        to `self.trainable_variables` through the whole unroll.

        Args:
          num_iter: number of steps to run.
          obj_weights: sequence of length num_iter, the weighted-objective
            multiplier for each step.
          batch_indices: sequence of length num_iter, each entry itself a
            sequence of dataset row indices for that step's minibatch. None
            for problems without a dataset (full-batch objective is used
            directly as the per-step objective in that case).
          jacob_switch: whether reg_optimizer's regularization term
            contributes to `regular` for this whole segment (a single value
            applied uniformly across all num_iter steps, matching the
            original -- see metaopt.py's per-unroll-segment scheduling of
            this via regularize_time/reg_scale). Ignored if reg_optimizer is
            False.

        Returns:
          scaled_meta_objective: the scale_objective()-normalized accumulated
            objective for this segment (plus the accumulated reg_optimizer
            regularization term, if any -- see `regular` below).
          problem_objectives: list of per-step objective values (length
            num_iter + 1; the first entry is the objective at the params as
            they stood before this segment's first step).
        """
        problem = self._problem
        data, labels = self._dataset if self._dataset is not None else (None, None)

        params = self._params
        attend_params = self._attend_params
        states = self._state
        global_state = self._global_state

        # HALO's auxiliary convex regularization: one random target tensor per
        # enabled term, drawn once for this whole segment (matching the
        # original, which fed a fresh np.random.uniform draw into a
        # tf.placeholder once per train()/tf.while_loop-execution call, not
        # once per step) and passed to every objective() call below.
        aux_kwargs = {}
        if self.use_aux_convex_l1:
            aux_kwargs["aux_conv_labels_l1"] = tf.random.uniform(
                [self.aux_convex_dim], -1.0, 1.0)
        if self.use_aux_convex_l2:
            aux_kwargs["aux_conv_labels_l2"] = tf.random.uniform(
                [self.aux_convex_dim], -1.0, 1.0)
        if aux_kwargs:
            aux_kwargs["convex_l1_ratio"] = self.convex_l1_ratio

        problem_objectives = [problem.objective(params, data, labels, **aux_kwargs)]
        obj_accum = tf.constant(0., dtype=tf.float32)
        # Resets every partial_unroll call (unlike self._state, which persists
        # across calls) -- matches the original's init_regular = tf.constant(0.)
        # inside its per-tf.while_loop-execution setup.
        regular = tf.constant(0., dtype=tf.float32)

        for itr in range(num_iter):
            if batch_indices is not None:
                idx = tf.constant(batch_indices[itr], dtype=tf.int32)
                batch_data = tf.gather(data, idx)
                batch_labels = tf.gather(labels, idx)
            else:
                batch_data, batch_labels = data, labels

            # Full-batch objective: what gets accumulated into the meta-loss.
            obj = problem.objective(params, data, labels, **aux_kwargs)

            # Mini-batch objective: what gets differentiated for the update.
            attend_target = attend_params if self.use_attention else params

            if self.reg_optimizer or self.reg_optimizee:
                if self.reg_optimizee:
                    # The regularizer's internal HVP tapes are nested *inside*
                    # this outer tape's still-open context, so ops they record
                    # are also recorded by outer_tape (TF's tape stack records
                    # into every currently-active tape, not just the innermost
                    # one) -- outer_tape.gradient() below can then
                    # differentiate all the way through, including through the
                    # regularizer's own double-backprop. Matches the original,
                    # which recomputed raw (non-noisy) tf.gradients() of
                    # current_obj + beta*reg directly, bypassing
                    # problem.gradients()'s noise injection.
                    with tf.GradientTape() as outer_tape:
                        outer_tape.watch(attend_target)
                        reg = problem.regularizer(
                            attend_target, batch_data, batch_labels,
                            self.reg_option, self.hessian_itrs)
                        current_obj = problem.objective(
                            attend_target, batch_data, batch_labels, **aux_kwargs)
                        adjusted_obj = current_obj + self.beta * reg
                    grads = outer_tape.gradient(adjusted_obj, attend_target)
                else:
                    reg = problem.regularizer(
                        attend_target, batch_data, batch_labels,
                        self.reg_option, self.hessian_itrs)
                    with tf.GradientTape() as inner_tape:
                        inner_tape.watch(attend_target)
                        current_obj = problem.objective(attend_target, batch_data, batch_labels, **aux_kwargs)
                    grads = problem.gradients(current_obj, attend_target, inner_tape)

                if self.reg_optimizer and jacob_switch:
                    regular = regular + self.alpha * reg
            else:
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(attend_target)
                    current_obj = problem.objective(attend_target, batch_data, batch_labels, **aux_kwargs)
                grads = problem.gradients(current_obj, attend_target, inner_tape)

            if not self.use_second_derivatives:
                new_grads = []
                for grad in grads:
                    if isinstance(grad, tf.IndexedSlices):
                        new_grads.append(
                            tf.IndexedSlices(tf.stop_gradient(grad.values), grad.indices))
                    else:
                        new_grads.append(tf.stop_gradient(grad))
                grads = new_grads

            problem_objectives.append(obj)
            obj_accum = obj_accum + obj_weights[itr] * obj

            new_params, new_states, global_state, attend_params = self._compute_updates(
                params, grads, states, global_state)
            params, states = new_params, new_states

            if not bool(tf.math.is_finite(obj_accum)):
                break
            if self.obj_train_max_multiplier > 0:
                max_diff = (self.obj_train_max_multiplier - 1) * tf.abs(self._initial_obj)
                max_obj = self._initial_obj + max_diff
                if bool(obj >= max_obj):
                    break

        self._params = params
        self._attend_params = attend_params
        self._state = states
        self._global_state = global_state

        scaled_meta_objective = self.scale_objective(
            obj_accum, tf.stack(problem_objectives), self._initial_obj)
        scaled_meta_objective = scaled_meta_objective + regular

        return scaled_meta_objective, problem_objectives

    def scale_objective(self, total_obj, all_objs, initial_obj,
                        obj_scale_eps=1e-6):
        """Normalizes the objective based on the initial objective value.

        Args:
          total_obj: The total accumulated objective over the training run.
          all_objs: A tensor of all the individual objectives over the training run.
          initial_obj: The initial objective value.
          obj_scale_eps: The epsilon value to use in computations for stability.

        Returns:
          The scaled objective as a single value.
        """
        if self.use_log_objective:
            if self.use_numerator_epsilon:
                scaled_problem_obj = ((all_objs + obj_scale_eps) /
                                      (initial_obj + obj_scale_eps))
                log_scaled_problem_obj = tf.math.log(scaled_problem_obj)
            else:
                scaled_problem_obj = all_objs / (initial_obj + obj_scale_eps)
                log_scaled_problem_obj = tf.math.log(scaled_problem_obj + obj_scale_eps)
            return tf.reduce_mean(log_scaled_problem_obj)
        else:
            return total_obj / (initial_obj + obj_scale_eps)

    # ------------------------------------------------------------------
    # Save / restore
    # ------------------------------------------------------------------

    def save(self, path=None):
        """Saves the optimizer's own trainable weights to disk."""
        result = {v.name: v.numpy() for v in self.trainable_variables}
        if path:
            with open(path, "wb") as f:
                dill.dump(result, f)
        return result

    def load(self, path):
        """Loads previously-saved weights (must be called after the optimizer's
        sub-layers have been built at least once -- see subclass docs)."""
        with open(path, "rb") as f:
            saved = dill.load(f)
        var_by_name = {v.name: v for v in self.trainable_variables}
        for name, value in saved.items():
            if name in var_by_name:
                var_by_name[name].assign(value)
