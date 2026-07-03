"""Learning to optimize in swarms — meta-optimizer (eager / TF2).

A population of `num_lstm` particles is optimized jointly: each step, per-particle
gradients are combined with an inter-particle attention (across the population) and
an intra-particle attention (combining gradient, momentum, distance-to-best and
inter-particle attraction features) before being fed to a CoordinateWiseDeepLSTM
that proposes the parameter update. The meta-loss is `entropy_loss.self_loss`,
which balances exploitation (mean loss) against exploration (population entropy).

Training follows the original (TF1) truncated-BPTT scheme:
    setup()       -- called once: fixes the problem instance (loss_fn/const_vars)
                      and builds the optimizer networks + attention weights.
    reset_state() -- called at the start of every training epoch: redraws random
                      particle positions/state (the problem instance itself stays
                      fixed, matching the original graph-mode "reset" op, which only
                      re-randomized x/state, never the problem's own constants).
    meta_minimize() -- runs one `len_unroll`-step segment from the current
                      particle state and applies one meta-gradient update,
                      continuing the trajectory (self.x/self.state) across calls.
"""

import os
import pickle

import numpy as np
import tensorflow as tf

import entropy_loss
import networks


def _make_nets(x_vars, config, net_assignments):
    """Creates the optimizer networks.

    Returns (nets, keys, subsets) where nets is a dict of created optimizer nets
    such that the net with key keys[i] should be applied to the subset of
    variables listed in subsets[i].
    """
    name_to_index = {v.name.split(":")[0]: i for i, v in enumerate(x_vars)}

    if net_assignments is None:
        if len(config) != 1:
            raise ValueError("Default net_assignments can only be used if there is "
                             "a single net config.")
        key = next(iter(config))
        net = networks.factory(**config[key])
        return {key: net}, [key], [list(range(len(x_vars)))]

    nets, keys, subsets = {}, [], []
    for key, names in net_assignments:
        if key in nets:
            raise ValueError("Repeated netid in net_assignments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))
    return nets, keys, subsets


class MetaOptimizer:
    """Learning to learn (meta) optimizer over a swarm of particles."""

    def __init__(self, **kwargs):
        self._nets = None
        self._net_keys = None
        self._optimizer = None

        self.num_lstm = 4
        self.intra_features = 4
        self.fc_kernel = []
        self.fc_bias = []
        self.fc_va = []

        self._loss_fn = None
        self._x_vars = None
        self.x = None
        self.state = None
        self.x_minimal = None
        self.pre_deltas = None
        self.pre_gradients = None
        self.fx_minimal = None

        if not kwargs:
            self._config = {
                "coordinatewise": {
                    "net": "CoordinateWiseDeepLSTM",
                    "net_options": {
                        "layers": (20, 20),
                        "preprocess_name": "LogAndSign",
                        "preprocess_options": {"k": 5},
                        "scale": 0.01,
                    }}}
        else:
            self._config = kwargs

    @property
    def trainable_variables(self):
        variables = [v for net in self._nets.values() for v in net.trainable_variables]
        return variables + list(self.fc_kernel) + list(self.fc_bias)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _intra_init(self, x_vars, model_path):
        """Initializes the intra-particle attention weights (one per optimizee variable)."""
        fc_columns = 10
        if model_path is None:
            for xi in x_vars:
                flat_dim = int(np.prod(xi.shape))
                self.fc_kernel.append(tf.Variable(
                    tf.random.normal([fc_columns, flat_dim * 2]), name="fc_kernel"))
                self.fc_bias.append(tf.Variable(
                    tf.random.normal([fc_columns, self.intra_features]), name="fc_bias"))
                self.fc_va.append(tf.Variable(
                    tf.ones([1, fc_columns]), trainable=False, name="fc_va"))
        else:
            with open("./{}/loss_record.pickle".format(model_path), "rb") as f:
                data = pickle.load(f)
            self.fc_kernel = [tf.Variable(item) for item in data["fc_weights"]]
            self.fc_bias = [tf.Variable(item) for item in data["fc_bias"]]
            for _ in x_vars:
                self.fc_va.append(tf.Variable(tf.ones([1, fc_columns]), trainable=False))

    def setup(self, make_loss, net_assignments=None, model_path=None):
        """Fixes a problem instance and builds the optimizer networks (call once)."""
        x_vars, const_vars, loss_fn = make_loss()
        self._loss_fn = loss_fn
        self._x_vars = x_vars

        nets, net_keys, subsets = _make_nets(x_vars, self._config, net_assignments)
        if len(subsets) != 1:
            raise NotImplementedError(
                "The swarm intra/inter-attention update only supports a single "
                "optimizer network covering all variables (net_assignments=None).")
        self._nets = nets
        self._net_keys = net_keys

        self._intra_init(x_vars, model_path)
        self.reset_state()

    def reset_state(self):
        """Redraws random particle positions and resets recurrent state.

        Call at the start of every training epoch; the problem instance
        (loss_fn/const_vars fixed by `setup`) is left untouched.
        """
        self.x = self._vars_init(self._x_vars)
        self.x_minimal = self._vars_init(self._x_vars)
        self.pre_deltas = self._vars_init(self._x_vars)
        self.pre_gradients = self._vars_init(self._x_vars)
        self.fx_minimal = tf.constant(float("inf"))

        net = self._nets[self._net_keys[0]]
        self.state = [net.initial_state_for_inputs(xj, dtype=tf.float32) for xj in self.x]

    def _vars_init(self, x_vars):
        return [tf.random.normal([self.num_lstm] + list(xi.shape), mean=0.0, stddev=0.01)
                for xi in x_vars]

    # ------------------------------------------------------------------
    # Attention helpers
    # ------------------------------------------------------------------

    def _attraction_init(self, mat):
        """Softmax-weighted attraction of a particle towards the rest of the population."""
        alpha = 1
        norm = []
        for i in range(self.num_lstm):
            mat_norm = tf.reshape(mat[i], [self.num_lstm, -1])
            mat_l2 = tf.reduce_sum(tf.multiply(mat_norm, mat_norm), axis=1)
            mat_l2 = -alpha * tf.reshape(mat_l2, [1, -1])
            mat_softmax = tf.nn.softmax(mat_l2)
            norm.append(tf.matmul(mat_softmax, mat_norm))
        return tf.reshape(tf.stack(norm, axis=0), tf.shape(mat[0]))

    def _inter_attention(self, mat_a, mat_b):
        """Attention across particles, applied to their (population-flattened) gradients."""
        l = 1
        gama = 1.0 / self.num_lstm
        origin = mat_a
        grad = mat_b

        def x_res_init(mat):
            mat_split = tf.split(mat, self.num_lstm)
            norm = []
            for i in range(self.num_lstm):
                sub_norm = tf.tile(mat_split[i], [self.num_lstm, 1])
                sub_norm = sub_norm - mat
                norm.append(tf.matmul(mat_split[i], tf.transpose(sub_norm)))
            result = (-1.0 / (2 * l)) * tf.concat(norm, axis=0)
            return tf.nn.softmax(tf.transpose(result))

        matmul1 = tf.matmul(grad, tf.transpose(grad))
        softmax_grad = tf.nn.softmax(tf.transpose(matmul1))
        softmax_origin = x_res_init(origin)
        input_mul_x = tf.matmul(softmax_grad, softmax_origin)
        result = gama * tf.matmul(input_mul_x, grad)
        return tf.add(grad, result)

    def _intra_attention(self, grad, pre_grad, x, x_min, sub_x_attraction, ht,
                         fc_kernel, fc_bias, fc_va):
        """Combines gradient/momentum/distance-to-best/attraction features per particle."""
        beta = 0.9
        intra_feature = tf.concat([grad, beta * pre_grad, x - x_min, sub_x_attraction], axis=0)
        intra_feature = tf.reshape(intra_feature, [self.num_lstm * self.intra_features, -1])
        ht_concat = tf.concat([ht for _ in range(self.intra_features)], axis=0)
        ht_concat = tf.reshape(ht_concat, [self.num_lstm * self.intra_features, -1])
        intra_concat = tf.concat([intra_feature, ht_concat], axis=1)
        intra_fc_bias = tf.tile(fc_bias, [1, self.num_lstm])
        intra_fc = tf.tanh(tf.matmul(fc_kernel, tf.transpose(intra_concat)) + intra_fc_bias)
        b_ij = tf.matmul(fc_va, intra_fc)
        p_ij = tf.nn.softmax(tf.reshape(b_ij, [self.num_lstm, -1]))
        p_ij = tf.reshape(p_ij, [self.num_lstm * self.intra_features, 1])
        gradient = tf.multiply(p_ij, intra_feature)
        gradient = tf.reshape(gradient, [self.intra_features, -1])
        gradient = tf.reduce_sum(gradient, axis=0)
        return tf.reshape(gradient, tf.shape(grad))

    # ------------------------------------------------------------------
    # Unroll
    # ------------------------------------------------------------------

    def unroll_step(self, len_unroll, second_derivatives=False):
        """Runs one `len_unroll`-step segment from the current particle state.

        Continues the trajectory: `self.x`/`self.state` are updated in place to the
        segment's final values, so consecutive calls perform truncated BPTT.

        Returns (loss, fx_final) where fx_final is the list of per-particle final
        losses (length num_lstm).
        """
        loss_fn = self._loss_fn
        net = self._nets[self._net_keys[0]]
        x = self.x
        state = self.state

        x_array, fx_array = [], []

        for _ in range(len_unroll):
            update_fx = []
            per_particle_grads = []
            for z in range(self.num_lstm):
                sub_x = [item[z] for item in x]
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(sub_x)
                    sub_fx_batch = loss_fn(sub_x)
                    sub_fx = tf.reduce_mean(sub_fx_batch)
                sub_grad = inner_tape.gradient(sub_fx, sub_x)
                x_array.append(sub_x[0])
                fx_array.append(sub_fx_batch)
                update_fx.append(sub_fx)
                per_particle_grads.append(sub_grad)

            gradients = [tf.stack([per_particle_grads[z][vi] for z in range(self.num_lstm)], axis=0)
                        for vi in range(len(x))]

            # Pairwise (raw) attraction: for each particle, the displacement towards
            # every other particle that has a lower (better) loss.
            attraction = []
            for j in range(len(x)):
                attraction_x = []
                for ind1 in range(self.num_lstm):
                    sub_attraction_x = []
                    for ind2 in range(self.num_lstm):
                        keep_zero = update_fx[ind1] > update_fx[ind2]
                        val = tf.where(keep_zero, x[j][ind1] - x[j][ind1], x[j][ind1] - x[j][ind2])
                        sub_attraction_x.append(val)
                    attraction_x.append(tf.stack(sub_attraction_x, axis=0))
                attraction.append(attraction_x)

            fx_sum = tf.reduce_sum(tf.stack(update_fx))
            if float(fx_sum) <= float(self.fx_minimal):
                self.fx_minimal = fx_sum
                self.x_minimal = x

            x_attraction = [self._attraction_init(sub) for sub in attraction]

            # Stopping the gradient here corresponds to what was done in the original
            # L2L NIPS submission: avoid differentiating through the loss function's
            # own Hessian, while still training the optimizer via BPTT.
            if not second_derivatives:
                gradients = [tf.stop_gradient(g) for g in gradients]

            for i in range(len(x)):
                shape = tf.shape(x[i])
                mat_x = tf.reshape(x[i], [self.num_lstm, -1])
                mat_grads = tf.reshape(gradients[i], [self.num_lstm, -1])
                inter_grads = self._inter_attention(mat_x, mat_grads)
                gradients[i] = tf.reshape(inter_grads, shape)

            x_min = self.x_minimal
            ht = self.pre_deltas
            pre_grads = self.pre_gradients

            deltas = []
            new_state = []
            for grad, pre_grad, xj, xminj, sub_att, htj, fck, fcb, fcv, s in zip(
                    gradients, pre_grads, x, x_min, x_attraction, ht,
                    self.fc_kernel, self.fc_bias, self.fc_va, state):
                feature = self._intra_attention(grad, pre_grad, xj, xminj, sub_att, htj, fck, fcb, fcv)
                delta, s_next = net(feature, s)
                deltas.append(delta)
                new_state.append(s_next)

            self.pre_deltas = deltas
            self.pre_gradients = gradients

            x = [x[j] + deltas[j] for j in range(len(x))]
            state = new_state

        # Final evaluation of the segment's end point (no parameter update).
        fx_final = []
        for z in range(self.num_lstm):
            sub_x = [item[z] for item in x]
            sub_fx_final_batch = loss_fn(sub_x)
            sub_fx_final = tf.reduce_mean(sub_fx_final_batch)
            fx_array.append(sub_fx_final_batch)
            x_array.append(sub_x[0])
            fx_final.append(sub_fx_final)

        loss = entropy_loss.self_loss(
            tf.stack(x_array, axis=0), tf.stack(fx_array, axis=0),
            (len_unroll + 1) * self.num_lstm)

        self.x = x
        self.state = state

        return loss, fx_final

    def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01,
                      net_assignments=None, model_path=None, second_derivatives=False):
        """Runs one unroll segment and applies one meta-gradient update.

        `setup()` is called automatically on the first invocation. Subsequent calls
        continue the particle trajectory (matching the truncated-BPTT training loop);
        call `reset_state()` between training epochs to redraw random particle
        positions.
        """
        if self._loss_fn is None:
            self.setup(make_loss, net_assignments, model_path)
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.Adam(learning_rate)

        with tf.GradientTape() as tape:
            loss, fx_final = self.unroll_step(len_unroll, second_derivatives)
            reg = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
            total_loss = loss + reg

        grads = tape.gradient(total_loss, self.trainable_variables)
        self._optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, self.trainable_variables) if g is not None)

        return total_loss, fx_final, self.x

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path=None):
        """Saves the LSTM network weights to disk (fc_kernel/fc_bias/fc_va are saved
        separately by the caller, alongside the loss record — see train.py)."""
        result = {}
        for k, net in self._nets.items():
            filename = None if path is None else os.path.join(path, "{}.l2l".format(k))
            key = k if path is None else filename
            net_vars = networks.save(net, filename=filename)
            result[key] = net_vars
        return result
