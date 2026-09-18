"""Layer-wise curriculum training for the unrolled models (LISTA / ALISTA).

Port of ``Model_Base_L2O/train.py``'s training path (lines 139-154 datasets,
312-406 curriculum) with ``utils.LassoLoss`` / ``utils.LassoObjective`` /
``utils.Adam`` copied verbatim. For each layer k:

  1. ``model.create_cell(k)``;
  2. stage 1: ``Adam(var_list, freeze_layer=True, lr)`` -- only the variables
     introduced by this layer receive gradients;
  3. stages 2-3: ``Adam(var_list, lr*0.2)``, ``Adam(var_list, lr*0.02)`` -- every
     variable trains, earlier layers' gradients scaled by ``0.3**age``;
  each stage a Keras ``fit()`` with ``EarlyStopping(monitor="val_lasso",
  patience, restore_best_weights=False)`` and ``epochs`` as a safety cap;
  4. ``tf.train.Checkpoint(model=model).save(<model_dir>/layer_<k+1>/model)``.

The saved checkpoint has the SAME object graph as the library's (the compiled
optimizer's variable list is what carries the per-layer weights, see
``unrolled.py``), so ``Model_Base_L2O/train.py --test`` and the ``lista`` /
``alista`` adapters read unified-trained checkpoints unchanged.

Faithful, not fixed (see ``unified/README.md``): ``LassoLoss`` uses
``tf.nn.l2_loss`` (= sum/2), i.e. the training objective is
``0.25*||Ax-b||^2 + lam*||x||_1`` summed over the batch, while every scoring
path uses ``0.5*||Ax-b||^2`` per instance. ``keras_train_loss="paper"`` keeps
that; ``"scoring"`` is available as an explicit, non-parity option.

Training is not bit-reproducible even in the library (the training
``tf.data`` pipeline sets ``experimental_deterministic=False``); the tests
check structure, value ranges and checkpoint interchangeability instead.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from unified.tasks.lasso import LassoProblem


# ----------------------------------------------------------- Model_Base utils.py

class LassoLoss(tf.keras.losses.Loss):

  def __init__(self, A, lam, N, F=None, **kwargs):
    super(LassoLoss, self).__init__(**kwargs)
    # self._A = tf.Variable(A, trainable=False, name='A_const_loss')
    self._A = A
    self._lam = lam
    self.F = F
    self.N = N

  def call(self, y_true, y_pred):
    sparse_pred = y_pred[:, -self.N:]
    measure_pred = tf.matmul(y_pred[:, -self.N:], self._A, transpose_b=True)
    l2_loss = tf.nn.l2_loss(measure_pred - y_true[:, :-self.N])
    l1_loss = tf.reduce_sum(tf.abs(sparse_pred))
    return 0.5 * l2_loss + self._lam * l1_loss


class ScoringLassoLoss(tf.keras.losses.Loss):
  """NOT the library's loss: per-instance ``0.5||Ax-b||^2 + lam||x||_1``, batch
  mean -- the objective everything is *scored* with. Opt-in via
  ``keras_train_loss="scoring"``; never used for parity."""

  def __init__(self, A, lam, N, **kwargs):
    super().__init__(**kwargs)
    self._A, self._lam, self.N = A, lam, N

  def call(self, y_true, y_pred):
    x = y_pred[:, -self.N:]
    residual = tf.matmul(x, self._A, transpose_b=True) - y_true[:, :-self.N]
    return tf.reduce_mean(0.5 * tf.reduce_sum(residual ** 2, axis=-1)
                          + self._lam * tf.reduce_sum(tf.abs(x), axis=-1))


class LassoObjective(tf.keras.metrics.Mean):

  def __init__(self, name, A, lam, M, N, layer_id=-1, dtype=None):
    super(LassoObjective, self).__init__(name=name, dtype=dtype)
    # self._A = tf.Variable(A, trainable=False, name='A_const_obj')
    self._A = A
    self._lam = lam
    self.M = M
    self.N = N
    self.layer_id = layer_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.layer_id == -1:
      sparse_pred = y_pred[:, -self.N:]
    elif self.layer_id >= 0:
        idx = self.M + self.layer_id * self.N
        sparse_pred = y_pred[:, idx:idx+self.N]
    else:
      raise ValueError('Invalid layer index {}'.format(self.layer_id))
    measure_pred = tf.matmul(sparse_pred, self._A, transpose_b=True)
    l2_loss = tf.reduce_sum(tf.square(measure_pred - y_true[:, :self.M]), axis=-1)
    l1_loss = tf.reduce_sum(tf.abs(sparse_pred), axis=-1)
    lasso_obj = 0.5 * l2_loss + self._lam * l1_loss
    return super(LassoObjective, self).update_state(lasso_obj, sample_weight=sample_weight)


class Adam(tf.keras.optimizers.Adam):

  def __init__(self, var_list, freeze_layer=False, **kwargs):
    super(Adam, self).__init__(**kwargs)
    self.var_list = var_list
    self.freeze_layer = freeze_layer

  def apply_gradients(self, grads_and_vars):
    grads_and_vars_multiplied = []
    for g, v in grads_and_vars:
      if g is None:
        continue
      if self.freeze_layer:
        if self.var_list[v.name] == 0:
          grads_and_vars_multiplied.append((g, v))
      else:
        g_mul = g * 0.3**self.var_list[v.name]
        grads_and_vars_multiplied.append((g_mul, v))
    super(Adam, self).apply_gradients(grads_and_vars_multiplied)


# ------------------------------------------------- Model_Base data_preprocessing.py

def make_dataset(problem: LassoProblem, batch_size: int, is_training: bool) -> tf.data.Dataset:
    """``input_fn`` for task='lasso': records ``[b; x_true; x0]`` -> (input ``[b, x0]``, label ``[b, x_true]``)."""
    m, n = problem.m, problem.n
    data = np.concatenate([problem._data, problem._x0], axis=1)

    def dataset_parser(value):
        inputs = tf.concat([value[:m], value[m + n:]], axis=-1)
        return inputs, value[:m + n]

    dataset = tf.data.Dataset.from_tensor_slices(data).cache()
    if is_training:
        dataset = dataset.shuffle(buffer_size=data.shape[0]).repeat()
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_threading.private_threadpool_size = 16
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        dataset_parser, batch_size=batch_size, num_parallel_batches=2, drop_remainder=False))
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ------------------------------------------------------------------ curriculum

def train_curriculum(model, train_instances: LassoProblem, val_instances: LassoProblem,
                     model_dir: str, *, lasso_lam: float, epochs: int, patience: int, base_lr: float,
                     train_batch_size: int, val_batch_size: int = 1024,
                     keras_train_loss: str = "paper", log: Callable[[str], None] = print,
                     verbose: int = 0) -> List[Dict]:
    """Train ``model`` (an un-built ``Lista``/``Alista``) layer by layer; returns per-layer history."""
    if os.path.isdir(os.path.join(model_dir, "layer_1")):
        raise FileExistsError("{} already holds layer checkpoints; the unified slice does not "
                              "resume partial curricula".format(model_dir))
    m, n = train_instances.m, train_instances.n
    a_const = tf.constant(train_instances.a, name="A_lasso_const")
    if keras_train_loss == "paper":
        loss = LassoLoss(a_const, lasso_lam, n, None)
    elif keras_train_loss == "scoring":
        loss = ScoringLassoLoss(a_const, lasso_lam, n)
    else:
        raise ValueError("keras_train_loss must be 'paper' or 'scoring'")
    monitor = "val_lasso"
    train_ds = make_dataset(train_instances, train_batch_size, True)
    val_ds = make_dataset(val_instances, val_batch_size, False)
    # train.py: integer division, passed to fit() as-is (0 when the val split is
    # smaller than one val batch, which Keras treats as "the whole dataset").
    steps_per_epoch = int(train_instances.num_samples // train_batch_size)
    val_steps = int(val_instances.num_samples // val_batch_size)
    if steps_per_epoch == 0:
        raise ValueError("train split ({} rows) smaller than train_batch_size {}: zero steps per epoch".format(
            train_instances.num_samples, train_batch_size))

    var_list: Dict[str, int] = {}
    checkpoint = tf.train.Checkpoint(model=model)
    histories = []
    for layer_id in range(model._T):
        model.create_cell(layer_id)
        for name in var_list:
            var_list[name] += 1
        for v in model.layers[layer_id].trainable_variables:
            if v.name not in var_list:
                var_list[v.name] = 0

        def fit(optimizer):
            model.compile(optimizer=optimizer, loss=loss,
                          metrics=[LassoObjective("lasso", a_const, lasso_lam, m, n, -1)])
            cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=patience,
                                                  mode="min", restore_best_weights=False)
            h = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[cb],
                          validation_data=val_ds, validation_steps=val_steps, verbose=verbose)
            return h.history

        stage_hist = [fit(Adam(var_list, True, learning_rate=base_lr))]
        for i in range(2):
            stage_hist.append(fit(Adam(var_list, learning_rate=base_lr * 0.2 * 0.1 ** i)))
        val_metric = stage_hist[-1][monitor][-1]
        save_prefix = os.path.join(model_dir, "layer_{}".format(layer_id + 1), "model")
        checkpoint.save(save_prefix)
        log("layer {}: stage epochs {} val_lasso {:.6g} saved {}".format(
            layer_id + 1, [len(h[monitor]) for h in stage_hist], val_metric, save_prefix))
        histories.append({"layer": layer_id, "stage_epochs": [len(h[monitor]) for h in stage_hist],
                          "val_lasso": float(val_metric)})
    return histories
