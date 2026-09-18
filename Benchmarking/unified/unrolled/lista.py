"""LISTA / ALISTA models (verbatim) + the ``IterativeSolver`` view of them: layer k IS iteration k.

``ListaCell``/``Lista``, ``AlistaCell``/``Alista`` and the two shrinkage
functions are verbatim copies of ``Model_Base_L2O/models/{lista,alista,utils}.py``
(module-relative imports rewritten). Verbatim so the models present the same
variables in the same order as the ones that wrote Experiment 2's
checkpoints -- which matters because of HOW those checkpoints hold the
weights:

see ``loader.py`` for how the checkpoints hold the weights and why the
verbatim variable order matters.
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp

from unified.problem import Batch
from unified.solver import IterativeSolver


# ---------------------------------------------------------------- models/utils.py

def shrink_free(data, theta):
  return tf.sign(data) * keras.layers.ReLU()(tf.abs(data) - theta)


def shrink_ss(inputs_, theta_, q, return_index=False):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.
    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO
    """
    abs_ = tf.abs(inputs_)
    thres_ = tfp.stats.percentile(abs_, 100.0-q, axis=1, keepdims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.cast(index_, tf.float32)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    output = (tf.multiply(index_, inputs_) +
              shrink_free(tf.multiply(cindex_, inputs_), theta_ ))
    if return_index:
        return output, cindex_
    else:
        return output


# ---------------------------------------------------------------- models/lista.py

class ListaCell(keras.layers.Layer):
  """Lista cell."""

  def __init__(self,
               A,
               w_1,
               W,
               share_W,
               step_size,
               theta,
               layer_id,
               name=None):
    super(ListaCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self.w_1 = w_1
    self.w_2 = W
    self.share_W = share_W
    self.step_size = step_size
    self.theta = theta
    self.layer_id = layer_id
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def call(self, inputs):
    # output = B * y
    output = tf.matmul(inputs[:, :self._M], self.w_1, transpose_b=True)
    # inputs_ is x0 for layer 0 (the seeded starting point data_preprocessing.py)
    inputs_ = inputs[:, -self._N:]
    if self.layer_id != 0 and self.share_W:
      inputs_ = inputs_ * self.step_size[self.layer_id - 1]
    output = output + tf.matmul(inputs_, self.w_2, transpose_b=True)
    output = shrink_free(output, self.theta)
    return tf.concat([inputs, output], 1)


class Lista(keras.Sequential):
  """Lista model."""

  def __init__(self,
               A,
               T,
               lam,
               share_W=False,
               D=None,
               name="Lista"):
    super(Lista, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._T = int(T)
    self._lam = lam
    self.share_W = share_W
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self._B = (np.transpose(self._A) / self._scale).astype(np.float32)
    _W = np.eye(self._N, dtype=np.float32) - np.matmul(self._B, self._A)
    self.w0_fixed = tf.constant(_W, dtype=tf.float32, name=name + "_W0_fixed")
    if share_W:
      self._W = keras.Variable(_W, trainable=True, name=name + "_W")
    else:
      self._W = [
          keras.Variable(_W, trainable=True, name=name + "_W" + str(i + 1))
          for i in range(1, self._T)
      ]
    self.theta = [
        keras.Variable(
            self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    if share_W:
      self.step_size = [
          keras.Variable(1.0, trainable=True,
                      name=name + "_step_size" + str(i + 1)) for i in range(self._T)
      ]
    else:
      self.step_size = [1.0] * self._T

    self.w_1 = keras.Variable(self._B, trainable=True, name=name + "_B")
    if D is not None:
      self._D = D
      self._W_D_constant = keras.Variable(self._D, trainable=False, name=name + "_W_D_constant")
      self._W_D = keras.Variable(self._D, trainable=True, name=name + "_W_D")
    else:
      self._D = None
      self._W_D = None

  def create_cell(self, layer_id):
    if layer_id == 0:
      w = self.w0_fixed
    elif self.share_W:
      w = self._W
    else:
      w = self._W[layer_id - 1]
    if self._D is None:
      w_d = self._W_D
      F = 0
    else:
      F = self._D.shape[0]
      if layer_id == self._T - 1:
        w_d = self._W_D
      else:
        w_d = self._W_D_constant

    w_1 = self.w_1
    cell = ListaCell(self._A, w_1, w, self.share_W,
                     self.step_size[layer_id],
                     self.theta[layer_id],
                     layer_id, "Lista_layer" + str(layer_id + 1))

    self.add(cell)


# --------------------------------------------------------------- models/alista.py

class AlistaCell(keras.layers.Layer):
  """Alista cell."""

  def __init__(self,
               A,
               W,
               step_size,
               theta,
               q,
               layer_id,
               name=None):
    super(AlistaCell, self).__init__(name=name)
    self._A = A
    self._W = W
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]
    self.step_size = step_size
    self.theta = theta
    self.q = q
    self.layer_id = layer_id

  def call(self, inputs):
    y = inputs[:, :self._M]
    xk = inputs[:, -self._N:]
    res = y - tf.matmul(xk, self._A, transpose_b=True)
    output = xk + self.step_size * tf.matmul(res, self._W, transpose_b=False)
    output = shrink_ss(output, self.theta, self.q)
    return tf.concat([inputs, output], 1)


class Alista(keras.Sequential):
  """Alista model."""

  def __init__(self,
               A,
               W,
               T,
               lam,
               q_per_layer,
               maxq,
               D=None,
               name="Alista"):
    super(Alista, self).__init__(name=name)
    self._A = tf.constant(A.astype(np.float32), name=name+'_A_const')
    self._W = tf.constant(W.astype(np.float32), name=name+'_W_const')
    self._T = int(T)
    self._lam = lam
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self.q = np.clip([(t+1) * q_per_layer for t in range(self._T)], 0.0, maxq)

    # keras.Variable, not tf.Variable: Keras 3 as it does not
    #  auto-track bare tf.Variable attributes as layer weights anymore.
    self.theta = [
        keras.Variable(self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    self.step_size = [
        keras.Variable(1.0, trainable=True, name=name + "_step_size" + str(i + 1))
        for i in range(self._T)
    ]

    if D is not None:
      self._D = D
      self._W_D_constant = keras.Variable(self._D, trainable=False, name=name + "_W_D_constant")
      self._W_D = keras.Variable(self._D, trainable=True, name=name + "_W_D")
    else:
      self._D = None
      self._W_D = None

  def create_cell(self, layer_id):
    if self._D is None:
      w_d = self._W_D
      F = 0
    else:
      F = self._D.shape[0]
      if layer_id == self._T - 1:
        w_d = self._W_D
      else:
        w_d = self._W_D_constant

    cell = AlistaCell(self._A, self._W,
                      self.step_size[layer_id],
                      self.theta[layer_id],
                      self.q[layer_id],
                      layer_id,
                      "Alista_layer" + str(layer_id + 1))

    self.add(cell)


# ------------------------------------------------------------------- unified glue

MODELS = {}   # model_name -> factory(a, num_layers, model_lam, w_alista, ss_q_per_layer, ss_maxq, share_W, ident)


def _lista(a, num_layers, model_lam, w_alista, ss_q_per_layer, ss_maxq, share_W, ident):
    return Lista(a, num_layers, model_lam, share_W, None, name="Lista")


def _alista(a, num_layers, model_lam, w_alista, ss_q_per_layer, ss_maxq, share_W, ident):
    if w_alista is None:
        raise ValueError("alista needs W.npy next to A.npy (data/lasso.py compute_alista_w=True)")
    return Alista(a, w_alista, num_layers, model_lam, ss_q_per_layer, ss_maxq, None, name="Alista")


def _lfista(a, num_layers, model_lam, w_alista, ss_q_per_layer, ss_maxq, share_W, ident):
    from unified.unrolled.lfista import Lfista
    return Lfista(a, num_layers, model_lam, share_W, x0_at_layer0=bool(ident.get("x0_at_layer0", False)),
                  name="Lfista")


MODELS.update({"lista": _lista, "alista": _alista, "lfista": _lfista})


def build_model(model_name: str, a: np.ndarray, num_layers: int, model_lam: float,
                w_alista=None, ss_q_per_layer: float = 1.2, ss_maxq: float = 13.0,
                share_W: bool = False, create_cells: bool = True, ident=None):
    """The model exactly as ``Model_Base_L2O/train.py`` constructs it; with every
    cell created (evaluation) or none (the curriculum adds them layer by layer)."""
    if model_name not in MODELS:
        raise ValueError("unified unrolled models: one of {}, got {!r}".format(sorted(MODELS), model_name))
    model = MODELS[model_name](a, num_layers, model_lam, w_alista, ss_q_per_layer, ss_maxq, share_W, ident or {})
    if create_cells:
        for layer_id in range(num_layers):
            model.create_cell(layer_id)
    return model


class UnrolledSolver(IterativeSolver):
    """Runs the created cells one at a time, yielding x_k = the last N columns."""

    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.layers)
        self.n = model._N

    def iterate(self, batch: Batch, num_steps: int) -> Iterator:
        if num_steps != self.num_layers:
            raise ValueError("an unrolled model has a fixed depth: num_steps={} but the model "
                             "has {} layers".format(num_steps, self.num_layers))
        # Model input is [b, x0] (data_preprocessing.dataset_parser); every cell
        # appends its output, so the last N columns are always the current iterate.
        h = tf.concat([tf.constant(batch.data["b"]), tf.constant(batch.x0)], axis=1)
        yield h[:, -self.n:]
        for cell in self.model.layers:
            h = cell(h)
            yield h[:, -self.n:]
