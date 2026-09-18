"""LFISTA (learned FISTA) -- the second unrolled family.

Ported from ``Model_Base_L2O/models/lfista.py`` (Chen et al.'s toolbox). The
cell computes

    x_{k+1} = shrink_free(We y + Wg_k x_k + Wm_k x_{k-1}, theta_k)

with ``Wg_k = (1+m_k)(I - BA)``, ``Wm_k = -m_k (I - BA)``, ``B = A^T / L`` and
``m_k`` from the FISTA ``t_k`` recursion: momentum folded into per-layer
matrices, so a cell reads TWO previous iterates from the tape
(``inputs[:, -N:]`` and ``inputs[:, -2N:-N]``). ``We`` is one matrix tied
across all layers (the library builds ``[tf.Variable(B)] * T`` -- one object
repeated).

Two deliberate deviations from the library file, both identity keys of the
model so an evaluation can never silently differ from its training:

* **Trainable.** The library declares every parameter as ``tf.Variable``,
  which Keras 3 does not track: those weights are never trained, never
  saved, and the layer-wise curriculum's freeze/decay never sees them (the
  same defect ``lista.py``/``alista.py`` had before commit ``2b2264c``). Here
  they are ``keras.Variable``s.
* **``x0_at_layer0``** (default ``False`` = the library's behaviour, the parity
  target). The library's layer 0 computes ``shrink(We y)`` -- it starts from
  x = 0 and ignores the seeded x0 every other benchmarked method starts from,
  and its layer 1 has no ``Wm`` term because there is no x_{-1}. With
  ``x0_at_layer0=True`` layer 0 adds a fixed ``(I - BA) x0`` term (LISTA's
  layer-0 treatment) and layer 1 uses ``Wm_1 x0``, so every layer follows the
  same rule and the start-point fairness of the platform holds.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import keras

from unified.unrolled.lista import shrink_free


class LfistaCell(keras.layers.Layer):
  """Lfista cell."""

  def __init__(self, A, Wg, Wm, We, theta, layer_id, w0_fixed=None, name=None):
    super(LfistaCell, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self.Wg = Wg
    self.Wm = Wm
    self.We = We
    self.theta = theta
    self.layer_id = layer_id
    self.w0_fixed = w0_fixed          # (I - BA) constant for layer 0 when x0 is consumed
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

  def call(self, inputs):
    # output = We * y
    output = tf.matmul(inputs[:, :self._M], self.We, transpose_b=True)
    if self.layer_id > 0:
      inputs_ = inputs[:, -self._N:]
      output += tf.matmul(inputs_, self.Wg, transpose_b=True)
    elif self.w0_fixed is not None:
      # x0_at_layer0: the seeded starting point enters through the fixed ISTA matrix
      output += tf.matmul(inputs[:, -self._N:], self.w0_fixed, transpose_b=True)
    if self.Wm is not None:
      prev_inputs_ = inputs[:, -self._N*2:-self._N]
      output += tf.matmul(prev_inputs_, self.Wm, transpose_b=True)

    output = shrink_free(output, self.theta)
    return tf.concat([inputs, output], 1)


class Lfista(keras.Sequential):
  """Lfista model."""

  def __init__(self, A, T, lam, share_W=False, x0_at_layer0=False, D=None, name="Lfista"):
    super(Lfista, self).__init__(name=name)
    self._A = A.astype(np.float32)
    self._T = int(T)
    self._lam = lam
    self.share_W = share_W          # accepted like the library does; unused there too
    self.x0_at_layer0 = bool(x0_at_layer0)
    self._M = self._A.shape[0]
    self._N = self._A.shape[1]

    self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
    self._theta = (self._lam / self._scale).astype(np.float32)

    self._B = (np.transpose(self._A) / self._scale).astype(np.float32)
    _W = np.eye(self._N, dtype=np.float32) - np.matmul(self._B, self._A)
    self.w0_fixed = tf.constant(_W, dtype=tf.float32, name=name + "_W0_fixed")

    _tk = [1.0, 1.0]
    self._mk = []
    for i in range(self._T):
        _tk.append((1 + math.sqrt(1 + 4*_tk[-1]**2.0)) / 2)
        self._mk.append((_tk[-2] - 1) / _tk[-1])

    self._Wg = [
        keras.Variable(_W * (1 + self._mk[i]), trainable=True, name=name + "_Wg" + str(i + 1))
        for i in range(1, self._T)
    ]
    self._Wm = [
        keras.Variable(- self._mk[i] * _W, trainable=True, name=name + "_Wm" + str(i + 1))
        for i in range(1, self._T)
    ]
    self.theta = [
        keras.Variable(self._theta, trainable=True, name=name + "_theta" + str(i + 1))
        for i in range(self._T)
    ]
    # One tied We (the library's ``[tf.Variable(B)] * T`` is a single object repeated)
    self._We = keras.Variable(self._B, trainable=True, name=name + "_We1")
    self._D = None
    self._W_D = None

  def create_cell(self, layer_id):
    if layer_id != 0:
      Wg = self._Wg[layer_id - 1]
      # the library has no x_{k-1} at layer 1; with x0 consumed, x0 plays that role
      Wm = self._Wm[layer_id - 1] if (layer_id > 1 or self.x0_at_layer0) else None
    else:
      Wg = None
      Wm = None
    cell = LfistaCell(self._A, Wg, Wm, self._We, self.theta[layer_id], layer_id,
                      w0_fixed=self.w0_fixed if (layer_id == 0 and self.x0_at_layer0) else None,
                      name="Lfista_layer" + str(layer_id + 1))
    self.add(cell)
