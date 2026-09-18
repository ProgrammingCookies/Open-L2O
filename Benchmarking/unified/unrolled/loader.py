"""Loading the layer-wise ``tf.train.Checkpoint`` files that
``Model_Base_L2O/train.py`` writes (``<model_dir>/layer_<k>/model-<k>``).

``train.py`` saves ``tf.train.Checkpoint(model=model)``. Keras 3 wraps list
attributes (``theta``, ``_W``, ``step_size``) in a ``TrackedList`` that TF's
checkpointing does not descend into, so the per-layer weights are NOT stored
under ``model/theta/...``. They reach the file only through the compiled
optimizer's variable list, as ``model/optimizer/_trainable_variables/<i>`` in
``model.trainable_variables`` order (LISTA: ``[_W[1..T-1], theta[1..T], w_1]``,
ALISTA: ``[theta[1..T], step_size[1..T]]``); ``w_1`` is a bare attribute, so it
appears once under the shorter path ``model/w_1``. ``load_layerwise_checkpoint``
reads exactly that layout, with the count and shape checks the DM/RNNProp
loader has, instead of relying on Keras rebuilding an optimizer to trigger a
deferred restore. It works for any model whose trainable variables are
Keras Variables held in attributes or lists, in construction order.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
import keras


_TV_KEY = "model/optimizer/_trainable_variables/{}/.ATTRIBUTES/VARIABLE_VALUE"
_ATTR_KEY = "model/{}/.ATTRIBUTES/VARIABLE_VALUE"


def layer_checkpoint_prefix(model_dir: str, num_layers: int) -> str:
    """``<model_dir>/layer_<T>/model-<k>`` as ``utils.check_and_load_partial`` finds it."""
    layer_dir = os.path.join(model_dir, "layer_{}".format(num_layers))
    prefix = tf.train.latest_checkpoint(layer_dir)
    if not prefix:
        raise FileNotFoundError("no tf.train checkpoint under {} -- model not fully trained "
                                "to {} layers".format(layer_dir, num_layers))
    return prefix


def load_layerwise_checkpoint(model, model_dir: str, num_layers: int) -> Dict[str, str]:
    """Assign every trainable variable of ``model`` from the layer-``num_layers``
    checkpoint; returns ``{variable name: checkpoint key}`` for inspection.
    Raises if any variable has no entry or the shapes disagree."""
    prefix = layer_checkpoint_prefix(model_dir, num_layers)
    reader = tf.train.load_checkpoint(prefix)
    shapes = reader.get_variable_to_shape_map()
    # Bare Variable attributes are stored under their attribute path (the
    # object graph keeps the first/shortest path to a shared object).
    attr_paths = {}
    for attr, value in vars(model).items():
        if isinstance(value, keras.Variable):
            attr_paths[id(value)] = _ATTR_KEY.format(attr)

    variables: List[keras.Variable] = list(model.trainable_variables)
    used: Dict[str, str] = {}
    for i, v in enumerate(variables):
        key = _TV_KEY.format(i)
        if key not in shapes:
            key = attr_paths.get(id(v), key)
        if key not in shapes:
            raise RuntimeError("{}: no entry for trainable variable {} ({!r}); checkpoint keys "
                               "under model/: {}".format(prefix, i, v.name,
                                                         sorted(k for k in shapes if k.startswith("model/"))[:8]))
        value = reader.get_tensor(key)
        if tuple(np.shape(value)) != tuple(v.shape):
            raise RuntimeError("{}: {} has shape {} but variable {!r} has shape {}".format(
                prefix, key, np.shape(value), v.name, tuple(v.shape)))
        v.assign(value)
        used[v.name] = key

    # Same count discipline as nets_dm._assign_saved_weights: the file must
    # hold exactly one trained tensor per model variable, no more.
    n_indexed = sum(1 for k in shapes if k.startswith("model/optimizer/_trainable_variables/"))
    n_attr = sum(1 for k in used.values() if not k.startswith("model/optimizer/"))
    if n_indexed + n_attr != len(variables):
        raise RuntimeError("{}: holds {} optimizer-tracked + {} attribute tensors but the model "
                           "has {} trainable variables".format(prefix, n_indexed, n_attr, len(variables)))
    return used


