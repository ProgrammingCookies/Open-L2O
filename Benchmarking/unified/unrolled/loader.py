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

    # Variables no cell references (e.g. LFISTA's layer-1 Wm in the library's
    # faithful mode) never reach Keras' optimizer list and are therefore not in
    # the file; they keep their construction values, which is exactly what the
    # library's model has too. Everything a cell uses must be found.
    used_by_cells = set()
    for cell in model.layers:
        for value in vars(cell).values():
            if isinstance(value, keras.Variable):
                used_by_cells.add(id(value))
            elif isinstance(value, (list, tuple)):
                used_by_cells.update(id(v) for v in value if isinstance(v, keras.Variable))

    variables: List[keras.Variable] = list(model.trainable_variables)
    used: Dict[str, str] = {}
    skipped: List[str] = []
    next_index = 0
    for v in variables:
        key = _TV_KEY.format(next_index)
        if key in shapes and tuple(shapes[key]) == tuple(v.shape) and id(v) not in attr_paths:
            next_index += 1
        elif id(v) in attr_paths and attr_paths[id(v)] in shapes:
            key = attr_paths[id(v)]
        elif id(v) not in used_by_cells:
            skipped.append(v.name)
            continue
        else:
            raise RuntimeError("{}: no entry for trainable variable {!r} (shape {}); next stored index {} has "
                               "shape {}; checkpoint keys under model/: {}".format(
                                   prefix, v.name, tuple(v.shape), next_index,
                                   shapes.get(_TV_KEY.format(next_index)),
                                   sorted(k for k in shapes if k.startswith("model/"))[:8]))
        value = reader.get_tensor(key)
        if tuple(np.shape(value)) != tuple(v.shape):
            raise RuntimeError("{}: {} has shape {} but variable {!r} has shape {}".format(
                prefix, key, np.shape(value), v.name, tuple(v.shape)))
        v.assign(value)
        used[v.name] = key

    # Same count discipline as nets_dm._assign_saved_weights: the file must
    # hold exactly one trained tensor per (used) model variable, no more.
    n_indexed = sum(1 for k in shapes if k.startswith("model/optimizer/_trainable_variables/"))
    n_attr = sum(1 for k in used.values() if not k.startswith("model/optimizer/"))
    if n_indexed + n_attr != len(used) or len(used) + len(skipped) != len(variables):
        raise RuntimeError("{}: holds {} optimizer-tracked + {} attribute tensors, assigned {} of {} model "
                           "variables ({} unused by any cell: {})".format(
                               prefix, n_indexed, n_attr, len(used), len(variables), len(skipped), skipped))
    return used


