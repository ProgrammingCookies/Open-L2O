"""
Adapter registration.

Importing this package registers every adapter, so a
single import adapters makes core.registry.list_methods() complete.

REMEMBER!:
To add a new L2O model, write an adapter and add one register_method(...)
line.
"""

from core.registry import register_method

from adapters.baselines import ClassicalOptimizer
from adapters.dm_rnnprop import DMRNNPropMethod
from adapters.model_based import ModelBasedMethod
from adapters.scale_family import ScaleFamilyMethod
from adapters.swarm import SwarmMethod

# Learned Optimizers
register_method(ScaleFamilyMethod("l2o-scale", "L2O-Scale"))
register_method(ScaleFamilyMethod("l2o-entropy", "L2O-Entropy"))
register_method(ScaleFamilyMethod("l2o-jacobian", "L2O-Jacobian"))
register_method(DMRNNPropMethod("l2o-dm", variant="dm"))
register_method(DMRNNPropMethod("l2o-rnnprop", variant="rnnprop"))
register_method(SwarmMethod())
register_method(ModelBasedMethod("lista", model_name="lista"))
register_method(ModelBasedMethod("alista", model_name="alista"))

# Classical optimizers
register_method(ClassicalOptimizer("adam", classical="Adam"))
register_method(ClassicalOptimizer("sgd", classical="SGD"))
register_method(ClassicalOptimizer("adagrad", classical="Adagrad"))
