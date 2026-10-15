"""Problem and metric registration.

Importing this package registers every built-in problem type
(core/problems.py) and metric (core/metrics.py); ``adapters`` imports it, so
anything that can run a method can also score it.

REMEMBER!:
To add a new problem type or new metrics, write a module here that calls
register_problem(...) / @register_metric(...) and add one import line below.
More metrics for an existing problem can go in that problem's module.

For a new problem to be scored, the library's evaluation script must also
save its final iterate and the instance arrays (every name in the spec's
``required_arrays``/``fingerprint_arrays``, instances along axis 0) as an
.npz the adapter picks up -- no adapter code changes:

  L2O-DM / L2O-RNNProp  <output_path>/<optimizer>_recovery-<problem>.npz
  L2O-Scale             <save_dir>/L2o_recovery-<problem>.npz

(the model-based adapter only runs task="lasso").
"""

from scoring import lasso  # noqa: F401
