"""Verbatim copies of ``Model_Free_L2O/L2O-Scale/optimizer/{hierarchical_rnn,rnn_cells,utils,
trainable_optimizer}.py`` (byte-identical to L2O-Entropy's), with only the
``from optimizer import ...`` lines rewritten to this package. Verbatim so the
weight creation order -- and therefore the index-keyed dill checkpoint contract
of ``trainable_optimizer.save/load`` -- is the library's by construction.
"""
