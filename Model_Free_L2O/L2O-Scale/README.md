## L2O-Scale

### Overview

Learned optimizers from "Learned Optimizers that Scale and Generalize" (Wichrowska et al., 2017): a hierarchical RNN (`HierarchicalRNN`) that maintains per-parameter, per-tensor, and global recurrent state to propose optimizee parameter updates, plus a simpler `CoordinatewiseRNN` and a handful of minimal reference optimizers (`GlobalLearningRate`, `LearningRateSchedule`, `TrainableAdam`).

This directory unifies what used to be two separate `L2O-Scale-Training/` and `L2O-Scale-Evaluation/` trees (now migrated to TF2/Keras 3 — see `migration_changes.txt`).

### Code Overview

- `optimizer/` — `trainable_optimizer.py` (base class), `hierarchical_rnn.py`, `coordinatewise_rnn.py`, `global_learning_rate.py`, `learning_rate_schedule.py`, `trainable_adam.py`, `rnn_cells.py` (custom `BiasGRUCell`), `utils.py`.
- `problems/` — `problem_generator.py` (the problem catalog: Quadratic, classic 2D test functions, MNIST/CIFAR-10 classifiers, and various composite/wrapper problems), `problem_sets.py` (named collections used by `train.py`/`evaluate.py`), `datasets.py`, `problem_spec.py`.
- `metaopt.py` — `train_optimizer()` (the meta-training loop) and `test_optimizer()`/`run_wall_clock_test()` (apply an optimizer, trained or baseline, as a standalone drop-in optimizer against a fresh problem).
- `train.py` — meta-training entry point (was `metarun.py`).
- `evaluate.py` — evaluation entry point (was `metatest.py`).

### Environment

- TensorFlow >= 2.10, Keras 3 (`pip install -r requirements.txt`)

### Train

```shell
python train.py --train_dir=runs --optimizer=HierarchicalRNN \
    --include_quadratic_problems --include_bowl_problems \
    --num_problems=1 --num_meta_iterations=100 \
    --fix_unroll --fix_unroll_length=20 --fix_num_steps=100 \
    --evaluation_period=1 --evaluation_epochs=5
```

Checkpoints are written to `<train_dir>/<optimizer>_<cell_cls>_<cell_size>_<num_cells>/` as `model-best.l2o` (best-so-far by evaluation cost), `model-iter<k>.l2o` (periodic), and `model-final.l2o`.

### Evaluate

```shell
python evaluate.py --train_dir=runs --optimizer=HierarchicalRNN \
    --save_dir=runs_eval --include_mnist_mlp_problems \
    --restore_model_name=model-final.l2o --num_testing_itrs=10000
```

`--test_optimizer` also accepts `SGD`/`Adam`/`Adagrad` for a plain baseline (no checkpoint needed). Results are written to `<save_dir>/seed<N>_eval_loss_record.pickle-<model_name>` for 5 fixed seeds (6, 12, 18, 24, 30).

### Using a Learned Optimizer as a Drop-in Optimizer

Any `TrainableOptimizer` subclass exposes a plain `apply_gradients(grads_and_vars)` method (the same calling convention as `tf.keras.optimizers.Optimizer.apply_gradients`), so a trained instance can drive optimization of any model built with real `tf.Variable`s:

```python
opt = hierarchical_rnn.HierarchicalRNN([10, 20, 20])
# force weights to build (see migration_changes.txt), then:
opt.load("runs/HierarchicalRNN_GRUCell_20_2/model-final.l2o")
opt.apply_gradients(zip(gradients, variables))
```

### Citation

```
@inproceedings{wichrowska2017learned,
  title={Learned optimizers that scale and generalize},
  author={Wichrowska, Olga and Maheswaranathan, Niru and Hoffman, Matthew W and Colmenarejo, Sergio Gomez and Denil, Misha and Freitas, Nando and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning},
  pages={3751--3760},
  year={2017},
  organization={PMLR}
}
```
