# Results validity — which runs are safe to use

**Last updated: 2026-09-17.** Read this before using any number out of `runs/`.

Two separate issues are recorded here: the **model-free checkpoint save/load bug** (2026-09-06, below) and the **model-based per-layer trajectory accumulation** (2026-09-17, last section).

A checkpoint save/load bug (found 2026-09-06, fixed the same day) silently
corrupted **every model-free result that was measured by loading a checkpoint**.
This file records exactly which runs that does and does not affect, so an
invalidated number doesn't end up in the thesis by accident.

## The bug in one paragraph

All five model-free libraries saved network weights as
`{v.name: v.numpy() for v in trainable_variables}`. Under Keras 3 `v.name` is the
**bare** name (`"kernel"`, `"bias"`, `"Matrix"`, …), not TF1's unique scoped path,
so names are not unique within a network and the dict **silently kept only the
last tensor per name** — on save *and* on load, with no error. Restored
optimizers therefore ran with most of their weights still randomly initialised:

| library | tensors saved / total | lost |
|---|---|---|
| L2O-DM, L2O-RNNProp | 3 / 10 | **70 %** |
| L2O-Scale (HierarchicalRNN, GRUCell, 20, 2) | 12 / 31 | **61 %** |
| L2O-Swarm, L2O-Entropy, L2O-Jacobian | same code pattern, not separately measured | — |

Fixed in all five libraries: keys are now `"<index>:<name>"`, `save()` refuses to
write an incomplete checkpoint, and `load()` validates count + shapes and rejects
a provably-lossy legacy file instead of half-restoring it.

## The decisive rule

> **Did the number come from `evaluate_*.py` / an adapter's `evaluate()`?**
> → it loaded a checkpoint → **INVALID** (model-free only).
>
> **Did the number come from the training script's own in-memory evaluation**
> (`recovery.npz`, `profile_summary.json`, `train.log`)?
> → **VALID** — it never round-trips through save/load.

## Per-run status

### INVALIDATED — model-free rows only
| run | what is invalid |
|---|---|
| `exp2_real` | `l2o-dm`, `l2o-rnnprop` rows. **`lista`/`alista` rows are fine** (see below). |
| `exp2_try2` | `l2o-rnnprop` rows. `lista`/`alista` fine. |
| `exp2_try2_lastloss` | all (RNNProp only run) |
| `primer_replication` | all (RNNProp only run) |
| `scale_diag`, `scale_diag_cl` | all model-free eval metrics |
| `scale_diag_profile` | its **eval** metrics; its grad-norm **profiling** is training-time → valid |
| `l2o_scale_lasso_diag`, `l2o_scale_lasso_diag_clip5` | the NMSE numbers (computed from `model-best.l2o` / `model-final.l2o`). The grad-norm/divergence observations logged *during training* are valid. |

Consequence for the write-up: the "model-free L2O fails to converge on LASSO"
conclusion, and the finding that none of the five candidate fixes helped, both
rest on invalidated measurements. They must be re-established after a retrain
before being used. (The RNNProp loss-weighting bug fix is a genuine correctness
fix regardless of this.)

### VALID — not affected
| run | why |
|---|---|
| `exp1_calibration` | `experiment1.py` only ever calls `method.train(...)`, never `evaluate()`. Its `modified_relative_loss` comes from `recovery.npz`, written from the **live in-memory** optimizer — `train_rnnprop.py`/`train_dm.py` eval block (`ev_x[0].numpy()`) and `L2O-Scale/metaopt.py:311-316` (`opt._params[0].numpy()`, written *before* `opt.save`). All profile metrics (grad norm, meta-loss, peak memory, timings) are logged live during training. Verified 2026-09-06 — no `.load()`/`.restore()` exists anywhere in the Scale training path. |
| **All `lista`/`alista` results, everywhere** | `Model_Base_L2O` uses `tf.train.Checkpoint`, a completely separate and correct mechanism, and never imports `networks.py`. Verified 2026-09-06. |
| `exp2_try2_curve`, `exp2_try2_moredata` | model-based only → unaffected. The width-gap finding and all three follow-ups stand. |

### Separate, still-open caveat on `exp1_calibration` (not the checkpoint bug)
Its data directory contains **only `train_data.npy`** — no held-out split — and
`--lasso_split` defaults to it. So Experiment 1's `modified_relative_loss` is an
**in-sample** measurement, on the same pool the optimizer trained on. It is a
valid statement that the optimizer learned to solve those problems; it is **not**
a held-out generalisation number the way Experiment 2's is. Do not compare it
directly against Experiment 2's held-out figures without saying so.

Also note: checkpoints written by `exp1_calibration` **before** the 2026-09-06 fix
are lossy and cannot be re-evaluated later; rows finished after the fix write
complete ones. The results table is unaffected either way — nothing reads them
back — but that directory now holds a mix of both formats.

## CONFIRMED 2026-09-07 — the bug was the cause

The retest reran the exact config that previously failed (RNNProp, (25,50),
narrow, p=0.10, seed 0) with the fix in place and nothing else changed:

| metric | old (broken load) | new (fixed) |
|---|---|---|
| NMSE vs x_true | **+2.58 dB** | **−36.33 dB** |
| modified relative loss | ~780,000 % above optimal | **0.18 %** |
| final vs min loss | ~50× apart (diverged) | equal (converged) |

FISTA's own reference recovery error on that data is **−36.81 dB**, so RNNProp
came within **0.5 dB of the LASSO optimum**. The "model-free L2O fails to
converge on LASSO" conclusion is **overturned** — it was an artifact of the
checkpoint bug throughout.

## Re-establishing the invalidated results

Confirming retrain launched 2026-09-06, container `rnnprop_ckptfix_check` →
`runs/scale_diag_ckptfix` (the exact (25,50) config that previously failed;
RNNProp then DM). Prediction: the old broken-load result was **+2.6 dB**; if the
checkpoint bug is the explanation this should return roughly **−30 dB**, matching
Experiment 1's in-memory −32.4 dB. If it still fails, the bug was real but was
*not* the cause, and the original non-convergence conclusion survives.

## 2026-09-17: eval scripts also save `x0` (no metric affected)

From 2026-09-17 ~16:40 local, `evaluate_dm.py`, `evaluate_rnnprop.py` and
`L2O-Scale/evaluate.py` write an extra `x0` key into their recovery `.npz`
(the starting point actually used), and the LASSO-capable adapters report an
`instances_fingerprint` in `EvalOutput.metrics` (`core/fingerprint.py`).
Because the repo is bind-mounted into the containers, any sweep resumed after
this point (try3 seeds 26-30, `exp1_calibration_2`) runs the edited scripts.
The edits add one attribute and one output key and consume no random draws;
`tests/test_determinism.py` confirms same-seed runs stay bit-exact. Rows
written after this carry the fingerprint natively (`b+x0+x_true`); rows
before it were backfilled into `results_with_fingerprints.jsonl` over
`b+x_true` by `backfill_fingerprints_exp2.py` -- re-run the backfill (or
delete that copy) after a resume so the post-hoc test does not read a stale
file. Post-hoc check passed for exp2_try3_210x300 (495/495), exp2_try2
(748/748) and exp2_real (480/480) cells.

## 2026-09-17: model-based per-layer trajectories were running means across test files

**Affects:** every `lista` / `alista` row's `final_loss`, `min_loss`, `mean_loss`
(and the raw `lasso_layer{i}` values in each cell's `all_test_metrics.json`) for
every test sparsity **except the first one evaluated** (`test_sparsity_0.02`), in
**every** sweep -- `exp2_real`, `exp2_try2`, `exp2_try3_210x300`,
`exp2_try2_moredata`, `exp2_try2_curve`.

**Does NOT affect:** the four Table-4 recovery metrics (`suboptimality_gap_*`,
`modified_relative_loss`, `nmse_vs_true_*`, `lasso_optimal_recovery_error_*`) --
they are computed by the adapter from `<file>_final_output.npy`, which comes from
`predict()` and is correct -- nor any model-free row, nor the compute-matched
comparison (`plot_compute_matched_rf.py` reads `modified_relative_loss`).

**The defect.** `Model_Base_L2O/train.py`'s `--test` path built its 16
`lasso_layer{i}` metrics (Keras `Mean` objects, `utils.LassoObjective`) once and
reused the same list in the `compile()`/`evaluate()` of every test file. Under
Keras 3 their running `total`/`count` were not reset between files, so the value
stored for the k-th file is the mean over files 1..k. Since the files are
evaluated in increasing sparsity order, every later (harder) file's trajectory
was dragged **down** towards the easier ones -- e.g. the 0.40 file's final-layer
value was reported at roughly half its true value. The code structure is the
toolbox's original 2021 code (`git blame` lines 240-267); the Keras 3 environment
exposes it. Present from the very first sweep (`exp2_real`, seed 0 narrow LISTA:
stored `3.0048, 5.2929, 8.5769, ...` = running means of the true
`3.0048, 7.581, 15.1447, ...`).

**Fix.** `train.py` now builds fresh metric objects per test file
(`make_test_metrics()`); verified by re-running the script on the try3 seed-21
narrow LISTA checkpoint: `2.9461, 7.5514, 14.9623` for the first three files
(previously `2.9461, 5.2488, 8.4866`). Model-based cells evaluated after this
fix (e.g. try3 seeds 26-30 once resumed) store per-file values directly.

**Regeneration, no retraining.** `regenerate_modelbased_trajectories.py --run
runs/<run>` loads every model-based checkpoint with `unified-lista` /
`unified-alista` (`Benchmarking/unified`, the in-process re-implementation) and
re-evaluates layer by layer on CPU. Each cell is cross-checked against the
trustworthy parts of the stored files -- `x_pred` per row against
`<file>_final_output.npy` and the stored values de-accumulating exactly onto the
regenerated ones (`f_k = k*s_k - (k-1)*s_{k-1}`) -- so a wrong checkpoint or
config cannot pass. It writes, never touching the originals:

| file | content |
|---|---|
| `<model_dir>/all_test_metrics_perfile.json` (+ `.provenance.json`) | corrected per-file `lasso_layer{i}` |
| `<run>/results_modelbased_trajectories_fixed.jsonl` | `results.jsonl` with the model-based `final_loss`/`min_loss`/`mean_loss`/`num_points` replaced (+ `trajectory_source: "regenerated_perfile"`) |
| `<run>/summary_modelbased_trajectories_fixed.json` | the sweep's own `summarize()` over those rows; **`summary.json` still aggregates the wrong values** |

**Use `results_modelbased_trajectories_fixed.jsonl` / `summary_modelbased_trajectories_fixed.json`**
(or the `_perfile.json`) for any model-based trajectory number. `results.jsonl`,
`summary.json`, `results_with_fingerprints.jsonl` and `all_test_metrics.json`
keep the original (wrong) model-based `final_loss`/`min_loss`/`mean_loss` for
provenance. Sanity check on `exp2_real` (LISTA, narrow): `final_loss` seed-mean
at p=0.02 unchanged (0.0247), p=0.10 0.0698 -> 0.1232, p=0.40 0.2552 -> 0.5175;
`modified_relative_loss` identical at every cell. Runs regenerated 2026-09-17:
`exp2_try2_curve`, `exp2_try3_210x300`, `exp2_try2`, `exp2_real`,
`exp2_try2_moredata` (see each run's `RESULTS_STATUS.md`). Cells evaluated after
the fix hold per-file values already; the script recognises both forms
(`stored_form` in the provenance json), so **re-run it after a sweep resumes**
(e.g. try3 seeds 26-30) -- it skips cells it has already done.

## 2026-09-18: L2O-Scale evaluation is not reproducible run-to-run (allocator-dependent)

**Affects:** every `l2o-scale` *evaluation* (`evaluate.py --include_lasso_problems`), i.e.
the `exp2_try2_scale_*` runs and the `exp2_try2_scale` sweep in progress. Training is not
affected (it uses the functional update path with explicit state).

**Mechanism** (found while porting Scale into `Benchmarking/unified`): for each chunk,
`evaluate.py:223-246` builds the optimizer, runs a dummy `apply_gradients` on a throw-away
`tf.Variable` (to create the lazily built weights) and loads the checkpoint; the chunk is then
run on a *new* `tf.Variable`. `TrainableOptimizer` keys its per-variable state by
`id(variable)` (`optimizer/trainable_optimizer.py:185-193`). When CPython reuses the freed
dummy's address for the new variable -- observed in **2 of 10 chunks** of one 1280-row split,
different chunks in different processes -- the real run silently inherits the dummy's
post-step state (log learning rates, gradient accumulators, hidden state) instead of a
fresh one. Two runs of the script with the same seed then differ: measured 6.5e-2 on
trajectory values of ~2.4 (~3 %) from step 1 on `exp2_try2_scale_narrow`'s checkpoint.

**Consequence:** Scale evaluation numbers carry an unseeded random component of that
order; same-seed reruns are not guaranteed to reproduce them. The four Table-4 metrics
are computed from `x_pred`, which is affected the same way. The intended behaviour
(a fresh state for the real variable) is what happens in the majority of chunks and
what the unified implementation (`unified-scale`, deterministic) does.

**Fix (applied 2026-09-18, `L2O-Scale/evaluate.py` `make_test_optimizer`):** the dummy
variable's slot is dropped after `opt.load(checkpoint_path)`, so the real variable always
starts fresh. Verified: three same-seed runs on `exp2_try2_scale_narrow`'s checkpoint are
now identical and equal to `unified-scale` bit for bit. Scale cells evaluated **before** the
fix (`exp2_try2_scale*`) should be re-evaluated (evaluation only -- the checkpoints are
valid) before their numbers are used; `unified-scale` gives the same numbers as the fixed
script and needs no subprocess. `tests/test_unified_parity.py` asserts Scale parity per chunk
(each chunk's script output must be bit-identical to the fresh or the inherited-slot variant
of the unified policy), which is also how the mechanism was pinned down.
