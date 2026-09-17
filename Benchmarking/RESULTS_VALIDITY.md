# Results validity — which runs are safe to use

**Last updated: 2026-09-06.** Read this before using any number out of `runs/`.

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
