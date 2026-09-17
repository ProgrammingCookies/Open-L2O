# Benchmarking Platform — Supervisor Meeting Prep (2026-09-06)

## How it works
- **Thin adapter layer, not a rewrite.** Each library's existing `train.py`/`evaluate.py` stays untouched — an adapter just wraps it.
- **Subprocess-based, not imported.** The libraries ship colliding module names (`meta.py`, `util.py`...), so they can't share one Python process. Each adapter shells out to the library's script in its own directory and normalizes what comes back (checkpoint path, loss trajectory, final/min loss, provenance).
- **Registry pattern.** Adding a new method = one adapter class + one `register_method(...)` line. Nothing else in the harness changes.

## Other platform-level features
- **Seed control** — a single top-level seed threads into every adapter's eval (and train, for the learned methods), for reproducible problem instances across methods.
- **Shared LASSO metrics module** — a verified FISTA reference solver + recovery metrics used to score every method on the same footing.
- **Config-driven, no silent defaults** — required fields must be explicit per adapter; missing config fails loudly rather than guessing.

## Status
Feature-complete and verified end-to-end across all 8 registered methods; it's what the experiment sweeps run on top of.

## Files to show for a quick overview
Four files carry the architecture — everything else repeats the same pattern:

1. **`Benchmarking/core/method.py`** (~50 lines) — the whole contract in one file: `L2OMethod` ABC with `train()`/`evaluate()`. Shows the two-flavor design (`trainable=True/False`) in the docstring alone. Best single file if only one gets looked at.
2. **`Benchmarking/core/shell.py`** — the subprocess mechanism: `run_script()` shells out to a library's own script with `sys.executable`, plus `check_known_flags()`, which statically scans the target script for its real argparse/absl flags and fails fast on a typo'd/unsupported flag before ever launching the subprocess.
3. **`Benchmarking/adapters/__init__.py`** (~35 lines) — the registry in one glance: every method is a single `register_method(...)` line, makes "add a model = one adapter + one line" concrete rather than a claim.
4. **`Benchmarking/adapters/dm_rnnprop.py`** or **`adapters/scale_family.py`** — one real adapter: `method.py`'s abstract contract turned into an actual `train()`/`evaluate()` that calls `shell.run_script(...)` and parses the result back into `TrainOutput`/`EvalOutput`.

If there's time for exactly one more: **`Benchmarking/run.py`** — the CLI entry point, ties it together end-to-end (`--list-methods`, `--config`, `--suite`).

---

# Experiments — Progress Notes

## Experiment 1 — model-free scalability/truncation sweep (Table 1/2)
**Status: ACTIVE.** Real 10-seed sweep running unattended in Docker (RTX 2070), launched 2026-09-02, up continuously since then with zero errors. Checkpoint 2026-09-06: 452/1080 rows (~42%), currently on seed 4/10. Rough pace suggests a few more days to finish (not a firm ETA — later seeds' larger-dim runs cost more).
Before launch, found+fixed two real GPU-memory scaling bugs: the largest problem dimension (5000,10000) was infeasible on this GPU and dropped entirely; the next two largest also OOM'd at the longest unroll length and needed a per-dimension batch-size reduction.

**Preliminary findings, seeds 0-3 (complete), placeholder training budgets — trends only, not final numbers.** Figures in `Figs/`: `experiment1_gradnorm_scaling`, `experiment1_quality_scaling_{l2o-dm,l2o-rnnprop,l2o-scale}`, `experiment1_cost_scaling`.
- **Meta-gradient norm vs. unroll length T diverges sharply by method**: L2O-DM's grad norm explodes 5 orders of magnitude (T=2→100), L2O-RNNProp grows mildly (~460x), L2O-Scale stays tiny throughout — a clean, controlled confirmation of the BPTT-instability story from earlier profiling work.
- **L2O-Scale's gradient clipping never engages** in this config — post-clip norm exactly equals raw norm at every T.
- **Seed-to-seed stability tracks the gradient story**: L2O-DM is wildly unstable run-to-run (std often exceeds the mean), RNNProp is comparatively steady, L2O-Scale gets progressively less stable as T grows even though its gradients stay small.
- **Cost**: L2O-Scale is ~4x slower per step than L2O-DM, ~3.5x slower than RNNProp, but uses *less* peak memory than either (~40-45% of their footprint) at every dimension tested.

## Experiment 2 — main 4-method LASSO sweep (Table 3/4)
**Status: COMPLETE.** 120/120 (method, width, seed) runs finished ~2026-08-21, zero errors. Eval fairness independently verified — all 4 methods score on identical test batches and identical starting points, immune to any train/eval RNG drift.
**Headline finding:** the model-free methods (L2O-DM, L2O-RNNProp) fail to converge on this LASSO task — end up worse than just guessing zero, on average. LISTA/ALISTA succeed (~-32dB recovery). This split became its own investigation (next section).

## ✅ Model-free non-convergence was A CHECKPOINT BUG — CONFIRMED 2026-09-07
**Lead with this. The headline result of the whole model-free investigation has flipped.**

Reran the exact config that "failed" (RNNProp, (25,50), narrow, p=0.10), fix in place, nothing else changed:

| metric | before (broken load) | after (fixed) |
|---|---|---|
| NMSE vs x_true | **+2.58 dB** (worse than guessing zero) | **−36.33 dB** |
| modified relative loss | ~780,000 % above optimal | **0.18 %** |
| final vs min loss | ~50× apart (diverged) | equal (converged) |

FISTA's own reference error on that data is **−36.81 dB** — so RNNProp lands **within 0.5 dB of the LASSO optimum**. Model-free L2O doesn't fail on this problem; the measurement did. A broader rerun (3 seeds × 3 widths, DM + RNNProp) is running now.


Experiment 1 shows RNNProp reaching **−32.4 dB NMSE** at (25,50) — genuinely excellent recovery — while Experiment 2 shows it failing (**+2.6 to +20 dB**, worse than guessing zero) on the *same* dimensions, noise, sparsity and λ. Chasing that contradiction found the cause:

`networks.save`/`load` build their dict as `{v.name: ...}`. Under **Keras 3, `v.name` is the bare name** (`"kernel"`, `"bias"`, `"recurrent_kernel"`), not TF1's unique scoped path — so a 10-tensor network has only 3 distinct names and **the dict silently keeps one tensor per name**. Verified by round-trip test: **3 of 10 tensors restored, 7 lost** (input projection + both LSTM cells' input kernels/biases), no error or warning.

**So every evaluation that loads a checkpoint ran a ~70% randomly-initialised optimizer.** The two experiments differ precisely here:
- **Exp 1** reads `recovery.npz` written by the *training* script's in-memory eval → never round-trips → shows true trained quality.
- **Exp 2** runs `evaluate_rnnprop.py`, which *loads a checkpoint* → mostly-random net → catastrophic divergence.

This affects exp2_real, exp2_try2, primer_replication and scale_diag — i.e. the entire model-free side of Experiment 2.

**L2O-Scale is worse still:** measured like-for-like, its checkpoints held **12 of 31 tensors — 61% lost** (its `Matrix`/`Bias` names repeat once per RNN cell). So the earlier L2O-Scale-on-LASSO side investigation ("best checkpoint NMSE = −0.01 dB") also measured a mostly-random optimizer and is invalid.

**Fixed (2026-09-06, uncommitted) in all five model-free libraries** — DM/RNNProp, Swarm, Scale, Entropy, Jacobian. Keys are now `"<index>:<name>"`; `save()` refuses to write an incomplete checkpoint; `load()` validates count + shapes, still accepts old checkpoints that are provably complete, and rejects lossy ones with an explanatory error instead of silently half-restoring. Verified: RNNProp round-trips 10/10, Scale saves 31/31, and a full `experiment1.py --smoke` across all three libraries passes with zero errors.

**Confirming retrain is RUNNING** (container `rnnprop_ckptfix_check`, `runs/scale_diag_ckptfix`): the exact (25,50) config that previously failed, RNNProp then DM, ~2h each. Old broken-load result was **+2.6 dB**; if the bug is the explanation this should come back near **−30 dB**, matching Exp 1's in-memory −32.4 dB.

**What survives:** the training-side instability findings (gradient-norm explosion with T, seed variance) are measured in-memory during training and are unaffected. All of Experiment 1's metrics are in-memory-based and stand.

## Model-free non-convergence investigation (the earlier, now-suspect story)
Root cause as understood before today: eval loss is lowest at step 0 and increases monotonically for the rest of the run — the learned optimizer never makes a net-beneficial step. Corroborated by the primer paper, which reports model-free failure at its own tested scale.
Tested 5 candidate fixes (curriculum learning, L2O-Scale-style gradient clipping, matching the paper's exact scale/noise setting, a confirmed-and-fixed RNNProp loss-weighting bug, and a full paper-protocol replication). Each gave a measurable improvement, none fixed it qualitatively.
**Caveat now:** every one of those evaluations went through the broken checkpoint load, so "none of the fixes worked" may simply mean "all of them were measured on a mostly-random network." The RNNProp loss-weighting bug fix is still a genuine correctness fix regardless.

## exp2_try2 — model-based re-run at the primer paper's own scale
LISTA/ALISTA complete (10 seeds), results tables + OOD-degradation figures done. Finding: both methods perform slightly worse at their own training-sparsity center as the training-width increases, and their out-of-distribution behavior differs structurally — ALISTA's degradation curve is essentially width-independent, LISTA's isn't. Prompted 3 follow-ups to explain the gap.

## Follow-ups on the width-dependent gap (model-based only)
1. **Training-curve instrumentation** — done: training is smooth and stable at every width, for both methods. Rules out training instability as the explanation.
2. **Train-to-convergence (higher patience)** — implemented and ready, but the real pilot was skipped by decision: (1) already showed no sign that more patience would help.
3. **More training data (2x)** — tested across 3 seeds now (21, 22, 23), confirmed 2026-09-06: the gap did not close — if anything it's slightly wider (LISTA 0.6→0.86dB, ALISTA 2.0→2.25dB) — firm evidence against a data-scarcity explanation.

Current read: the gap looks like a genuine structural effect of training-distribution width, not an artifact of unstable training or too little data — a candidate thesis conclusion for this part.

## L2O-Scale as a candidate fix (side investigation)
Tested as a structural fix for the model-free non-convergence problem, since it has built-in gradient clipping. Its raw meta-gradient norm is genuinely ~1000x tamer than L2O-DM's, but it still diverges late in training, clipped or not. Manually stopped mid-run at the user's call; inconclusive, a real follow-up candidate, not yet integrated into the harness's Experiment 2 protocol.
