# Experiments analysis log

Running record of findings, interpretations and open questions about the thesis
experiments (pre-study Experiments 1-5 and any follow-ups). This file is the
cross-session context: each chat session that analyses or plans an experiment
should read this first and append to it, so conclusions reached in one chat are
available in the next.

Conventions:
- One `## Experiment N` section per experiment; dated `###` entries underneath,
  newest at the bottom of each section.
- Record the *interpretation* and the evidence for it (which run dir, which
  rows, which script/code line), not just raw numbers -- raw numbers live in
  `Benchmarking/runs/<run>/results.jsonl`.
- Validity caveats per run are in `Benchmarking/RESULTS_VALIDITY.md` and the
  `RESULTS_STATUS.md` marker inside each affected `runs/` dir; cross-reference
  rather than duplicate.
- Keep "Open questions / next steps" per experiment current -- delete items
  when resolved, and say what resolved them.

---

## Experiment 1 -- Model-free L2O scalability and truncation (T sweep)

Orchestrator: `Benchmarking/experiments/experiment1.py`. Run dir:
`Benchmarking/runs/exp1_calibration` (the real 10-seed sweep despite the name).
Design: 4 dims x 9 unroll lengths T x 3 methods (L2O-DM, L2O-RNNProp,
L2O-Scale) x 10 seeds = 1080 rows. (5000,10000) dropped -- OOMs all three
methods on the 8 GB RTX 2070 SUPER (module docstring point 6).

### Metrics (one row per (method, dim, T, seed) in `results.jsonl`)

All logged live during meta-training -- Experiment 1 never calls
`evaluate()`. Read via `_read_profile` (`experiment1.py:338-356`) from
`profiling.py`'s `profile_summary.json`, and `_read_modified_relative_loss`
(`:359-377`) from the training script's `recovery.npz`. `summary.json`
aggregates every metric column as mean/std across seeds per (method, dim, T).

| column | what it is | notes |
|---|---|---|
| `peak_allocated_mb` | TF BFC allocator peak (`get_memory_info()['peak']`) | excludes CUDA-context overhead (~7.8 GB on this card); see `gpu_mem_sampler.py` |
| `peak_reserved_mb` | **same number** as above | TF exposes one peak figure; both pre-study memory columns are filled with it (docstring point 1) |
| `meta_loss_final` / `meta_loss_mean` | L2O training objective (weighted optimizee loss over the unrolled window), last logged point / mean over all logged points | DM/RNNProp log once per epoch, Scale once per unroll segment -- `_mean` averages over different-sized populations per method |
| `grad_norm_final` / `grad_norm_mean` | global L2 norm of the meta-gradient w.r.t. optimizer params, pre-clip | all three methods |
| `grad_norm_post_clip_final` / `grad_norm_post_clip_mean` | same, after clipping | **L2O-Scale only** (`null` for DM/RNNProp, which don't clip). Scale uses elementwise `clip_by_value`, so this is the global norm of the elementwise-clipped gradient, not a norm-rescale |
| `modified_relative_loss` | Eq 10: `mean(f(x_pred) - f*) / mean(f*)` over the batch; `f*` from FISTA (`solve_xstar`, cached by hash of the `(A, b)` bytes) | **in-sample** -- `x_pred` is the live optimizer's final iterate on the training pool (no held-out split in the Exp-1 data dir). See `RESULTS_VALIDITY.md` |
| `avg_time_per_iteration_s` | wall time per *optimizee step* | the only timing figure comparable across methods (per-logged-point time is not, see above) |
| `total_time_s` | wall time of the meta-training loop as measured by `profiling.py` | |
| `train_sec` | orchestrator-side wall time of the whole `method.train()` subprocess | bookkeeping, not a metric (`_NON_METRIC_ROW_KEYS`, `:136`); includes process start, data load, checkpointing, eval block |

Matches pre-study §7.1's metric list one-to-one, with two deviations: a single
peak-memory figure for both memory columns, and in-sample modified relative
loss.

### 2026-09-13 -- Peak GPU memory vs. dimension is a batch-size artifact (481/1080 rows)

**Observation that prompted this:** for all three methods, `peak_allocated_mb`
roughly doubles from (25,50) to (50,100), rises only ~25% to (250,500), then is
flat to (500,1000). Looked like memory "saturating" with problem size.

**Explanation:** batch size is not constant across the dim axis. The sweep uses
`DEFAULT_BATCH_SIZE_BY_DIM` (`experiment1.py:151`, rationale in module
docstring point 7, lines ~77-95): batch 128 at the two small dims, **32** at
(250,500), **16** at (500,1000). This was set before launch because the two
large dims OOM at T=100 with batch 128; the empirical ceiling is
`batch_size * n <= ~16000`. Confirmed in the run's recorded `config.json`.

The quantity that drives memory is `batch_size * n`:

| dim | batch | batch*n | predicted ratio | measured (DM, T=100) |
|---|---|---|---|---|
| 25x50 | 128 | 6 400 | 1.00 | 1.00 |
| 50x100 | 128 | 12 800 | 2.00 | 2.00 |
| 250x500 | 32 | 16 000 | 2.50 | 2.50 |
| 500x1000 | 16 | 16 000 | 2.50 | 2.50 |

Identical ratios for RNNProp and L2O-Scale (2.00 / 2.48 / 2.48). The last two
dims sit on the same 16000 ceiling by construction, hence the flat line.

**Memory model (fits all rows so far):** `peak ~= c_method * batch * n * T +
small constant`.
- Per-coordinate-per-unrolled-step cost is constant across dims to ~0.2%:
  DM ~3.33 KB, RNNProp ~3.44 KB, L2O-Scale ~1.43 KB. This is BPTT activation
  storage of a coordinatewise RNN -- each optimizee coordinate has its own RNN
  state, and all T unrolled steps are kept for the backward pass.
- Linear in T: DM at 25x50 fits `20.7 MB * T + 14.6 MB`, R^2 = 0.999997.
- Problem data (A matrix, dataset) is negligible: at T=2, (250,500) vs
  (500,1000) differ by ~1.5 MB total.
- Nothing odd about the allocator or the measurement (`tf_gpu_peak_mb`, TF
  BFC allocator peak -- see `gpu_mem_sampler.py`).

**Implications for the write-up:**
1. The raw memory column vs. dimension measures the batch-size schedule, not
   problem-size scaling. Report **per-instance memory** (`peak / batch_size`),
   which recovers the expected linear-in-n trend at T=100: DM 16.3 / 32.6 /
   163 / 326 MB; RNNProp 16.9 / 33.6 / 168 / 336 MB; Scale 7.0 / 14.0 / 70 /
   140 MB. At minimum the table must state the batch sizes per dim.
2. Cross-method comparison IS valid at every dim (batch identical across
   methods within a dim). Headline: L2O-Scale needs ~2.3x less memory per
   coordinate-step than DM/RNNProp, consistent with its smaller per-coordinate
   RNN state (HierarchicalRNN).
3. Smaller batch at the two large dims also means a noisier meta-gradient
   estimate there -- a confound on the dim axis for the loss / grad-norm
   columns too, not only memory. Docstring point 7 already states this.

### 2026-09-16 -- Time, meta-gradient norm and R_f vs T from the first 481 rows (seeds 0-3 complete, seed 4 partial)

Medians over seeds per (method, dim, T); recomputable from `results.jsonl`.
Seed count is 4-5 per cell, so treat the exponents as first-cut; re-check
at 540 rows (seed 4 complete, the agreed stop point) before quoting.

**Horizon caveat that governs the whole R_f column.** `num_unrolls = 10` is
fixed and the horizon is `T * num_unrolls` (docstring point 2,
`experiment1.py:274`), for training AND for the single end-of-training
evaluation that writes `recovery.npz`. So a T=2 optimizer is scored after 20
optimizee steps and a T=100 optimizer after 1000. `modified_relative_loss`
vs T is therefore "quality after 10T steps of an optimizer trained with
unroll T", not "optimizer quality at a fixed budget". It is the pre-study's
own design (fixed num_unrolls keeps the horizon a multiple of T for every T),
but the write-up must say it: a decreasing R_f-vs-T curve is partly just
more iterations. Together with the in-sample caveat (`RESULTS_VALIDITY.md`),
the Exp-1 R_f column answers "did meta-training at unroll T produce an
optimizer that solves its own training pool in 10T steps".

**1. Time per optimizee step is flat in dimension and fits `t = a + b/T`.**
Per-method fit pooled over dims (median rows):
DM a = 16.8 ms/step, b = 40 ms per unroll; RNNProp 20.8 / 47; L2O-Scale
84.6 / 37. The `b/T` term is the per-unroll meta-update (backward pass +
Adam step) amortised over T steps; it is why time/step *falls* from T=2 to
T~20 and then flattens. `total_time_s` is linear in T (8.5 / 10.5 / 42.7 s
per unit T for DM / RNNProp / Scale at T=100), as it must be with a 10T
horizon. Across dims the per-step time varies by < 10% (e.g. DM T=100:
16.9 / 17.3 / 18.0 / 16.9 ms) even though `batch * n` goes 6400 -> 12800 ->
16000 -> 16000: the coordinatewise RNNs are so small that the GPU is
launch/latency-bound, not throughput-bound, at every dim tested. The
per-*instance* cost therefore rises only through the batch-size schedule
(128 / 128 / 32 / 16), the same artefact as the memory column (2026-09-13
entry): per-instance time at T=100 is 0.13 / 0.14 / 0.56 / 1.06 ms for DM.
L2O-Scale's step is ~4-5x DM's (HierarchicalRNN: per-tensor and global RNNs
on top of the per-coordinate one) despite using ~2.3x less memory per
coordinate-step.

**2. Meta-gradient norm grows as a power of T, with a method-specific
exponent, and scales with dimension only for DM/RNNProp.** Log-log slopes
of `grad_norm_mean` vs T (T = 2..100, per dim):

| method | 25x50 | 50x100 | 250x500 | 500x1000 | norm at T=2 -> T=100 (25x50) |
|---|---|---|---|---|---|
| L2O-DM | 1.90 | 1.92 | 1.77 | 1.67 | 42 -> 6.8e4 |
| L2O-RNNProp | 1.51 | 1.49 | 1.49 | 1.50 | 0.026 -> 9.2 |
| L2O-Scale | 1.45 | 1.40 | 1.45 | 1.47 | 3.4e-4 -> 0.12 |

- RNNProp and Scale share T^1.5 at every dim; DM is steeper (T^1.7-1.9) and
  its norms are 3-5 orders of magnitude larger. `grad_norm_final` shows the
  same ordering.
- Dimension: at fixed T, RNNProp's norm scales as n^1.0 (slope 0.97-1.02 at
  T = 2, 16, 100) and so does its `meta_loss_final` (n^0.95-1.02) -- the
  meta-objective is the raw LASSO value `0.5||Ax-b||^2 + lam||x||_1`, whose
  magnitude grows with m (= n/2), so the gradient does too. Scale's norm is
  dimension-free (slope -0.07 to 0.01) because its meta-objective is the
  log-ratio `log f_t / f_0`, which is scale-invariant; its `meta_loss_final`
  is also dim-flat (-0.012 at T=2 to -1.2..-1.5 at T=100). DM sits in
  between (n^0.6 at T=2, ~n^0 at T=16, n^0.2 at T=100) with erratic
  meta-loss scaling (n^-0.16 to n^0.96) -- consistent with DM's training
  being unstable rather than with any clean law. Caveat: batch size also
  changes across dims (128 -> 16), which changes gradient *noise*; the n^1.0
  law for RNNProp is clean enough that this is not what drives it.
- Adam normalises gradient scale, so a large norm is not by itself
  instability; the T-exponent (how fast BPTT gradients grow with unroll
  length) is the stability-relevant number, and DM's is the worst.

**3. L2O-Scale's gradient clipping never engaged.** `grad_norm_post_clip_*`
equals `grad_norm_*` in all 160 Scale rows (to 1e-9 relative). The clip is
elementwise `clip_by_value(g, -5, 5)` (`L2O-Scale/metaopt.py:264`); the
largest *global* mean norm observed is 18.9 and typical values are 1e-4 to
0.1, so no element approaches 5. The pre-study's before/after-clip metric
is therefore vacuous on LASSO at this configuration -- report one column
and state this. (Contrast: the L2O-Scale-on-LASSO diagnostic in
`project_l2o_scale_lasso_followup` saw clipping engage under a different,
longer training budget; here `scale_num_meta_iterations = 50`.)

**4. R_f (in-sample, after 10T steps), pooled over dims, median [IQR]:**

| T | L2O-DM | L2O-RNNProp | L2O-Scale |
|---|---|---|---|
| 2 | 20.7 [5.3, 31] | 20.6 [20, 23] | 120 [120, 120] |
| 8 | 18.2 [7, 93] | 1.30 [1.2, 1.4] | 114 [100, 120] |
| 16 | 5.4 [2.2, 20] | 0.84 [0.77, 0.90] | 107 [83, 120] |
| 30 | 2.4 [0.8, 6] | 0.34 [0.26, 0.38] | 91 [50, 110] |
| 50 | 3.5 [0.9, 11] | 0.086 [0.05, 0.21] | 68 [22, 94] |
| 75 | 1.5 [0.8, 9] | 0.076 [0.06, 0.21] | 42 [9, 72] |
| 100 | 0.94 [0.8, 5.8] | 0.124 [0.09, 0.20] | 23 [10, 51] |

- RNNProp improves monotonically and tightly (IQR ~10%) to T=50 and then
  plateaus at R_f ~0.08-0.12 for T = 50-100 (25x50 per-seed at T=100:
  0.047-0.21) -- at 500-1000 steps it is 2-3 orders of magnitude below its
  20-step value but still ~1 order above Exp 2's 1000-step values
  (0.001-0.03, `exp2_try2`); Exp 1's training budget (50 epochs on 12,800
  instances at (25,50), `dm_rnnprop_num_epochs`) is smaller. No sign of
  T-driven divergence at T=100 with 4-5 seeds.
- DM is erratic at every T: per-seed R_f at (25,50) spans 0.6-22 at T=100
  and 1.7-399 at T=16; medians are not monotone in T. Same training-side
  failure as in Exp 2 (`project_exp2_try2_modelfree_partial`): it is DM's
  meta-training, not the horizon, that fails. Any DM-vs-T claim needs the
  IQR next to it.
- Scale at T=2 (20 steps) has R_f ~120 across all dims, i.e. it has not
  moved from the starting point (R_f(x0) ~127 at (25,50) in the
  compute-matched analysis); by T=100 (1000 steps) it reaches only 10-50.
  L2O-Scale does not learn to solve LASSO in this budget at any T -- the
  same conclusion as the manual follow-up in `project_l2o_scale_lasso_followup`,
  now with 4-5 seeds x 4 dims x 9 T.
- Dimension has little effect on RNNProp's R_f at any T (25x50 vs
  500x1000 at T=100: 0.126 vs 0.109), so on the quality axis the
  coordinatewise design scales as advertised; the cost of dimension is the
  memory / batch-size / gradient-magnitude side, not solution quality.

**What this gives RQ1 (unroll length vs cost/stability/quality):** cost is
linear in T (time) and in T (memory, 2026-09-13 entry) with method
constants; the BPTT gradient grows as T^1.5 (RNNProp, Scale) to T^1.9 (DM);
quality (RNNProp, the one method that works) improves to T ~ 50 and
saturates -- so on LASSO the T tradeoff has a knee near T = 30-50 for
RNNProp, beyond which longer unrolls buy no in-sample quality but cost
proportionally more time, memory and gradient magnitude. DM and Scale do
not reach a usable optimizer at any T with these budgets, so for them the
T axis is moot.

### Open questions / next steps (Experiment 1)
- Sweep resumed 2026-09-16 as container `exp1_calibration_2` (new name, same
  `runs/exp1_calibration`; the old container is kept, never removed), queued
  behind the Primer rerun by the watcher in `Benchmarking/runs/_watcher/`.
  Auto-stops at 540 rows (seeds 0-4). Re-check the 2026-09-16 tables then;
  decide seeds 5-9 vs try3 seeds 26-30 with the deadline in view.
- Table presentation: per-instance memory AND per-instance time (both are
  batch-schedule artefacts in raw form); one grad-norm column for Scale
  (clipping inactive); R_f with IQR, and the 10T-horizon + in-sample caveats
  under it.
- Figures to build (no GPU): time/step vs T (a + b/T fit), grad-norm vs T
  log-log with the fitted exponents, R_f vs T with IQR bands, all three
  methods per panel.
- Optional: an `--num_unrolls` variation at one dim would separate "longer
  unroll" from "more steps" in the R_f column (the horizon caveat above);
  not on the spine, only if a reviewer asks.

---

## Experiment 2 -- LASSO generalisation (ID/OOD sparsity width)

Pre-study §7.2 question: *How does the concentration of the training task
distribution trade ID quality against OOD robustness, and do model-free and
model-based L2O families differ in this trade?* Methods L2O-DM, L2O-RNNProp
(model-free), LISTA, ALISTA (model-based); LASSO, λ=0.005; Table 4 metrics
(suboptimality gap, modified relative loss R_f, NMSE vs x_true in dB,
LASSO-optimal recovery error = NMSE of x* itself, the recoverability floor).

### 2026-09-13 -- Consolidation: run inventory and the role of each run

Seven run directories touch Experiment 2. Each has exactly one job in the
write-up; the sprawl is only confusing if a run is quoted outside its role.

| run dir | dims / noise | widths (train p) | valid rows | role |
|---|---|---|---|---|
| `exp2_try3_210x300` | (210,300), noiseless | **centred**: narrow 0.175 fixed, medium U(0.1125,0.2375), wide U(0.05,0.3); 11 test p | LISTA/ALISTA/RNNProp, seeds 21-30, **3/90 triples done** (narrow s21 x 3 methods), ~10.6 GPU-h per seed measured => ~4 more days | **intended headline**: pre-study dims + the deconfounded width design |
| `exp2_try2` | (25,50), noiseless | same centred scheme | LISTA/ALISTA 10 seeds complete; RNNProp seeds 21-22 (6/30 triples); DM s21 narrow+medium only | fallback headline / cross-scale robustness check (the Primer's own validated model-free scale) |
| `exp2_real` | (210,300), 40 dB SNR | **confounded**: narrow 0.10 fixed, medium U(0.075,0.20), wide U(0.05,0.30); centres 0.100/0.1375/0.175; 8 test p | LISTA/ALISTA 10 seeds complete; model-free rows INVALID (checkpoint bug) | appendix: model-based at 40 dB + the confounded-design lesson. NOT a noise comparison against try3 (differs in noise AND centring). **Never the width claim** -- "wide is better at dense p" there is partly "wide was trained nearer dense p" |
| `exp2_try2_curve` | (25,50) | centred | LISTA/ALISTA seed 21, per-layer training curves | rules out training instability as the cause of the own-centre gap |
| `exp2_try2_moredata` | (25,50), train_size 64,000 | centred | LISTA/ALISTA seeds 21-23 | rules out sample density as the cause of the own-centre gap |
| `exp2_try2_lastloss`, `scale_diag*`, `primer_replication`, `l2o_scale_lasso_diag_clip5` | various | -- | all evaluated through the broken checkpoint load | INVALID as results; citable only as the reproducibility case study (how the bug was found) |
| `runs/scale_diag_ckptfix` | (25,50), 40 dB | narrow 0.10 | RNNProp seed 0 | the single before/after row proving the checkpoint fix (+2.58 -> -36.33 dB) |

Validity per run: `Benchmarking/RESULTS_VALIDITY.md`. Provenance traps that
still hold (from the 2026-09-06 cross-run analysis): both `summary.json`
files are stale and still contain invalidated model-free rows -- build every
table from `results.jsonl`; the `num_fista_iters: 2000` flag in `config.json`
is wrong (x* cache keys prove 50,000 iterations / gap_tol 1e-10 were used, and
the 2000-vs-50k f* bias is <= 2e-8 relative, irrelevant); the pre-study text
says "2000 iterations of FISTA" for f* -- the thesis should state 50,000 and
why.

### 2026-09-13 -- What is established for model-based (LISTA/ALISTA), 10 seeds

1. **The ID/OOD width tradeoff exists and its size depends on scale/noise**
   (deconfounded, paired by seed, n=10, p<0.05; from
   `project_exp2_modelbased_crossrun_analysis`):

   | method / run | ID cost of widening | OOD benefit at +0.20 offset | net |
   |---|---|---|---|
   | LISTA (210,300) 40 dB [exp2_real] | +1.40 dB | -4.51 dB | favourable |
   | LISTA (25,50) noiseless [try2] | +0.72 dB | -0.70 dB | a wash |
   | ALISTA (210,300) 40 dB [exp2_real] | -1.19 dB (gain) | -1.26 dB | free both ways |
   | ALISTA (25,50) noiseless [try2] | +1.98 dB | -2.12 dB | a wash |

   Defensible: OOD benefit of widening is ~6x larger for LISTA at (210,300)/40 dB
   than at (25,50)/noiseless. NOT settled: whether the sign of the tradeoff
   flips with scale/noise -- exp2_real's ID comparison is not centre-vs-centre
   and the two runs differ in dims AND noise AND centring at once. `try3` is
   the run that separates these (same centring as try2, same dims as exp2_real).

2. **The own-centre gap is structural, not a training artefact** (try2, p=0.175):
   LISTA -19.1/-19.0/-18.4 dB, ALISTA -21.8/-21.2/-19.8 dB for
   narrow/medium/wide. `exp2_try2_curve`: all 6 per-layer curves smooth and
   monotone, stage epochs 9-143 under the 200 cap, patience not binding.
   `exp2_try2_moredata` (3 seeds, 2x data): gap does not close, slightly
   widens (LISTA 0.6 -> 0.86 dB, ALISTA 2.0 -> 2.25 dB). So: fitting one
   fixed-depth unrolled network across a wider sparsity range costs accuracy
   at any single sparsity, and more data/longer training does not buy it back.

3. **ALISTA vs LISTA disagree by metric, consistently**: ALISTA better on
   NMSE (1.1-5.0 dB, 0/10 seeds contradicting), worse on R_f (10/10 seeds),
   every width, both runs. Mechanism identified: ALISTA's support selection
   exempts the largest entries from shrinkage, i.e. a debiasing step relative
   to the LASSO minimiser -- it moves toward x_true and away from x*. At
   (210,300) medium/wide it beats the exact LASSO minimiser on recovery
   (excess-over-floor -0.42/-0.94 dB at p=0.15). Thesis-relevant: "ID quality"
   has two different answers depending on whether the target is the objective
   or the signal, and the pre-study's four metrics were chosen well enough to
   expose that.

4. **Width sensitivity: ALISTA is the more width-sensitive of the two** (on
   frame-independent measures: 1.87 vs 0.57 dB NMSE spread at (25,50); log-ratio
   R_f spread 0.733 vs 0.333 decades at (210,300)). The earlier "ALISTA is
   width-insensitive" reading was a per-method-autoscaled-axis artefact; use
   `plot_experiment2_metrics_modelbased.py` (shared y-axis) for the thesis.

5. **Recoverability floor**: at (25,50) NMSE(x*, x_true) is only -9.6/-6.3/-5.2
   dB at p=0.30/0.35/0.40 -- those cells measure problem difficulty, not
   method quality. At (210,300) the floor is -30.7/-23.3/-16.1 dB (try3 s21),
   far more benign. Tables should grey out cells within ~1 dB of the floor.

6. **Unexplained**: ALISTA-narrow is the worst width at its own training
   point at (210,300)/40 dB but the best at (25,50)/noiseless. try3 will say
   whether that is dims or noise.

### 2026-09-13 -- Model-free status (post checkpoint fix): what 2 seeds already show

`exp2_try2` RNNProp, seeds 21-22, all 3 widths, `--last_step_loss` on,
1000 evaluation steps. Recomputed this session from `results.jsonl`
(LISTA/ALISTA columns from all 10 seeds):

| test p | RNNProp n/m/w NMSE dB | LISTA n/m/w | ALISTA n/m/w | floor (seeds 21-22) |
|---|---|---|---|---|
| 0.02 | -37.3 / -37.1 / -36.5 | -28.4 / -29.0 / -29.5 | -85.1 / -97.4 / -94.5 | -40.6 |
| 0.10 | -39.3 / -39.5 / -39.1 | -29.2 / -29.7 / -29.7 | -62.4 / -61.7 / -61.1 | -42.2 |
| 0.175 | -28.7 / -28.8 / -28.4 | -19.1 / -19.0 / -18.4 | -21.8 / -21.2 / -19.8 | -30.4 |
| 0.30 | -9.3 / -9.3 / -9.1 | -8.1 / -8.1 / -7.9 | -6.0 / -5.9 / -5.8 | -9.5 |
| 0.40 | -5.2 / -5.2 / -5.2 | -5.2 / -5.2 / -5.2 | -3.3 / -3.4 / -3.5 | -5.2 |

Width spread (max-min over widths, dB) at p=0.05-0.20: RNNProp 0.3-0.7,
LISTA 0.4-1.1, ALISTA 1.3-2.8. RNNProp's R_f is 0.001-0.006 at p >= 0.10
(LISTA 0.02-0.03, ALISTA 0.007-0.22, rising with p): at 1000 steps it sits
on the LASSO optimum at every width.

**Interpretation -- and the caveat that decides the thesis claim.** The naive
reading "model-free is width-insensitive, model-based is width-sensitive"
would answer the pre-study's family question, but it is confounded by the
**evaluation regime**: RNNProp runs 1000 iterations on a convex problem and
lands near x* regardless of what it was trained on; LISTA/ALISTA are fixed at
16 layers. At iteration 16, RNNProp's R_f is 25-76 (LISTA 0.022 after 16
layers) and it needs ~290-320 steps to match LISTA (from
`project_exp2_try2_modelfree_partial`). So the family comparison is
the pre-study's *per-protocol* comparison (§7.2: each family "trained to
its own standard, fixed budget" -- 1000 steps vs 16 layers is the design, not
an oversight). A matched-budget comparison has to be *added* because the
width question is otherwise confounded by convergence.

**This is computable from existing data, no GPU.** `evaluate_rnnprop.py`
saves `<opt>_eval_loss_record.pickle-lasso_dataset` = batch-mean f(x_k) for
every k in 1..1000 (only the final iterate x_1000 is saved, so NMSE-vs-k is
NOT available, R_f-vs-k is). Per row, mean f* = `final_loss / (1 +
modified_relative_loss)`, hence R_f(k) = loss_record[k] / mean_f* - 1 with no
FISTA recompute (verified 2026-09-13: `problems.py`'s `loss_fn` is
`0.5*||Ax-b||^2 + lam*||x||_1` batch-averaged, same scale as
`lasso_metrics.lasso_objective`; `final_loss` is `trajectory[-1]` via
`core/result.py:summarize_trajectory`; the x* cache exists anyway as a
fallback). Deliverables: (a) R_f-vs-iteration curves per width and
test p, with LISTA/ALISTA's 16-layer R_f as horizontal lines; (b) a
"model-free at 16 steps / at first-match / at 1000 steps" column set in
every model-free-vs-model-based table. Needs `dill` to unpickle -> run inside
the Docker image.

**L2O-DM** fails training-side at Table 3 hyperparameters (in-memory eval ==
loaded eval; internal eval bounces 140 -> 3338 -> 127 -> 1154 -> 96; NMSE
-0.4 to -3.6 dB at (25,50)), and its best-internal-eval checkpoint is epoch 5
for medium vs 100 for narrow, so any DM width comparison is an epoch
comparison. Dropped from both running sweeps 2026-09-11. Decision pending
(user's): report as a documented failure, drop with justification, or
diagnose further.

**First (210,300) data point** (`exp2_try3_210x300`, narrow, seed 21, 1000
steps): RNNProp -41.0 (p=0.05) to -16.1 dB (p=0.40), R_f 0.001-0.006 at
p >= 0.15; LISTA -38.1 to -10.7 dB; ALISTA -63.0 to -9.4 dB. RNNProp beats
LISTA on NMSE at every p and beats ALISTA for p >= 0.20; on R_f, RNNProp is
worst at p <= 0.15 (0.056 at p=0.02, small-denominator effect on near-null
instances) and best from p >= 0.175. One seed -- direction only.

### 2026-09-13 -- What Experiment 2 can and cannot claim right now

Can claim (10-seed, deconfounded, model-based): a real but modest ID-cost /
OOD-benefit tradeoff for fixed-depth unrolled networks, whose size and
possibly sign depend on scale and noise; the ID cost is structural; the two
unrolled architectures differ mainly in *what* they converge to (objective
vs signal), not in how width-sensitive they are.

Cannot claim yet: anything about the family difference (needs the
compute-matched analysis above, and try3's RNNProp seeds); anything from
exp2_real's model-free rows or the pre-fix diagnostic runs.

### 2026-09-13 -- Compute-matched comparison + classical baselines (gaps #1 and #2 closed)

Two analysis-time additions, no GPU time. Scripts (both in `Benchmarking/`):
- `classical_baselines.py --run runs/<run>` -- textbook ISTA and FISTA (fixed
  step 1/L, no restart) from the shared seeded x0 on the exact per-seed test
  files, scored with `evaluate_lasso_recovery` + the run's own `xstar_cache`.
  Writes `<run>/classical_baselines.jsonl` (rows at k = 16, 100, 1000; same
  metric columns as `results.jsonl`, `width="all"`) and
  `<run>/classical_trajectories/<method>_seed<N>_p<p>.npy` (mean f(x_k),
  k = 0..1000). Done for exp2_try2 (10 seeds), exp2_try3_210x300 (seed 21),
  exp2_real (10 seeds, 480 rows, completed later the same day).
- `plot_compute_matched_rf.py --run <run>` -- R_f(k) for RNNProp from the
  per-step `L2L_eval_loss_record` pickles (mean f* exact from the x* cache,
  not back-solved), ISTA/FISTA from the trajectories, LISTA/ALISTA-16 from
  `results.jsonl`. Outputs `Figs/<run>/compute_matched_rf_<run>.{pdf,png}`,
  `Figs/<run>/iterations_to_match_<run>.{pdf,png}` and the numbers in
  `Benchmarking/compute_matched_<run>.md`. Iteration k = k learned steps =
  k solver iterations = k layers, all from the same x0.

**Findings, (25,50) noiseless [exp2_try2; RNNProp 2 seeds, LISTA/ALISTA 10]:**

| at k = 16 | R_f |
|---|---|
| RNNProp | 23-72 (R_f(0) = 127; it has barely moved) |
| ISTA | 0.61-2.2 |
| FISTA | 0.23-1.6 |
| LISTA | 0.022-0.026 |
| ALISTA | 0.003 (p=0.02) rising to 0.18 (p=0.40) |

- **LISTA-16 beats FISTA-16 by 10-60x on R_f; FISTA needs 78-118 iterations
  to reach LISTA-16's R_f (~5-7x iteration speed-up for LISTA), 26-128 to
  reach ALISTA-16's.** The classical L2O claim holds on this problem.
- **RNNProp needs 301-335 steps to reach LISTA-16 (never at p=0.02) and
  204-389 to reach ALISTA-16 (never at p <= 0.05) -- ~20x the model-based
  budget.** At k=16 it is 10-100x WORSE than plain ISTA-16.
- **FISTA's R_f curve lies below RNNProp's at every k <= 1000**: at k=1000
  FISTA is < 1e-6, RNNProp 0.001-0.03. RNNProp only overtakes ISTA around
  k ~ 200-300 and only ends below ISTA-1000 for p >= 0.10. So on the LASSO
  objective, model-free RNNProp is dominated by FISTA at every budget on
  this problem; its "reaches the LASSO floor" result is a 1000-step result
  that FISTA reaches by k ~ 100-300.
- **The slow start is not a `--last_step_loss` artefact**: the one valid
  post-fix RNNProp run WITHOUT the flag (`runs/scale_diag_ckptfix`, 40 dB,
  p=0.10, seed 0) has the same shape -- R_f 122 -> 31 at k=16 -> 0.83 at
  k=100 -> 0.008 at k=300 (try2 with the flag: 127 -> 35 -> 1.07 -> 0.025).
- **Width response on R_f differs between the families in direction, not
  only size** (numbers from `compute_matched_exp2_try2.md`):
  - RNNProp's three curves coincide at every k: relative spread ~2% at k=16,
    ~10% at k=100, <= 12% at k=1000 for p <= 0.30 (29% at p=0.35-0.40 where
    R_f is 0.0014 vs 0.0018, i.e. numerically tiny). Where a difference
    exists, **narrow is best** (p=0.02: 0.0290 / 0.0316 / 0.0325 n/m/w).
    So its width-insensitivity is not a convergence artefact -- it holds
    while R_f is still ~50.
  - LISTA-16: 10-12% spread at p=0.10-0.30 (inside the training supports)
    but **30-37% at the OOD sparse end, with wide best** (p=0.02: 0.0216 /
    0.0257 / 0.0310 w/m/n; p=0.05: 30%) and 23% at p=0.40. ALISTA-16: up to
    36% at p=0.1125, 30% at p=0.40.
  - Read: the OOD-benefit-of-widening shows up on the objective for the
    unrolled networks (30-40%, at the sparsities farthest from the 0.175
    centre) and not at all for the model-free optimizer (<= 12%, reverse
    sign). Small in absolute R_f, but a genuine family difference in width
    response, consistent with the NMSE story above (ALISTA up to 2 dB).

**Findings, (210,300) noiseless [exp2_try3_210x300; seed 21, narrow only so far]:**
- LISTA-16 R_f 0.005 for p <= 0.175 vs FISTA-16 0.28-1.7 and ISTA-16 1.2-3.1:
  **100-300x** at sparse p. FISTA needs 57-95 iterations to match LISTA-16;
  ISTA 228-961 or never.
- RNNProp-16 R_f 21-67. RNNProp reaches LISTA-16 only for p >= 0.175 (254-369
  steps) and **never for p <= 0.15**: its 1000-step plateau (R_f 0.056 at
  p=0.02, 0.010 at p=0.10, 0.008 at p=0.1125) sits above LISTA-16's 0.005.
  FISTA is again below RNNProp at every k.
- One seed. Re-run the script as try3 seeds land (it picks up whatever is in
  `results.jsonl`; extend `classical_baselines.py` to new seeds first).

**Interpretation for the thesis (supersedes the "cannot claim the family
difference yet" line above):**
1. On the objective, at any matched budget, the ranking on LASSO is
   FISTA/LISTA/ALISTA >> ISTA > RNNProp until k ~ 200-300; RNNProp catches
   ISTA late and never catches FISTA. The model-free optimizer's headline
   "within 0.5 dB of the LASSO optimum" is true and is also what FISTA does
   in a fraction of the iterations. This reconciles the earlier "contradicts
   the Primer" worry: the Primer's LASSO figures are loss-vs-iteration over
   the same 1000-step horizon, where RNNProp *is* poor for most of the
   trajectory -- there is no contradiction, only a different summary point.
2. The families do differ, but not primarily in width sensitivity on R_f: the
   fixed-depth unrolled networks buy a 5-7x (25,50) to ~6x+ (210,300)
   iteration speed-up over FISTA and 10-300x lower R_f at matched depth; the
   model-free optimizer buys nothing over FISTA at any budget on this convex
   problem. Width then acts on the unrolled networks' *recovery* quality
   (NMSE, ALISTA's support-selection debiasing), which is the ID/OOD story.
3. Classical rows are now in every Experiment-2 table (FISTA-16 and
   FISTA-1000), with the baseline-vs-x*-solver distinction stated under each.
   exp2_real (40 dB, uncentred): LISTA-16 R_f 0.0066-0.0086 at p <= 0.10 vs
   FISTA-16 0.89-1.69, i.e. 100-250x -- same order as try3's noiseless
   (210,300) result, so the speed-up is not a noiseless artefact.

Caveats: R_f only for the per-iteration view (NMSE-vs-k is not recoverable:
only x_1000 is saved); 2 RNNProp seeds at (25,50), 1 at (210,300); L2O-DM not
in this analysis (no valid RNNProp-style rows; its s21 narrow/medium records
exist in `runs/exp2_try2` if wanted). **Iteration-matched is not
compute-matched**: one RNNProp step is a per-coordinate LSTM forward pass,
one ISTA/FISTA step is a matrix product, one LISTA layer is a matrix product
plus a threshold -- so the comparison is already generous to RNNProp. State
this in the thesis; a FLOP count is not needed.

### 2026-09-14 -- Recoverability floor: excess-over-floor tables and a wording correction

Recomputed from `results.jsonl` (LISTA/ALISTA, 10 seeds for exp2_try2 and
exp2_real; try3 had 2 model-based seeds at the time). Floor = NMSE(x*, x_true),
seed std <= 0.7 dB at (25,50), <= 0.13 dB at (210,300). "Excess" = method NMSE
minus floor (dB); negative = beats the LASSO minimiser on recovery.

(25,50) noiseless [exp2_try2], m = 25:

| p | nonzeros | floor | LISTA-n excess | LISTA-w | ALISTA-n | ALISTA-w |
|---|---|---|---|---|---|---|
| 0.02 | 1 | -40.4 | 12.1 | 10.9 | -44.6 | -54.0 |
| 0.10 | 5 | -42.1 | 12.9 | 12.4 | -20.3 | -19.1 |
| 0.15 | 8 | -34.5 | 12.5 | 13.1 | 6.1 | 8.0 |
| 0.175 | 9 | -30.4 | 11.2 | 12.0 | 8.6 | 10.6 |
| 0.2375 | 12 | -17.5 | 5.3 | 5.8 | 7.3 | 8.1 |
| 0.30 | 15 | -9.6 | 1.5 | 1.7 | 3.7 | 3.8 |
| 0.35 | 18 | -6.3 | 0.3 | 0.3 | 2.3 | 2.1 |
| 0.40 | 20 | -5.2 | 0.0 | 0.0 | 1.9 | 1.7 |

- LISTA's raw 24 dB "degradation" from p=0.10 to 0.40 is entirely the floor
  (which drops 37 dB); against the floor LISTA goes from +12.9 to 0.0 dB.
  At p=0.40 LISTA-16 solves LASSO as well as it can be solved.
- Width differences at p >= 0.30 are <= 0.2 dB: nothing to interpret. RQ2a's
  width claims live at p <= 0.2375 and survive floor subtraction (~1 dB).
- LISTA's ~12-13 dB excess is flat over p <= 0.15: a 16-layer quality ceiling
  that width does not move, bounding any possible width effect at (25,50).

(210,300) 40 dB [exp2_real], m = 210:

| p | nonzeros | floor | LISTA-n | LISTA-w | ALISTA-n | ALISTA-w |
|---|---|---|---|---|---|---|
| 0.05 | 15 | -42.9 | 5.6 | 7.2 | 1.4 | 3.1 |
| 0.10 | 30 | -40.1 | 3.9 | 5.3 | 2.8 | 1.6 |
| 0.20 | 60 | -35.0 | 13.4 | 3.5 | 15.1 | 0.4 |
| 0.30 | 90 | -27.9 | 15.6 | 8.1 | 15.8 | 6.8 |
| 0.40 | 120 | -15.8 | 6.7 | 4.2 | 7.1 | 5.0 |

- Floor stays benign (-16 dB at 120 nonzeros), so excess is the real RQ2a
  picture: narrow-trained networks fall 13-16 dB behind the optimum outside
  the training support; wide cuts that to 0.4-8 dB. (Centre confound of this
  run still applies to the *size* of the effect.)
- try3 (2 seeds, centred, noiseless) shows the same shape: LISTA-n excess
  5-7 dB at p <= 0.175 rising to 12-15 at p=0.2375-0.30; ALISTA-w negative
  (-19 to -1.5 dB) for p <= 0.20.

**Wording correction:** earlier entries/memories say the (25,50) dense end is
"information-theoretically unrecoverable". Too strong. The floor measures how
well the LASSO objective *at lambda=0.005* identifies x_true, not a limit on
any estimator -- ALISTA beats it (negative excess) by up to 54 dB at sparse p.
Use "LASSO-optimal recovery floor" and "unrecoverable by LASSO at this
lambda". Mechanism for ALISTA < floor: support selection = debiasing relative
to the shrinkage-biased x*.

Write-up: report excess-over-floor next to raw NMSE (or overlay the floor on
every NMSE figure); grey out (25,50) cells at p >= 0.35 (within 0.3 dB of the
floor); state that (25,50)'s dense grid (k >= 15 nonzeros for m = 25) is
outside LASSO's recovery regime -- a consequence of pairing the Primer's
small scale with the pre-study's dense test sparsities, and a reason
(210,300) is the headline scale.

### 2026-09-14 -- Floor decomposition: shrinkage bias vs support failure (verified, seed 21)

Checked on `exp2_try2` seed 21 (noiseless, (25,50)): exact-support recovery
rate of x* and the floor predicted from LASSO's shrinkage bias alone on the
true support (KKT fixed point x*_S = x_S - lam (A_S^T A_S)^{-1} sign(x_S)).

| p | k | exact support | actual floor | bias-only prediction |
|---|---|---|---|---|
| 0.02 | 1 | 99.5% | -40.3 dB | -39.7 |
| 0.05 | 2 | 97.7% | -43.2 | -42.3 |
| 0.10 | 5 | 29.2% | -42.6 | -43.7 |
| 0.175 | 9 | 0.9% | -31.3 | -41.3 |
| 0.30 | 15 | 0.0% | -9.8 | -35.7 |

=> at k <= 5 the floor IS the shrinkage bias (~lam per nonzero); from k = 9
the excess over the bias prediction (10 dB at p=0.175, 26 dB at p=0.30) is
support failure. The knee sits where the Donoho-Tanner l1 phase transition
would put it for delta = m/n = 0.5 (rho ~ 0.4 => k ~ 10; from memory, verify).
At (210,300) delta = 0.7 and p=0.40 gives rho = 0.57 -- NEAR the threshold,
not comfortably inside: say "degraded, not collapsed" (-16 dB). Do not write
"k <~ m / log(n/k)" (meaningless at n/k ~ 3) or "shrinks by lam/L" (that is
ISTA's per-step threshold, not the solution bias).

References to check before citing: Candes-Tao 2005; Donoho 2006; Baraniuk et
al. 2008 (RIP, m >= C k log(n/k)); Donoho-Tanner 2009 (phase transition);
Wainwright 2009 IEEE TIT (LASSO support-recovery thresholds); Tibshirani
1996, Fan-Li 2001 (LASSO bias), Belloni-Chernozhukov 2013 (post-LASSO OLS);
Chen-Liu-Wang-Yin NeurIPS 2018 (support selection, LISTA-CPSS), Liu et al.
ICLR 2019 (ALISTA). ALISTA-below-floor "because of" support selection is
still an inference; a 1-seed ALISTA run with support selection off (~25 min
GPU) would make it an ablation.

### 2026-09-14 -- Citation verification (`citations_verified.md`) reconciled with the code and checkpoints

A web-search agent checked `citations_to_verify.md`; its report is
`citations_verified.md`. Accepted/rejected against this repo's code and data:

**Accepted (use these in the thesis):**
- Donoho-Tanner weak threshold, signed signals, Gaussian A, computed from
  Amelunxen-Lotz-McCoy-Tropp 2014 Prop. 4.5: **rho_W(0.5) = 0.386 -> k* = 9.6
  at (25,50), i.e. p* = 0.19; rho_W(0.7) = 0.499 -> k* = 105 at (210,300),
  i.e. p* = 0.35.** (My remembered 0.55 for delta=0.7 was wrong.) Matches the
  floor knees in the data: (25,50) floor -26 dB at k=10, -17.5 at k=12, -9.6
  at k=15; (210,300) floor -28/-22/-16 dB at p=0.30/0.35/0.40 -- a ramp of
  O(sqrt n) width around p*=0.35, exactly where the pre-study's test grid
  brackets it. Cite Donoho & Tanner JAMS 2009 (curve) / Proc. IEEE 2010
  (readable); Phil. Trans. 2009 only for universality.
- Sample-complexity bound: cite Baraniuk et al. 2008 Thm 5.2 (stated as
  k <= c1 m / log(n/k)) or Foucart-Rauhut Ch. 9 (m >= C s ln(eN/s)); NOT
  Donoho 2006, which does not contain it. RIP bounds are not sharp at
  proportional sparsity (Donoho-Tanner 2010 Sec. X) -- that, not "heuristic",
  is the reason to use the phase transition.
- Drop Wainwright 2009 for Exp 2 (noisy model, sublinear sparsity k = o(n);
  at (25,50) k=5 it would demand 38 measurements yet recovery works with 25).
- Shrinkage bias: derive the KKT fixed point inline (two lines); magnitude
  O(lambda) is stated in Chen-Liu-Wang-Yin 2018 (discussion after Thm 2).
  Fan-Li 2001 for "LASSO biases large coefficients"; Zou 2006 for "lacks the
  oracle property"; Belloni-Chernozhukov 2013 says "smaller bias", not
  "removes".
- Support selection: introduced in Chen et al. 2018 Sec. 2.2, motivated by
  "kicking" in linearized Bregman (Osher et al. 2010), NOT by false-positive
  rate; schedule p^k = min(1.2k, 13) in both LISTA-CPSS and ALISTA's
  reference code. Their simulation setup (m=250, n=500, N(0,1/m) columns
  normalised, Bernoulli(0.1) support, N(0,1) nonzeros, NMSE dB) is nearly
  Experiment 2's.
- Primer F1: the exact sentence is that sub-gradient and model-free methods
  perform poorly at (25,50) **"only L2O-RNNprop can converge faster than
  ISTA and comparably to FISTA, though reaching lower precision."** Do not
  quote the unqualified version. Metric = Eq. 21 modified relative loss
  (ratio of expectations), f* from 2,000 FISTA iterations, 1000 iterations
  for model-free and classical, 16 layers for LISTA/ALISTA, model-based
  trained on the LASSO loss.

**Rejected / qualified -- the agent could not see this repo:**
- E3 ("ALISTA beats x* because theta^k -> 0, not because of support
  selection"): assumes MSE-trained networks as in Chen et al. This repo
  trains LISTA/ALISTA on the LASSO objective (`Model_Base_L2O/train.py:222`,
  `utils.LassoLoss`). Read the learned thresholds from the seed-21 layer-16
  checkpoints (`model/optimizer/_trainable_variables/*`; theta initialised
  at lambda/L = 0.00088):
  LISTA narrow theta^1..16 = 0.336, 0.239, 0.181, 0.138, 0.118, 0.103,
  0.089, 0.075, 0.063, 0.053, 0.043, 0.035, 0.028, 0.021, 0.017, 0.009;
  LISTA wide 0.332 -> 0.009 (same geometric decay); ALISTA narrow/wide
  0.33 -> 0.005-0.011 with learned step sizes 0.4-2.3.
  => threshold annealing IS learned by BOTH networks (even under the LASSO
  loss), so it is the general mechanism by which an unrolled network can
  escape the O(lambda) floor -- but it cannot be what separates ALISTA
  (below floor) from LISTA (+5-13 dB above floor). The only structural
  difference between the two here is `shrink_ss` (ALISTA) vs `shrink_free`
  (LISTA). Support selection stays the best explanation for the
  ALISTA-specific result; still an inference pending an ss-off ablation
  (~25 min GPU, 1 seed). Nice thesis figure: learned theta^k per layer.
- New discrepancy with the Primer (from the corrected F1 quote): the Primer
  reports RNNProp "faster than ISTA, comparable to FISTA" at (25,50); our
  compute-matched curves show RNNProp slower than ISTA until k ~ 200-300 and
  never below FISTA. Their RNNProp was trained on 12,800 Bernoulli-sparsity
  pairs with random x0 and their own hyperparameters; ours on Table 3's.
  This upgrades the post-fix `primer_replication.py` rerun (~3 GPU-h) from
  optional to worth doing: it tests exactly this.

### 2026-09-15 -- Citable artifact for the RQ2a width tradeoff (statement 1)

`Benchmarking/plot_experiment2_width_tradeoff.py` -> `Figs/experiment2/
width_tradeoff_nmse.{pdf,png}` (combined) + four standalone panels
`width_tradeoff_nmse_<run>_<method>.{pdf,png}` (shared y-range, no title) +
`Benchmarking/experiment2_width_tradeoff_figure.tex` (subcaption multi-panel
figure, `fig:exp2-width-tradeoff`, image paths relative to repo root -- adjust
`\graphicspath` in Overleaf) + `Benchmarking/experiment2_width_tradeoff_table.tex`
(`\label{tab:exp2-width-tradeoff}`). Built because nothing in the repo
supported statement 1: the 2026-09-06 paired numbers lived only in a chat
artifact, and the raw per-width NMSE figures cannot show a 1-2 dB effect on
a 40-90 dB axis.

**Definitions (one consistent set, dB, positive = widening hurts; paired by
seed; on excess-over-floor; at offsets d from each width's OWN training
centre, linearly interpolated; runs need >= 5 complete seeds):**
- ID cost = paired difference at d = 0.
- robustness benefit at d = paired difference in *degradation from own
  centre* = net(d) - ID cost (negative: the wider training degrades less).
- net at d = paired difference of the absolute excess at d.
- d = -0.075 (sparse side) and +0.20 (dense side); cells where either width
  is in the exact-recovery regime (raw NMSE < -60 dB) are n/a.

**Continuity with the 2026-09-06 table:** the raw-NMSE-at-fixed-p ID costs
reproduce exactly (+0.72 / +1.98 try2, +1.40 / -1.19 exp2_real), and for
try2 (shared centres, floor cancels) the "OOD benefit at +0.20" = the new
robustness column exactly (-0.70 LISTA, -2.12 ALISTA). For exp2_real the
old OOD numbers (-4.51 / -1.26) were raw NMSE with ID measured at fixed
p=0.10; the new ones are on excess-over-floor at own-centre offsets, which is
the deconfounded quantity, so **quote the new table, not the old memory**.

| run | method | wide - narrow | ID cost | robustness (-0.075) | net (-0.075) | robustness (+0.20) | net (+0.20) |
|---|---|---|---|---|---|---|---|
| (25,50) noiseless | LISTA | | +0.72*** | -1.22*** | -0.50*** | -0.70*** | +0.02*** |
| (25,50) noiseless | ALISTA | | +1.98*** | n/a | n/a | -2.12*** | -0.14*** |
| (210,300) 40 dB | LISTA | | -0.13*** | -1.38*** | -1.51*** | -9.75*** | -9.88*** |
| (210,300) 40 dB | ALISTA | | -3.10*** | +3.38*** | +0.28** | -6.50*** | -9.60*** |

(medium - narrow rows in the .tex; *** p < 0.001 paired t, n = 10; Wilcoxon
agrees everywhere it is significant.)

**Reading for statement 1:**
- (25,50): widening costs 0.7 (LISTA) / 2.0 dB (ALISTA) at the centre and
  buys back the same amount of robustness by d = +0.20, so the net at the
  dense OOD point is ~0 -- but that point (p = 0.375) is on the floor for
  everyone, so the (25,50) dense side cannot show a benefit. On the sparse
  side (d = -0.075, p = 0.10) LISTA's net is -0.50 dB in wide's favour.
- (210,300)/40 dB: no ID cost at own centre (LISTA -0.13, ALISTA -3.10 --
  wide is *better* at its own centre, on excess-over-floor), and a large
  dense-side robustness benefit (-9.75 / -6.50 dB) once the offset is 0.20
  past the training centre. Sparse side: LISTA wide better (-1.38), ALISTA
  wide worse (+3.38) -- ALISTA-narrow (trained at p = 0.10) is nearer the
  exact-recovery regime, which is the ALISTA-narrow sign flip already noted.
- Caveats that stay: exp2_real's widths differ in centre (own-centre offsets
  handle the *comparison*, but "wide" still saw denser data); the dense-side
  offset lands narrow at p = 0.30 and wide at p = 0.375, where excess-over-
  floor is non-monotone (narrow's excess peaks at 0.30), so the -9.8 dB is
  sensitive to the choice d = 0.20 -- the figure shows the whole curve, cite
  it alongside the table. try3 (centred, (210,300)) will replace exp2_real
  in this table once it has >= 5 seeds; re-run the script then.

### 2026-09-15 -- Claim 3 sharpened: width response on R_f, per width (exp2_try2, 10 seeds)

Earlier shorthand ("30-37% at p=0.02-0.05 with wide best") was LISTA-only.
Full pattern (R_f 10-seed means; spread = (max-min)/min across widths;
supports narrow {0.175}, medium [0.1125,0.2375], wide [0.05,0.30]):

| p | LISTA n/m/w | spread | best | wide<narrow | ALISTA n/m/w | spread | best | wide<narrow |
|---|---|---|---|---|---|---|---|---|
| 0.02 | .0310/.0257/.0216 | 44% | wide | 10/10 | .0032/.0032/.0032 | 1% | -- | 5/10 |
| 0.05 | .0302/.0255/.0221 | 37% | wide | 10/10 | .0033/.0033/.0034 | 2% | -- | 5/10 |
| 0.10 | .0245/.0223/.0222 | 10% | wide | 10/10 | .0071/.0075/.0095 | 34% | narrow | 0/10 |
| 0.1125 | .0235/.0221/.0228 | 7% | medium | 9/10 | .0136/.0148/.0196 | 44% | narrow | 0/10 |
| 0.15 | .0238/.0233/.0244 | 5% | medium | 2/10 | .0598/.0610/.0673 | 13% | narrow | 0/10 |
| 0.175 | .0237/.0234/.0242 | 3% | medium | 2/10 | .0878/.0887/.0923 | 5% | narrow | 0/10 |
| 0.20 | .0238/.0234/.0238 | 1% | -- | 4/10 | .1118/.1106/.1119 | 1% | -- | 4/10 |
| 0.2375 | .0231/.0225/.0217 | 7% | wide | 10/10 | .1419/.1375/.1329 | 7% | wide | 10/10 |
| 0.30 | .0232/.0220/.0202 | 15% | wide | 10/10 | .1651/.1557/.1452 | 14% | wide | 10/10 |
| 0.35 | .0242/.0225/.0200 | 21% | wide | 10/10 | .1840/.1661/.1501 | 23% | wide | 10/10 |
| 0.40 | .0261/.0238/.0204 | 28% | wide | 10/10 | .2196/.1790/.1532 | 43% | wide | 10/10 |

- Inside the support overlap (p ~ 0.15-0.20): 1-5% spread, narrow/medium
  slightly ahead = the ID cost on the objective.
- LISTA: symmetric U -- wide best on both sides, growing with distance
  (37-44% sparse, 21-28% dense; 10/10 seeds).
- ALISTA: asymmetric -- no effect at p <= 0.05 (exact recovery, R_f ~
  0.003); wide WORSE at p = 0.10-0.175 (up to 44% at p=0.1125, 0/10 seeds
  contradicting); wide best on the dense side (43% at p=0.40, 10/10).
- Dense-end OOD benefit is visible on R_f but not on NMSE at (25,50): at
  p >= 0.35 every width is on the recovery floor (NMSE net ~ 0 in the
  width-tradeoff table), while R_f still separates them by 20-43%. R_f is
  the more informative OOD metric where the floor collapses.
Cite: `fig:width-ratio-rf-exp2-try2` (`width_ratio_rf_exp2_try2_figure.tex`,
panels `Figs/exp2_try2/width_ratio_rf_exp2_try2_{lista,alista}`; built
2026-09-15 by `plot_experiment2_width_ratio_rf.py` -- paired R_f ratio
wide/narrow and medium/narrow per seed, geometric mean + 95% t-interval, log
axis centred on 1, supports shaded) for the *shape* (parts 2-3), and
`tab:exp2try2mb-rel-loss` for the absolute values. The script also emits
exp2_real (`fig:width-ratio-rf-exp2-real`): there the ratios reach 0.07-0.5
on the dense side and 1.2-4 on the sparse side -- the uncentred-width
confound in raw p, which is why exp2_real's width claims go through the
own-centre-offset figure instead.

### 2026-09-16 -- Why the width gap on the suboptimality gap (Eq. 9) is so much larger in exp2_real than in exp2_try2

Question: `Figs/experiment2/experiment2_modelbased_subopt_gap.png` (exp2_real)
shows widths separated by up to 1.2 decades, while
`Figs/exp2_try2/experiment2_try2_modelbased_subopt_gap.png` shows the three
curves almost on top of each other. Both from
`plot_experiment2_metrics_modelbased.py` (`RUNS`, line 72), field
`suboptimality_gap_mean`, median over seeds, shared y-axis across method
panels. Not an axis artefact: both frames span ~3 decades.

**Answer: two stacked causes, separated with `exp2_try3_210x300` (4 complete
seeds, 21-24) as the like-for-like control.** Width spread = mean over test
p of log10(max/min) of the per-width median gap:

| run | dims / noise / centring | LISTA | ALISTA |
|---|---|---|---|
| exp2_real | (210,300), 40 dB, **uncentred** | 0.33 dec | 0.73 dec |
| exp2_try3 (n=4) | (210,300), noiseless, centred | 0.15 | 0.18 |
| exp2_try2 | (25,50), noiseless, centred | 0.06 | 0.06 |

(try2 restricted to seeds 21-24 gives 0.066 / 0.051 -- the seed subset is
not what differs.) Training config is identical across all three (16 layers,
lr 5e-4, batch 128, 32k instances, lam 0.005; only patience 5 vs 8 in try3,
which cannot produce a width-dependent effect).

**Cause 1 -- exp2_real's uncentred design and/or 40 dB noise (exp2_real vs
try3: ~2x LISTA, ~4x ALISTA).** These two differences cannot be separated
with existing runs -- do NOT attribute this step to centring alone. The
centring part has a visible signature in the exp2_real figure: LISTA's curves
*cross* at p ~ 0.12 -- wide is worst at p <= 0.10 (N/W = 0.58-0.73) and best
at p >= 0.20 (N/W = 2-4.5). That is exactly what "wide was trained nearer the
dense end" (centre 0.175 vs narrow 0.10) produces; no such crossover exists in
try2/try3 where all centres are 0.175. Also, narrow at 0.10 puts its cliff
0.075 earlier (p=0.15 instead of 0.2375) and leaves a 0.30 offset to the
dense end instead of 0.225.

**Cause 2 -- problem scale: at (25,50) the 16-layer nets sit against a
width-independent quality ceiling, so there is no ID specialisation to lose
OOD (try3 vs try2: ~2.5-3x).** R_f = gap/f* makes this plain:

- try2 LISTA: R_f = 0.020-0.026 at EVERY width and EVERY p (own-centre
  0.0236 / 0.0235 / 0.0240). The gap curve is just 0.023 * f*(p); training
  width has no purchase on it.
- try3 LISTA: R_f = 0.005 ID (4-5x better than (25,50)), flat inside each
  width's support, then a cliff once p leaves it -- narrow 0.0055 (p=0.175)
  -> 0.0234 (0.2375) -> 0.0385 (0.30), a 7x rise, while wide holds ~0.007 to
  0.2375 and only rises at 0.30 (past its own support). The width gap IS the
  different position of that cliff.
- Corollary, testable in the same numbers: narrow's ID advantage over wide
  at the own training point is **1%** (try2, p=0.175: 8.43e-4 vs 8.53e-4),
  **16%** (try3, p=0.175: 1.14e-3 vs 1.36e-3), **27%** (exp2_real, p=0.10:
  1.03e-3 vs 1.42e-3, confounded). No specialisation -> no cliff.
- ALISTA at try2 is the cleanest version: all three widths lie on top of each
  other while R_f rises 0.003 -> 0.2 with p -- the *test* problem's density,
  not the *training* width, sets the error.

Hypothesis (consistent with, not proven by, this data): the ceiling is the
Donoho-Tanner transition. At (25,50) p* = 0.19, so try2's training centre
0.175 sits on the threshold and even ID instances are near-unrecoverable
(matches "LISTA at (25,50) never gets within 11 dB of the optimum", entry
2026-09-06); at (210,300) p* = 0.35, so the training range is deep in the
recoverable regime where a 16-layer net can specialise -- and then pay for it.

Consequences for the thesis: the two figures are not "the width effect at
two scales"; they differ in centring, noise and dims at once. try3's figure
is the like-for-like (210,300) comparison and should replace exp2_real's for
any width statement once more seeds land. exp2_real's figure is the
confounded-design lesson (see run inventory).

Side observation, not an outlier: try3 ALISTA-**medium** at p=0.40 is 2x
worse than narrow AND wide in all 4 seeds (6.3e-2 to 1.1e-1 vs 3.3e-2 to
4.5e-2). Unexplained; added to open questions (n=4, dense end -- do not build
on it yet).

### 2026-09-17 -- L2O-Scale as the third model-free arm: its Exp-1 "failure" was the lr prior, not the method

**Decision (user, 2026-09-16):** add L2O-Scale (Wichrowska et al. 2017) to
Experiment 2 as the third model-free method. L2O-enhanced (DM + curriculum +
imitation, `--if_cl --if_mt`, already wired) was proposed and rejected on the
literature record (never learned well, like DM). Entropy/Jacobian (Scale +
NN-generalisation regularisers), Swarm (DM's networks) and Minimax (other
problem class) were not candidates.

**Why Exp 1's L2O-Scale column (R_f 10-50 at 1000 steps, every T, 2026-09-16
entry) is not evidence that L2O-Scale cannot solve LASSO -- two code facts:**
1. `hierarchical_rnn.py` initialises every coordinate's log-learning-rate
   uniformly on `[log min_lr, log max_lr]` with the library default
   `(1e-6, 1e-2)`, log-centre **1e-4**. For (25,50) LASSO `1/L = 0.21`
   (seed-21 narrow A). The optimizer starts ~1000x too slow and must learn
   three decades of lr growth; Exp 1 gave it 500 meta-updates.
2. Exp 1 used `num_problems=1`: `pg.Lasso` draws its 128-instance batch once
   in `__init__`, so all 50 meta-iterations trained on ONE batch, whereas
   DM/RNNProp refresh the batch every epoch (12,800 distinct instances).

**lr-prior pilot (CPU, 3 arms, seed 21, `runs/exp2_try2/data/seed21/narrow`,
DM/RNNProp-matched recipe: 100 fresh-batch problems x 1000 steps, unroll 20,
validation every problem; in-sample on the last training batch, f* = 0.0357):**

| `[min_lr, max_lr]` | validation cost | R_f | NMSE | meta-grad norm (median) |
|---|---|---|---|---|
| (1e-6, 1e-2) = Exp 1 | bounces 0.2-4.2, never settles (82 problems; crashed on a file read under host memory pressure) | 4.19 | +0.13 dB | -- |
| **(1e-3, 1e-1)** | **0.038, flat from problem 1** (0.042 -> 0.038, min at 71, last/min 1.08) | **0.075** | -18.0 dB | 0.15 |
| (1e-2, 1.0) | 0.04-0.23, noisy | 0.39 | -18.8 dB | 2.3 |

- The default prior reproduces Exp 1's failure exactly; one decade-and-a-half
  shift of the prior turns L2O-Scale into a working optimizer (R_f 0.075 --
  between RNNProp's 0.001-0.04 and LISTA's 0.022-0.027, far from DM's
  0.22-5.7). Read for Exp 1: the Scale column measures a prior mismatch, not
  the architecture; say so wherever that column is discussed.
- With the right prior L2O-Scale is good almost from initialisation (first
  validation 0.042 after one problem); 100 problems polish it by ~10%. Its
  learned per-coordinate lr + RMS-normalised update is close to a tuned
  adaptive first-order method before any meta-training -- a different
  starting point from DM/RNNProp's random LSTM, worth stating when comparing
  "how much did meta-training buy".
- Not compute-matched against LISTA/ALISTA here (1000 steps vs 16 layers);
  the compute-matched script applies once the harness run lands.

**Integration (done, smoke-verified in both `experiments/experiment2_try2.py`
and `experiments/v2/experiment2_try2.py`; `v2/experiment2.py` accepts the
name but raises -- no Scale branch there):**
- `L2O-Scale/problems/problem_generator.py::Lasso`: `indices=` (exact rows)
  and `x0_mode="aligned"` (seeded `<split>_x0.npy` start, same file DM/
  RNNProp/LISTA/ALISTA use). Training path unchanged.
- `L2O-Scale/evaluate.py`: `--include_lasso_problems` walks the test file
  deterministically in chunks of the TRAINING batch size (the tensor-level
  RNN input and the output RMS pool over the whole (batch, n) tensor, so
  chunk size is part of the input distribution), fresh reload per chunk,
  writes the per-step mean objective + `L2o_recovery-lasso_dataset.npz`.
  Also now seeds random/numpy/TF from `--seed` like `train.py` -- without it
  the lr prior is redrawn per run (`tf.random.uniform(seed=None)`): verified
  two reruns differ by up to 0.38 in x_pred unseeded, bit-identical seeded.
- `adapters/scale_family.py`: `lasso_dataset` in PROBLEM_MAP; `evaluate()`
  derives `train_dir`/`restore_model_name` from the `checkpoint` argument
  (Exp 2 evaluates into per-sparsity `eval_p*/` subdirs, where the old
  output_dir-based derivation finds nothing); recovery metrics recomputed
  with `core.lasso_metrics.evaluate_lasso_recovery` exactly as
  `dm_rnnprop.py`; `select_checkpoint="best"` returns `model-best.l2o`.
- Orchestrator: `scale_*` config fields (README table has the sources);
  `num_problems = model_free_num_epochs`, 1 meta-iteration/problem,
  validation every problem with 1 restart (period must be 1 with 1
  meta-iteration; DM gets 20 x 20). Caveat: validation is on the batch just
  trained on, so `model-best` is partly a batch-easiness pick -- record the
  winning problem index from `train.log`.
- Verified: 31/31 tensors round-trip; eval step-0 objective == mean f(x0)
  over the test file to float precision; `x_true`/`b` row-aligned with the
  split file; 10-step smoke trajectory length correct.

**Cost:** ~50 ms/optimizee-step on CPU at (25,50)/batch 128 (faster than the
GPU's 85 ms -- eager overhead dominates at this size), so a (width, seed)
triple is ~3 h train+validation + ~1.5 h test eval on CPU and can run without
the GPU. Note the pilot's three containers overlapped `exp1_calibration_2`
rows ~482-509 (recorded in that run's `RESULTS_STATUS.md`).

**Next:** one seed x 3 widths straight into `runs/exp2_try2` (new skip key
`(l2o-scale, width, seed)`, shared data/x0/x*-cache, `alista` kept in
`--methods`), then seeds 22-30 if held-out R_f holds up; add a Scale column
to the Exp 2 tables/figures and the compute-matched view.

### 2026-09-17 -- Model-based per-layer trajectories were running means across test files (all sweeps); regenerated from checkpoints

Found while building the in-process re-implementation (`Benchmarking/unified`,
branch `unified-slice`) and checking it against the stored try3 LISTA
evaluation: the unified per-layer trajectory for `test_sparsity_0.10` started
at f(x0) = 14.962 (the same number RNNProp's stored trajectory starts at on that
split) while the stored `lasso_layer0` was 8.4866. `x_pred` matched the stored
`final_output.npy`, so the iterates were identical -- only the metric differed.

**Mechanism.** `Model_Base_L2O/train.py` `--test` builds its 16 `lasso_layer{i}`
Keras `Mean` metrics once and reuses them across the `compile()`/`evaluate()` of
each test file; under Keras 3 their totals are never reset, so the value stored
for the k-th file is the running mean over files 1..k. Confirmed three ways: the
running means of the true per-file f(x0) reproduce the stored values to 4
decimals on try3 (`2.9461, 5.2488, 8.4866, 10.5895 ...`) and on exp2_real seed 0
(`3.0048, 5.2929, 8.5769 ...`); reproduced locally on CPU with two different
files; and the stored values de-accumulate exactly (`f_k = k*s_k - (k-1)*s_{k-1}`)
onto the unified trajectories for all 11 try3 files (now a pytest,
`tests/test_unified_parity.py::test_real_unrolled_checkpoint_parity`).
Original 2021 toolbox code; present in every sweep from exp2_real onward.

**Scope.** Wrong: model-based `final_loss`/`min_loss`/`mean_loss` and raw
`lasso_layer{i}` for every test sparsity after the first (0.02), all sweeps.
Because files run in increasing sparsity order, harder files were dragged down
(0.40 final layer ~ half its true value). Correct: the four Table-4 recovery
metrics (from `final_output.npy`), all model-free rows, the compute-matched
comparison (reads `modified_relative_loss`). So the RQ2a/claim-3 artifacts and
tables built on the recovery metrics stand; anything that plotted a
model-based per-layer curve or quoted a model-based `final_loss` does not.

**Fix + regeneration.** `train.py` builds fresh metrics per file (verified on
the try3 checkpoint: `2.9461, 7.5514, 14.9623`). No retraining needed:
`Benchmarking/regenerate_modelbased_trajectories.py --run runs/<run>` re-evaluates
every model-based checkpoint with `unified-lista`/`unified-alista` on CPU,
cross-checks per cell (x_pred per row vs `final_output.npy`, de-accumulation),
and writes `all_test_metrics_perfile.json` next to each stored file plus
`<run>/results_modelbased_trajectories_fixed.jsonl`; originals untouched.
Run for exp2_try2_curve, exp2_try3_210x300, exp2_try2, exp2_real,
exp2_try2_moredata (2026-09-17). Recorded in `RESULTS_VALIDITY.md` and each
run's `RESULTS_STATUS.md`.

**Side result of the same check:** on CPU the unified RNNProp evaluation is
bit-identical to the library script's (0.0 difference over 1000 steps x 1280
instances); the stored GPU files differ from both by up to 8e-3 in x_pred
(float32 GPU/CPU drift), 1e-5 in the trajectory.

### Open questions / next steps (Experiment 2)
- Primer discrepancy, now sharper (see 2026-09-14 citation entry): the Primer
  says RNNProp is faster than ISTA / comparable to FISTA at (25,50); ours is
  not. Post-fix rerun LAUNCHED 2026-09-16 by the watcher as container
  `primer_replication_ckptfix` -> `runs/primer_replication_ckptfix` (fresh
  dir; the old `runs/primer_replication` rows are invalid and would have been
  skip-tracked). Compare its 10 replicate rows to the F1 claim when done;
  R_f(k) from its `L2L_eval_loss_record` pickle, but note random x0 per
  replicate, so `classical_baselines.py`'s shared-x0 comparison does not
  apply as-is.
- ALISTA support-selection ablation (ss off, 1 seed, ~25 min GPU) to turn
  the below-floor mechanism from inference into evidence.
- ~~Compute-matched R_f-vs-iteration~~ DONE 2026-09-13 (entry above). Re-run
  `plot_compute_matched_rf.py` as try3 seeds land.
- ~~Classical baseline rows~~ DONE 2026-09-13 for all three runs, and
  FISTA-16 / FISTA-1000 columns are now in both `.tex` result tables
  (`generate_result_tables.py`, `generate_result_tables_try2_modelbased.py`).
  Both generators now aggregate `results.jsonl` + `classical_baselines.jsonl`
  directly -- the stale-`summary.json` trap is closed, and the exp2_real
  table no longer carries the invalid DM/RNNProp columns. Every table prints
  a note distinguishing the plain Beck-Teboulle FISTA baseline (step 1/L, no
  restart, from x0, exactly k iterations) from the restart + certified-gap
  FISTA that computes x* (from 0, gap 1e-10, cap 50,000); sub-1e-9 cells
  render as a bound because they sit inside x*'s certification tolerance.
- `exp2_try3_210x300`: STOPPED at the seed-25 boundary on 2026-09-16 (5 seeds,
  45 cells) by the watcher in `Benchmarking/runs/_watcher/`, to hand the GPU
  to the Primer rerun and then Exp 1 (user decision). Seeds 26-30 resume with
  `docker start exp2_try3_210x300` when the GPU is next free. At 5 seeds:
  re-run `plot_compute_matched_rf.py`, `plot_experiment2_width_tradeoff.py`
  (try3 replaces exp2_real in `tab:exp2-width-tradeoff`), and the two
  ALISTA re-examinations below. `classical_baselines.py` extended to seeds 22-25 on 2026-09-16 (330 rows
  in `classical_baselines.jsonl`, 5 seeds x 66, verified).
- DM decision (user).
- L2O-Scale harness run (2026-09-17 entry): launch 1 seed x 3 widths, then extend; re-read Exp 1's Scale column as a prior-mismatch measurement.
- Resume `exp2_try2_modelfree` (seeds 23+): not recommended (try3
  supersedes it, thesis outline §4); keep the container.
- Fix `generate_result_tables.py` to read `results.jsonl`, not `summary.json`.
- Correct both `config.json` `num_fista_iters` caveats to 50,000.
- ALISTA-narrow sign flip between runs: re-examine once try3 has >= 5 seeds.
- try3 ALISTA-medium 2x worse than narrow and wide at p=0.40 (all 4 seeds, 2026-09-16 entry): re-examine with >= 5 seeds.
- Separate centring from noise for the exp2_real -> try3 width-spread drop (2026-09-16 entry): would need a (210,300) 40 dB centred run; low priority unless the noise story matters.

---

## Experiments 3-5

Not started. Sections to be added when planning begins.
