# Thesis thread and outline (working draft, 2026-09-13; rescoped 2026-09-16)

Companion to `experiments_analysis.md` (evidence) and `DA233X_Prestudy.pdf`
(the plan). This file is the *argument*: what the thesis claims, which
chapter carries which claim, and what each claim rests on. Update it as
claims move between "pending" and "established".

## 0. Scope decision (2026-09-16, from the thesis advisor)

**Experiment 2 and its sub-experiments, plus the platform work, are
sufficient for the thesis. Experiment 1 is an optional add-on if time
allows.** Exp 1's established findings (memory/time/grad-norm laws) are
kept in the ledger and can become an extra chapter or appendix later;
nothing else in the thesis depends on them.

**Not in the thesis (author's decision, 2026-09-16):** the reproducibility
*case study* -- the checkpoint save/load bug, the "published-code results
were silently wrong" story, the Primer's unseeded comparisons, and the
Primer-replication discrepancy. That material is for the author's own
learning only. The platform's reproducibility *features* (seeding,
provenance, fair shared eval batches; pre-study §7.3) stay in ch. 3 as
platform work. `RESULTS_VALIDITY.md` stays as internal bookkeeping for
which runs may be quoted.

The spine is therefore **one platform, one tradeoff**.

## 1. The thread in one paragraph

The pre-study already contains the spine; the experiments feel like sauce
only because the sub-runs were never assigned to it. The spine is:

> L2O's promise (§2 of the pre-study) is that training on a task
> distribution lets an optimizer discover distribution-specific shortcuts.
> That same mechanism is the source of one of its headline open problems:
> a shortcut learned on a concentrated distribution may not survive outside
> it (§5.2 "generalization: interpolation vs extrapolation"). Since the
> 2021 Primer there has been no maintained platform on which this question
> can be answered fairly and reproducibly (§6). This thesis (a) extends the
> Primer's code into that platform (unified train/eval pipeline, seeded
> data + training, LASSO generator, Table-4 metrics; §7.4), and (b) uses
> it to measure the ID-quality / OOD-robustness tradeoff of
> training-distribution width, for model-free and model-based L2O on LASSO
> (Experiment 2), including how the two families differ.

So: **one platform, one tradeoff.** Every chapter and every run should be
attachable to one of those two.

## 2. Chapter skeleton -> what it rests on

| chapter | pre-study section | evidence | status |
|---|---|---|---|
| 1 Introduction: L2O, shortcuts, the ID/OOD tradeoff, RQ, contributions | §1-2, §7.2 | -- | write from the pre-study |
| 2 Background: classical (ISTA/FISTA), model-free (DM/RNNProp), model-based (LISTA/ALISTA), open challenges (§5.2 generalization is the one that matters; §5.1 scalability only as context) | §1-5 | -- | pre-study text is already most of it |
| 3 The benchmarking platform: unified train/eval harness, seeding + provenance + shared eval batches, sweep driver, LASSO generator, Table-4 metrics, Docker (migration = one paragraph + appendix) | §6, §7.3, §7.4 | `Benchmarking/README*.md`, `CONTRIBUTING.md`; **writing plan: `ch3_platform_writing_plan.md`; drafts: `Benchmarking/thesis_ch3/`** | plan approved 2026-09-16; drafting |
| 4 Experiment 2 -- method: design (Table 3, centred widths -- and why the centring was changed from the pre-study's Table 3), metrics (Table 4), classical baselines, compute-matched view, recoverability floor | §7.2 | `experiments/v2/`, `classical_baselines.py`, `plot_compute_matched_rf.py` | design done; needs writing |
| 5 Experiment 2 -- results: model-based width tradeoff; model-free (RNNProp) vs model-based | §7.2 | `exp2_try3_210x300` (headline), `exp2_try2` (cross-scale), `_curve`/`_moredata` (mechanism), `exp2_real` (appendix at most: model-based at 40 dB) | model-based half established; model-free half at 5 try3 seeds |
| 6 Discussion: what the tradeoff says; how the families differ; limitations (RNNProp = the model-free family, DM failed; iteration-matched not compute-matched; in-sample floor) | -- | compute-matched + classical baselines | pending |
| 7 Conclusion, future work | -- | -- | -- |
| (optional) Experiment 1: unroll length vs memory / time / stability across dims | §7.1 | `runs/exp1_calibration` (481/1080 rows) | only if time allows; findings in ledger below |

Out of the thesis (own learning only): the reproducibility case study
(checkpoint bug, zero-parameter LISTA under Keras 3, RNNProp loss weighting,
Primer's unseeded comparisons, `primer_replication_ckptfix`).

## 3. Claims ledger

Status: **E** established (>= 10 seeds or a proof), **E\*** established in
direction with < 10 seeds (re-check when the run completes), **P** pending a
running job, **N** needs new work, **U** user decision.

### Platform
- E: Six L2O codebases run under one harness with seeded data, seeded
  training, identical eval batches/x0 across methods, provenance records
  (git state, diffs, untracked-file hashes).
- E: LASSO generator (three sparsity widths, per-sparsity test sets),
  Table-4 metric functions, FISTA x* solver with certified gap, classical
  ISTA/FISTA baselines, per-iteration R_f, sweep template
  (`experiments/v2/`).
- (Own learning, NOT thesis: checkpoint round-trip bug, Primer's unseeded
  comparison -- see `project_checkpoint_save_load_bug`,
  `project_paper_reproducibility_claims`.)

### Experiment 1 (optional chapter/appendix)
- E: Peak memory = c_method * batch * n * T; per-coordinate-step constant
  (DM 3.33 KB, RNNProp 3.44 KB, Scale 1.43 KB). Report per-instance memory.
- E* (4-5 seeds, 481 rows; re-check at 540): time/step = a + b/T, flat in
  dim (GPU latency-bound), per-instance time is the batch-schedule artefact
  again; grad-norm ~ T^1.5 (RNNProp, Scale) / T^1.7-1.9 (DM), ~ n^1.0 for
  RNNProp (raw-LASSO meta-objective) vs dim-free for Scale (log-ratio);
  Scale's clipping never engages (one grad-norm column, not two); R_f:
  RNNProp improves to T~50 then saturates (knee), DM erratic, Scale never
  learns LASSO. Log entry 2026-09-16 (Exp 1). Caveat: R_f is after 10T
  steps (fixed num_unrolls), so T also sets the eval budget.
- Caveat to state: R_f is in-sample in Exp 1; batch size differs across dims
  (16-128), which also changes meta-gradient noise on the dim axis.

### Experiment 2 -- model-based
- E: ID cost / OOD benefit tradeoff exists; size depends on scale/noise.
  CITE: `Figs/experiment2/width_tradeoff_nmse` + `tab:exp2-width-tradeoff`
  (`experiment2_width_tradeoff_table.tex`, built 2026-09-15: paired by seed,
  own-centre offsets, excess-over-floor). Numbers: (25,50) ID cost +0.72 /
  +1.98 dB (LISTA/ALISTA) with robustness benefit -0.70 / -2.12 at d=+0.20;
  (210,300)/40 dB no ID cost (-0.13 / -3.10) and robustness -9.75 / -6.50.
  Re-run the script when try3 has >= 5 seeds.
- E: Own-centre gap is structural (not instability, not sample density).
- E: Width response on R_f (exp2_try2, 10 seeds; log entry 2026-09-15):
  1-5% inside the support overlap (ID cost), growing with distance outside.
  LISTA symmetric U (wide best both sides, 37-44% sparse / 21-28% dense);
  ALISTA asymmetric (wide worse by up to 44% at p=0.10-0.175, best by 43%
  at p=0.40). R_f shows the dense-end OOD benefit that NMSE cannot (floor).
  CITE: `fig:width-ratio-rf-exp2-try2` (`width_ratio_rf_exp2_try2_figure.tex`)
  + `tab:exp2try2mb-rel-loss`.
- E: ALISTA vs LISTA disagree by metric (NMSE vs R_f); ALISTA lands below
  the LASSO-optimal floor at sparse p. Both networks learn annealed
  thresholds (verified from checkpoints, 2026-09-14), so the differentiator
  is support selection -- an inference until an ss-off ablation.
- E: ALISTA is the more width-sensitive architecture.
- E: dense-end degradation at (25,50) is the floor, not the method (LISTA
  excess over floor 12.9 -> 0.0 dB from p=0.10 to 0.40); floor = shrinkage
  bias at k <= 5, support failure from k ~ 9, at the Donoho-Tanner threshold
  (k* = 9.6 at (25,50), 105 at (210,300) => p* = 0.19 / 0.35). Wording:
  "unrecoverable by LASSO at this lambda". Citations: see
  `citations_verified.md` + the 2026-09-14 reconciliation entry.
- P (try3): whether the tradeoff's *sign* flips with scale vs noise; the
  ALISTA-narrow sign flip.

### Experiment 2 -- model-free
- E (2 seeds, (25,50)): RNNProp at 1000 steps reaches the LASSO optimum at
  every width, width spread <= 0.7 dB.
- E (2 seeds (25,50), 1 seed (210,300)): **compute-matched view** (done
  2026-09-13, `compute_matched_<run>.md`, `Figs/<run>/compute_matched_rf_*`).
  RNNProp needs ~300 steps to reach LISTA-16's R_f (never at p<=0.15 at
  (210,300)); FISTA is below RNNProp at every budget; RNNProp's width curves
  coincide at every k. The 1000-vs-16 comparison stays the pre-study's
  per-protocol design; this is the added matched-budget view.
- E: **classical baselines** (done 2026-09-13, `classical_baselines.jsonl` per
  run): LISTA-16 is 10-60x (25,50) / 100-300x (210,300) below FISTA-16 on
  R_f; FISTA needs 5-7x the iterations to match. The L2O speed-up claim holds
  for the unrolled networks and not at all for the model-free optimizer.
- OPEN (sharpened 2026-09-14): the Primer's exact sentence is "only
  L2O-RNNprop can converge faster than ISTA and comparably to FISTA, though
  reaching lower precision" at (25,50) -- the opposite of our compute-matched
  result (RNNProp slower than ISTA until k ~ 200-300, never below FISTA).
  Post-fix rerun running 2026-09-16 (`runs/primer_replication_ckptfix`).
- Caveat to state: iteration-matched is not compute-matched (LSTM step vs
  matrix product); generous to RNNProp.
- P (try3): RNNProp at (210,300): 5 seeds done 2026-09-16 (stopped at the
  boundary to hand the GPU to the Primer rerun + Exp 1); seeds 26-30 later.
- U: L2O-DM -- fails training-side at Table 3 hyperparameters. Options:
  (a) report as documented failure with the epoch-selection trap;
  (b) drop with justification (Primer itself reports DM failing on LASSO);
  (c) diagnose (unbounded cost).

### Discussion (the "families differ" question)
- N: depends on the compute-matched analysis. Two candidate readings, to be
  decided by data, not preference:
  1. "At matched depth (16), model-based wins on ID and the width tradeoff
     is a model-based phenomenon; model-free needs ~20x the iterations to
     match and then is width-insensitive because it has converged."
  2. "Model-free learns a general-purpose solver (converges to x* from any
     width); model-based learns a distribution-specific estimator (ALISTA
     beats x* on recovery at its training sparsity) -- the shortcut in §2 is
     literally visible in the metrics."
  **Data (2026-09-13): both hold, and reading 1 dominates on the objective**
  -- at matched budget model-based wins by 10-300x and FISTA beats RNNProp
  at every k; reading 2 is where the width story lives (NMSE, not R_f).

## 4. Gaps, ranked by value per hour (rescoped 2026-09-16)

1. **DM decision** (see ledger) -- it sets the wording of every
   "model-free" sentence; zero compute for (a)/(b).
2. **try3 to 10 seeds** (seeds 26-30, ~10.6 GPU-h each, ~2.2 days) -- the
   headline run; the only GPU work on the spine. At 5 seeds (tonight),
   re-run `classical_baselines.py`, `plot_compute_matched_rf.py`,
   `plot_experiment2_width_tradeoff.py` and the try3 result tables.
3. **Write ch. 3 (platform)** -- no data dependency, fully established.
4. **Write ch. 4 (Exp 2 method)** -- design is fixed; only numbers pending.
5. Housekeeping before final tables: both `config.json` `num_fista_iters`
   notes -> 50,000; grey out cells within ~1 dB of the recoverability floor.
6. Optional, ~25 min GPU: ALISTA support-selection-off ablation (turns the
   below-floor mechanism from inference into evidence).
7. Optional: Exp 1 seeds 5-9 / figures -- only after the above.

Off the spine: Primer replication rerun (own learning), L2O-Scale on LASSO,
proximal-auxiliary-task lead, more (25,50) model-free seeds.

## 5. Open decisions for the author

- Thesis deadline -> try3 seed count, whether Exp 1 is included at all.
- DM: (a) documented failure / (b) drop with justification / (c) diagnose.
- GPU after try3 seed 25: the 2026-09-16 watcher chain hands the GPU to the
  Primer rerun and then Exp 1; under the rescope both are off the spine and
  try3 seeds 26-30 should probably go first instead.
