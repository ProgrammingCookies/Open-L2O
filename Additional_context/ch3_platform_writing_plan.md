# Chapter 3 -- The benchmarking platform: writing plan (2026-09-16)

Companion to `thesis_thread_outline.md` (ch. 3 row). Format: LaTeX/Overleaf.
Target length ~10-14 pages incl. 2 figures + 3 tables.

## The one-sentence job of this chapter

Show that the Primer's Open-L2O code was turned into a platform on which
Experiment 2 could be run fairly and reproducibly -- and on which someone
else could run their own experiment -- by delivering the four things the
pre-study §7.4 promised:

| §7.4 goal | delivered as | ch. 3 section |
|---|---|---|
| (i) a unified pipeline (one train/eval interface instead of six) | `Benchmarking/core` + `adapters` | 3.3 |
| (ii) reproducibility and fairness | seed threading, provenance, shared x0, deterministic eval, Docker | 3.4 |
| (iii) atomized, reusable, easy for new people to modify | thin adapters, `CONTRIBUTING.md`, `experiment_template.py` + `sweep.py` | 3.3, 3.5 |
| (iv) metric functions and data generator for the experiments | `data/lasso.py`, `core/lasso_metrics.py`, classical baselines, per-iteration R_f | 3.6 |

Anything not in that table is either ch. 4 (an *Experiment 2 decision*,
e.g. the hyperparameter table, the centred widths, 50k FISTA iterations)
or out of the thesis (the reproducibility case study, per the 2026-09-16
scope decision).

**Rule for the whole chapter: mechanism, not narrative.** "The platform
guarantees every method sees the same test batch and the same x0" is in.
"We discovered the old eval scored a random optimizer" is out.

## Sections

### 3.1 Starting point (~1 page)
- What Open-L2O is: the Primer's [2] code release, six model-free libraries
  (L2O-DM/RNNProp, L2O-Scale, L2O-Entropy, L2O-Jacobian, L2O-Swarm) and one
  model-based library (LISTA family), each a separate TF1-era project with
  its own `train.py`/`evaluate.py`, own flag conventions (argparse vs absl),
  own result files, and colliding module names (`meta.py`, `util.py`,
  `problems.py`, `networks.py`).
- Pre-study §6: no maintained alternative since 2021. Pre-study §7.4: the
  four goals above.
- Framing sentence to reuse: *this thesis extends the Primer's code into a
  platform; it does not replace it.*
- Evidence: pre-study §6/§7.4; `README.md` "Why a harness (and why thin)".
- **Table 3.1 -- inherited vs added.** The single most important artifact
  in the chapter: one row per component, columns *inherited from Open-L2O /
  modified / added*. Rows: each library's training + evaluation code
  (inherited, migrated); harness core (added); adapters (added); sweep
  driver + template (added); LASSO generator (added; the repo's old
  per-library generators inherited); Table-4 metrics + FISTA x* solver
  (added); classical ISTA/FISTA baselines (added); seed/provenance (added);
  Docker image (added); profiling (added, `profiling.py` x2 -- Exp 1 only,
  mention or drop with Exp 1).

### 3.2 Migration to TensorFlow 2 / Keras 3 (ONE paragraph + Appendix A; decided 2026-09-16)
- The paragraph: why it was necessary before any of §7.4 was possible (TF1
  graph-mode code does not run on a current stack; six libraries with
  incompatible pins cannot share one environment, and one environment is a
  precondition for one pipeline); the principle followed (mechanical 1:1
  port, no functional changes -- what makes the migrated libraries still be
  *the Primer's methods*); each library verified end to end after the port;
  pointer to Appendix A.
- **Appendix A -- migrated libraries table.** Library | family
  (coordinatewise RNN / hierarchical RNN / population / unrolled) |
  structural change (Scale: Training+Evaluation split unified; Entropy and
  Jacobian rebuilt on the unified Scale; DM/RNNProp: LSTMCell API; Swarm:
  meta API reshape; Model_Base: imports normalised) | entry points |
  verification.
- Evidence: `project_scale_migration`, `project_entropy_migration`,
  `project_jacobian_migration` (+ scope note: 7 HALO exclusions),
  `project_swarm_migration`, `project_keras3_compat_fixes`,
  `project_structure`.
- **Out, decided:** Keras-3 API breakages fixed during the port (variable
  tracking, optimizer signatures) are NOT mentioned anywhere; nor any "bug
  found in published code" story.

### 3.3 The harness (~2.5 pages) -- goals (i) and (iii)
- The contract: `L2OMethod.train(config, output_dir) -> TrainOutput`,
  `evaluate(config, checkpoint, output_dir) -> EvalOutput`; the
  `trajectory` convention (one scalar per step) and
  `summarize_trajectory` -> `final_loss`/`min_loss`/`mean_loss`. Numbers
  (verified 2026-09-16, incl. blank lines): 7 modules in `core/` (~700
  lines, `lasso_metrics.py` ~200 of them; `method.py` 49), one adapter
  file per family,
  11 registered methods (8 learned: dm, rnnprop, scale, entropy, jacobian,
  swarm, lista, alista; 3 classical: adam, sgd, adagrad).
- Why adapters are thin subprocess wrappers: module-name collision -> the
  libraries cannot be imported into one process; also keeps the migrated
  code untouched (the fairness argument: the platform compares the Primer's
  methods, not re-implementations of them).
- What "one step" means per family: an optimizer step (model-free) vs one
  unrolled layer (model-based) -- the layer-as-iteration mapping in
  `adapters/model_based.py`; state that iteration != compute here (this
  caveat is reused in ch. 4/6).
- Classical optimizers as first-class registered methods delegating to a
  host library's eval harness.
- Config: explicit-required keys, no hidden defaults (`core.config.require`)
  and *why* (train and eval must agree on what they evaluate); flag
  validation against the target script (`discover_script_flags`).
- **Figure 3.1 -- architecture.** Boxes: config -> registry -> adapter ->
  subprocess in library dir -> artifacts -> `BenchmarkResult`; the six
  libraries below; experiments/sweep on top as a consumer. Keep to one
  diagram.
- Evidence: `README.md` (layout, config schema, required-keys table),
  `core/method.py`, `CONTRIBUTING.md` §1-4, §9.

### 3.4 Reproducibility and fairness (~2 pages) -- goal (ii), pre-study §7.3
- Seeding: one top-level `seed` threaded to training (weight init, unroll
  sampling) and to evaluation (instance selection, x0) for every method;
  verified bit-exact across processes for training. State the one known
  gap honestly (neural-net eval problems ~0.1% nondeterminism; irrelevant
  to LASSO).
- Fairness mechanisms, as *mechanisms*: (a) evaluation never draws from
  the mutable global RNG -- a deterministic full pass over every test
  instance for all four Exp-2 methods; (b) one seeded x0 per instance,
  stored with the dataset and consumed by every method's first step/layer
  (for LISTA/ALISTA this required a layer-0 input -- describe the fixed
  `(I-BA)` layer-0 for LISTA so the network keeps exactly T learnable
  layers); (c) identical A, b, x_true across methods by construction.
  Cite the empirical check (byte-identical `x_true`/`b` across methods on
  a real sweep) as verification, without the history around it.
- Provenance per run: exact subprocess command, git commit/branch/status,
  diff sidecar, sha256 of untracked `.py` files, config snapshot, timings.
  Why: most sweep code is uncommitted while an experiment runs.
- Environment: one conda env locally, one Docker image (`open-l2o:latest`,
  TF 2.21 / Keras 3.15, GPU) for the sweeps; note the two differ and that
  all thesis results come from the image.
- Evidence: `README.md` "Seed control", `project_eval_fairness_verification`
  (mechanism part only), `project_todo_seed_reproducibility`,
  `project_docker_gpu_access`, `Dockerfile`.

### 3.5 Running experiments: the sweep driver (~1 page) -- goal (iii)
- `experiments/sweep.py`: grid over axes, resume from `results.jsonl`,
  per-cell error isolation, provenance, seed-aggregated `summary.json`
  with pluggable aggregators, CLI overrides derived from the config
  dataclass. `experiment_template.py`: the four marked places.
- Deliberately outside `core`: experiments are consumers of the platform.
- Evidence: `CONTRIBUTING.md` §11, `experiments/v2/`, commit 45f2f0a.
- Optional: a short listing of the template's `SweepSpec` (8 lines).

### 3.6 Experiment support: data and metrics (~1.5 pages) -- goal (iv)
- LASSO dataset generator (`data/lasso.py`): shared dictionary A per seed,
  Bernoulli-Gaussian x_true, three training-width modes (fixed p / p drawn
  per instance from an interval), per-sparsity test files, seeded x0
  sibling files, ALISTA's analytic W (exact LP). Describe the *knobs*; the
  *values* (Table 3, centring) are ch. 4.
- Metric functions (`core/lasso_metrics.py`): Eq. 9 suboptimality gap,
  Eq. 10 modified relative loss (ratio of expectations), Eq. 11 NMSE vs
  x_true, LASSO-optimal recovery floor NMSE(x*, x_true); f*/x* from a
  restarted FISTA with certified duality gap (mechanism; the 50,000 cap and
  the 2000-iteration discrepancy with the pre-study are ch. 4). Cached by
  hash of (A, b).
- Classical baselines (`classical_baselines.py`): textbook ISTA/FISTA from
  the same x0 on the same test files, scored with the same functions;
  per-iteration R_f from the model-free eval record
  (`plot_compute_matched_rf.py`) -- the machinery that makes the
  matched-budget view in ch. 5 possible.
- Evidence: `project_lasso_generator`, `project_experiment2_metrics`,
  `project_exp2_compute_matched_baselines`.

### 3.7 Limitations (~0.5 page)
- `per_seed` shape not standardised across adapters; suites can mix
  problems; iteration != compute across families; classical-optimizer
  coverage differs per host; only 2 of 9 model-based variants wired; NN
  eval nondeterminism; single-GPU/8 GB ceiling (batch schedule) if Exp 1
  is included.
- Evidence: `README.md` "Known limitations", `CONTRIBUTING.md` §5-6.

## Figures and tables (all buildable now, no GPU)
- Table 3.1 inherited / modified / added (write by hand from §3.1 list).
- Table 3.2 migrated libraries (from `project_structure` + migration memories).
- Figure 3.1 architecture diagram (draw; TikZ or exported PNG into `Figs/`).
- Table 3.3 (optional) required config keys per adapter -- lift from README.

## Decisions (author, 2026-09-16)
1. **Migration depth**: one paragraph in 3.2 + Appendix A table. Table 3.2
   is therefore the appendix table; ch. 3 keeps Table 3.1 only.
2. **Keras-3 API fixes**: omit entirely.
3. **Exp 1 profiling** (`profiling.py`, GPU peak memory): mention in Table
   3.1 only if Exp 1 ends up in the thesis -- still open.

## Drafts (LaTeX, untracked, in `Benchmarking/thesis_ch3/`)
See that directory: one `.tex` per section plus `table_inherited_added.tex`,
`appendix_migrated_libraries.tex`, `fig_architecture.tex` (TikZ).

## Suggested drafting order
3.3 -> 3.4 -> 3.6 (the substance; all evidence is in README/CONTRIBUTING)
-> 3.5 -> 3.1 + Table 3.1 -> 3.2 -> 3.7. Draft each section directly in
LaTeX; the READMEs are already close to prose for 3.3-3.5.

## Status (2026-09-16)
All seven sections + Table 3.1 + Appendix A + Figure 3.1 (TikZ) drafted in
`Benchmarking/thesis_ch3/` (`chapter3.tex` inputs them; header lists the
labels/cite keys to define). Not compiled locally (no LaTeX install);
braces/environments balanced by script. Next: paste into Overleaf, fix
labels/bib keys, read through. Open: Exp 1 profiling row in Table 3.1 (only
if Exp 1 is included).

Stale-doc fix applied: `CONTRIBUTING.md` §1 core size (7 files / ~700 lines).

2026-09-17: §3.4 now cites the automated fairness/determinism suite
(`Benchmarking/tests/`, instances-fingerprint contract in
`core/fingerprint.py`) instead of the one-off manual check; Table 3.1 has a
row for it. See memory `project_fairness_tests`.
