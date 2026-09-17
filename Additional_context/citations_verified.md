# Citation verification — Experiment 2 (LASSO recovery floor)

Checked 2026-09-14 against primary sources. Responds item-by-item to
`citations_to_verify.md`. Empirical statements about the repo's own runs are
out of scope, as requested.

**Result: 14 of 18 items are correct modulo wording. Four have substantive
errors: A2, B2, D2, E1. One of them (B2) changes a number.**

---

## Verdict summary

| Item | Verdict | What to change |
|---|---|---|
| A1 | OK | Note it's framed as error-correction, not compressed sensing |
| A2 | **Claim wrong** | Donoho 2006 does not state O(k log(n/k)) — cite A4 instead |
| A3 | OK | — |
| A4 | OK | Paper states the bound in *inverted* form |
| A5 | OK | Use `ln(eN/s)`, not `ln(N/s)`; confirm theorem number in the book |
| A6 | Overstated | Inverted form is a published theorem, not a heuristic |
| B1 | Wrong paper | Cite Donoho & Tanner JAMS 2009 for the weak curve |
| B2 | **Number wrong** | ρ_W(0.7) ≈ **0.50**, not 0.55 |
| B3 | OK | Better citation exists (Donoho–Maleki–Montanari 2011) |
| C1 | OK but misapplied | Noisy model, sublinear sparsity — probably not your regime |
| D1 | OK | — |
| D2 | **Partly wrong** | Fan & Li did not say LASSO lacks the oracle property |
| D3 | OK | λ ≠ λ/L confirmed numerically; citation needs a check |
| D4 | OK | "smaller bias", not "removes the bias" |
| E1 | **Motivation wrong** | Motivated by "kicking", not false-positive rate |
| E2 | OK | ss confirmed in reference implementation, p^k = min(1.2k, 13) |
| E3 | Right conclusion, wrong mechanism | It's θ^k → 0, and the paper states it |
| F1 | OK | Quoted sentence is stronger than the paper's |

---

## A. Compressed-sensing sample complexity

### A1 — Candès & Tao (2005), "Decoding by Linear Programming" — **confirmed**

IEEE Trans. Inf. Theory **51**(12), 4203–4215, December 2005.
DOI `10.1109/TIT.2005.858979`

RIP *is* introduced here — §1.D, under the name "restricted isometry numbers"
δ_S. Baraniuk et al. (A4) confirm the attribution explicitly in their §3. Not
the 2006 "Near-optimal signal recovery from random projections" paper.

Two caveats worth knowing:

- The paper is framed as an **error-correcting problem**: recover input f from
  corrupted y = Af + e where e is sparse. It is not framed as compressed
  sensing. The ℓ₁-recovery-of-sparse-signals reading is equivalent but indirect.
- The Gaussian result (Thm. 1.6, Cor. 1.7) concerns a matrix F with i.i.d.
  N(0, 1/p) entries acting as the annihilator of A.

### A2 — Donoho (2006), "Compressed Sensing" — **reference OK, claim wrong**

IEEE Trans. Inf. Theory **52**(4), 1289–1306, April 2006.
DOI `10.1109/TIT.2006.871582`

Correct as one of the two founding papers. **It does not contain the
O(k log(n/k)) statement.** The paper's machinery is Gelfand n-widths, optimal
recovery and information-based complexity; the abstract's headline
sample-complexity claim is n = O(m^{1/4} log^{5/2} m) for specific image
classes. Its listed keywords are "Optimal Recovery, Information-Based
Complexity, Gel'fand n-widths, Basis Pursuit."

Note also that **Donoho's notation is reversed from yours**: m = signal
dimension, n = number of measurements.

**Action:** cite A4 for the k log(n/k) bound; keep A2 only as a founding
reference.

### A3 — Candès, Romberg & Tao (2006) — **confirmed**

IEEE Trans. Inf. Theory **52**(2), 489–509, February 2006.
DOI `10.1109/TIT.2005.862083`

Exact statement from the abstract: for f a superposition of |T| spikes obeying

    |T| ≤ C_M · (log N)^{-1} · |Ω|

f is reconstructed exactly by ℓ₁ minimisation with probability ≥ 1 − O(N^{-M}).
Rearranged, |Ω| ≳ C |T| log N — so O(k log n) measurements, confirmed. And they
are a **randomly chosen set Ω of Fourier frequencies** (partial DFT), not
Gaussian. Both parts of the attributed claim hold.

### A4 — Baraniuk, Davenport, DeVore & Wakin (2008) — **confirmed**

Constructive Approximation **28**(3), 253–263, December 2008.
DOI `10.1007/s00365-007-9003-x`

Read in full. **Theorem 5.2**, stated in the *inverted* form:

> Suppose that n, N, and 0 < δ < 1 are given. If the probability distribution
> generating the n × N matrices Φ(ω) satisfies the concentration inequality
> (4.3), then there exist constants c₁, c₂ > 0 depending only on δ such that the
> RIP (3.8) holds for Φ(ω) with the prescribed δ and any
> **k ≤ c₁ n / log(N/k)** with probability ≥ 1 − 2e^{−c₂n}.

Notation map: their `n` = your `m` (rows/measurements), their `N` = your `n`
(columns/ambient). So in your notation: k ≤ c₁ m / log(n/k), equivalently
m ≥ (k/c₁) log(n/k).

On the `log(N/k)` vs `log(eN/k)` question: the **theorem** is stated with
`log(N/k)`. The `e` appears only inside the proof, via the binomial bound
C(N,k) ≤ (eN/k)^k. The paper also remarks that validity for
k ≤ c₁n/log(N/k) implies validity for k ≤ c₁′ n/[log(N/n) + 1].

Their Gaussian ensemble (4.4) is φ_ij ~ N(0, 1/n) with n = number of rows —
**exactly your A**. Section 3 also notes that with RIP of order 3k, ℓ₁ decoding
achieves best-possible n-width performance precisely when k ~ n/(log(N/n)+1).

### A5 — the standard textbook formulation — **confirmed, with a correction**

Foucart & Rauhut, *A Mathematical Introduction to Compressive Sensing*,
Birkhäuser/Springer, 2013. **Chapter 9, "Sparse Recovery with Random
Matrices", §9.1 "Restricted Isometry Property for Subgaussian Matrices",
Theorem 9.2.**

⚠️ I could not open the book to confirm the theorem *number* — verify 9.2
against your copy before citing it. The chapter and section titles are
confirmed from the published table of contents.

The statement is confirmed in equivalent form from two open sources:

- Foucart, *Flavors of Compressive Sensing*, Theorem 1 — A ∈ ℝ^{m×N} with
  independent Gaussian entries of mean zero and variance 1/m; if
  **m ≥ C δ^{-2} s ln(eN/s)** then with probability ≥ 1 − 2exp(−cδ²m) the matrix
  satisfies RIP of order s with constant δ. *This matches your setup exactly
  (variance 1/m, unit-norm columns).*
- Krahmer, Mendelson & Rauhut, *Suprema of Chaos Processes and the RIP*,
  Thm. A.1 — m ≥ Cδ^{-2} max{s log(en/s), log(ε^{-1})}.

**Use `ln(eN/s)`.** The `e` matters: without it the bound degenerates as
s → N, whereas `ln(eN/s)` stays valid across the whole range.

### A6 — "the inverted form is only a heuristic" — **too dismissive**

k ≲ m / log(n/k) is literally how Baraniuk et al. state Theorem 5.2 (A4), so
it is a published theorem, not folklore.

The correct objection is different and stronger: **the constants c₁, c₂ are
unspecified and known to be badly conservative in the proportional-sparsity
regime.** Donoho & Tanner (*Precise Undersampling Theorems*, §X and Fig. 6)
show that the best available RIP-based bounds for Gaussian matrices sit far
below the true polytope threshold, and comment that the RIP approach is
"seemingly not precise". Blanchard, Cartis & Tanner computed the sharpest known
RIP constants and found the same gap.

So: yes, the phase transition (item B) is the right tool in your regime. But
frame it as *"RIP bounds are not sharp at proportional sparsity"* rather than
*"the inverted form is a heuristic"*.

---

## B. Donoho–Tanner phase transition

### B1 — which paper — **cite a different one**

For the **weak threshold on signed signals**, the derivation is in:

> Donoho, D. L. & Tanner, J. (2009). "Counting faces of randomly projected
> polytopes when the projection radically lowers dimension."
> *Journal of the American Mathematical Society* **22**(1), 1–53.

The two papers listed in the original note are the wrong ones for this purpose:

- Phil. Trans. R. Soc. A **367**(1906), 4273–4293 (2009) — this is the
  **universality** paper (empirical, across Gaussian / Fourier / Hadamard /
  Rademacher / Bernoulli / ternary / expander ensembles). Cite it for
  universality, not for the curve.
- PNAS 2005, "Sparse nonnegative solution…" — this is the **nonnegative** case.

**Signed vs nonnegative matters.** From *Precise Undersampling Theorems* §III:

| Signal model | Algorithm | Polytope | Curve |
|---|---|---|---|
| signed k-sparse | (P1) ℓ₁ min | cross-polytope C | ρ(δ; **C**) |
| nonnegative k-sparse | (LP) | simplex T | ρ(δ; T) — strictly higher |
| k-simple in [0,1] | (Feas) | hypercube I | ρ(δ; I) = max(0, 2−δ⁻¹) |

**You want ρ(δ; C).** Coordinates match yours: δ = m/n, ρ = k/m.

Weak vs strong: weak = holds for *most* k-sparse vectors (typical signals);
strong = holds for *all* k-sparse vectors. ρ(δ; Q, strong) < ρ(δ; Q, weak).
You want weak.

For a single readable citation covering all of it: Donoho & Tanner (2010),
"Precise Undersampling Theorems", *Proceedings of the IEEE* **98**(6), 913–924,
DOI `10.1109/JPROC.2010.2045630`.

### B2 — numerical values — **ρ_W(0.7) is wrong**

Rather than read a figure, I computed ρ_W(δ) from the published closed form.

**Method.** Amelunxen, Lotz, McCoy & Tropp, "Living on the Edge: Phase
Transitions in Convex Programs with Random Data", *Information and Inference*
**3**(3), 224–294 (2014), **Proposition 4.5**: the normalised statistical
dimension of the descent cone of the ℓ₁ norm at a vector with fraction ρ of
nonzeros is

    ψ(ρ) = inf_{τ≥0} { ρ(1+τ²) + (1−ρ) ∫_τ^∞ (u−τ)² φ(u) du },
    φ(u) = √(2/π) e^{−u²/2}  supported on [0,∞)

with the transition at m = d·ψ(k/d) (their Theorem II). Their **Remark 4.6**
states this reproduces the Donoho–Tanner weak threshold exactly, and Bayati,
Lelarge & Montanari proved the transition occurs there asymptotically.

Converting to Donoho–Tanner coordinates: with ε = k/n, δ = ψ(ε) and
ρ_W = ε/ψ(ε).

**Computed weak threshold ρ_W(δ) for signed signals, Gaussian A:**

| δ = m/n | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | **0.50** | 0.60 | **0.70** | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ρ_W = k/m | 0.154 | 0.189 | 0.243 | 0.291 | 0.337 | **0.3857** | 0.438 | **0.4988** | 0.573 | 0.678 | 0.760 |
| k/n | 0.008 | 0.019 | 0.049 | 0.087 | 0.135 | 0.193 | 0.263 | 0.349 | 0.459 | 0.610 | 0.722 |

- **ρ_W(0.5) = 0.3857** — the remembered ≈0.39 was right. For m = 25 this gives
  k* ≈ 9.6.
- **ρ_W(0.7) = 0.4988** — the remembered ≈0.55 is **wrong**. Correct value
  ≈ 0.50. For m = 210 this gives k* ≈ 105.
- **ρ_W → 1 as δ → 1** — confirmed: 0.916 at δ=0.995, 0.988 at δ=0.9999,
  0.996 at δ=0.99999.

**Validation.** The implementation reproduces the published asymptote
ρ(δ; C) ~ [2 log(1/δ)]^{-1} as δ → 0 (Theorem 3 of *Precise Undersampling
Theorems*): at δ = 10⁻⁶, computed 0.0373 vs asymptote 0.0362; at δ = 0.005,
0.0941 vs 0.0944.

**At your two configurations:**

| (m, n) | δ | ρ_W | k* = ρ_W·m | transition at k/n |
|---|---|---|---|---|
| (25, 50) | 0.500 | 0.3857 | **9.6** | 0.193 |
| (210, 300) | 0.700 | 0.4988 | **104.8** | **0.349** |

### B3 — LASSO at the basis-pursuit transition — **fair, better citation available**

The best citation is **not** Bayati & Montanari, it is:

> Donoho, D. L., Maleki, A. & Montanari, A. (2011). "The Noise-Sensitivity
> Phase Transition in Compressed Sensing."
> *IEEE Trans. Inf. Theory* **57**(10), 6920–6941.

Their abstract states directly that for x̂^{1,λ} solving
min ‖y−Ax‖²/2 + λ‖x‖₁ with i.i.d. Gaussian A in the large-system limit
(n/N → δ, k/n → ρ), the phase boundary ρ = ρ_MSE(δ) is *identical to* the
previously-known ℓ₁–ℓ₀ equivalence transition in the noiseless k-sparse case —
"a single phase boundary describes the fundamental phase transitions both for
the noiseless and noisy cases."

**Caveat to state explicitly in the draft.** Their result concerns *noise
sensitivity* (MSE/σ² bounded vs unbounded) with λ optimally tuned. You have
noiseless data and a fixed small λ. The bridge is the standard fact that the
LASSO minimiser → the basis-pursuit solution as λ → 0. Write one sentence
making that step rather than asserting equivalence.

Bayati & Montanari, "The LASSO Risk for Gaussian Matrices", *IEEE TIT*
**58**(4), 1997–2017 (2012) is a real and relevant paper, but it characterises
the LASSO *risk*, not the transition location.

---

## C. LASSO support-recovery thresholds

### C1 — Wainwright (2009) — **formula confirmed, but likely the wrong tool here**

> Wainwright, M. J. (2009). "Sharp Thresholds for High-Dimensional and Noisy
> Sparsity Recovery Using ℓ₁-Constrained Quadratic Programming (Lasso)."
> *IEEE Trans. Inf. Theory* **55**(5), 2183–2202. DOI `10.1109/TIT.2009.2016018`

Everything attributed is correct:

- For a broad class of Gaussian ensembles satisfying mutual incoherence
  conditions, there exist thresholds 0 < θ_ℓ ≤ 1 ≤ θ_u < ∞ such that for any
  δ > 0: n > 2(θ_u + δ)k log(p − k) ⟹ success w.p. → 1;
  n < 2(θ_ℓ − δ)k log(p − k) ⟹ failure w.p. → 1.
- For the **uniform** Gaussian ensemble (Σ = I), θ_ℓ = θ_u = 1, so
  "the precise threshold n = 2k log(p − k) is exactly determined."
- The constant **2** and the argument **log(p − k)** are both right.
- The arXiv version (math/0605740) carries the extra additive terms:
  n > 2(θ_u + ν) s log(p − s) + s + 1.
- Conditions: mutual incoherence on Σ, a lower bound on the minimum nonzero
  magnitude, and a prescribed choice of λ_n. The property is **signed support
  recovery**, not ℓ₂ accuracy.

### ⚠️ Why this probably does not belong in Experiment 2

Two mismatches with your setup:

1. **Noise.** Wainwright's model is y = Xβ* + w with sub-Gaussian noise. Your
   samples are noiseless. Without noise there is nothing to confuse the
   support, so this is not the binding constraint.
2. **Sparsity regime.** n = 2k log(p−k) lives in the sublinear regime
   k = o(n). Your sweep is proportional sparsity (k ∝ n), which is
   Donoho–Tanner territory. Donoho & Tanner make this contrast themselves in
   §IX of *Precise Undersampling Theorems*, noting that Wainwright's boundary
   "does not allow the sparsity control parameter k to be of size c·n".

Concretely at (m, n) = (25, 50), k = 5: the Wainwright threshold demands
n > 2·5·log(45) ≈ 38 measurements. You have 25, and recovery works fine.

**Recommendation:** either drop it, or keep one sentence contrasting
*exact signed-support recovery under noise* against *typical ℓ₁–ℓ₀
equivalence*, to show the distinction is deliberate.

---

## D. LASSO shrinkage bias

### D1 — Tibshirani (1996) — **confirmed**

*Journal of the Royal Statistical Society, Series B* **58**(1), 267–288.
Original LASSO paper; continuous shrinkage toward zero is its defining
property ("least absolute shrinkage and selection operator").

### D2 — Fan & Li (2001) — **reference correct, parenthetical wrong**

*Journal of the American Statistical Association* **96**(456), 1348–1360.
MR1946581.

**Confirmed:** they advocate penalties yielding three properties —
unbiasedness, sparsity, continuity. Unbiasedness requires p′_λ(t) → 0 for
large t. The ℓ₁ penalty has p′_λ(t) = λ for all t > 0, so it over-penalises
large coefficients and yields a soft-thresholding rule that is biased. This
motivates SCAD, whose derivative is

    p′_λ(t) = λ [ I(t ≤ λ) + (aλ − t)₊ / ((a−1)λ) · I(t > λ) ],   a = 3.7

They also *introduce* the oracle property and prove SCAD attains it.

❌ **They did not say "LASSO lacks the oracle property."** That claim belongs
to:

> Zou, H. (2006). "The Adaptive Lasso and Its Oracle Properties."
> *JASA* **101**(476), 1418–1429.

with the supporting irrepresentable-condition results in Zhao & Yu (2006) and
Meinshausen & Bühlmann (2006).

**Action:** split the citation. Fan & Li for *bias on large coefficients*;
Zou for *lack of the oracle property*.

### D3 — the KKT fixed point — **formula correct; derive it, don't cite a book**

The formula is right and takes two lines from stationarity. With b = A_S x_S
(noiseless, x_true supported on S), correct support and preserved signs:

    A_Sᵀ(A_S x*_S − b) + λ sign(x*_S) = 0
    A_SᵀA_S (x*_S − x_S) + λ sign(x_S) = 0
    ⟹  x*_S = x_S − λ (A_SᵀA_S)^{-1} sign(x_S)          ✓

This is the primal-dual witness construction — published in **Wainwright
(2009) §3**. Hastie, Tibshirani & Wainwright, *Statistical Learning with
Sparsity* (CRC, 2015) covers it in **Ch. 11, "Theoretical Results for the
Lasso"**; ⚠️ I could not confirm the subsection number from open sources —
check the PDF before citing one. Given how short the derivation is, deriving
it inline is cleaner than citing a textbook.

Better citation for the *magnitude* claim: Chen, Liu, Wang & Yin (2018) state
for the noiseless LASSO solution x̄(λ) that **‖x̄(λ) − x*‖ = O(λ)** — exactly
the "≈ λ per nonzero" scaling, in a paper you already cite.

#### λ vs λ/L — **confirmed different**

Numerically, at your actual parameters (λ = 0.005, A with i.i.d. N(0,1/m)
entries, unit-norm columns, bias = λ(A_SᵀA_S)^{-1}sign(x_S)):

| (m, n) | L = λ_max(AᵀA) | λ/L | mean \|bias\| k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| (25, 50) | 5.65 | 0.00089 | 0.0047 | 0.0056 | 0.0084 |
| (210, 300) | 4.56 | 0.0011 | 0.0050 | 0.0058 | 0.0044 |

**λ/L is ~4.5–5.6× smaller than λ here.** λ/L is the per-iteration
soft-threshold in ISTA (the Primer uses step 1/L and threshold λ/L); the bias
of the LASSO *solution* is ≈ λ per nonzero at low k, drifting above λ as k
grows and (A_SᵀA_S)^{-1} departs from the identity. Two different quantities —
the original note's concern was justified.

### D4 — Belloni & Chernozhukov (2013) — **confirmed, soften the wording**

*Bernoulli* **19**(2), 521–547. DOI `10.3150/11-BEJ410`. arXiv:1001.0188.

Abstract: OLS post-Lasso "performs at least as well as Lasso in terms of the
rate of convergence, and has the advantage of a smaller bias."

Two wording corrections:

- **"smaller bias", not "removes the shrinkage bias".**
- The setting is nonparametric regression with approximate sparsity, and the
  striking result is that this holds *even when Lasso-based model selection
  "fails"* in the sense of missing components. Post-Lasso can be strictly
  faster when selection includes all true components plus sufficient sparsity.

---

## E. Support selection in unrolled ISTA

### E1 — Chen, Liu, Wang & Yin (2018), LISTA-CPSS — **mechanism right, motivation wrong**

NeurIPS 2018 (spotlight), pp. 9061–9071. arXiv:1808.10038.
Note: **Chen and Liu are co-first authors, listed alphabetically.**

**Confirmed** — support selection is introduced here, §2.2. At each layer,
before soft-thresholding, the p^k% largest-magnitude entries are trusted as
"true support" and passed through *without* thresholding; the rest are
soft-thresholded normally. Formally (their eq. 11–12), with S_{p^k}(v) the
index set of the p^k% largest magnitudes:

    (η^ss_{p^k, θ^k}(v))_i = v_i                 if |v_i| > θ^k and i ∈ S_{p^k}(v)
                           = v_i − θ^k sign(v_i) if |v_i| > θ^k and i ∉ S_{p^k}(v)
                           = 0                   if |v_i| ≤ θ^k

**Schedule:** p^k = min(p·k, p_max), both hand-tuned hyperparameters.
Simulation values: p = 1.2, p_max = 12 (LISTA-SS) or 13 (LISTA-CPSS), at
p_b = 0.1, K = 16 layers. Image CS experiment: p% = 0.4% per layer, selected
across the whole batch rather than per sample.

❌ **The motivation attributed to them is not in the paper.** They credit
"kicking" in linearized Bregman iteration (Osher, Mao, Dong & Yin, 2010) and
claim advantages on *recoverability and convergence rate*. There is no
false-positive-rate framing anywhere in the paper.

**Action:** delete the false-positive clause; say "inspired by kicking in
linearized Bregman iteration" instead.

Useful for you: their simulation setup is m = 250, n = 500, A_ij ~ N(0, 1/m)
with columns normalised to unit ℓ₂ norm, support Bernoulli(p_b = 0.1),
nonzeros N(0,1), 1000 test samples, NMSE in dB. **Nearly identical to your
Experiment 2.**

### E2 — Liu, Chen, Wang & Yin (2019), ALISTA — **confirmed**

ICLR 2019, OpenReview `B1lnzn0ctQ`.
Author order: Jialin Liu*, Xiaohan Chen*, Zhangyang Wang, Wotao Yin
(* co-first, alphabetical).

- **Analytic weights:** W is the solution to a **data-free optimisation problem
  based on coherence minimisation**, leaving only stepsize and threshold to
  data-driven learning. "Minimising mutual coherence with A" is close but
  loose — it is the *generalised* mutual coherence between W and A subject to a
  normalisation constraint. Worth saying precisely.
- **Support selection confirmed.** The reference implementation's own
  invocation carries `-p 1.2 -maxp 13` at `--M 250 --N 500 --pnz 0.1` —
  i.e. the identical p^k = min(1.2k, 13) schedule inherited from LISTA-CPSS.
- Caveat: ALISTA's headline theorem (retaining the optimal linear convergence
  of Chen et al. 2018) is proved for the plain ALISTA model. Support selection
  is an experimental add-on, not part of that theorem.

### E3 — "beats x* on the true support" — **right conclusion, wrong mechanism, and it IS in the paper**

This is the biggest upgrade available. The reasoning is **stated**, not merely
inferable — but via a different mechanism.

In the **Discussion after Theorem 2** of Chen et al. (2018), for the noiseless
LASSO solution x̄(λ):

    x^k → x̄(λ) sublinearly,  ‖x̄(λ) − x*‖ = O(λ),  λ > 0
    x^k → x̄(λ) linearly,     ‖x̄(λ) − x*‖ = O(λ),  λ large enough

— an inherent trade-off: larger λ gives faster convergence and a less accurate
solution. Then the key move: **LISTA's per-layer thresholds {θ^k} correspond
to a path of LASSO parameters {λ^k}**, and their **Theorem 1** proves that
θ^k → 0 is *necessary* for convergence to x*. **Theorem 2** then gives, in the
noiseless case,

    ‖x^k(x*, 0) − x*‖₂ ≤ sB exp(−ck) → 0

which is strictly below the O(λ) floor of any fixed-λ LASSO minimiser.

So: **the network escapes the LASSO bias primarily by annealing its threshold
to zero, not by support selection.** Support selection improves the rate
(c̃^k_ss > c for large k) and the noise constant (C̃_ss < C) *on top of* that —
Theorem 3.

**Action:** rewrite E3 around θ^k → 0 (Theorem 1) plus the O(λ) LASSO floor
(Discussion after Theorem 2), citing Theorems 1–3. That converts the strongest
interpretive claim in the experiment into a cited theorem, and it explains
something the benchmark will actually measure.

---

## F. Primer / benchmark cross-check

### F1 — Chen et al. (2022), "Learning to Optimize: A Primer and A Benchmark"

*JMLR* **23**(189), 1–59. Checked directly against the copy in the project,
§4.1.2.

**(a) Dimensions — confirmed**, (m,n) = (5,10) and (25,50).

⚠️ **But the quoted sentence is stronger than the paper.** The actual text
says all the sub-gradient descent **and** model-free L2O methods perform
poorly at (25,50), then immediately qualifies: *only L2O-RNNprop can converge
faster than ISTA and comparably to FISTA, though reaching lower precision.*
Writing "all the model-free L2O methods perform poorly" unqualified will get
flagged by anyone who knows the paper.

**(b) Confirmed.** 1,000 iterations for both model-free L2O methods and
classical optimizers; both axes logarithmic. Model-based (LISTA, ALISTA) run
at 16 layers, optionally extended by appending FISTA iterations at test time.

**(c) Confirmed, near-verbatim.** They did not go larger due to the "high
memory cost of LSTM-based model-free L2O methods."

**(d) Confirmed.** ISTA and FISTA, both with step size 1/L (L = largest
eigenvalue of AᵀA) and soft-threshold λ/L, at matched 1,000 iterations.
GD / Adam / RMSProp also appear as "natural baselines" at step size 10⁻³.

### The y-axis metric — get this exactly right

It is their **Eq. (21), the "modified relative loss"**:

    R_{f,Q}(x) = E_{q~Q}[ f_q(x) − f*_q ] / E_{q~Q}[ f*_q ]

Two details that are easy to get wrong and that matter, since this metric is
reused in both of your experiments:

- It is a **ratio of expectations**, not the expectation of a ratio.
- **f*_q is generated by 2,000 iterations of FISTA**, not by an exact solver.

Also: samples are **noiseless**; 12,800 training pairs, 1,280 for validation
and test; model-based L2O is trained on the **LASSO loss** in this test (not
MSE against the true solution, as in §4.1.1).

---

## What actually changes in the pre-study

Three things worth acting on, in order of value.

**1. The ρ_W correction gives you a citable experiment design.**
With ρ_W(0.7) = 0.499, the transition for (210, 300) sits at k/n ≈ **0.349**.
Your planned test sparsity list is p ∈ [0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
0.35, 0.40], so:

- p = 0.30 → just below the transition
- p = 0.35 → **essentially on it**
- p = 0.40 → above it

That is a well-placed sweep, and you can now justify it with a citation instead
of intuition. It also predicts where the LASSO-optimal recovery error should
start climbing — which is exactly the diagnostic the design calls for. Add the
finite-size caveat: at n = 300 the transition has O(√n) width in k
(Donoho & Tanner, *Precise Undersampling Theorems*, §II), so expect a ramp,
not a step.

**2. Decide whether Wainwright belongs at all.** My read: no for Experiment 2
(wrong noise model, wrong sparsity regime). It may earn one sentence
contrasting support recovery against ℓ₁–ℓ₀ equivalence, if you want to show
the distinction is deliberate rather than overlooked. Your call.

**3. The E3 rewrite.** Reframing around θ^k → 0 turns the strongest
interpretive claim in the experiment into a cited theorem from a paper already
in the bibliography, and it gives a mechanism the benchmark can actually
instrument (log the learned θ^k per layer alongside NMSE).

---

## Reproducing the ρ_W table

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq

def psi(rho):
    """Normalised statistical dimension of the descent cone of the l1 norm
    at a vector with fraction rho of nonzeros.
    Amelunxen, Lotz, McCoy & Tropp (2014), Proposition 4.5."""
    def f(tau):
        tail = (1 + tau**2) * norm.sf(tau) - tau * norm.pdf(tau)
        return rho * (1 + tau**2) + 2 * (1 - rho) * tail
    return minimize_scalar(f, bounds=(0, 40), method='bounded',
                           options={'xatol': 1e-14}).fun

def rho_W(delta):
    """Donoho-Tanner weak threshold rho(delta; C), signed signals, Gaussian A.
    delta = m/n, returns rho = k/m."""
    eps = brentq(lambda e: psi(e) - delta, 1e-12, 0.999999, xtol=1e-14)
    return eps / delta

# rho_W(0.5) -> 0.3857 ;  rho_W(0.7) -> 0.4988
```

The `∫_τ^∞ (u−τ)²φ(u)du` term is evaluated in closed form as
`(1+τ²)(1−Φ(τ)) − τφ(τ)`, doubled because ALMT's kernel
φ(u) = √(2/π)e^{−u²/2} is twice the standard normal density on [0,∞).

---

## Full reference list (verified)

- Amelunxen, D., Lotz, M., McCoy, M. B. & Tropp, J. A. (2014). Living on the edge: phase transitions in convex programs with random data. *Information and Inference* **3**(3), 224–294. DOI 10.1093/imaiai/iau005. arXiv:1303.6672
- Baraniuk, R., Davenport, M., DeVore, R. & Wakin, M. (2008). A simple proof of the restricted isometry property for random matrices. *Constructive Approximation* **28**(3), 253–263. DOI 10.1007/s00365-007-9003-x
- Bayati, M. & Montanari, A. (2012). The LASSO risk for Gaussian matrices. *IEEE Trans. Inf. Theory* **58**(4), 1997–2017.
- Belloni, A. & Chernozhukov, V. (2013). Least squares after model selection in high-dimensional sparse models. *Bernoulli* **19**(2), 521–547. DOI 10.3150/11-BEJ410
- Candès, E. J., Romberg, J. & Tao, T. (2006). Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information. *IEEE Trans. Inf. Theory* **52**(2), 489–509. DOI 10.1109/TIT.2005.862083
- Candès, E. J. & Tao, T. (2005). Decoding by linear programming. *IEEE Trans. Inf. Theory* **51**(12), 4203–4215. DOI 10.1109/TIT.2005.858979
- Chen, T., Chen, X., Chen, W., Heaton, H., Liu, J., Wang, Z. & Yin, W. (2022). Learning to Optimize: A Primer and A Benchmark. *JMLR* **23**(189), 1–59.
- Chen, X., Liu, J., Wang, Z. & Yin, W. (2018). Theoretical linear convergence of unfolded ISTA and its practical weights and thresholds. *NeurIPS 31*, 9061–9071. arXiv:1808.10038
- Donoho, D. L. (2006). Compressed sensing. *IEEE Trans. Inf. Theory* **52**(4), 1289–1306. DOI 10.1109/TIT.2006.871582
- Donoho, D. L., Maleki, A. & Montanari, A. (2011). The noise-sensitivity phase transition in compressed sensing. *IEEE Trans. Inf. Theory* **57**(10), 6920–6941. arXiv:1004.1218
- Donoho, D. L. & Tanner, J. (2009). Counting faces of randomly projected polytopes when the projection radically lowers dimension. *J. Amer. Math. Soc.* **22**(1), 1–53.
- Donoho, D. L. & Tanner, J. (2009). Observed universality of phase transitions in high-dimensional geometry… *Phil. Trans. R. Soc. A* **367**(1906), 4273–4293.
- Donoho, D. L. & Tanner, J. (2010). Precise undersampling theorems. *Proc. IEEE* **98**(6), 913–924. DOI 10.1109/JPROC.2010.2045630
- Fan, J. & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *JASA* **96**(456), 1348–1360.
- Foucart, S. & Rauhut, H. (2013). *A Mathematical Introduction to Compressive Sensing.* Birkhäuser/Springer. Ch. 9, §9.1, Thm. 9.2 *(theorem number unverified)*.
- Liu, J., Chen, X., Wang, Z. & Yin, W. (2019). ALISTA: Analytic weights are as good as learned weights in LISTA. *ICLR 2019.*
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B* **58**(1), 267–288.
- Wainwright, M. J. (2009). Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ₁-constrained quadratic programming (Lasso). *IEEE Trans. Inf. Theory* **55**(5), 2183–2202. DOI 10.1109/TIT.2009.2016018
- Zou, H. (2006). The adaptive lasso and its oracle properties. *JASA* **101**(476), 1418–1429.
