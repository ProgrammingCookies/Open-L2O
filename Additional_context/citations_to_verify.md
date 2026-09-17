# Citations and theoretical statements to verify

Context: these statements were made from memory (by Claude, 2026-09-14) while
explaining the "LASSO-optimal recovery floor" in the thesis's Experiment 2
(LASSO, dictionary A with i.i.d. N(0, 1/m) entries and unit-norm columns,
lambda = 0.005, dims (m, n) = (25, 50) and (210, 300), signal with exactly
k = round(p * n) nonzeros drawn N(0, 1) on a uniformly random support).
None of them were checked against a primary source. For each item, please:

1. confirm the reference exists with the stated authors / venue / year (give
   the DOI or arXiv id), or correct it;
2. confirm the paper actually contains the claim attributed to it, and quote
   or paraphrase the precise statement (including any conditions such as
   Gaussian matrices, noiseless vs noisy, exact vs approximate recovery);
3. flag anything that is folklore / commonly repeated but not in that paper.

Only web-verifiable facts are asked for here. Empirical statements about
this repo's own runs are NOT in scope (they were verified on the data).

---

## A. Compressed-sensing sample complexity and the m >= C k log(n/k) bound

**A1.** Candes, E. J. and Tao, T. (2005). "Decoding by linear programming."
IEEE Transactions on Information Theory 51(12).
Claim attributed: introduces the Restricted Isometry Property (RIP) and shows
that l1 minimisation recovers k-sparse signals exactly when A satisfies RIP
of suitable order. Verify: year, venue, and that RIP is introduced here (not
in a different Candes-Tao paper such as "Near-optimal signal recovery from
random projections", 2006).

**A2.** Donoho, D. L. (2006). "Compressed sensing." IEEE Transactions on
Information Theory 52(4).
Claim attributed: one of the two founding compressed-sensing papers; states
that O(k log(n/k)) random measurements suffice for l1 recovery of k-sparse
signals. Verify the exact form of the sample-complexity statement in this
paper (it may be phrased via n log(m) or via "n-widths", not k log(n/k)).

**A3.** Candes, E. J., Romberg, J. and Tao, T. (2006). "Robust uncertainty
principles: exact signal reconstruction from highly incomplete frequency
information." IEEE Trans. Inf. Theory 52(2).
Claim attributed: exact l1 recovery from O(k log n) random Fourier
measurements. Verify the measurement-count statement and that it is Fourier
(partial DFT), not Gaussian, measurements.

**A4.** Baraniuk, R., Davenport, M., DeVore, R. and Wakin, M. (2008). "A
simple proof of the restricted isometry property for random matrices."
Constructive Approximation 28(3).
Claim attributed: for i.i.d. sub-Gaussian (incl. Gaussian) matrices, RIP of
order k holds with high probability when m >= C k log(n/k). Verify that this
is the stated bound and its form (some sources write m >= C k log(n/k),
others C k log(en/k)).

**A5. Statement to check (no single source):** "LASSO / l1 minimisation
recovers the support reliably only while k is well below m; a commonly quoted
sufficient condition is m >= C k log(n/k) for Gaussian A." Is this the
standard textbook formulation (e.g. Foucart & Rauhut, *A Mathematical
Introduction to Compressive Sensing*, 2013, Ch. 9)? Please give the
chapter/theorem number if so.

**A6. Statement to check:** the inverted form "k <~ m / log(n/k)" is only a
heuristic and is not meaningful when n/k is small (e.g. n/k ~ 3, where
log(n/k) ~ 1). Is this a fair characterisation, and is the phase-transition
description (item B) the standard tool in that regime?

## B. Donoho-Tanner phase transition

**B1.** Donoho, D. L. and Tanner, J. (2009). "Observed universality of phase
transitions in high-dimensional geometry, with implications for modern data
analysis and signal processing." Philosophical Transactions of the Royal
Society A 367(1906). (There is also Donoho & Tanner, PNAS 2005, "Sparse
nonnegative solution of underdetermined linear equations by linear
programming" and PNAS 2009 "Counting faces of randomly projected polytopes
when the projection radically lowers dimension".)
Claim attributed: for Gaussian A, l1 minimisation exhibits a sharp phase
transition in the (delta, rho) plane, delta = m/n, rho = k/m, with a "weak"
threshold curve rho_W(delta) below which recovery succeeds for typical
signals. Verify which paper is the right one to cite for the *weak*
threshold curve, and whether it applies to signed (not just nonnegative)
signals.

**B2. Numerical values to check** (quoted from memory, may be wrong):
- rho_W(delta = 0.5) ~ 0.39 (i.e. k ~ 0.39 m; for m = 25 that is k ~ 10).
- rho_W(delta = 0.7) ~ 0.55.
- rho_W(delta) -> 1 as delta -> 1.
Please read the values off a published table or figure of the weak threshold
(Donoho-Tanner tabulate it) and report them, with the source.

**B3. Statement to check:** the Donoho-Tanner transition is derived for
noiseless l1 minimisation (basis pursuit), and LASSO with a small lambda on
noiseless data is expected to sit at essentially the same transition. Is
that a fair statement, or is there a specific result relating the LASSO
(penalised) transition to the basis-pursuit one? (Possibly relevant: Bayati
& Montanari 2012 "The LASSO risk for Gaussian matrices", IEEE TIT 58(4).)

## C. LASSO support-recovery thresholds

**C1.** Wainwright, M. J. (2009). "Sharp thresholds for high-dimensional and
noisy sparsity recovery using l1-constrained quadratic programming (Lasso)."
IEEE Transactions on Information Theory 55(5).
Claim attributed: for Gaussian designs, exact support recovery by the LASSO
undergoes a sharp threshold at n = 2 k log(p - k) (in that paper's notation:
n = number of observations, p = dimension, k = sparsity), with recovery
succeeding above and failing below, under conditions on the minimum
nonzero magnitude and the regularisation parameter. Verify the constant (2),
the argument of the log (p - k vs p), and the conditions.

## D. LASSO shrinkage bias

**D1.** Tibshirani, R. (1996). "Regression shrinkage and selection via the
lasso." Journal of the Royal Statistical Society B 58(1).
Claim attributed: original LASSO paper; notes that the l1 penalty shrinks
coefficient estimates toward zero. Verify.

**D2.** Fan, J. and Li, R. (2001). "Variable selection via nonconcave
penalized likelihood and its oracle properties." JASA 96(456).
Claim attributed: states explicitly that the LASSO produces biased estimates
for large coefficients (motivating SCAD), i.e. LASSO lacks the "oracle
property". Verify the statement and its wording.

**D3. Formula to check:** on the true support S, in the noiseless case and
with the support correctly identified, the LASSO minimiser satisfies the KKT
fixed point
    x*_S = x_S - lambda (A_S^T A_S)^{-1} sign(x_S),
so the bias is roughly lambda per nonzero when columns are unit-norm and
nearly orthogonal. Is this the standard form (e.g. in Buhlmann & van de Geer,
*Statistics for High-Dimensional Data*, 2011, or Hastie, Tibshirani &
Wainwright, *Statistical Learning with Sparsity*, 2015)? Give a section
number.
Also confirm: lambda / L (L = largest eigenvalue of A^T A) is the
soft-threshold applied at each ISTA iteration, NOT the bias of the LASSO
solution -- the two are different quantities.

**D4.** Belloni, A. and Chernozhukov, V. (2013). "Least squares after model
selection in high-dimensional sparse models." Bernoulli 19(2).
Claim attributed: "post-LASSO" (OLS refit on the LASSO-selected support)
removes the shrinkage bias and can outperform the LASSO. Verify.

## E. Support selection in unrolled ISTA (LISTA-CPSS / ALISTA)

**E1.** Chen, X., Liu, J., Wang, Z. and Yin, W. (2018). "Theoretical linear
convergence of unfolded ISTA and its practical weights and thresholds."
NeurIPS 2018.
Claim attributed: introduces the "support selection" (ss) technique -- at
each layer the p% largest-magnitude entries are exempted from
soft-thresholding -- yielding LISTA-CPSS; motivated as reducing the "false
positive" rate / the bias of thresholding on the true support. Verify that
this is where support selection is introduced and how the paper motivates
it.

**E2.** Liu, J., Chen, X., Wang, Z. and Yin, W. (2019). "ALISTA: Analytic
weights are as good as learned weights in LISTA." ICLR 2019.
Claim attributed: ALISTA uses the analytic weight matrix W (minimising
mutual coherence with A) and adopts the support-selection step from E1.
Verify that ALISTA's reference implementation uses support selection (ss)
and what percentage schedule it uses.

**E3. Statement to check (interpretive):** because support selection exempts
the largest entries from shrinkage, the resulting estimate is *less biased
toward zero than the LASSO minimiser x\** on those entries -- so an unrolled
network with support selection can have LOWER error against the true signal
x_true than x* itself, while having HIGHER LASSO objective than x*. Is this
reasoning stated anywhere in E1/E2 or in follow-up work (e.g. discussions of
"debiasing" in unrolled networks), or is it only an inference?

## F. Primer / benchmark paper cross-check

**F1.** Chen, T., Chen, X., Chen, W., Heaton, H., Liu, J., Wang, Z. and Yin,
W. (2022). "Learning to Optimize: A Primer and A Benchmark." JMLR 23(189).
Claims attributed: (a) Section 4.1.2 runs the LASSO experiment at (m, n) =
(25, 50) (and (5, 10)) and reports that "all the model-free L2O methods
perform poorly" at (25, 50); (b) the figures there are loss (or relative
loss) vs. iteration over 1000 iterations; (c) the paper states it did not go
larger than (25, 50) "due to the high memory cost of LSTM-based model-free
L2O methods"; (d) the classical baselines in that experiment are ISTA and
FISTA at matched iteration counts. Verify each and quote the exact metric
used on the y-axis of the LASSO figure.
