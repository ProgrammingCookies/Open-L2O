"""
Experiment-2 LASSO recovery metrics.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# Certified-duality-gap stopping tolerance. Hard coded.
_GAP_TOL = 1e-10
_GAP_CHECK_EVERY = 25

def soft_threshold(z: np.ndarray, thresh: float) -> np.ndarray:
    """S_eta(z) = sign(z) * max(|z| - eta, 0) : pre-study Eq 2."""
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def lasso_objective(a: np.ndarray, x: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """f_q(x) = 0.5*||Ax - b||_2^2 + lam*||x||_1
    x: (batch, N)
    b: (batch, M)
    """
    residual = x @ a.T - b
    l2 = 0.5 * np.sum(residual ** 2, axis=-1)
    l1 = lam * np.sum(np.abs(x), axis=-1)
    return l2 + l1


def _certified_gap(a64: np.ndarray, b64: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """Per-instance certified upper bound on f(x) - f*, from LASSO duality.

    Dual point: r = Ax - b, s = min(1, lam / ||A^T r||_inf), nu = s*r. Then
    f(x) - f* <= f(x) - D(nu) = f(x) + 0.5*||nu||^2 + nu^T b for any x (weak
    duality), with equality at x=x*.
    """
    r = x @ a64.T - b64  # (batch, M)
    at_r = r @ a64  # (batch, N), == A^T r per row
    dual_norm = np.max(np.abs(at_r), axis=-1)
    s = np.minimum(1.0, lam / np.maximum(dual_norm, 1e-300))
    nu = s[:, None] * r
    f = 0.5 * np.sum(r ** 2, axis=-1) + lam * np.sum(np.abs(x), axis=-1)
    return f + 0.5 * np.sum(nu ** 2, axis=-1) + np.sum(nu * b64, axis=-1)


def fista_solve(a: np.ndarray, b: np.ndarray, lam: float, num_iters: int = 50_000,
                x0: Any = None, gap_tol: float = _GAP_TOL) -> np.ndarray:
    """Batched FISTA for LASSO, with per-instance gradient-scheme adaptive
    restart ("Adaptive Restart for Accelerated Gradient Schemes" O'Donoghue and Candes, 2012) and certified-duality-gap early
    stopping.

    a: (M, N) shared dictionary.
    b: (batch, M).
    Returns x*: (batch, N), float32.
    """
    m, n = a.shape
    a64 = a.astype(np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    batch = b64.shape[0]

    # Lipschitz constant of grad(0.5||Ax-b||^2) = A^T(Ax-b) is the largest
    # eigenvalue of A^T A.
    l = float(np.linalg.eigvalsh(a64.T @ a64)[-1])
    eta = 1.0 / l
    thresh = lam * eta

    x = np.zeros((batch, n), dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    y = x.copy()
    t = np.ones(batch, dtype=np.float64)

    for it in range(num_iters):
        grad = (y @ a64.T - b64) @ a64  # shape: (batch, N)
        x_new = soft_threshold(y - eta * grad, thresh)

        # Gradient-scheme restart, evaluated per-instance since different
        # rows of the batch reach their own restart points at different
        # iterations: restart momentum whenever the extrapolation step (y ->
        # x_new) stopped moving in a descent-consistent direction relative
        # to the previous iterate.
        restart = np.sum((y - x_new) * (x_new - x), axis=-1) > 0
        t = np.where(restart, 1.0, t)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = ((t - 1.0) / t_new)[:, None]
        y = np.where(restart[:, None], x_new, x_new + momentum * (x_new - x))
        x, t = x_new, t_new

        if gap_tol is not None and ((it + 1) % _GAP_CHECK_EVERY == 0 or it == num_iters - 1):
            gap = _certified_gap(a64, b64, x, lam)
            f = 0.5 * np.sum((x @ a64.T - b64) ** 2, axis=-1) + lam * np.sum(np.abs(x), axis=-1)
            if np.max(gap / np.maximum(f, 1e-300)) < gap_tol:
                break

    # Deliberately not cast to float32 as to not lose precision.
    return x


def _xstar_cache_path(cache_dir: str, a: np.ndarray, b: np.ndarray,
                      lam: float, num_iters: int, gap_tol: float) -> str:
    """Content-addressed path: keyed on the actual A/b bytes (not a filename).
    This is done so that if multiple callers have the same A/b/lam/num_iters but
    actually differ (for example a random eval batch subset vs. the full file) can never accidentally read back a wrong cache
    entry as different content always produces a different key.

    gap_tol is part of the key, not just num_iters: as fista_solve now
    stops on the certified gap rather than always running num_iters
    iterations.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(b, dtype=np.float32).tobytes())
    h.update(np.array([lam, num_iters, gap_tol], dtype=np.float64).tobytes())
    return os.path.join(cache_dir, h.hexdigest() + ".npy")


def solve_xstar(a: np.ndarray, b: np.ndarray, lam: float, num_iters: int,
                cache_dir: Optional[str] = None, gap_tol: float = _GAP_TOL) -> np.ndarray:
    """x* only depends on (A, b, lam, num_iters, gap_tol)
    so it's safe to share across every method/width/seed and therefore saved as a file to be called over and over easily.
    """
    if cache_dir is None:
        return fista_solve(a, b, lam, num_iters=num_iters, gap_tol=gap_tol)

    cache_path = _xstar_cache_path(cache_dir, a, b, lam, num_iters, gap_tol)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    x_star = fista_solve(a, b, lam, num_iters=num_iters, gap_tol=gap_tol)
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cache_path[:-len(".npy")] + ".tmp-{}.npy".format(os.getpid())
    np.save(tmp_path, x_star)
    os.replace(tmp_path, cache_path)
    return x_star


def nmse_db(x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """10*log10(||x - x_ref||^2 / ||x_ref||^2): pre-study Eq 11."""
    num = np.sum((x - x_ref) ** 2, axis=-1)
    den = np.sum(x_ref ** 2, axis=-1)
    den = np.where(den == 0, 1e-12, den)
    return 10.0 * np.log10(np.maximum(num, 1e-20) / den)


@dataclass
class Experiment2Metrics:
    suboptimality_gap_mean: float
    suboptimality_gap_std: float
    modified_relative_loss: float
    nmse_vs_true_mean: float
    nmse_vs_true_std: float
    lasso_optimal_recovery_error_mean: float
    lasso_optimal_recovery_error_std: float
    num_instances: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "suboptimality_gap_mean": self.suboptimality_gap_mean,
            "suboptimality_gap_std": self.suboptimality_gap_std,
            "modified_relative_loss": self.modified_relative_loss,
            "nmse_vs_true_mean": self.nmse_vs_true_mean,
            "nmse_vs_true_std": self.nmse_vs_true_std,
            "lasso_optimal_recovery_error_mean": self.lasso_optimal_recovery_error_mean,
            "lasso_optimal_recovery_error_std": self.lasso_optimal_recovery_error_std,
            "num_instances": self.num_instances,
        }


def compute_lasso_metrics(a: np.ndarray, b: np.ndarray, x_true: np.ndarray,
                          x_pred: np.ndarray, lam: float,
                          num_fista_iters: int = 50_000,
                          xstar_cache_dir: Optional[str] = None) -> Experiment2Metrics:
    """Computes all four Experiment-2 metrics for one test set.

    a: (M, N).
    b: (batch, M)
    x_true: (batch, N)
    x_pred: (batch, N)
    """
    x_star = solve_xstar(a, b, lam, num_fista_iters, xstar_cache_dir)
    f_star = lasso_objective(a, x_star, b, lam)
    f_pred = lasso_objective(a, x_pred, b, lam)

    gap = f_pred - f_star  # Eq 9
    worst_relative_gap = float(np.min(gap) / max(float(np.mean(f_star)), 1e-30))
    if worst_relative_gap < -1e-4:
        warnings.warn(
            "compute_lasso_metrics: x_pred beats the certified-optimal "
            "x_star by relative gap {:.3e} on at least one instance (lam={}) "
            "-- check x_pred was produced/scored with this same lam.".format(
                worst_relative_gap, lam))
    modified_relative_loss = float(np.mean(gap) / np.mean(f_star))  # Eq 10

    nmse_vs_true = nmse_db(x_pred, x_true)  # Eq 11
    lasso_optimal_recovery_error = nmse_db(x_star, x_true)  # same formula, x=x*

    return Experiment2Metrics(
        suboptimality_gap_mean=float(np.mean(gap)),
        suboptimality_gap_std=float(np.std(gap)),
        modified_relative_loss=modified_relative_loss,
        nmse_vs_true_mean=float(np.mean(nmse_vs_true)),
        nmse_vs_true_std=float(np.std(nmse_vs_true)),
        lasso_optimal_recovery_error_mean=float(np.mean(lasso_optimal_recovery_error)),
        lasso_optimal_recovery_error_std=float(np.std(lasso_optimal_recovery_error)),
        num_instances=int(b.shape[0]),
    )
