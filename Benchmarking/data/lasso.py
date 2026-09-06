"""Sparse-signal LASSO dataset generator adapted for Experiment 2

Generates, for one shared random dictionary A, three training sets that differ only
in how concentrated their sparsity distribution is (narrow/medium/wide) plus a fixed
battery of test sets at specific sparsity levels, so that training-distribution width
is the only thing varying across the "narrow"/"medium"/"wide" runs.

Optionally (``compute_alista_w=True`` / ``--compute_alista_w``,  also generates ALISTA's analytic
weight matrix W and writes ``W.npy`` alongside ``A.npy`` in every width dir
see ``compute_alista_w()``'s docstring for the exact LP being solved.

Every split file also gets a seeded ``<name>_x0.npy`` sibling (see
``_x0_filename()``): the shared starting point every method's recovery
trajectory begins from, so a given seed produces the same x0 per instance
regardless of which of the four methods is being trained/evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog

SparsitySpec = Union[float, Tuple[float, float]]

# Default values for Experiment 2
DEFAULT_M = 25
DEFAULT_N = 50
DEFAULT_LAM = 0.005
DEFAULT_TRAIN_SIZE = 32_000
DEFAULT_VAL_SIZE = 1_024
DEFAULT_TEST_SIZE = 1_280
DEFAULT_SNR_DB = float("inf")  # noiseless: _add_noise_for_snr's noise_power = signal_power/10**(inf/10) == 0
DEFAULT_TRAIN_SPARSITIES: Dict[str, SparsitySpec] = {
    "narrow": 0.175,
    "medium": (0.1125, 0.2375),
    "wide": (0.05, 0.3),
}
DEFAULT_TEST_SPARSITIES: List[float] = [0.02, 0.05, 0.10, 0.1125, 0.15, 0.175, 0.20, 0.2375, 0.30, 0.35, 0.40]


def sample_dictionary(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """A ~ N(0, 1/m) entries, columns rescaled to unit L2 norm."""
    a = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, n)).astype(np.float32)
    norms = np.linalg.norm(a, axis=0, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a (measure-zero) all-zero column
    return (a / norms).astype(np.float32)


def compute_alista_w(a: np.ndarray) -> np.ndarray:
    """The ALISTA matrix W (same (m, n) shape as A, columns
    aligned to A's columns.

    Solves the ALISTA paper's own mutual-coherence-minimization problem exactly,
    independently per column i:

        w_i = argmin_w  max_{j != i} |w^T a_j|   s.t.  w^T a_i = 1
    """
    m, n = a.shape
    a64 = a.astype(np.float64)
    w_cols = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        a_i = a64[:, i]
        a_other = a64[:, mask]  # shape: (m, n-1)
        k = n - 1

        c = np.zeros(m + 1)
        c[-1] = 1.0
        a_eq = np.concatenate([a_i, [0.0]])[None, :]
        b_eq = np.array([1.0])
        if k > 0:
            a_ub = np.concatenate([
                np.concatenate([a_other.T, -np.ones((k, 1))], axis=1),
                np.concatenate([-a_other.T, -np.ones((k, 1))], axis=1),
            ], axis=0)
            b_ub = np.zeros(2 * k)
        else:
            a_ub = b_ub = None
        bounds = [(None, None)] * m + [(0, None)]

        result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(
                "ALISTA coherence-minimization LP failed for column {}: {}".format(
                    i, result.message))
        w_cols.append(result.x[:m])
    return np.stack(w_cols, axis=1).astype(np.float32)


def _draw_p(sparsity_spec: SparsitySpec, rng: np.random.Generator) -> float:
    if isinstance(sparsity_spec, (tuple, list)):
        lo, hi = sparsity_spec
        return float(rng.uniform(lo, hi))
    return float(sparsity_spec)


def sample_sparse_signal(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Exactly round(p*n) nonzero coordinates (uniform random support), values ~N(0,1)."""
    k = int(round(p * n))
    x = np.zeros(n, dtype=np.float32)
    if k > 0:
        support = rng.choice(n, size=k, replace=False)
        x[support] = rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)
    return x


def sample_sparse_signal_bernoulli(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Ber(p)*N(0,1) per coordinate -- nonzero count varies per
    instance (binomial, mean p*n), matching Chen et al.'s primer/JMLR-paper
    ("Learning to Optimize: A Primer and a Benchmark") sections 4.1.1/4.1.2's
    own stated sparsity model. Deliberately different from
    sample_sparse_signal's exact-round(p*n)-count convention, which is
    Experiment 2's own dataset-generator design decision (see
    project_lasso_generator memory), not what the primer paper itself does --
    use this one specifically when replicating the paper's own protocol.
    """
    mask = rng.random(n) < p
    x = np.zeros(n, dtype=np.float32)
    k = int(mask.sum())
    if k > 0:
        x[mask] = rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)
    return x


def _add_noise_for_snr(b_clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Per-instance additive Gaussian noise calibrated to the given signal ratio (in dB)."""
    signal_power = np.mean(b_clean ** 2, axis=-1, keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(loc=0.0, scale=1.0, size=b_clean.shape).astype(np.float32) * noise_std
    return (b_clean + noise).astype(np.float32)


def generate_split(
    a: np.ndarray,
    num_samples: int,
    sparsity_spec: SparsitySpec,
    snr_db: float,
    x0_stddev: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the LASSO samples as well as the x0 starting points.
    Returns (data, p_used, x0): data is (num_samples, M+N) rows shape [b; x_true];
    x0 is (num_samples, N), a dense small-magnitude seeded starting point.

    x0 is what every method's recovery trajectory begins from, which is LISTA/ALISTA's
    layer 0 and the model-free optimizers' first step both consume it, so for a all four methods start each instance from the identical point
    (written about in Benchmarking/README.md's "x0 alignment" part).
    """
    m, n = a.shape
    x_true = np.stack(
        [sample_sparse_signal(n, _draw_p(sparsity_spec, rng), rng) for _ in range(num_samples)],
        axis=0,
    )
    p_used = np.array([float(np.count_nonzero(row)) / n for row in x_true], dtype=np.float32)
    b_clean = x_true @ a.T  # shape: (num_samples, m)
    b = _add_noise_for_snr(b_clean, snr_db, rng)
    data = np.concatenate([b, x_true], axis=1).astype(np.float32)
    x0 = rng.normal(loc=0.0, scale=x0_stddev, size=(num_samples, n)).astype(np.float32)
    return data, p_used, x0


def _x0_filename(split_filename: str) -> str:
    """The seeded-x0 sibling file for a given split file, e.g.
    'train_data.npy' -> 'train_data_x0.npy'. Kept as a separate file (not
    extra columns on the split file) so every existing consumer of the split
    files' [b; x_true] row layout (problems.py's width check,
    model_based.py's b/x_true slicing, the metadata's documented row_layout)
    stays untouched.

    Model_Free_L2O/.../problems.py and Model_Base_L2O/data_preprocessing.py
    both derive this same filename independently -- they're separate
    processes/environments and can't import this function -- so if this
    naming rule ever changes, update it in all three places.
    """
    assert split_filename.endswith(".npy")
    return split_filename[:-len(".npy")] + "_x0.npy"


@dataclass
class Experiment2Config:
    #The necessary configs for experiment 2
    seed: int
    m: int = DEFAULT_M
    n: int = DEFAULT_N
    lam: float = DEFAULT_LAM
    train_size: int = DEFAULT_TRAIN_SIZE
    val_size: int = DEFAULT_VAL_SIZE
    test_size: int = DEFAULT_TEST_SIZE
    snr_db: float = DEFAULT_SNR_DB
    train_sparsities: Dict[str, SparsitySpec] = None
    test_sparsities: List[float] = None
    compute_alista_w: bool = False
    # Shared seeded starting point every method's recovery trajectory.
    x0_stddev: float = 0.01

    def __post_init__(self):
        if self.train_sparsities is None:
            self.train_sparsities = dict(DEFAULT_TRAIN_SPARSITIES)
        if self.test_sparsities is None:
            self.test_sparsities = list(DEFAULT_TEST_SPARSITIES)


def _test_filename(p: float) -> str:
    return "test_sparsity_{:.2f}.npy".format(p)


def _dataset_fingerprint(cfg: Experiment2Config) -> Dict[str, Any]:
    """The subset of metadata.json that fully determines generate_split's output
    given the same seed. This is used to detect a resumed run whose config silently
    changed from the one that produced the data already."""
    return {
        "m": cfg.m, "n": cfg.n, "lam": cfg.lam,
        "train_size": cfg.train_size, "val_size": cfg.val_size, "test_size": cfg.test_size,
        "snr_db": cfg.snr_db,
        "train_sparsities": cfg.train_sparsities, "test_sparsities": cfg.test_sparsities,
        "alista_w": cfg.compute_alista_w,
        "x0_stddev": cfg.x0_stddev,
    }


def generate_experiment2_dataset(config: Experiment2Config, out_dir: str) -> str:
    """Writes one self-contained data directory per training-distribution, (A.npy, train_data.npy, val_data.npy,
    test_sparsity_*.npy). All three widths share the same dictionary A and the same
    test sets, generated once, so training-distribution width is the only thing that
    differs between them.

    If metadata.json already exists and its params match the config, generation is skipped and the existing directory is reused.
    """
    seed_dir = os.path.join(out_dir, "seed{}".format(config.seed))
    metadata_path = os.path.join(seed_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            existing = json.load(f)
        # Dopuble check config so both sides use the same types for comparison.
        wanted = json.loads(json.dumps(_dataset_fingerprint(config)))
        mismatches = {k: (existing.get(k), v) for k, v in wanted.items() if existing.get(k) != v}
        if mismatches:
            raise RuntimeError(
                "{} already exists but was generated with different parameters than "
                "the current config, refusing to silently overwrite or reuse it. "
                "Mismatched fields (existing, requested): {}. Use a different out_dir "
                "or delete the stale seed directory if this change was intentional."
                .format(metadata_path, mismatches))
        return seed_dir

    rng = np.random.default_rng(config.seed)
    a = sample_dictionary(config.m, config.n, rng)

    # ALISTA's W depends only on A, so compute it once per seed
    alista_w = compute_alista_w(a) if config.compute_alista_w else None

    # Shared test sets with fixed sparsity per file
    test_sets = {}
    test_x0 = {}
    for p in config.test_sparsities:
        data, p_used, x0 = generate_split(a, config.test_size, p, config.snr_db,
                                          config.x0_stddev, rng)
        test_sets[p] = data
        test_x0[p] = x0
        assert np.allclose(p_used, p, atol=1.0 / config.n), (
            "test sparsity {} did not round-trip through round(p*n) exactly".format(p))

    width_p_used: Dict[str, np.ndarray] = {}
    for width_name, spec in config.train_sparsities.items():
        width_dir = os.path.join(seed_dir, width_name)
        os.makedirs(width_dir, exist_ok=True)

        train_data, train_p, train_x0 = generate_split(
            a, config.train_size, spec, config.snr_db, config.x0_stddev, rng)
        val_data, _, val_x0 = generate_split(
            a, config.val_size, spec, config.snr_db, config.x0_stddev, rng)
        width_p_used[width_name] = train_p

        np.save(os.path.join(width_dir, "A.npy"), a)
        if alista_w is not None:
            np.save(os.path.join(width_dir, "W.npy"), alista_w)
        np.save(os.path.join(width_dir, "train_data.npy"), train_data)
        np.save(os.path.join(width_dir, _x0_filename("train_data.npy")), train_x0)
        np.save(os.path.join(width_dir, "val_data.npy"), val_data)
        np.save(os.path.join(width_dir, _x0_filename("val_data.npy")), val_x0)
        np.save(os.path.join(width_dir, "train_sparsity_used.npy"), train_p)
        for p, data in test_sets.items():
            fname = _test_filename(p)
            np.save(os.path.join(width_dir, fname), data)
            np.save(os.path.join(width_dir, _x0_filename(fname)), test_x0[p])

    metadata = {
        "seed": config.seed,
        "m": config.m,
        "n": config.n,
        "lam": config.lam,
        "train_size": config.train_size,
        "val_size": config.val_size,
        "test_size": config.test_size,
        "snr_db": config.snr_db,
        "train_sparsities": config.train_sparsities,
        "test_sparsities": config.test_sparsities,
        "sparsity_semantics": "exact round(p*n) nonzeros, uniform random support",
        "nonzero_magnitude_dist": "N(0, 1)",
        "dictionary_dist": "N(0, 1/m) entries, columns rescaled to unit L2 norm",
        "noise_convention": "per-instance additive Gaussian noise at fixed SNR (dB) on b",
        "row_layout": "[b (M,); x_true (N,)], length M+N matches "
                       "Model_Base_L2O/utils.py's LassoObjective/NMSE slicing",
        "x0_stddev": config.x0_stddev,
        "x0_layout": "each split file <name>.npy has a sibling <name>_x0.npy, "
                     "shape (num_samples, N) -- the seeded starting point "
                     "every method's recovery trajectory begins from, see "
                     "README's 'x0 alignment' decision",
        "test_files": [_test_filename(p) for p in config.test_sparsities],
        "train_sparsity_mean_actual": {
            w: float(np.mean(p)) for w, p in width_p_used.items()
        },
        "alista_w": config.compute_alista_w,
        "alista_w_method": ("per-column mutual-coherence-minimization LP "
                            "(scipy.optimize.linprog, HiGHS)" if config.compute_alista_w else None),
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return seed_dir


def generate_primer_lasso_dataset(
    seed: int, m: int, n: int, lam: float, p: float,
    train_size: int, val_size: int, test_size: int, snr_db: float,
    out_dir: str,
) -> str:
    """Flat (no narrow/medium/wide width triad) LASSO dataset replicating
    Chen et al.'s primer/JMLR-paper ("Learning to Optimize: A Primer and a
    Benchmark") section 4.1.2 exactly: a single fixed Bernoulli(p) sparsity
    shared by train/val/test (the width/OOD axis is a DA233X pre-study
    addition, not present in the primer's own experiment), i.i.d. Bernoulli
    sparsity per sample_sparse_signal_bernoulli (not Experiment 2's
    exact-round(p*n)-count convention), and no x0 sibling files -- this
    dataset is meant to be consumed with
    Model_Free_L2O/.../problems.py's lasso_from_dataset(x0_mode="random"),
    matching the paper's own "average over 10 random starting points"
    evaluation protocol rather than Experiment 2's cross-method x0 alignment.

    Writes A.npy, train_data.npy, val_data.npy, test_data.npy (single file,
    not one per test sparsity), metadata.json, directly under
    out_dir/seed{seed}/ (no width subdirectory).
    """
    rng = np.random.default_rng(seed)
    a = sample_dictionary(m, n, rng)
    seed_dir = os.path.join(out_dir, "seed{}".format(seed))
    os.makedirs(seed_dir, exist_ok=True)

    def _gen(num_samples: int) -> np.ndarray:
        x_true = np.stack(
            [sample_sparse_signal_bernoulli(n, p, rng) for _ in range(num_samples)], axis=0)
        b_clean = x_true @ a.T
        b = _add_noise_for_snr(b_clean, snr_db, rng)
        return np.concatenate([b, x_true], axis=1).astype(np.float32)

    train_data = _gen(train_size)
    val_data = _gen(val_size)
    test_data = _gen(test_size)

    np.save(os.path.join(seed_dir, "A.npy"), a)
    np.save(os.path.join(seed_dir, "train_data.npy"), train_data)
    np.save(os.path.join(seed_dir, "val_data.npy"), val_data)
    np.save(os.path.join(seed_dir, "test_data.npy"), test_data)

    metadata = {
        "seed": seed, "m": m, "n": n, "lam": lam, "sparsity_p": p,
        "train_size": train_size, "val_size": val_size, "test_size": test_size,
        "snr_db": snr_db,
        "sparsity_model": "i.i.d. Ber(p)*N(0,1) per coordinate, count varies "
                          "per instance -- Chen et al. primer/JMLR-paper "
                          "sections 4.1.1/4.1.2's own stated convention "
                          "(NOT Experiment 2's exact-round(p*n)-count model)",
        "row_layout": "[b (M,); x_true (N,)], length M+N",
        "note": "Replicates Chen et al. 'Learning to Optimize: A Primer and a "
                "Benchmark' (JMLR 2022) section 4.1.2's LASSO-minimization "
                "experiment: single fixed sparsity (no width/OOD axis), "
                "Bernoulli sparsity, train_size=12800 (paper-stated, not "
                "Table 3's 32000). No x0 sibling files -- see "
                "problems.lasso_from_dataset(x0_mode='random').",
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return seed_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="One dataset directory is generated per seed")
    p.add_argument("--m", type=int, default=DEFAULT_M)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    p.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    p.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    p.add_argument("--snr_db", type=float, default=DEFAULT_SNR_DB)
    p.add_argument("--compute_alista_w", action="store_true",
                   help="Also solve the ALISTA analytic-weight LP and write W. Off by default.")
    p.add_argument("--x0_stddev", type=float, default=0.01,
                   help="Stddev of the shared seeded x0 every method's recovery "
                        "trajectory starts from.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for seed in args.seeds:
        cfg = Experiment2Config(
            seed=seed, m=args.m, n=args.n, lam=args.lam,
            train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
            snr_db=args.snr_db, compute_alista_w=args.compute_alista_w,
            x0_stddev=args.x0_stddev,
        )
        seed_dir = generate_experiment2_dataset(cfg, args.out_dir)
        print("seed {}: wrote {}".format(seed, seed_dir))


if __name__ == "__main__":
    main()
