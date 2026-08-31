"""Non-negative HSIC Lasso (plan M4; Yamada et al., 2014 — the GraphLIME solver).

Each feature ``k`` gets a centered, Frobenius-normalised RBF kernel K̄_k; the
model output gets L̄ likewise. Solving

    min_β  ½‖vec(L̄) − Σ_k β_k vec(K̄_k)‖²  +  ρ‖β‖₁,   β ≥ 0

with sklearn's ``Lasso(positive=True)`` yields β_k ∝ how much feature k's
similarity structure explains the output's similarity structure:
βᵀ-objective expands to Σ_k β_k HSIC(x_k, y) − ½Σ_{k,l} β_k β_l HSIC(x_k, x_l).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from graphlime_rdf.explain.hsic import FloatArray, centering_matrix, rbf_kernel


def _centered_normalised_kernel(values: FloatArray, sigma: float | None) -> FloatArray:
    """H K H, scaled to unit Frobenius norm (zero matrix if degenerate)."""
    n = values.shape[0]
    h = centering_matrix(n)
    kc = h @ rbf_kernel(values, sigma) @ h
    norm = float(np.linalg.norm(kc))
    if norm < 1e-12:
        return np.zeros_like(kc)
    return kc / norm


def hsic_lasso(
    x: FloatArray,
    y: FloatArray,
    rho: float,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> FloatArray:
    """Return β ≥ 0, one weight per column of ``x`` (n samples × d features)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"bad shapes: x{x.shape}, y{y.shape}")
    if x.shape[0] < 2:
        raise ValueError("HSIC Lasso needs at least 2 samples")

    n, d = x.shape
    design = np.empty((n * n, d))
    for k in range(d):
        design[:, k] = _centered_normalised_kernel(x[:, [k]], sigma_x).ravel()
    target = _centered_normalised_kernel(y, sigma_y).ravel()

    solver = Lasso(alpha=rho / (n * n), positive=True, fit_intercept=False, max_iter=10_000)
    solver.fit(design, target)
    beta: FloatArray = np.asarray(solver.coef_, dtype=np.float64)
    return beta
