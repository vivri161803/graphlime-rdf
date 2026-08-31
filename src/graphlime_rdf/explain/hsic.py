"""Empirical HSIC with RBF kernels (plan M4; Gretton et al., 2005).

Implements the pieces GraphLIME's HSIC Lasso is built from: the RBF kernel
with median-heuristic bandwidth, the centering matrix, empirical HSIC and its
normalised variant. Pure NumPy — no torch dependency in the math core.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def median_heuristic_sigma(x: FloatArray) -> float:
    """Median of pairwise Euclidean distances; falls back to 1.0 if degenerate."""
    sq = _squared_distances(np.asarray(x, dtype=np.float64))
    upper = sq[np.triu_indices_from(sq, k=1)]
    if upper.size == 0:
        return 1.0
    median = float(np.sqrt(np.maximum(np.median(upper), 0.0)))
    return median if median > 0 else 1.0


def _squared_distances(x: FloatArray) -> FloatArray:
    norms = np.sum(x * x, axis=1, keepdims=True)
    sq = norms + norms.T - 2.0 * (x @ x.T)
    return np.maximum(sq, 0.0)


def rbf_kernel(x: FloatArray, sigma: float | None = None) -> FloatArray:
    """K_ij = exp(-‖x_i − x_j‖² / (2σ²)); σ from the median heuristic if None."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if sigma is None:
        sigma = median_heuristic_sigma(x)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    kernel: FloatArray = np.exp(-_squared_distances(x) / (2.0 * sigma**2))
    return kernel


def centering_matrix(n: int) -> FloatArray:
    """H = I − (1/n)·11ᵀ, symmetric and idempotent."""
    if n < 1:
        raise ValueError("n must be >= 1")
    h: FloatArray = np.eye(n) - np.full((n, n), 1.0 / n)
    return h


def hsic_from_kernels(k: FloatArray, latent: FloatArray) -> float:
    """Empirical HSIC = trace(K H L H) / (n−1)²."""
    n = k.shape[0]
    if n < 2:
        raise ValueError("HSIC needs at least 2 samples")
    h = centering_matrix(n)
    kc = h @ k @ h
    lc = h @ latent @ h
    return float(np.sum(kc * lc) / (n - 1) ** 2)


def hsic(x: FloatArray, y: FloatArray, sigma_x: float | None = None, sigma_y: float | None = None) -> float:
    """Empirical HSIC between samples ``x`` (n, dx) and ``y`` (n, dy)."""
    return hsic_from_kernels(rbf_kernel(x, sigma_x), rbf_kernel(y, sigma_y))


def nhsic(x: FloatArray, y: FloatArray, sigma_x: float | None = None, sigma_y: float | None = None) -> float:
    """Normalised HSIC ∈ [0, 1]: HSIC(x,y) / √(HSIC(x,x)·HSIC(y,y))."""
    k = rbf_kernel(x, sigma_x)
    latent = rbf_kernel(y, sigma_y)
    hxy = hsic_from_kernels(k, latent)
    hxx = hsic_from_kernels(k, k)
    hyy = hsic_from_kernels(latent, latent)
    # HSIC(x,x) is a sum of squares: values this small mean the centered
    # kernel is numerically zero (constant sample) and the ratio is noise.
    if hxx < 1e-15 or hyy < 1e-15:
        return 0.0
    return float(hxy / np.sqrt(hxx * hyy))
