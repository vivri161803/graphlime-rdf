"""M4 property tests for the non-negative HSIC Lasso — written before the code."""

from __future__ import annotations

import numpy as np

from graphlime_rdf.explain.hsic_lasso import hsic_lasso


def _data(n: int = 60, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """d noise features; y depends (nonlinearly) on feature 2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = np.tanh(2 * x[:, [2]]) + 0.05 * rng.normal(size=(n, 1))
    return x, y


def test_all_betas_nonnegative() -> None:
    x, y = _data()
    beta = hsic_lasso(x, y, rho=0.01)
    assert beta.shape == (x.shape[1],)
    assert np.all(beta >= 0)


def test_planted_target_feature_gets_largest_beta() -> None:
    """Planting y itself as a column ⇒ that column dominates."""
    x, y = _data()
    x_planted = np.hstack([x, y])
    beta = hsic_lasso(x_planted, y, rho=0.01)
    assert beta.argmax() == x.shape[1]
    assert beta[x.shape[1]] > 0


def test_relevant_feature_beats_noise() -> None:
    x, y = _data()
    beta = hsic_lasso(x, y, rho=0.01)
    assert beta.argmax() == 2


def test_duplicated_feature_splits_weight_not_double_counts() -> None:
    x, y = _data()
    beta_single = hsic_lasso(x, y, rho=0.01)
    x_dup = np.hstack([x, x[:, [2]]])  # duplicate the relevant feature
    beta_dup = hsic_lasso(x_dup, y, rho=0.01)
    combined = beta_dup[2] + beta_dup[-1]
    # The duplicated pair together carries about the weight the original had —
    # allow slack, but rule out double counting (≈ 2x) and vanishing.
    assert combined <= 1.5 * beta_single[2] + 1e-9
    assert combined >= 0.5 * beta_single[2] - 1e-9


def test_sparsity_nonincreasing_in_rho() -> None:
    x, y = _data(n=80, d=8, seed=1)
    nnz = [
        int(np.sum(hsic_lasso(x, y, rho=rho) > 1e-10))
        for rho in [0.001, 0.01, 0.1, 1.0]
    ]
    assert nnz == sorted(nnz, reverse=True)
    assert nnz[-1] <= 1  # huge rho kills (almost) everything


def test_constant_feature_gets_zero_weight() -> None:
    x, y = _data()
    x_const = np.hstack([x, np.ones((x.shape[0], 1))])
    beta = hsic_lasso(x_const, y, rho=0.01)
    assert beta[-1] < 1e-10


def test_deterministic() -> None:
    x, y = _data()
    b1 = hsic_lasso(x, y, rho=0.01)
    b2 = hsic_lasso(x, y, rho=0.01)
    np.testing.assert_array_equal(b1, b2)
