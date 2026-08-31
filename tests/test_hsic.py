"""M4 property tests for HSIC — written BEFORE the implementation (plan M4).

These encode the mathematical invariants the whole method rests on. They are
never weakened to make progress.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from graphlime_rdf.explain.hsic import (
    centering_matrix,
    hsic,
    median_heuristic_sigma,
    nhsic,
    rbf_kernel,
)


# Bounded, finite float arrays: n samples × d dims.
def samples(n_min: int = 5, n_max: int = 25, d: int = 2) -> st.SearchStrategy[np.ndarray]:
    return hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(n_min, n_max), st.just(d)),
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    )


def paired(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    x = draw(samples())
    y = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(x.shape[0], 2),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        )
    )
    return x, y


paired_samples = st.composite(paired)()


@given(st.integers(2, 50))
def test_centering_matrix_symmetric_idempotent(n: int) -> None:
    h = centering_matrix(n)
    np.testing.assert_allclose(h, h.T, atol=1e-12)
    np.testing.assert_allclose(h @ h, h, atol=1e-12)


@given(paired_samples)
@settings(max_examples=50, deadline=None)
def test_hsic_symmetric(xy: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = xy
    np.testing.assert_allclose(hsic(x, y), hsic(y, x), rtol=1e-9, atol=1e-12)


@given(paired_samples)
@settings(max_examples=50, deadline=None)
def test_hsic_nonnegative(xy: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = xy
    assert hsic(x, y) >= -1e-12


@given(paired_samples)
@settings(max_examples=50, deadline=None)
def test_nhsic_in_unit_interval(xy: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = xy
    value = nhsic(x, y)
    assert -1e-9 <= value <= 1.0 + 1e-9


@given(paired_samples, st.randoms(use_true_random=False))
@settings(max_examples=30, deadline=None)
def test_hsic_invariant_under_joint_permutation(
    xy: tuple[np.ndarray, np.ndarray], rng: object
) -> None:
    x, y = xy
    perm = np.arange(x.shape[0])
    getattr(rng, "shuffle")(perm)  # noqa: B009 - hypothesis Random object
    np.testing.assert_allclose(
        hsic(x, y), hsic(x[perm], y[perm]), rtol=1e-9, atol=1e-12
    )


def test_hsic_detects_dependence_over_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y_dep = np.sin(3 * x) + 0.05 * rng.normal(size=(200, 1))
    y_noise = rng.normal(size=(200, 1))
    assert hsic(x, y_dep) > 5 * hsic(x, y_noise)


def test_hsic_detects_nonlinear_dependence_where_pearson_fails() -> None:
    """y = x² with x symmetric about 0: Pearson ≈ 0, HSIC clearly > 0.

    The test that justifies the whole method (plan M4).
    """
    rng = np.random.default_rng(1)
    x = np.concatenate([-np.linspace(0.1, 3, 100), np.linspace(0.1, 3, 100)])
    rng.shuffle(x)
    y = x**2
    pearson = float(np.corrcoef(x, y)[0, 1])
    assert abs(pearson) < 0.05

    x2, y2 = x.reshape(-1, 1), y.reshape(-1, 1)
    dependent = nhsic(x2, y2)
    independent = nhsic(x2, rng.permutation(y).reshape(-1, 1))
    assert dependent > 0.2
    assert dependent > 3 * independent


def test_rbf_kernel_basic_properties() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 3))
    k = rbf_kernel(x, sigma=1.0)
    np.testing.assert_allclose(np.diag(k), 1.0, atol=1e-12)
    np.testing.assert_allclose(k, k.T, atol=1e-12)
    assert np.all(k > 0) and np.all(k <= 1 + 1e-12)


def test_median_heuristic_positive_even_for_constant_input() -> None:
    assert median_heuristic_sigma(np.zeros((10, 2))) > 0
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 2))
    assert median_heuristic_sigma(x) > 0


def test_hsic_zero_for_constant_feature() -> None:
    rng = np.random.default_rng(4)
    x = np.ones((30, 1))
    y = rng.normal(size=(30, 1))
    assert abs(hsic(x, y)) < 1e-10
