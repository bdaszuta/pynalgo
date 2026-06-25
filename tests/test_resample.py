"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.resample: interpolation,
           extrapolation, and Lebesgue function.
"""

import numpy as np
from numpy import array, linspace, meshgrid
from numpy.testing import assert_allclose

from pynalgo.resample import (
    extrap_Richardson,
    extrap_Richardson_err,
    interp_barycentric_1d,
    interp_barycentric_nd,
    interp_Neville,
    Lebesgue_func,
    lerp_1d,
    uniform_filter_1d,
    gaussian_filter_1d,
)
from pynalgo.special_functions.grids import grid_ChebyshevT

_rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Barycentric 1D interpolation
# ---------------------------------------------------------------------------

def test_barycentric_1d_exp_BT():
    """Interpolate exp(-x^2) on [-3, 3] with BT method, error < 1e-3."""

    # Source grid: N=20 Chebyshev-Lobatto nodes on [-3, 3]
    x_src = 3.0 * grid_ChebyshevT(N=20, variety="GL")
    f_src = np.exp(-x_src ** 2)

    # Target grid: 50 random points in [-3, 3]
    x_tgt = _rng.uniform(-3.0, 3.0, size=50)
    x_tgt.sort()

    f_interp = interp_barycentric_1d(x_src, f_src, x_tgt, method="BT")
    f_exact = np.exp(-x_tgt ** 2)

    assert_allclose(f_interp, f_exact, atol=1e-3)


def test_barycentric_1d_exp_FH():
    """Interpolate exp(-x^2) on [-3, 3] with FH method, error < 1e-3."""

    x_src = 3.0 * grid_ChebyshevT(N=14, variety="GL")
    f_src = np.exp(-x_src ** 2)

    x_tgt = _rng.uniform(-3.0, 3.0, size=50)
    x_tgt.sort()

    f_interp = interp_barycentric_1d(x_src, f_src, x_tgt, method="FH")
    f_exact = np.exp(-x_tgt ** 2)

    assert_allclose(f_interp, f_exact, atol=1e-3)


# ---------------------------------------------------------------------------
# Barycentric ND interpolation
# ---------------------------------------------------------------------------

def test_barycentric_nd_2d_exp():
    """2D tensor-product interpolation of exp(-(x^2 + y^2))."""

    # Source grids: N=10 Chebyshev nodes on [-2, 2]
    N_src = 10
    x_src = 2.0 * grid_ChebyshevT(N=N_src, variety="GL")
    y_src = 2.0 * grid_ChebyshevT(N=N_src, variety="GL")

    X_src, Y_src = meshgrid(x_src, y_src, indexing="ij")
    F_src = np.exp(-(X_src ** 2 + Y_src ** 2))

    # Target grid: 20 x 20 uniform
    x_tgt = linspace(-2.0, 2.0, 20)
    y_tgt = linspace(-2.0, 2.0, 20)

    F_interp = interp_barycentric_nd(
        x_src, y_src, F_src, x_tgt, y_tgt, method="FH"
    )

    X_tgt, Y_tgt = meshgrid(x_tgt, y_tgt, indexing="ij")
    F_exact = np.exp(-(X_tgt ** 2 + Y_tgt ** 2))

    # Verify shape
    assert F_interp.shape == (20, 20)

    # Verify error
    assert_allclose(F_interp, F_exact, atol=1e-3)


# ---------------------------------------------------------------------------
# Lebesgue function
# ---------------------------------------------------------------------------

def test_Lebesgue_Chebyshev_bound():
    """Lebesgue constant for Chebyshev nodes (N=5) should be <= 2."""
    N = 5
    x_s = grid_ChebyshevT(N=N, variety="GL")          # 6 source nodes
    x_t = grid_ChebyshevT(N=5 * N, variety="GL")      # fine target grid

    lam = Lebesgue_func(x_s, x_t)
    lebesgue_const = float(np.max(lam))

    # For Chebyshev extrema nodes with n=5, the Lebesgue constant is
    # L_5 ~ 1.94  ->  under 2.
    assert lebesgue_const <= 2.0


def test_Lebesgue_equidistant_grows():
    """Lebesgue constant for equidistant nodes grows with N."""

    # Fine target grid -- random to avoid exact hits on source nodes
    x_t = _rng.uniform(-1.0, 1.0, size=300)
    x_t.sort()

    lam_small = Lebesgue_func(linspace(-1.0, 1.0, 6), x_t)
    lam_large = Lebesgue_func(linspace(-1.0, 1.0, 12), x_t)

    max_small = float(np.max(lam_small))
    max_large = float(np.max(lam_large))

    # Equidistant-node Lebesgue constant grows (exponentially) with N
    assert max_large > max_small


def test_Lebesgue_at_source_node():
    """Lebesgue function equals 1 at every source node (sanity check)."""
    x_s = grid_ChebyshevT(N=4, variety="GL")
    lam = Lebesgue_func(x_s, x_s)

    assert_allclose(lam, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def test_Richardson_extrap_known():
    """Richardson extrapolation: (k=2, n0=2, Ac=1.0, Af=1.25) -> 4/3."""
    k = 2.0
    n0 = 2.0
    Ac = 1.0
    Af = 1.25

    result = extrap_Richardson(k, n0, Ac, Af)
    expected = 4.0 / 3.0  # = 1.333...

    assert_allclose(result, expected, atol=1e-10)


def test_Richardson_err_convergence():
    """Richardson error estimator recovers known n0 = 2."""
    # Simulate A(h) = A* + C*h^2  with A* = 1, C = 1, h = 0.5
    A_star = 1.0
    h = 0.5

    Ak = A_star + (h / 2) ** 2   # k = 2
    A = A_star + h ** 2           # base grid
    Al = A_star + (h / 4) ** 2   # l = 4

    n_est = extrap_Richardson_err(2.0, 4.0, Ak, A, Al, n0=1)

    # Should recover n0 ~ 2
    assert_allclose(n_est, 2.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Neville interpolation
# ---------------------------------------------------------------------------

def test_Neville_polynomial_exact():
    """Neville interpolation of x^4 - 2x^2 + 0.2x matches exact values."""
    pp_f = lambda x: x ** 4 - 2 * x ** 2 + 0.2 * x

    xx = array([0.1, 0.2, 0.3, 1.0, 3.0])
    pp = pp_f(xx)
    x = array([0.125, 2.5])

    result = interp_Neville(xx, pp, x)
    expected = pp_f(x)

    assert_allclose(result, expected, atol=1e-10)


###############################################################################
# Column utility kernels
###############################################################################

def test_lerp_1d_exact():
    """Piecewise linear interpolation recovers x^2 exactly at source nodes."""
    x = array([0.0, 1.0, 2.0, 3.0])
    f = x ** 2
    assert_allclose(lerp_1d(x, f, x), f, atol=1e-14)


def test_lerp_1d_interior():
    """Linear interpolation at midpoints."""
    x = array([0.0, 2.0, 4.0])
    f = x ** 2
    xt = array([1.0, 3.0])
    result = lerp_1d(x, f, xt)
    assert_allclose(result, array([2.0, 10.0]), atol=1e-14)


def test_lerp_1d_extrapolate_left():
    """Outside left: clamped to first source value."""
    x = array([0.0, 1.0, 2.0])
    f = x ** 2
    xt = array([-1.0])
    result = lerp_1d(x, f, xt)
    assert_allclose(result, array([0.0]), atol=1e-14)


def test_lerp_1d_extrapolate_right():
    """Outside right: clamped to last source value."""
    x = array([0.0, 1.0, 2.0])
    f = x ** 2
    xt = array([3.0])
    result = lerp_1d(x, f, xt)
    assert_allclose(result, array([4.0]), atol=1e-14)


def test_uniform_filter_1d_identity():
    """Boxcar with size=1 returns input unchanged."""
    data = np.column_stack([np.arange(5.0),
                             np.arange(5.0) ** 2])
    result = uniform_filter_1d(data, 1)
    assert_allclose(result, data, atol=1e-14)


def test_uniform_filter_1d_constant():
    """Boxcar on constant columns returns constants."""
    data = np.ones((5, 3), dtype=np.float64) * 3.0
    result = uniform_filter_1d(data, 3)
    assert_allclose(result, data, atol=1e-14)


def test_uniform_filter_1d_linear():
    """Boxcar of linear ramp preserves linearity at interior."""
    x = np.arange(10.0)
    data = np.column_stack([x, x, 2 * x])
    result = uniform_filter_1d(data, 3)
    assert_allclose(result[1:-1, 0], x[1:-1], atol=1e-14)


def test_gaussian_filter_1d_identity():
    """Gaussian with sigma -> 0 returns input."""
    data = np.column_stack([np.arange(5.0),
                             np.arange(5.0) ** 2])
    result = gaussian_filter_1d(data, 1e-8)
    assert_allclose(result, data, atol=1e-6)


def test_gaussian_filter_1d_constant():
    """Gaussian on constant columns returns constants."""
    data = np.ones((5, 3), dtype=np.float64) * 3.0
    result = gaussian_filter_1d(data, 1.0)
    assert_allclose(result, data, atol=1e-14)


def test_gaussian_filter_1d_smoothing():
    """Gaussian reduces variance of noisy data."""
    rng = np.random.default_rng(12345)
    x = np.arange(20.0)
    data = np.column_stack([x, np.sin(x / 3) + 0.5 * rng.standard_normal(20)])
    result = gaussian_filter_1d(data, 2.0)
    orig_var = np.sum(np.abs(np.diff(data[:, 1])))
    smooth_var = np.sum(np.abs(np.diff(result[:, 1])))
    assert smooth_var < orig_var
