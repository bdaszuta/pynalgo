"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test rat_D1 and rat_D2 for barycentric rational derivatives.
"""
import numpy as np
from numpy import array, cos, linspace, sin, tanh
from numpy.testing import assert_allclose

from pynalgo.rational import (aaa, aaa_real, eval_rat, rat_D1, rat_D2,
                              rat_D, diff_mat_nodal_rat, rat_eval)


# ---------------------------------------------------------------------------
# Helper: build rational approximant of a known function
# ---------------------------------------------------------------------------

def _make_aaa(fn, x_range=(-1, 1), n_pts=200, tol=1e-10, **kw):
    """Build AAA approximant, return (z, f, w, x_test, f_test, df_test)."""
    x = linspace(*x_range, n_pts)
    f = fn(x)
    z, fv, w = aaa(x, f, tol=tol, **kw)
    return z, fv, w


def _make_aaa_real(fn, x_range=(-1, 1), n_pts=200, tol=1e-10, **kw):
    x = linspace(*x_range, n_pts)
    f = fn(x)
    z, fv, w = aaa_real(x, f, tol=tol, **kw)
    return z, fv, w


# ===========================================================================
# rat_D1 tests
# ===========================================================================

def test_rat_D1_sw_polynomial():
    """x^2 via 3 support points: r'(x) = 2x everywhere."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    df = rat_D1(z, f, w, x_test, method='sw')
    assert_allclose(df, 2 * x_test, atol=1e-14)


def test_rat_D1_sw_at_support():
    """Derivative at support nodes matches known values."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    df = rat_D1(z, f, w, z, method='sw')
    assert_allclose(df, 2 * z, atol=1e-14)


def test_rat_D1_sw_sin():
    """AAA on sin(3x): derivative matches 3*cos(3x)."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-1, 1, 50)
    df = rat_D1(z, fv, w, x_test, method='sw')
    expected = 3 * cos(3 * x_test)
    assert np.max(np.abs(df - expected)) < 1e-5


def test_rat_D1_sw_coincident_mixed():
    """Some target points coincident with support, some not."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    df = rat_D1(z, f, w, x_test, method='sw')
    assert_allclose(df, 2 * x_test, atol=1e-14)


def test_rat_D1_matrix_identity():
    """x^2: D1 @ f = 2*z."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    df = rat_D1(z, f, w, z, method='matrix')
    assert_allclose(df, 2 * z, atol=1e-14)


def test_rat_D1_matrix_requires_x_equals_z():
    """ValueError when x != z for matrix method."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_wrong = array([-1.0, 0.0, 0.5])
    try:
        rat_D1(z, f, w, x_wrong, method='matrix')
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rat_D1_quotient_sin():
    """Cross-check quotient vs SW on sin(3x) AAA."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-1, 1, 30)
    df_sw = rat_D1(z, fv, w, x_test, method='sw')
    df_q = rat_D1(z, fv, w, x_test, method='quotient')
    assert_allclose(df_q, df_sw, rtol=1e-12)


def test_rat_D1_quotient_at_support():
    """Quotient returns derivative (not f-value) at coincident nodes."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    df = rat_D1(z, f, w, z, method='quotient')
    # Bug check: must NOT return f values
    assert np.max(np.abs(df - 2 * z)) < 1e-14
    assert np.max(np.abs(df - f)) > 1e-10  # clearly not f values


def test_rat_D1_crosscheck_all_methods():
    """All three D1 methods agree on well-resolved AAA."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-1, 1, 30)
    df_sw = rat_D1(z, fv, w, x_test, method='sw')
    df_m = rat_D1(z, fv, w, z, method='matrix')
    df_q = rat_D1(z, fv, w, x_test, method='quotient')
    # SW vs quotient on general grid
    assert_allclose(df_q, df_sw, rtol=1e-12)
    # SW at support nodes vs matrix
    df_sw_at_z = rat_D1(z, fv, w, z, method='sw')
    assert_allclose(df_sw_at_z, df_m, rtol=1e-12)


def test_rat_D1_invalid_method():
    """ValueError for unknown method string."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    try:
        rat_D1(z, f, w, z, method='nonexistent')
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ===========================================================================
# rat_D2 tests
# ===========================================================================

def test_rat_D2_sw_polynomial():
    """x^2 via 3 support points: r''(x) = 2 everywhere."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, 2 * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_sw_sin():
    """AAA on sin(3x): D2 matches -9*sin(3x)."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-0.9, 0.9, 30)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    expected = -9 * sin(3 * x_test)
    assert np.max(np.abs(d2f - expected)) < 5e-5


def test_rat_D2_sw_at_support():
    """D2 at support nodes is correct for x^2."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    d2f = rat_D2(z, f, w, z, method='sw')
    assert_allclose(d2f, 2 * np.ones_like(z), atol=1e-14)


def test_rat_D2_matrix_identity():
    """x^2: D1 @ D1 @ f = 2 everywhere."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    d2f = rat_D2(z, f, w, z, method='matrix')
    assert_allclose(d2f, 2 * np.ones_like(z), atol=1e-14)


def test_rat_D2_sw_coincident_mixed():
    """D2 with mixed coincident/non-coincident targets."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, 2 * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_quotient_sin():
    """Cross-check quotient D2 vs SW D2 on sin(3x) AAA."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-0.8, 0.8, 20)
    d2f_sw = rat_D2(z, fv, w, x_test, method='sw')
    d2f_q = rat_D2(z, fv, w, x_test, method='quotient')
    assert_allclose(d2f_q, d2f_sw, rtol=1e-10)


def test_rat_D2_quotient_quadratic():
    """Quotient D2 on exact quadratic: r'' = 2a to machine precision."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    d2f = rat_D2(z, f, w, x_test, method='quotient')
    assert_allclose(d2f, 2 * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_crosscheck_all_methods():
    """All three D2 methods agree on well-resolved AAA."""
    z, fv, w = _make_aaa(lambda x: sin(3 * x))
    x_test = linspace(-0.8, 0.8, 20)
    d2f_sw = rat_D2(z, fv, w, x_test, method='sw')
    d2f_m = rat_D2(z, fv, w, z, method='matrix')
    d2f_q = rat_D2(z, fv, w, x_test, method='quotient')
    # SW vs quotient away from nodes
    assert_allclose(d2f_q, d2f_sw, rtol=1e-10)
    # SW at support nodes vs matrix
    d2f_sw_at_z = rat_D2(z, fv, w, z, method='sw')
    assert_allclose(d2f_sw_at_z, d2f_m, rtol=1e-3)


def test_rat_D2_invalid_method():
    """ValueError for unknown method in D2."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    try:
        rat_D2(z, f, w, z, method='nope')
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ===========================================================================
# Edge case tests
# ===========================================================================

def test_rat_D1_sw_n1_constant():
    """Single support point: derivative is zero everywhere."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    x_test = linspace(-1, 1, 10)
    df = rat_D1(z, f, w, x_test, method='sw')
    assert_allclose(df, np.zeros_like(x_test), atol=1e-14)


def test_rat_D2_sw_n1_constant():
    """Single support point: second derivative is zero."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    x_test = linspace(-1, 1, 10)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, np.zeros_like(x_test), atol=1e-14)


def test_rat_D1_sw_n2_linear():
    """n=2: derivative is constant slope (linear interpolation)."""
    z = array([-1.0, 1.0])
    f = 2 * z + 1  # f(x) = 2x + 1
    w = array([-0.5, 0.5])
    x_test = linspace(-1, 1, 10)
    df = rat_D1(z, f, w, x_test, method='sw')
    assert_allclose(df, 2 * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_sw_n2():
    """n=2: second derivative is zero (linear interpolation)."""
    z = array([-1.0, 1.0])
    f = 2 * z + 1
    w = array([-0.5, 0.5])
    x_test = linspace(-1, 1, 10)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, np.zeros_like(x_test), atol=1e-14)


def test_rat_D1_all_coincident():
    """All targets are support nodes: no NaN, exact for quadratic."""
    z = array([-1.0, 0.0, 1.0])
    f = 3 * z ** 2 + 2 * z + 1
    w = array([-0.5, 1.0, -0.5])
    df = rat_D1(z, f, w, z, method='sw')
    assert_allclose(df, 6 * z + 2, atol=1e-14)


def test_rat_D2_all_coincident():
    """All targets at support nodes: D2 exact for quadratic."""
    z = array([-1.0, 0.0, 1.0])
    f = 3 * z ** 2 + 2 * z + 1
    w = array([-0.5, 1.0, -0.5])
    d2f = rat_D2(z, f, w, z, method='sw')
    assert_allclose(d2f, 6 * np.ones_like(z), atol=1e-14)


def test_rat_D1_complex_weights():
    """AAA with kappa: complex weights, real function."""
    x = linspace(-1, 1, 100)
    fn = lambda x: sin(3 * x)
    f = fn(x)
    z, fv, w = aaa(x, f, tol=1e-10, kappa=1.5)
    x_test = linspace(-1, 1, 20)
    df = rat_D1(z, fv, w, x_test, method='sw')
    expected = 3 * cos(3 * x_test)
    assert np.max(np.abs(df - expected)) < 1e-5


def test_rat_D2_complex_weights():
    """AAA with kappa: D2 with complex weights."""
    x = linspace(-1, 1, 100)
    fn = lambda x: sin(3 * x)
    f = fn(x)
    z, fv, w = aaa(x, f, tol=1e-10, kappa=1.5)
    x_test = linspace(-0.9, 0.9, 15)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    expected = -9 * sin(3 * x_test)
    assert np.max(np.abs(d2f - expected)) < 5e-5


def test_rat_D1_aaa_real():
    """aaa_real weights: D1 works with real-only AAA."""
    z, fv, w = _make_aaa_real(lambda x: sin(3 * x))
    x_test = linspace(-1, 1, 30)
    df = rat_D1(z, fv, w, x_test, method='sw')
    expected = 3 * cos(3 * x_test)
    assert np.max(np.abs(df - expected)) < 1e-5


def test_rat_D2_aaa_real():
    """aaa_real weights: D2 works."""
    z, fv, w = _make_aaa_real(lambda x: sin(3 * x))
    x_test = linspace(-0.9, 0.9, 15)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    expected = -9 * sin(3 * x_test)
    assert np.max(np.abs(d2f - expected)) < 5e-5


def test_rat_D1_quotient_n1_constant():
    """Quotient D1: n=1 returns zeros."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    x_test = linspace(-1, 1, 10)
    df = rat_D1(z, f, w, x_test, method='quotient')
    assert_allclose(df, np.zeros_like(x_test), atol=1e-14)


def test_rat_D2_quotient_n1_constant():
    """Quotient D2: n=1 returns zeros."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    x_test = linspace(-1, 1, 10)
    d2f = rat_D2(z, f, w, x_test, method='quotient')
    assert_allclose(d2f, np.zeros_like(x_test), atol=1e-14)


def test_rat_D1_matrix_n1():
    """Matrix D1: n=1 produces zero."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    df = rat_D1(z, f, w, z, method='matrix')
    assert_allclose(df, np.zeros_like(z), atol=1e-14)


def test_rat_D2_matrix_n1():
    """Matrix D2: n=1 produces zero."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    d2f = rat_D2(z, f, w, z, method='matrix')
    assert_allclose(d2f, np.zeros_like(z), atol=1e-14)


def test_rat_D1_sin_reconstruction_quality():
    """AAA on sin: D1 error should be small relative to function scale."""
    z, fv, w = _make_aaa(lambda x: sin(5 * x), n_pts=300, tol=1e-12)
    x_test = linspace(-1, 1, 100)
    df = rat_D1(z, fv, w, x_test, method='sw')
    df_exact = 5 * cos(5 * x_test)
    err = np.max(np.abs(df - df_exact))
    assert err < 1e-4, f"D1 error {err} too large"


def test_rat_D2_sin_reconstruction_quality():
    """AAA on sin: D2 error should be small."""
    z, fv, w = _make_aaa(lambda x: sin(5 * x), n_pts=300, tol=1e-12)
    x_test = linspace(-0.8, 0.8, 80)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    d2f_exact = -25 * sin(5 * x_test)
    err = np.max(np.abs(d2f - d2f_exact))
    assert err < 1e-2, f"D2 error {err} too large"


# ===========================================================================
# Used by D2 linear/quadratic analytic verification
# ===========================================================================

def test_rat_D1_quadratic_exact():
    """Quadratic function with n=3: exact reconstruction, D1 = 2ax+b."""
    # f(x) = 3x^2 + 2x + 1
    z = array([-1.0, 0.0, 1.0])
    a, b, c = 3.0, 2.0, 1.0
    f = a * z**2 + b * z + c
    # Barycentric weights for degree-2 interpolation on these 3 nodes
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    df = rat_D1(z, f, w, x_test, method='sw')
    assert_allclose(df, 2 * a * x_test + b, atol=1e-14)


def test_rat_D2_linear_exact():
    """Linear function: D2 = 0 exactly."""
    z = array([-1.0, 0.0, 1.0])
    f = 2 * z + 1
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, np.zeros_like(x_test), atol=1e-14)


def test_rat_D2_quadratic_exact():
    """Quadratic: D2 = 2a everywhere."""
    z = array([-1.0, 0.0, 1.0])
    a = 3.0
    f = a * z**2 + 2 * z + 1
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 20)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, 2 * a * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_cubic_exact():
    """With 3 nodes, cubic f reduces to quadratic interpolant. D2 = 2b."""
    z = array([-1.0, 0.0, 1.0])
    a, b = 2.0, 3.0
    f = a * z**3 + b * z**2 + z + 1
    w = array([-0.5, 1.0, -0.5])
    # With 3 nodes, rational function = quadratic interpolant through
    # (-1, f(-1)), (0, f(0)), (1, f(1)) = b*x^2 + (a+c)*x + d.
    # r'' = 2b = 6 everywhere.
    x_test = linspace(-1, 1, 20)
    d2f = rat_D2(z, f, w, x_test, method='sw')
    assert_allclose(d2f, 2 * b * np.ones_like(x_test), atol=1e-14)


def test_rat_D2_constant_function():
    """Constant function: D2 = 0."""
    z, fv, w = _make_aaa(lambda x: 5 * np.ones_like(x))
    x_test = linspace(-1, 1, 10)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    assert_allclose(d2f, np.zeros_like(x_test), atol=1e-14)
    # Also verify D1
    df = rat_D1(z, fv, w, x_test, method='sw')
    assert_allclose(df, np.zeros_like(x_test), atol=1e-14)


def test_rat_D2_tanh():
    """tanh(10x): D2 works through sharp transition."""
    z, fv, w = _make_aaa(lambda x: tanh(10 * x), n_pts=400, tol=1e-10)
    x_test = linspace(-0.5, 0.5, 30)
    d2f = rat_D2(z, fv, w, x_test, method='sw')
    # D2 of tanh(10x) = -200*tanh(10x)*sech(10x)^2
    t = tanh(10 * x_test)
    expected = -200 * t * (1 - t**2)
    # Relaxed tolerance: AAA has many points near transition
    assert np.max(np.abs(d2f - expected)) < 0.5


def test_rat_D1_abs():
    """abs(x): D1 at points away from x=0 matches sign."""
    z, fv, w = _make_aaa(lambda x: np.abs(x), n_pts=400, tol=1e-8)
    x_test = linspace(-1, -0.1, 15)
    df = rat_D1(z, fv, w, x_test, method='sw')
    assert np.all(df < -0.9)  # derivative ~ -1 for x < 0
    x_test = linspace(0.1, 1, 15)
    df = rat_D1(z, fv, w, x_test, method='sw')
    assert np.all(df > 0.9)   # derivative ~ +1 for x > 0


# ===========================================================================
# Phase A: diff_mat_nodal_rat tests
# ===========================================================================

def test_diff_mat_nodal_rat_k1_quadratic():
    """k=1 on x^2: D1 @ f = 2z."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    D1 = diff_mat_nodal_rat(z, w, k=1)
    df = D1 @ f
    assert_allclose(df, 2 * z, atol=1e-14)


def test_diff_mat_nodal_rat_k2_quadratic():
    """k=2 on x^2: D2 @ f = 2 everywhere."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    D2 = diff_mat_nodal_rat(z, w, k=2)
    d2f = D2 @ f
    assert_allclose(d2f, 2 * np.ones_like(z), atol=1e-14)


def test_diff_mat_nodal_rat_k3_cubic():
    """k=3 on x^3: D3 @ f = 6 everywhere (at least 4 nodes needed)."""
    z = array([-1.0, -0.5, 0.5, 1.0])
    f = z ** 3
    # BT weights: w_k = 1/prod_{j!=k}(z_k - z_j)
    w = array([-2/3, 4/3, -4/3, 2/3])
    D3 = diff_mat_nodal_rat(z, w, k=3)
    d3f = D3 @ f
    assert_allclose(d3f, 6 * np.ones_like(z), atol=1e-14)


def test_diff_mat_nodal_rat_n1():
    """n=1 returns [[0]]."""
    D1 = diff_mat_nodal_rat(array([0.5]), array([1.0]), k=1)
    assert D1.shape == (1, 1)
    assert D1[0, 0] == 0.0
    D3 = diff_mat_nodal_rat(array([0.5]), array([1.0]), k=3)
    assert D3[0, 0] == 0.0


def test_diff_mat_nodal_rat_aaa_sin():
    """Dk on AAA sin(3x) matches analytic at support nodes."""
    x = linspace(-1, 1, 200)
    f = sin(3 * x)
    z, fv, w = aaa_real(x, f, tol=1e-10)
    D1 = diff_mat_nodal_rat(z, w, k=1)
    df = D1 @ fv
    df_exact = 3 * cos(3 * z)
    assert np.max(np.abs(df - df_exact)) < 1e-4
    D2 = diff_mat_nodal_rat(z, w, k=2)
    d2f = D2 @ fv
    d2f_exact = -9 * sin(3 * z)
    assert np.max(np.abs(d2f - d2f_exact)) < 5e-3


# ===========================================================================
# Phase B: arbitrary-order rat_D tests
# ===========================================================================

def test_rat_D_k3_quartic():
    """k=3 on x^4 via 5 BT points."""
    z = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    f = z ** 4
    # BT weights
    w = np.empty(5)
    for k in range(5):
        w[k] = 1.0 / np.prod([z[k] - z[j] for j in range(5) if j != k])
    x_test = linspace(-1, 1, 10)
    d3f = rat_D(z, f, w, x_test, k=3, method='sw')
    assert_allclose(d3f, 24 * x_test, atol=1e-13)
    # Crosscheck matrix at support nodes
    d3f_mat = rat_D(z, f, w, z, k=3, method='matrix')
    assert_allclose(d3f_mat, 24 * z, atol=1e-13)


def test_rat_D_k4_quintic():
    """k=4 on x^5 via 6 BT points."""
    z = array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    f = z ** 5
    w = np.empty(6)
    for k in range(6):
        w[k] = 1.0 / np.prod([z[k] - z[j] for j in range(6) if j != k])
    x_test = linspace(-1, 1, 10)
    d4f = rat_D(z, f, w, x_test, k=4, method='sw')
    assert_allclose(d4f, 120 * x_test, atol=1e-12)


def test_rat_D_k3_sin_aaa():
    """AAA on sin(3x): k=3 matches -27*cos(3x) within tolerance."""
    x = linspace(-1, 1, 300)
    f = sin(3 * x)
    z, fv, w = aaa_real(x, f, tol=1e-12)
    x_test = linspace(-0.8, 0.8, 20)
    d3f = rat_D(z, fv, w, x_test, k=3, method='sw')
    expected = -27 * cos(3 * x_test)
    assert np.max(np.abs(d3f - expected)) < 1e-3


def test_rat_D_k1_matches_rat_D1():
    """k=1 via rat_D matches rat_D1."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    assert_allclose(rat_D(z, f, w, x_test, k=1, method='sw'),
                    rat_D1(z, f, w, x_test, method='sw'), atol=1e-14)
    assert_allclose(rat_D(z, f, w, z, k=1, method='matrix'),
                    rat_D1(z, f, w, z, method='matrix'), atol=1e-14)


def test_rat_D_k2_matches_rat_D2():
    """k=2 via rat_D matches rat_D2."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    assert_allclose(rat_D(z, f, w, x_test, k=2, method='sw'),
                    rat_D2(z, f, w, x_test, method='sw'), atol=1e-14)


def test_rat_D_n1():
    """n=1 returns zeros for any k."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    for k in [1, 2, 3]:
        df = rat_D(z, f, w, array([0.0, 1.0]), k=k, method='sw')
        assert_allclose(df, np.zeros(2), atol=1e-14)


def test_rat_D_node_k3():
    """k=3 at support nodes matches matrix method."""
    z = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    f = z ** 4
    w = np.empty(5)
    for k in range(5):
        w[k] = 1.0 / np.prod([z[k] - z[j] for j in range(5) if j != k])
    d3f_sw = rat_D(z, f, w, z, k=3, method='sw')
    d3f_mat = rat_D(z, f, w, z, k=3, method='matrix')
    assert_allclose(d3f_sw, d3f_mat, atol=1e-13)


# ===========================================================================
# Phase C: rat_eval tests
# ===========================================================================

def test_rat_eval_maxderiv0():
    """max_deriv=0 returns (r,) matching eval_rat."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    r = rat_eval(z, f, w, x_test, max_deriv=0, method='sw')
    assert isinstance(r, list)
    assert len(r) == 1
    assert_allclose(r[0], eval_rat(z, f, w, x_test), atol=1e-14)


def test_rat_eval_maxderiv1():
    """max_deriv=1 returns (r, dr) matching individual calls."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    r, dr = rat_eval(z, f, w, x_test, max_deriv=1, method='sw')
    assert_allclose(r, eval_rat(z, f, w, x_test), atol=1e-14)
    assert_allclose(dr, rat_D1(z, f, w, x_test, method='sw'), atol=1e-14)


def test_rat_eval_maxderiv2():
    """max_deriv=2 returns (r, dr, d2r)."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    r, dr, d2r = rat_eval(z, f, w, x_test, max_deriv=2, method='sw')
    assert_allclose(r, eval_rat(z, f, w, x_test), atol=1e-14)
    assert_allclose(dr, rat_D1(z, f, w, x_test, method='sw'), atol=1e-14)
    assert_allclose(d2r, rat_D2(z, f, w, x_test, method='sw'), atol=1e-14)


def test_rat_eval_maxderiv3():
    """max_deriv=3 on x^4."""
    z = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    f = z ** 4
    w = np.empty(5)
    for k in range(5):
        w[k] = 1.0 / np.prod([z[k] - z[j] for j in range(5) if j != k])
    x_test = linspace(-1, 1, 5)
    result = rat_eval(z, f, w, x_test, max_deriv=3, method='sw')
    assert len(result) == 4
    assert_allclose(result[1], 4 * x_test**3, atol=1e-12)
    assert_allclose(result[2], 12 * x_test**2, atol=1e-12)
    assert_allclose(result[3], 24 * x_test, atol=1e-12)


def test_rat_eval_n1():
    """n=1 returns zeros."""
    z = array([0.5])
    f = array([3.0])
    w = array([1.0])
    result = rat_eval(z, f, w, array([0.0, 1.0]), max_deriv=2, method='sw')
    assert_allclose(result[0], 3 * np.ones(2), atol=1e-14)
    assert_allclose(result[1], np.zeros(2), atol=1e-14)
    assert_allclose(result[2], np.zeros(2), atol=1e-14)


def test_rat_eval_coincident():
    """Coincident nodes handled correctly."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    r, dr, d2r = rat_eval(z, f, w, z, max_deriv=2, method='sw')
    assert_allclose(r, f, atol=1e-14)
    assert_allclose(dr, 2 * z, atol=1e-14)
    assert_allclose(d2r, 2 * np.ones_like(z), atol=1e-14)


def test_rat_eval_vs_rat_D():
    """rat_eval[max_deriv=k][k] == rat_D(k=k) for multiple k."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    x_test = linspace(-1, 1, 10)
    result = rat_eval(z, f, w, x_test, max_deriv=3, method='sw')
    for k in range(1, 4):
        assert_allclose(result[k], rat_D(z, f, w, x_test, k=k, method='sw'),
                        atol=1e-14)


# :D
