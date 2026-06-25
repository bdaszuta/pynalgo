"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.resample.lagrange_uniform:
           Lagrange interpolation on uniform grids.
"""
from numpy import array, linspace
from numpy.testing import assert_allclose

from pynalgo.resample import (
    interp_barycentric_1d,
    interp_lagrange_uniform,
)


###############################################################################
# Exact polynomial reconstruction
###############################################################################

def test_order1_linear():
    """Order-1 Lagrange = linear interpolation."""
    x = linspace(0, 1, 10)
    f = 3 * x + 2
    xt = linspace(0, 1, 20)
    r = interp_lagrange_uniform(x, f, xt, order=1)
    assert_allclose(r, 3 * xt + 2, atol=1e-14)


def test_order2_quadratic():
    """Order-2 Lagrange recovers x^2 exactly."""
    x = linspace(-1, 1, 20)
    f = x ** 2
    xt = linspace(-0.8, 0.8, 15)
    r = interp_lagrange_uniform(x, f, xt, order=2)
    assert_allclose(r, xt ** 2, atol=1e-14)


def test_order3_cubic():
    """Order-3 Lagrange recovers x^3 exactly."""
    x = linspace(-1, 1, 25)
    f = x ** 3
    xt = linspace(-0.9, 0.9, 20)
    r = interp_lagrange_uniform(x, f, xt, order=3)
    assert_allclose(r, xt ** 3, atol=1e-12)


def test_order5_quintic():
    """Order-5 Lagrange recovers x^5 exactly."""
    x = linspace(-1, 1, 30)
    f = x ** 5
    xt = linspace(-0.5, 0.5, 10)
    r = interp_lagrange_uniform(x, f, xt, order=5)
    assert_allclose(r, xt ** 5, atol=1e-8)


###############################################################################
# Boundary targets
###############################################################################

def test_boundary_left():
    """Left-boundary target uses bias-left stencil."""
    x = linspace(-1, 1, 20)
    f = x ** 2
    xt = array([x[0]])  # first source node
    r = interp_lagrange_uniform(x, f, xt, order=2)
    assert_allclose(r, array([1.0]), atol=1e-14)


def test_boundary_right():
    """Right-boundary target uses bias-right stencil."""
    x = linspace(-1, 1, 20)
    f = x ** 2
    xt = array([x[-1]])  # last source node
    r = interp_lagrange_uniform(x, f, xt, order=2)
    assert_allclose(r, array([1.0]), atol=1e-14)


###############################################################################
# Crosscheck vs BT (arbitrary-grid Lagrange)
###############################################################################

def test_crosscheck_BT():
    """Uniform Lagrange matches BT on uniform grid."""
    x = linspace(-1, 1, 15)
    f = x ** 3 - 2 * x
    xt = linspace(-0.9, 0.9, 25)
    r_lag = interp_lagrange_uniform(x, f, xt, order=3)
    r_bt = interp_barycentric_1d(x, f, xt, method='BT')
    assert_allclose(r_lag, r_bt, atol=1e-12)


###############################################################################
# Edge cases
###############################################################################

def test_nonuniform_raises():
    """Non-uniform grid raises ValueError."""
    x = array([0.0, 1.0, 2.5])
    f = x ** 2
    try:
        interp_lagrange_uniform(x, f, x, order=1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_order_ge_N():
    """order >= N falls back to full BT."""
    x = linspace(-1, 1, 5)
    f = x ** 2
    xt = linspace(-0.5, 0.5, 3)
    r = interp_lagrange_uniform(x, f, xt, order=10)
    r_bt = interp_barycentric_1d(x, f, xt, method='BT')
    assert_allclose(r, r_bt, atol=1e-12)


#
# :D
#
