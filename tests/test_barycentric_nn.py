"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.resample.barycentric_nn:
           nearest-neighbor Floater-Hormann interpolation.
"""
import numpy as np
from numpy import array, linspace, sin, cos
from numpy.testing import assert_allclose

from pynalgo.resample import (
    interp_barycentric_1d,
    interp_nn_1d,
)


###############################################################################
# W=N: matches standard
###############################################################################

def test_nn_W_equals_N_fh():
    """W=N matches standard FH."""
    x = linspace(-1, 1, 12)
    f = sin(3 * x)
    xt = linspace(-0.9, 0.9, 20)
    r_nn = interp_nn_1d(x, f, xt, d=4, method=2, W=12)
    r_fh = interp_barycentric_1d(x, f, xt, d=4, method='FH')
    assert_allclose(r_nn, r_fh, atol=1e-12)


def test_nn_W_equals_N_bt():
    """W=N matches standard BT."""
    x = linspace(-1, 1, 10)
    f = cos(4 * x)
    xt = linspace(-0.8, 0.8, 15)
    r_nn = interp_nn_1d(x, f, xt, d=0, method=3, W=10)
    r_bt = interp_barycentric_1d(x, f, xt, d=0, method='BT')
    assert_allclose(r_nn, r_bt, atol=1e-12)


###############################################################################
# W < N: exact polynomial reconstruction
###############################################################################

def test_nn_W3_polynomial():
    """W=3 on quadratic: exact reconstruction."""
    x = linspace(-1, 1, 20)
    f = x ** 2
    xt = linspace(-0.5, 0.5, 5)
    r = interp_nn_1d(x, f, xt, d=2, method=2, W=3)
    assert_allclose(r, xt ** 2, atol=1e-12)


def test_nn_W5_polynomial():
    """W=5 on quartic: exact reconstruction."""
    x = linspace(-1, 1, 25)
    f = x ** 4
    xt = linspace(-0.3, 0.3, 7)
    r = interp_nn_1d(x, f, xt, d=3, method=2, W=5)
    assert_allclose(r, xt ** 4, atol=1e-12)


def test_nn_W2_linear():
    """W=2 on linear: exact reconstruction."""
    x = linspace(-1, 1, 10)
    f = 3 * x + 1
    xt = linspace(-0.9, 0.9, 10)
    r = interp_nn_1d(x, f, xt, d=1, method=2, W=2)
    assert_allclose(r, 3 * xt + 1, atol=1e-12)


###############################################################################
# Boundary
###############################################################################

def test_nn_boundary():
    """NN stencil shifts correctly at boundary."""
    x = linspace(-1, 1, 30)
    f = sin(4 * x)
    xt = array([-0.95, 0.95])  # near boundaries
    r = interp_nn_1d(x, f, xt, d=2, method=2, W=4)
    assert np.all(np.isfinite(r))


###############################################################################
# Coincident
###############################################################################

def test_nn_coincident():
    """Target == source node returns exact value."""
    x = linspace(-1, 1, 15)
    f = sin(6 * x)
    r = interp_nn_1d(x, f, x, d=2, method=2, W=5)
    assert_allclose(r, f, atol=1e-14)


###############################################################################
# Edge cases
###############################################################################

def test_nn_W_too_small():
    """W < 2 raises ValueError."""
    x = linspace(-1, 1, 10)
    f = x ** 2
    try:
        interp_nn_1d(x, f, array([0.5]), d=2, method=2, W=1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_nn_W_larger_than_N():
    """W > N falls back to full interpolation (same as W=N)."""
    x = linspace(-1, 1, 8)
    f = sin(3 * x)
    xt = linspace(-0.5, 0.5, 5)
    r_nn = interp_nn_1d(x, f, xt, d=2, method=2, W=20)
    r_fh = interp_barycentric_1d(x, f, xt, d=2, method='FH')
    assert_allclose(r_nn, r_fh, atol=1e-12)


#
# :D
#
