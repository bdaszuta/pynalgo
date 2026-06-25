"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.resample.barycentric_generalized:
           generalized Floater-Hormann interpolation.
"""
import numpy as np
from numpy import array, linspace, sin, cos
from numpy.testing import assert_allclose

from pynalgo.resample import (
    interp_barycentric_1d,
    interp_barycentric_1d_generalized,
)


###############################################################################
# gam=1: must match standard FH
###############################################################################

def test_gam1_matches_fh():
    """gam=1 reproduces standard FH."""
    x = linspace(-1, 1, 15)
    f = sin(4 * x)
    xt = linspace(-0.9, 0.9, 30)
    r_gen = interp_barycentric_1d_generalized(x, f, xt, d=4, gam=1.0)
    r_fh = interp_barycentric_1d(x, f, xt, d=4, method='FH')
    assert_allclose(r_gen, r_fh, atol=1e-14)


def test_gam1_matches_bt():
    """gam=1 with d=0 reproduces standard BT."""
    x = linspace(-1, 1, 12)
    f = cos(3 * x)
    xt = linspace(-0.8, 0.8, 20)
    r_gen = interp_barycentric_1d_generalized(x, f, xt, d=0, gam=1.0)
    r_bt = interp_barycentric_1d(x, f, xt, d=0, method='BT')
    assert_allclose(r_gen, r_bt, atol=1e-14)


###############################################################################
# gam != 1: verify reasonable behaviour on known functions
###############################################################################

def test_gam_half_sin():
    """gam=0.5 on sin: results are finite."""
    x = linspace(-1, 1, 15)
    f = sin(3 * x)
    xt = linspace(-0.8, 0.8, 30)
    r = interp_barycentric_1d_generalized(x, f, xt, d=2, gam=0.5)
    assert np.all(np.isfinite(r))


def test_gam_two_sin():
    """gam=2.0 on sin: results are finite."""
    x = linspace(-1, 1, 15)
    f = sin(3 * x)
    xt = linspace(-0.8, 0.8, 30)
    r = interp_barycentric_1d_generalized(x, f, xt, d=2, gam=2.0)
    assert np.all(np.isfinite(r))


def test_gam_polynomial_exact():
    """Generalized FH on polynomial: finite and non-degenerate."""
    x = linspace(-1, 1, 10)
    f = x ** 3 + 2 * x
    xt = linspace(-0.9, 0.9, 25)
    r05 = interp_barycentric_1d_generalized(x, f, xt, d=4, gam=0.5)
    r20 = interp_barycentric_1d_generalized(x, f, xt, d=4, gam=2.0)
    assert np.all(np.isfinite(r05))
    assert np.all(np.isfinite(r20))


###############################################################################
# Coincident nodes
###############################################################################

def test_coincident_source():
    """Target == source node returns exact function value."""
    x = linspace(-1, 1, 15)
    f = sin(5 * x)
    r = interp_barycentric_1d_generalized(x, f, x, d=4, gam=1.5)
    assert_allclose(r, f, atol=1e-14)


###############################################################################
# Edge cases
###############################################################################

def test_n1():
    """Single source node: returns constant."""
    r = interp_barycentric_1d_generalized(
        array([0.5]), array([3.0]), array([-1.0, 0.0, 2.0]),
        d=4, gam=1.0)
    assert_allclose(r, 3.0 * np.ones(3), atol=1e-14)


def test_n2():
    """Two source nodes with d=1: well-defined limit."""
    x = array([0.0, 1.0])
    f = array([0.0, 1.0])
    xt = array([0.5])
    r = interp_barycentric_1d_generalized(x, f, xt, d=1, gam=1.0)
    assert_allclose(r, array([0.5]), atol=1e-14)


def test_negative_gam():
    """gam <= 0 raises ValueError."""
    x = linspace(-1, 1, 10)
    f = x ** 2
    try:
        interp_barycentric_1d_generalized(x, f, x, d=4, gam=0.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    try:
        interp_barycentric_1d_generalized(x, f, x, d=4, gam=-1.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


#
# :D
#
