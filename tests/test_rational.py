"""
Test pynalgo.rational: AAA rational approximation.

@author: Boris Daszuta
"""
# pylint: skip-file

import numpy as np
from numpy import (array, linspace, sin)
from numpy.testing import assert_allclose

from pynalgo.rational import aaa, aaa_real, eval_rat, poles_residues


def test_eval_rat_constant():
    """Constant rational function evaluated anywhere gives the constant."""
    z = array([0.0, 1.0])
    f = array([3.0, 3.0])
    w = array([1.0, 1.0])
    z_test = linspace(-1, 2, 10)
    result = eval_rat(z, f, w, z_test)
    assert_allclose(result, 3.0 * np.ones_like(z_test), atol=1e-14)


def test_eval_rat_at_support():
    """Evaluate at support points recovers exact function values."""
    z = array([-1.0, 0.0, 1.0])
    f = array([2.0, 0.0, 1.0])
    w = array([1.0, -2.0, 1.0])
    result = eval_rat(z, f, w, z)
    assert_allclose(result, f, atol=1e-14)


def test_eval_rat_polynomial():
    """Reconstruct x^2 from three support points."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    z_test = linspace(-1, 1, 20)
    result = eval_rat(z, f, w, z_test)
    assert_allclose(result, z_test ** 2, atol=1e-14)


def test_eval_rat_coincident_mixed():
    """Coincident and non-coincident targets mixed -- no NaN, exact at nodes."""
    z = array([-1.0, 0.0, 1.0])
    f = z ** 2
    w = array([-0.5, 1.0, -0.5])
    z_test = array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = eval_rat(z, f, w, z_test)
    assert_allclose(result, z_test ** 2, atol=1e-14)


def test_aaa_sin():
    """AAA on sin(3x) on [-1,1] should converge to high accuracy."""
    x = linspace(-1, 1, 200)
    f = sin(3 * x)
    z, fv, w = aaa(x, f, tol=1e-10)
    assert z.size > 5
    x_test = linspace(-1, 1, 50)
    f_test = eval_rat(z, fv, w, x_test)
    err = np.max(np.abs(f_test - sin(3 * x_test)))
    assert err < 1e-6


def test_aaa_constant():
    """AAA on constant function should use 1 support point."""
    x = linspace(-1, 1, 100)
    f = 5.0 * np.ones_like(x)
    z, fv, w = aaa(x, f, tol=1e-13)
    x_test = linspace(-1, 1, 20)
    f_test = eval_rat(z, fv, w, x_test)
    assert_allclose(f_test, 5.0 * np.ones_like(x_test), atol=1e-12)


def test_aaa_required_nodes():
    """Required nodes should appear in the support set (within grid spacing)."""
    x = linspace(-1, 1, 101)
    f = np.abs(x)
    z_req = array([-1.0, 0.0, 1.0])
    z, fv, w = aaa(x, f, tol=1e-10, z_required=z_req, max_terms=20)
    dx = x[1] - x[0]
    for req in z_req:
        assert np.min(np.abs(z - req)) < dx


def test_aaa_returns_sorted():
    """AAA should return support points sorted ascending."""
    x = linspace(-1, 1, 100)
    f = sin(5 * x)
    z, fv, w = aaa(x, f, tol=1e-10, max_terms=10)
    assert np.all(np.diff(z) > 0)


def test_poles_residues_real():
    """Poles from real-valued rational function should be finite."""
    x = linspace(-3, 3, 200)
    f = 1.0 / (1.0 + x ** 2)
    z, fv, w = aaa(x, f, tol=1e-10, max_terms=20)
    pol, res = poles_residues(z, fv, w)
    assert np.all(np.isfinite(pol))


def test_aaa_real_sin():
    """aaa_real on sin(3x) should converge (real weights, no kappa)."""
    x = linspace(-1, 1, 200)
    f = sin(3 * x)
    z, fv, w = aaa_real(x, f, tol=1e-10)
    assert z.size > 5
    x_test = linspace(-1, 1, 50)
    f_test = eval_rat(z, fv, w, x_test)
    err = np.max(np.abs(f_test - sin(3 * x_test)))
    assert err < 1e-6


def test_aaa_kappa():
    """AAA with kappa=1.5 on sin(3x) converges (complex weight scaling)."""
    x = linspace(-1, 1, 200)
    f = sin(3 * x)
    z, fv, w = aaa(x, f, tol=1e-10, kappa=1.5)
    assert z.size > 5
    x_test = linspace(-1, 1, 50)
    f_test = eval_rat(z, fv, w, x_test)
    err = np.max(np.abs(f_test - sin(3 * x_test)))
    assert err < 1e-6
