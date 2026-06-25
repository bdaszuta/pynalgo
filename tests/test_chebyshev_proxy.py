"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for Chebyshev proxy rootfinder.
"""

import numpy as np
from numba import njit

from pynalgo.root_finding import chebyshev_proxy


@njit
def f_poly1(x):
  """x^2 - 4, roots at -2, 2."""
  return x ** 2 - 4.0


@njit
def f_poly2(x):
  """x^2 + x - 2 = (x-1)(x+2), roots at -2, 1."""
  return x ** 2 + x - 2.0


@njit
def f_sin(x):
  """sin(x), roots at k*pi."""
  return np.sin(x)


@njit
def f_exp_sin(x):
  """exp(-x^2) - 0.5, two symmetric roots."""
  return np.exp(-x ** 2) - 0.5


@njit
def f_none(x):
  """x^2 + 1, no real roots."""
  return x ** 2 + 1.0


@njit
def f_linear(x):
  """2*x - 3, root at 1.5."""
  return 2.0 * x - 3.0


@njit
def f_cubic(x):
  """(x+1)(x-2)(x-3) = x^3 - 4x^2 + x + 6."""
  return x ** 3 - 4.0 * x ** 2 + x + 6.0


@njit
def f_twisted(x):
  """Old test function: cos(2*(x^2*cos(cos(8*pi*x)) - 1/2))."""
  return np.cos(2.0 * (x ** 2 * np.cos(np.cos(x * 8.0 * np.pi)) - 0.5))


class TestChebyshevProxy:
  """Unit tests for chebyshev_proxy rootfinder."""

  def test_linear(self):
    roots = chebyshev_proxy(f_linear, -10.0, 10.0)
    assert len(roots) == 1
    assert abs(roots[0] - 1.5) < 1e-12

  def test_quadratic_symmetric(self):
    roots = chebyshev_proxy(f_poly1, -5.0, 5.0)
    assert len(roots) == 2
    assert abs(roots[0] - (-2.0)) < 1e-12
    assert abs(roots[1] - 2.0) < 1e-12

  def test_quadratic_asymmetric(self):
    roots = chebyshev_proxy(f_poly2, -3.0, 3.0)
    assert len(roots) == 2
    assert abs(roots[0] - (-2.0)) < 1e-12
    assert abs(roots[1] - 1.0) < 1e-12

  def test_sin_basic(self):
    roots = chebyshev_proxy(f_sin, -0.5, 7.0)
    expected = np.array([0.0, np.pi, 2.0 * np.pi])
    assert len(roots) == 3
    assert np.allclose(roots, expected, atol=1e-10)

  def test_exp_sin(self):
    roots = chebyshev_proxy(f_exp_sin, -3.0, 3.0)
    assert len(roots) == 2
    r_exp = np.sqrt(np.log(2.0))
    assert abs(abs(roots[0]) - r_exp) < 1e-10
    assert abs(abs(roots[1]) - r_exp) < 1e-10

  def test_no_roots(self):
    roots = chebyshev_proxy(f_none, -5.0, 5.0)
    assert len(roots) == 0

  def test_cubic_three(self):
    roots = chebyshev_proxy(f_cubic, -5.0, 5.0)
    assert len(roots) == 3
    expected = np.array([-1.0, 2.0, 3.0])
    assert np.allclose(roots, expected, atol=1e-10)

  def test_twisted_residuals(self):
    """All returned roots should satisfy |f(x)| < 1e-10."""
    roots = chebyshev_proxy(f_twisted, -4.0, 4.0)
    assert len(roots) > 0
    residuals = np.abs(f_twisted(roots))
    assert np.all(residuals < 1e-10)

  def test_partial_interval(self):
    """Root interval that captures only some roots."""
    roots = chebyshev_proxy(f_poly1, 0.0, 5.0)
    assert len(roots) == 1
    assert abs(roots[0] - 2.0) < 1e-12

  def test_narrow_interval_with_root(self):
    """Tight bracket containing a root."""
    roots = chebyshev_proxy(f_linear, 1.4, 1.6)
    assert len(roots) == 1
    assert abs(roots[0] - 1.5) < 1e-12

  def test_bisection_fallback(self):
    """Force bisection by setting N_max small."""
    roots = chebyshev_proxy(f_poly1, -5.0, 5.0, N_init=4, N_max=8)
    assert len(roots) == 2
    assert abs(roots[0] - (-2.0)) < 1e-10
    assert abs(roots[1] - 2.0) < 1e-10

#
# :D
#
