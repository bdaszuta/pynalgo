"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Chebyshev proxy root-extraction.

Refs:
[1] Computing Zeros on a Real Interval through Chebyshev Expansion
    and Polynomial Rootfinding

    Author(s): John P. Boyd
    Source: SIAM Journal on Numerical Analysis, Vol. 40, No. 5 (2003),
    pp. 1666-1682

[2] Finding the Zeros of a Univariate Equation: Proxy Rootfinders,
    Chebyshev Interpolation, and the Companion Matrix

    Boyd, 2013
"""
import numpy as np
from numpy import (arange, cos, float64, pi, sort, zeros)

from pynalgo.common_tools import JIT, TYPE_CHECKING
from pynalgo.spectral._dct import dct1


def _cheb_extrema_grid(a, b, N):
  """Chebyshev extrema grid on [a,b] with N intervals (N+1 points).

  Grid ordering: -1 -> +1 on the reference interval.
  """
  k = arange(N + 1)
  x_ref = -cos(pi * k / N)
  return (b - a) / 2.0 * x_ref + (b + a) / 2.0


def _cheb_coeffs_from_extrema(f_m):
  r"""Chebyshev coefficients a_0..a_{M-1} from M extrema-grid values.

  Maps values on the Chebyshev extrema grid
  :math:`x_j = -\\cos(\\pi j/(M-1))` for :math:`j=0..M-1`
  to expansion coefficients via DCT-I."""

  M = f_m.size
  k = arange(M)
  ph_e = (-1.0) ** k
  nor_e = 2.0 * zeros(M)
  nor_e[1:] = 2.0
  nor_e[0] = 1.0

  dct_val = dct1(f_m)
  a_k = ph_e * (dct_val - f_m[0] - ph_e * f_m[-1]) / 2.0
  a_k += (ph_e * f_m[0] + f_m[-1]) / 2.0
  a_k = nor_e * a_k / (M - 1)

  return a_k


def _coeff_strip_zero(a_n):
  """Strip trailing Chebyshev coefficients below machine epsilon."""
  if a_n.size == 0:
    return a_n
  a_n_abs_max = np.abs(a_n).max()
  if a_n_abs_max == 0.0:
    return a_n[:1]
  a_n_norm = np.abs(a_n) / a_n_abs_max
  eps = np.finfo(float64).eps

  last_good = 0
  for j in range(a_n.size - 1, -1, -1):
    if a_n_norm[j] > eps:
      last_good = j
      break

  return a_n[:last_good + 1]


def _cheb_companion_matrix(a_n):
  r"""Chebyshev companion matrix (Boyd 2013, Eq. 12-13).

  The eigenvalues of this matrix are the roots in :math:`[-1,1]` of the
  Chebyshev expansion :math:`\\sum_{k=0}^N a_k T_k(x) = 0`.

  References
  ----------
  Boyd, J.P. (2013). Finding the Zeros of a Univariate Equation:
  Proxy Rootfinders, Chebyshev Interpolation, and the Companion Matrix.
  """
  N = a_n.size - 1
  C = zeros((N, N), dtype=float64)
  if N <= 0:
    return C

  if N == 1:
    C[0, 0] = -a_n[0] / a_n[1]
    return C

  C[0, 1] = 1.0
  for j in range(1, N):
    C[j, j - 1] = 0.5
    if j + 1 < N:
      C[j, j + 1] = 0.5

  inv_denom = 1.0 / (2.0 * a_n[-1])
  for j in range(N):
    C[N - 1, j] = C[N - 1, j] - a_n[j] * inv_denom

  return C


def _init_coarse(a, b, N, func):
  """Evaluate func on N-point extrema grid, return (f_c_N, an_f_ci_N).

  f_c_N:    function values (size N+1)
  an_f_ci_N: zero-padded coarse coefficients (size 2*N+1)
  """
  x_gr_c = _cheb_extrema_grid(a, b, N)
  f_c_N = func(x_gr_c)
  an_f_ci_N = zeros(2 * N + 1, dtype=float64)
  an_f_ci_N[:N + 1] = _cheb_coeffs_from_extrema(f_c_N)
  return f_c_N, an_f_ci_N


def _chebyshev_proxy_impl(a, b, N, N_max, func, coeff_err_tol, im_abs_tol,
                           f_c_N, an_f_ci_N):
  """Recursive core of the Chebyshev proxy rootfinder.

  Carries coarse-grid function values and zero-padded coefficients
  through recursion levels to avoid recomputation.  All parameters
  are concrete (no Optional types) so the function is JIT-safe.
  """
  if 2 * N > N_max:
    mid = a + (b - a) / 2.0
    fc_1, aci_1 = _init_coarse(a, mid, 8, func)
    fc_2, aci_2 = _init_coarse(mid, b, 8, func)
    roots_1 = _chebyshev_proxy_impl(
        a, mid, 8, N_max, func, coeff_err_tol, im_abs_tol, fc_1, aci_1)
    roots_2 = _chebyshev_proxy_impl(
        mid, b, 8, N_max, func, coeff_err_tol, im_abs_tol, fc_2, aci_2)

    if roots_1.size == 0:
      return roots_2
    if roots_2.size == 0:
      return roots_1
    result = zeros(roots_1.size + roots_2.size, dtype=float64)
    result[:roots_1.size] = roots_1
    result[roots_1.size:] = roots_2
    return sort(result)

  N_fine = 2 * N
  x_gr_f = _cheb_extrema_grid(a, b, N_fine)
  f_f_N = zeros(N_fine + 1, dtype=float64)

  for j in range(N + 1):
    f_f_N[2 * j] = f_c_N[j]

  for j in range(N):
    f_f_N[2 * j + 1] = func(x_gr_f[2 * j + 1])

  an_f_f_N = _cheb_coeffs_from_extrema(f_f_N)
  diff_err = np.abs(an_f_ci_N - an_f_f_N).sum()

  if diff_err > coeff_err_tol:
    an_f_fi_N = zeros(4 * N + 1, dtype=float64)
    an_f_fi_N[:2 * N + 1] = an_f_f_N
    return _chebyshev_proxy_impl(
        a, b, N_fine, N_max, func, coeff_err_tol, im_abs_tol,
        f_f_N, an_f_fi_N)

  an_strip_z = _coeff_strip_zero(an_f_ci_N[:N + 1])

  if an_strip_z.size <= 1:
    return zeros(0, dtype=float64)

  # Modified Chebyshev companion matrix: bottom row adds
  # -a_j/(2*a_N) to the 0.5 superdiagonal entries from
  # the standard construction (Boyd 2013, Eq. 12-13).
  C = _cheb_companion_matrix(an_strip_z)
  if C.shape[0] == 0:
    return zeros(0, dtype=float64)

  eig = np.linalg.eigvals(C.astype(np.complex128))
  valid = (np.abs(eig.imag) <= im_abs_tol) & (np.abs(eig.real) <= 1.0)
  roots_cur = eig[valid].real

  if roots_cur.size == 0:
    return zeros(0, dtype=float64)

  return sort((b - a) / 2.0 * roots_cur + (b + a) / 2.0)


def chebyshev_proxy(func, a, b, N_init=8, N_max=100,
                    coeff_err_tol=1e-10, im_abs_tol=1e-10):
  r"""Find all roots of func on [a, b] via Chebyshev spectral proxy.

  Approximates func with a Chebyshev polynomial, resolves until
  spectral convergence, then finds all roots simultaneously via the
  companion-matrix eigenvalue method.

  Parameters
  ----------
  func : callable
    Callable f(x) -> 1D float64 array; should be compatible with Numba JIT.
  a, b : float
    Interval endpoints.
  N_init : int
    Initial Chebyshev polynomial degree (default 8).
  N_max : int
    Maximum degree before bisecting the domain (default 100).
  coeff_err_tol : float
    Spectral convergence tolerance (default 1e-10).
  im_abs_tol : float
    Threshold for imaginary-part root rejection (default 1e-10).

  Returns
  -------
  roots : ndarray[float64]
    Sorted roots on [a, b]. Empty if none found.

  References
  ----------
  Boyd, J.P. (2003). Computing Zeros on a Real Interval through
  Chebyshev Expansion and Polynomial Rootfinding. SIAM J. Numer.
  Anal. 40(5), 1666-1682.

  Boyd, J.P. (2013). Finding the Zeros of a Univariate Equation:
  Proxy Rootfinders, Chebyshev Interpolation, and the Companion
  Matrix.
  """
  fc, aci = _init_coarse(a, b, N_init, func)
  return _chebyshev_proxy_impl(
      a, b, N_init, N_max, func, coeff_err_tol, im_abs_tol, fc, aci)


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _cheb_extrema_grid = JIT(_cheb_extrema_grid)
  _cheb_coeffs_from_extrema = JIT(_cheb_coeffs_from_extrema)
  _coeff_strip_zero = JIT(_coeff_strip_zero)
  _cheb_companion_matrix = JIT(_cheb_companion_matrix)
  _init_coarse = JIT(_init_coarse)
  _chebyshev_proxy_impl = JIT(_chebyshev_proxy_impl)
  chebyshev_proxy = JIT(chebyshev_proxy)

#
# :D
#
