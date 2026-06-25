"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Calculation of Hermite polynomial and function families.

Refs:
[1]: Numerical Recipes in C, 2E
        Press, et al., 1997
"""
from numpy import (exp, ones, pi, sqrt, zeros)

from pynalgo.common_tools import (NDArray, Any, float64,
                                  JIT, TYPE_CHECKING)

###############################################################################
# Hermite psi functions (weighted, rescaled Hermite polynomials)
###############################################################################

def poly_Hermite_psi(N : int,
                     x : NDArray[Any],
                     ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of Hermite psi functions [1].

  Computes the normalized Hermite functions:

  .. math::

     \psi_n(x) = \exp(-x^2/2) \, \widetilde{H}_n(x)

  where :math:`\widetilde{H}_n(x)` are the rescaled Hermite polynomials
  from [1]:

  .. math::

     \widetilde{H}_0(x) &= \pi^{-1/4} \\
     \widetilde{H}_1(x) &= \sqrt{2} \, \pi^{-1/4} \, x \\
     \widetilde{H}_{n+1}(x) &= x \sqrt{\frac{2}{n+1}} \widetilde{H}_n(x)
                               - \sqrt{\frac{n}{n+1}} \widetilde{H}_{n-1}(x)

  Domain: :math:`x \in (-\infty, \infty)`.

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.

  Usage
  -----
  >>> from numpy import (allclose, array, linspace)
  >>> N = 4
  >>> x = linspace(-2, 2, num=5)
  >>> allclose(
  ...   poly_Hermite_psi(N, x, ret_ord_num=2),
  ...   array([
  ...     [-0.58689842042856,  0.26302962362333,  0.                ,
  ...      -0.26302962362333,  0.58689842042856],
  ...     [ 0.39424986030507, -0.46497507629251,  0.45996857917733,
  ...      -0.46497507629251,  0.39424986030507]
  ...   ])
  ... )
  True
  '''
  of = 1.
  two = 2.

  # pi^(-1/4)
  pi_m14 = of / sqrt(sqrt(pi))

  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  wei = exp(-x * x / two)

  # psi_0 = pi^(-1/4) * exp(-x^2/2)
  pnm1 = pi_m14 * wei
  if n_min == 0:
    poly_arr[ind_0, :] = pnm1

  pn = zeros(x.shape)
  if N >= 1:
    # psi_1 = sqrt(2) * pi^(-1/4) * x * exp(-x^2/2)
    pn = sqrt(two) * pi_m14 * x * wei
    if n_min <= 1:
      poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    nm1 = n_it - 1
    pTmp = (x * sqrt(two / n_it) * pn
            - sqrt(nm1 / n_it) * pnm1)
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr


def poly_Hermite_H(N : int,
                   x : NDArray[Any],
                   ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of (physicists') Hermite polynomials [1].

  Returns the standard Hermite polynomials :math:`H_n(x)`:

  .. math::

     H_0(x) &= 1 \\
     H_1(x) &= 2x \\
     H_{n+1}(x) &= 2x H_n(x) - 2n H_{n-1}(x)

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.

  Usage
  -----
  >>> from numpy import (allclose, array, linspace)
  >>> N = 4
  >>> x = linspace(-2, 2, num=5)
  >>> allclose(
  ...   poly_Hermite_H(N, x, ret_ord_num=2),
  ...   array([
  ...     [-40.,   4.,   0.,  -4.,  40.],
  ...     [ 76., -20.,  12., -20.,  76.]
  ...   ])
  ... )
  True
  '''
  two = 2.

  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  pnm1 = ones(x.shape)
  if n_min == 0:
    poly_arr[ind_0, :] = pnm1

  pn = zeros(x.shape)
  if N >= 1:
    pn = two * x
    if n_min <= 1:
      poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    nm1 = n_it - 1
    pTmp = two * x * pn - two * nm1 * pnm1
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr

###############################################################################
# Hermite derivatives
###############################################################################

def poly_der_Hermite_H(M : int,
                       N : int,
                       x : NDArray[Any],
                       ret_der_num : int=1,
                       ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of derivatives of Hermite polynomials [1].

  Uses the identity:

  .. math::
     \frac{d^m}{dx^m} H_n(x) = 2^m (n)_m H_{n-m}(x)

  where :math:`(n)_m = n(n-1)\cdots(n-m+1)` is the falling factorial.

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  x : NDArray[Any]
    The domain over which to compute.

  ret_der_num : int = 1
    Number of derivatives to return (M as highest included).

  ret_ord_num : int = 1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.

  Usage
  -----
  >>> from numpy import (allclose, array, linspace)
  >>> M, N = 2, 4
  >>> x = linspace(-2, 2, num=4)
  >>> dHH = poly_der_Hermite_H(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dHH.shape
  (3, 2, 4)
  >>> allclose(
  ...   dHH[-1, -1, :],
  ...   array([672., -10.66666666666667, -10.66666666666667, 672.])
  ... )
  True
  '''
  of = 1.

  res = zeros((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for m in range(M, M - ret_der_num, -1):
    m_ind = -(M - m + 1)
    pref = 2 ** m

    # Batch: compute all needed shifted Hermite polynomials at once
    max_shifted_order = N - m
    if max_shifted_order >= 0:
      H_shifted = poly_Hermite_H(max_shifted_order, x,
                                 ret_ord_num=max_shifted_order + 1)

      for n_c in range(N + 1 - ret_ord_num, N + 1):
        n_ind = n_c - (N + 1 - ret_ord_num)

        if n_c >= m:
          ff = of
          for k in range(0, m):
            ff = ff * (n_c - k)
          for i in range(x.size):
            res[m_ind, n_ind, i] = pref * ff * H_shifted[n_c - m, i]

  return res

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  poly_Hermite_psi = JIT(poly_Hermite_psi)
  poly_Hermite_H = JIT(poly_Hermite_H)
  poly_der_Hermite_H = JIT(poly_der_Hermite_H)

#
# :D
#
