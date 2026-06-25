"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Calculation of Laguerre polynomial families via recursion.

Refs:
[1]: NIST Handbook of Mathematical Functions;
        Olver, et al., 2013
"""
from numpy import (arange, array, empty, exp, log, ones, zeros)

from pynalgo.common_tools import (NDArray, Any, float64,
                                  JIT, JIT_NC, TYPE_CHECKING)
from pynalgo.special_functions.factorial import (special_log_abs_gamma, )

###############################################################################
# Associated Laguerre polynomials
###############################################################################

def poly_Laguerre(N : int,
                  alpha : float,
                  x : NDArray[Any],
                  ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of associated Laguerre polynomials [1].

  Uses the recurrence:

  .. math::

     L_0^{(\alpha)}(x) &= 1 \\
     L_1^{(\alpha)}(x) &= 1 + \alpha - x \\
     (n+1)L_{n+1}^{(\alpha)}(x) &= (2n+\alpha+1-x)L_n^{(\alpha)}(x)
                                 - (n+\alpha)L_{n-1}^{(\alpha)}(x)

  Domain: :math:`x \in [0, \infty)`.

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  alpha : float
    Associated parameter (alpha > -1).

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.
  '''
  of = 1.

  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  pnm1 = ones(x.shape)
  pn = of + alpha - x

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1
  if N >= 1 and n_min <= 1:
    poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    nm1 = n_it - 1
    pTmp = ((2 * nm1 + alpha + of - x) * pn
            - (nm1 + alpha) * pnm1) / n_it
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr

###############################################################################
# Internal helpers for lambda-function derivatives
###############################################################################

def _laguerre_til(N : int,
                  alpha : float,
                  x : NDArray[Any],
                  ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  "Tilted" Laguerre functions used in lambda derivative construction.

  .. math::

     \widetilde{L}_k^{(\alpha)}(x) =
       2^{\alpha/2 - k} L_k^{(\alpha/2 - k)}(x/2)
       \exp(-x/4) \, x^{\alpha/2 - k}

  .. note::

     The factor :math:`2^{\alpha/2 - k}` is intentionally omitted
     in the implementation; it is accounted for in the normalization
     of the calling function.
  '''
  res = zeros((ret_ord_num, x.size), dtype=float64)
  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))

  for n_c in range(n_min, N + 1):
    k = n_c
    shifted_alpha = alpha / 2 - k

    # Call poly_Laguerre for the polynomial part evaluated at x/2
    L_poly = poly_Laguerre(k, shifted_alpha, x / 2, ret_ord_num=k + 1)
    Lk_val = L_poly[k, :]

    # Tilted weight: exp(-x/4) * x^(alpha/2 - k)
    # Note: the 2^{alpha/2-k} factor is intentionally omitted;
    # it is accounted for in the normalization of the calling function.
    wei = exp(-x / 4) * x ** shifted_alpha
    res[n_c - nrdiff, :] = wei * Lk_val

  return res


def _exp_fa_vect(N : int,
                 alpha : float,
                 ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Compute the normalization factor for Laguerre lambda functions:

  .. math::

     \exp\_\mathrm{fa}[n] = \sqrt{\frac{n!}{\Gamma(n + \alpha + 1)}}

  for :math:`n = N - \mathrm{ret\_ord\_num} + 1 \ldots N`.
  '''
  of = 1.

  # Base factor: 1 / sqrt(Gamma(alpha + 1))
  gam0 = exp(-special_log_abs_gamma(array([alpha + of]))[0] / 2)

  # Full array for all orders 0..N
  fa = zeros(N + 1, dtype=float64)
  fa[0] = gam0

  for nn in range(1, N + 1):
    fa[nn] = fa[nn - 1] * exp(log(nn / (nn + alpha)) / 2)

  # Extract requested orders
  res = zeros(ret_ord_num, dtype=float64)
  for j in range(ret_ord_num):
    res[j] = fa[N - ret_ord_num + j + 1]
  return res

###############################################################################
# Laguerre lambda functions (weight-embedded, orthonormal)
###############################################################################

def poly_Laguerre_lambda(N : int,
                         alpha : float,
                         x : NDArray[Any],
                         ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of normalized associated Laguerre functions [1].

  Defined as:

  .. math::

     \lambda_n^{(\alpha)}(x) = \sqrt{\frac{n!}{\Gamma(n+\alpha+1)}}
       \exp(-x/2) \, x^{\alpha/2} \, L_n^{(\alpha)}(x)

  These functions are orthonormal on :math:`[0, \infty)` with unit weight.

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  alpha : float
    Associated parameter (alpha > -1).

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.
  '''
  of = 1.

  L = poly_Laguerre(N, alpha, x, ret_ord_num=N + 1)

  for n_c in range(N + 1):
    if n_c == 0:
      log_norm = -special_log_abs_gamma(array([alpha + of]))[0] / 2
    else:
      log_n_fact = sum(log(arange(1, n_c + 1)))
      log_norm = (log_n_fact
                  - special_log_abs_gamma(array([n_c + alpha + of]))[0]) / 2

    wei = exp(log_norm - x / 2)
    if alpha != 0:
      wei = wei * x ** (alpha / 2)

    for i in range(x.size):
      L[n_c, i] = wei[i] * L[n_c, i]

  res = empty((ret_ord_num, x.size), dtype=float64)
  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = L[N - ret_ord_num + j + 1, i]
  return res

###############################################################################
# Laguerre derivatives
###############################################################################

def poly_der_Laguerre(M : int,
                      N : int,
                      alpha : float,
                      x : NDArray[Any],
                      ret_der_num : int=1,
                      ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of derivatives of associated Laguerre polynomials [1].

  Uses the identity:

  .. math::
     \frac{d^m}{dx^m} L_n^{(\alpha)}(x) = (-1)^m L_{n-m}^{(\alpha+m)}(x)

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  alpha : float
    Associated parameter (alpha > -1).

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
  '''
  res = zeros((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for m in range(M, M - ret_der_num, -1):
    m_ind = -(M - m + 1)
    sign = (-1) ** m

    max_shifted_order = N - m
    if max_shifted_order >= 0:
      L_shifted = poly_Laguerre(max_shifted_order, alpha + m, x,
                                ret_ord_num=max_shifted_order + 1)

      for n_c in range(N + 1 - ret_ord_num, N + 1):
        n_ind = n_c - (N + 1 - ret_ord_num)
        if n_c >= m:
          for i in range(x.size):
            res[m_ind, n_ind, i] = sign * L_shifted[n_c - m, i]

  return res

def poly_der_Laguerre_lambda(M : int,
                             N : int,
                             alpha : float,
                             x : NDArray[Any],
                             ret_der_num : int=1,
                             ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of derivatives of Laguerre lambda functions [1].

  Uses the full product-rule expansion of the weighted Laguerre functions
  via the tilde-Laguerre decomposition from [1]:

  .. math::

     \frac{d^m}{dx^m} \lambda_n^{(\alpha)}(x) = \sum_{k=0}^m
       \binom{m}{k} (-1)^{m-k} k! \,
       \widetilde{L}_k^{(\alpha)}(x) \,
       L_{n-m+k}^{(\alpha+m-k)}(x) \,
       \exp\_\mathrm{fa}[n]

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  alpha : float
    Associated parameter (alpha > -1).

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
  '''
  of = 1.
  n_pts = x.size
  dL = zeros((ret_der_num, ret_ord_num, n_pts), dtype=float64)

  # Precompute normalization factors for all needed orders
  exp_fa_full = _exp_fa_vect(N, alpha, ret_ord_num=N + 1)

  for m in range(M, M - ret_der_num, -1):
    m_ind = -(M - m + 1)

    if m == 0:
      # Undifferentiated lambda function
      L_lam = poly_Laguerre_lambda(N, alpha, x, ret_ord_num=N + 1)
      for j in range(ret_ord_num):
        n_idx = N - ret_ord_num + j + 1
        for i in range(n_pts):
          dL[m_ind, j, i] = L_lam[n_idx, i]
      continue

    # Tilted Laguerre polynomials up to order m
    Ltil = _laguerre_til(m, alpha, x, ret_ord_num=m + 1)

    # Accumulate over k:
    #   d^m lambda_n = sum_{k=0}^m
    #     binom(m,k)*(-1)^(m-k)*k!*Ltil_k*L_{n-m+k}^(alpha+m-k)
    # scaled by exp_fa[n]
    for k in range(0, m + 1):
      # Binomial coefficient: binom(m, k) = m! / (k! * (m-k)!)
      # Compute binom(m, k) in a numba-compatible way:
      # binom(m, k) = m * (m-1) * ... * (m-k+1) / k!
      k_fact = of
      binom_val = of
      for j in range(1, k + 1):
        k_fact = k_fact * j
        binom_val = binom_val * (m - j + 1)
      binom_val = binom_val / k_fact

      # Combinatorial coefficient: binom(m, k) * (-1)^(m-k) * k!
      comb = binom_val * ((-1) ** (m - k)) * k_fact

      # Ltil_k(x): row k of the tilted array
      Ltil_k = Ltil[k, :]

      # Laguerre poly: L_{n-m+k}^(alpha+m-k)(x) with exp(-x/4)
      shifted_alpha = alpha + m - k
      max_shifted_n = N - m + k
      if max_shifted_n >= 0:
        L_shifted = poly_Laguerre(max_shifted_n, shifted_alpha, x,
                                  ret_ord_num=max_shifted_n + 1)
        wei = exp(-x / 4)

        for j in range(ret_ord_num):
          n_c = N - ret_ord_num + j + 1
          if n_c >= m:
            n_shifted = n_c - m + k
            for i in range(n_pts):
              dL[m_ind, j, i] += (comb * Ltil_k[i]
                                  * L_shifted[n_shifted, i] * wei[i])

    # Apply normalization: multiply each column by exp_fa[n]
    for j in range(ret_ord_num):
      n_c = N - ret_ord_num + j + 1
      if n_c >= m:
        for i in range(n_pts):
          dL[m_ind, j, i] = dL[m_ind, j, i] * exp_fa_full[n_c]

  return dL

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  poly_Laguerre = JIT(poly_Laguerre)
  poly_der_Laguerre = JIT(poly_der_Laguerre)
  _laguerre_til = JIT(_laguerre_til)
  _exp_fa_vect = JIT_NC(_exp_fa_vect)
  poly_Laguerre_lambda = JIT_NC(poly_Laguerre_lambda)
  poly_der_Laguerre_lambda = JIT_NC(poly_der_Laguerre_lambda)

#
# :D
#
