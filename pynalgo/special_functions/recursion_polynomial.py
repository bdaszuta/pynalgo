"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Calculation of polynomial families via recursion.

Refs:
[1]: Spectral Methods for Time-Dependent Problems;
        Hesthaven, et al., 2007
[2]: Numerical Recipes in C, 2E
        Press, et al., 1997
"""
from numpy import (arange, array, concatenate, cos, cumsum, empty,
                  exp, log, ones, sin, zeros)

from pynalgo.common_tools import (NDArray, Any, float64,
                                  JIT, JIT_NC, TYPE_CHECKING)

from pynalgo.special_functions.factorial import (special_exp_log_frac_sum,
                                              special_log_abs_gamma, )

###############################################################################
# Jacobi polynomials and derivatives
###############################################################################

def poly_JacobiP(N : int,
                 a : float,
                 b : float,
                 x : NDArray[Any],
                 ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Recursive computation of Jacobi polynomials [1,2].

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  a : float
    Alpha parameter.

  b : float
    Beta parameter.

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
  >>> from numpy import linspace
  >>> N, a, b = 10, 1.2, 0.8
  >>> x = linspace(-1, 1, num=4)
  >>> poly_JacobiP(N, a, b, x, ret_ord_num=2)
      array([[-6.72097987,  0.05723328, -0.35877722, 14.55419863],
             [ 7.25865825,  0.41301415,  0.26335232, 16.30070247]])
  '''
  poly_arr = zeros((ret_ord_num, x.size))

  # Orders to calculate
  nrdiff = N - ret_ord_num + 1

  tmp = zeros((2, 1), dtype=float64)
  tmp[0], tmp[1] = nrdiff, 0

  n_min = max(tmp)

  # Arr ind. offset
  ind_0 = abs(nrdiff) if nrdiff < 0 else 0

  # Start recursion
  pnm1 = ones(x.shape)

  # Split factors to avoid errors during numba caching
  xFa = x * (1 + (a + b) / 2)
  pn = (a - b) / 2 + xFa

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1

  for n_it in range(1, N + 1):
    j = n_it

    c_j = 2 * (j + 1) * (j + a + b + 1) * (2 * j + a + b)
    d_j = (2 * j + a + b + 1) * (a**2 - b**2)
    e_j = ((2 * j + a + b) *
            (2 * j + a + b + 1) *
            (2 * j + a + b + 2))
    f_j = 2 * (j + a) * (j + b) * (2 * j + a + b + 2)

    pTmp = ((d_j + e_j * x) * pn - pnm1 * f_j) / c_j
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pnm1

  return poly_arr

def poly_der_JacobiP(M : int,
                     N : int,
                     a : float,
                     b : float,
                     x : NDArray[Any],
                     ret_der_num : int=1,
                     ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Recursive computation of derivatives of Jacobi polynomials [1,2].

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  a : float
    Alpha parameter.

  b : float
    Beta parameter.

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
  >>> from numpy import linspace
  >>> M, N, a, b = 2, 10, 1.2, 0.8
  >>> x = linspace(-1, 1, num=4)
  >>> dJP = poly_der_JacobiP(M, N, a, b, x, ret_ord_num=2, ret_der_num=3)
  >>> dJP.shape
      (3, 2, 4)
  >>> dJP[-1,-1,:]
      array([5897.65983206,  -62.86169596,  -48.99087888, 9481.72963046])
  '''
  _fC0 = 0.
  poly_arr = zeros(((ret_der_num), (ret_ord_num), x.size))

  for k in range(M, (M - ret_der_num), -1):
    k_ind = -(M - k + 1)
    # Ensure indexing for requested return
    ind_ord_ret = max(array([-ret_ord_num, -(N + 1)]))

    if k >= 0:
      # Compute shifted Jacobi basis P^(a+k,b+k)_(n-k)
      # Need P^(a+k,b+k)_(n-k) [Return all orders]

      if (N-k) >= 0:  # only eval. for non-neg. orders
        P_Jac_c = poly_JacobiP((N-k), a+k, b+k, x, ret_ord_num=N+1)
        poly_arr[k_ind, ind_ord_ret:, :] = P_Jac_c[ind_ord_ret:, :]
      else:
        poly_arr[k_ind, ind_ord_ret:, :] = _fC0

    if k > 0:
      # Apply front factor: exp(sum(ln(a+b+2+j), {j,n-1,n+k-2}))/2^k
      f2k = 2**k

      for n_c in range(N + 1 + ind_ord_ret, N + 1):
        if n_c >= k:
          j = arange(n_c - 1, (n_c + k - 2) + 1)
          ff = exp(sum(log(j + a + b + 2))) / f2k

          ind_n_c = n_c - (N + 1 + ind_ord_ret)

          poly_arr[k_ind, ind_n_c, :] = ff * poly_arr[k_ind, ind_n_c, :]
  return poly_arr

###############################################################################
# Chebyshev first kind (ChebyshevT) definitions
###############################################################################

def poly_ChebyshevT(N : int,
                    x : NDArray[Any],
                    ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of Chebyshev polynomials of the first kind via Jacobi
  polynomials [1].  (Delegates to poly_JacobiP; not a direct recurrence.)

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  x : NDArray[Any]
    Evaluation points.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.

  Usage
  -----
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevT(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-1. ,  1. ,  0. , -1. ,  1. ],
             [ 1. , -0.5, -1. , -0.5,  1. ]])
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_JacobiP(N, -hf, -hf, x, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for j in range(Jac_pol.shape[0]):
    for i in range(Jac_pol.shape[1]):
      Jac_pol[j, i] = pf[j] * Jac_pol[j, i]

  res = empty((ret_ord_num, x.size), dtype=float64)

  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = Jac_pol[N - ret_ord_num + j + 1, i]
  return res

def poly_der_ChebyshevT(M : int,
                        N : int,
                        x : NDArray[Any],
                        ret_der_num : int=1,
                        ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of derivatives of Chebyshev polynomials of the first
  kind via Jacobi polynomials [1].  (Delegates to poly_der_JacobiP;
  not a direct recurrence.)

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
  >>> from numpy import linspace
  >>> M, N = 2, 10
  >>> x = linspace(-1, 1, num=4)
  >>> dCT = poly_der_ChebyshevT(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dCT.shape
      (3, 2, 4)
  >>> dCT[-1,-1,:]
      array([3300.        , -109.82167353, -109.82167353, 3300.        ])
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_der_JacobiP(M, N, -hf, -hf, x,
                                ret_der_num=M+1, ret_ord_num=N+1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for mix in range(Jac_pol.shape[0]):
    for j in range(Jac_pol.shape[1]):
      for i in range(x.size):
        Jac_pol[mix, j, i] = pf[j] * Jac_pol[mix, j, i]

  res = empty((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for mix in range(0, ret_der_num):
    for j in range(0, ret_ord_num):
      for i in range(x.size):
        _m = M - ret_der_num + mix + 1
        _j = N - ret_ord_num + j + 1
        res[mix, j, i] = Jac_pol[_m, _j, i]
  return res

###############################################################################
# Chebyshev second kind (ChebyshevU) definitions
###############################################################################

def poly_ChebyshevU(N : int,
                    x : NDArray[Any],
                    ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of Chebyshev polynomials of the second kind via Jacobi
  polynomials [1].  (Delegates to poly_JacobiP; not a direct recurrence.)

  U_n(x) is related to the Jacobi polynomials via
  :math:`U_n = \frac{(2)_n}{(3/2)_n} P_n^{(1/2, 1/2)}(x)`.

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
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevU(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-10.,   1.,   0.,  -1.,  10.],
             [ 11.,  -1.,  -1.,  -1.,  11.]])
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_JacobiP(N, hf, hf, x, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, 2 * of, of, 3 * hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for j in range(Jac_pol.shape[0]):
    for i in range(Jac_pol.shape[1]):
      Jac_pol[j, i] = pf[j] * Jac_pol[j, i]

  res = empty((ret_ord_num, x.size), dtype=float64)

  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = Jac_pol[N - ret_ord_num + j + 1, i]
  return res


def poly_der_ChebyshevU(M : int,
                        N : int,
                        x : NDArray[Any],
                        ret_der_num : int=1,
                        ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of derivatives of Chebyshev polynomials of the second
  kind via Jacobi polynomials [1].  (Delegates to poly_der_JacobiP;
  not a direct recurrence.)

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
  >>> from numpy import linspace
  >>> M, N = 2, 10
  >>> x = linspace(-1, 1, num=4)
  >>> dCU = poly_der_ChebyshevU(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dCU.shape
      (3, 2, 4)
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_der_JacobiP(M, N, hf, hf, x,
                             ret_der_num=M + 1, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, 2 * of, of, 3 * hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for mix in range(Jac_pol.shape[0]):
    for j in range(Jac_pol.shape[1]):
      for i in range(x.size):
        Jac_pol[mix, j, i] = pf[j] * Jac_pol[mix, j, i]

  res = empty((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for mix in range(0, ret_der_num):
    for j in range(0, ret_ord_num):
      for i in range(x.size):
        _m = M - ret_der_num + mix + 1
        _j = N - ret_ord_num + j + 1
        res[mix, j, i] = Jac_pol[_m, _j, i]
  return res

###############################################################################
# Chebyshev fourth kind (ChebyshevW) definitions
###############################################################################

def poly_ChebyshevW(N : int,
                    x : NDArray[Any],
                    ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of Chebyshev polynomials of the fourth kind via Jacobi
  polynomials [1].  (Delegates to poly_JacobiP; not a direct recurrence.)

  W_n(x) is related to the Jacobi polynomials via
  :math:`W_n = (2n+1) \, n! / (3/2)_n \, P_n^{(1/2, -1/2)}(x)`.

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
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevW(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-1.,  1.,  1., -1., 19.],
             [ 1.,  0., -1., -2., 21.]])
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_JacobiP(N, hf, -hf, x, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, 3 * hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for j in range(Jac_pol.shape[0]):
    for i in range(Jac_pol.shape[1]):
      Jac_pol[j, i] = (2 * j + 1) * pf[j] * Jac_pol[j, i]

  res = empty((ret_ord_num, x.size), dtype=float64)

  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = Jac_pol[N - ret_ord_num + j + 1, i]
  return res


def poly_der_ChebyshevW(M : int,
                        N : int,
                        x : NDArray[Any],
                        ret_der_num : int=1,
                        ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of derivatives of Chebyshev polynomials of the fourth
  kind via Jacobi polynomials [1].  (Delegates to poly_der_JacobiP;
  not a direct recurrence.)

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
  >>> from numpy import linspace
  >>> M, N = 2, 10
  >>> x = linspace(-1, 1, num=4)
  >>> dCW = poly_der_ChebyshevW(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dCW.shape
      (3, 2, 4)
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_der_JacobiP(M, N, hf, -hf, x,
                             ret_der_num=M + 1, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, 3 * hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for mix in range(Jac_pol.shape[0]):
    for j in range(Jac_pol.shape[1]):
      for i in range(x.size):
        Jac_pol[mix, j, i] = (2 * j + 1) * pf[j] * Jac_pol[mix, j, i]

  res = empty((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for mix in range(0, ret_der_num):
    for j in range(0, ret_ord_num):
      for i in range(x.size):
        _m = M - ret_der_num + mix + 1
        _j = N - ret_ord_num + j + 1
        res[mix, j, i] = Jac_pol[_m, _j, i]
  return res

###############################################################################
# Chebyshev third kind (ChebyshevV) definitions
###############################################################################

def poly_ChebyshevV(N : int,
                    x : NDArray[Any],
                    ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of Chebyshev polynomials of the third kind via Jacobi
  polynomials [1].  (Delegates to poly_JacobiP; not a direct recurrence.)

  V_n(x) is related to the Jacobi polynomials via
  :math:`V_n = n! / (1/2)_n \, P_n^{(-1/2, 1/2)}(x)`.

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
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevV(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-19.,   1.,  -1.,  -1.,   1.],
             [ 21.,  -2.,  -1.,   0.,   1.]])
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_JacobiP(N, -hf, hf, x, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for j in range(Jac_pol.shape[0]):
    for i in range(Jac_pol.shape[1]):
      Jac_pol[j, i] = pf[j] * Jac_pol[j, i]

  res = empty((ret_ord_num, x.size), dtype=float64)

  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = Jac_pol[N - ret_ord_num + j + 1, i]
  return res


def poly_der_ChebyshevV(M : int,
                        N : int,
                        x : NDArray[Any],
                        ret_der_num : int=1,
                        ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of derivatives of Chebyshev polynomials of the third
  kind via Jacobi polynomials [1].  (Delegates to poly_der_JacobiP;
  not a direct recurrence.)

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
  >>> from numpy import linspace
  >>> M, N = 2, 10
  >>> x = linspace(-1, 1, num=4)
  >>> dCV = poly_der_ChebyshevV(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dCV.shape
      (3, 2, 4)
  '''
  of = 1.
  hf = of / 2

  Jac_pol = poly_der_JacobiP(M, N, -hf, hf, x,
                             ret_der_num=M + 1, ret_ord_num=N + 1)

  # exponential prefactors
  pf = special_exp_log_frac_sum(N, of, of, of, hf)

  # pad size for n=0
  pf = concatenate((array([of]), pf))

  for mix in range(Jac_pol.shape[0]):
    for j in range(Jac_pol.shape[1]):
      for i in range(x.size):
        Jac_pol[mix, j, i] = pf[j] * Jac_pol[mix, j, i]

  res = empty((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for mix in range(0, ret_der_num):
    for j in range(0, ret_ord_num):
      for i in range(x.size):
        _m = M - ret_der_num + mix + 1
        _j = N - ret_ord_num + j + 1
        res[mix, j, i] = Jac_pol[_m, _j, i]
  return res

###############################################################################
# Gegenbauer polynomial definitions
###############################################################################

def poly_Gegenbauer(N : int,
                    lam : float,
                    x : NDArray[Any],
                    ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of Gegenbauer polynomials (via Jacobi polynomials) [1].

  :math:`C_n^{(\lambda)}(x)` is computed via the relation to Jacobi polynomials
  :math:`P_n^{(\lambda-1/2, \lambda-1/2)}(x)` with the standard normalization
  :math:`C_n^{(\lambda)}(1) = \binom{n + 2\lambda - 1}{n}`.

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  lam : float
    Lambda parameter (lam > -1/2).

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
  >>> N, lam = 4, 2.5
  >>> x = linspace(-1, 1, num=5)
  >>> allclose(
  ...   poly_Gegenbauer(N, lam, x, ret_ord_num=2),
  ...   array([
  ...     [-35.,  2.1875, -0.   , -2.1875, 35. ],
  ...     [ 70., -6.2891,  4.375, -6.2891, 70. ]
  ...   ])
  ... )
  True
  '''
  of = 1.
  a = lam - of / 2

  Jac_pol = poly_JacobiP(N, a, a, x, ret_ord_num=N + 1)

  # Normalization factor via cumulative product (avoids Gamma calls)
  j_max = N - 1  # 0 to N-2 inclusive, or empty for N<=1
  nff = zeros(N + 1, dtype=float64)
  nff[0] = 1.0
  nff[1] = 2.0 - of / (of + a)
  if j_max > 0:
    j = arange(0, j_max)
    ff = exp(cumsum(log((2 * a + j + 2) / (a + 1 + j))))
    ff = ff * (1 + 2 * a)
    nff[2:] = ff / (a + 1 + j + 1)

  for n_c in range(N + 1):
    for i in range(x.size):
      Jac_pol[n_c, i] = nff[n_c] * Jac_pol[n_c, i]

  res = empty((ret_ord_num, x.size), dtype=float64)

  for k in range(0, ret_ord_num):
    for i in range(x.size):
      res[k, i] = Jac_pol[N - ret_ord_num + k + 1, i]
  return res


def poly_der_Gegenbauer(M : int,
                        N : int,
                        lam : float,
                        x : NDArray[Any],
                        ret_der_num : int=1,
                        ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of Gegenbauer polynomial derivatives (via Jacobi polynomials) [1].

  Uses the relation:

  .. math::
     \frac{d^m}{dx^m} C_n^{(\lambda)}(x) = 2^m (\lambda)_m C_{n-m}^{(\lambda+m)}(x)

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  lam : float
    Lambda parameter (lam > -1/2).

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
  >>> from numpy import linspace
  >>> M, N, lam = 2, 4, 2.5
  >>> x = linspace(-1, 1, num=4)
  >>> dCG = poly_der_Gegenbauer(M, N, lam, x, ret_ord_num=2, ret_der_num=3)
  >>> dCG.shape
      (3, 2, 4)
  '''
  of = 1.

  res = zeros((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for m in range(M, M - ret_der_num, -1):
    m_ind = -(M - m + 1)

    # Rising factorial: (lam)_m = lam * (lam+1) * ... * (lam+m-1)
    rf = of
    for k in range(0, m):
      rf = rf * (lam + k)

    pref = (2 ** m) * rf

    # Batch: compute all needed shifted Gegenbauer polynomials at once
    max_shifted_order = N - m
    if max_shifted_order >= 0:
      C_shifted = poly_Gegenbauer(max_shifted_order, lam + m, x,
                                  ret_ord_num=max_shifted_order + 1)

      for n_c in range(N + 1 - ret_ord_num, N + 1):
        n_ind = n_c - (N + 1 - ret_ord_num)

        if n_c >= m:
          # C_{n-m}^(lam+m)
          for i in range(x.size):
            res[m_ind, n_ind, i] = pref * C_shifted[n_c - m, i]

  return res

###############################################################################
# Ultraspherical polynomial definitions
###############################################################################

def poly_Ultraspherical(N : int,
                        lam : float,
                        x : NDArray[Any],
                        ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Evaluation of normalized Ultraspherical polynomials (via Gegenbauer polynomials) [1].

  Defined as the Gegenbauer polynomial normalized to unity at x = 1:
  :math:`P_n^{(\lambda)}(x) = C_n^{(\lambda)}(x) / C_n^{(\lambda)}(1)`.

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  lam : float
    Lambda parameter (lam > -1/2).

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
  >>> N, lam = 4, 2.5
  >>> x = linspace(-1, 1, num=5)
  >>> allclose(
  ...   poly_Ultraspherical(N, lam, x, ret_ord_num=2),
  ...   array([
  ...     [-1.,  0.0625    , -0.    , -0.0625    , 1.],
  ...     [ 1., -0.08984375,  0.0625, -0.08984375, 1.]
  ...   ])
  ... )
  True
  '''
  C = poly_Gegenbauer(N, lam, x, ret_ord_num=N + 1)

  # C_n^(lam)(1) = binom(n + 2*lam - 1, n)
  # Use exp-log-gamma for numerical stability
  for n_c in range(N + 1):
    if n_c == 0:
      norm = 1.0
    else:
      # log(C_n^(lam)(1)) = log Gamma(n+2*lam)
      #                     - log Gamma(n+1) - log Gamma(2*lam)
      log_norm = (special_log_abs_gamma(array([n_c + 2 * lam]))[0]
                  - special_log_abs_gamma(array([n_c + 1]))[0]
                  - special_log_abs_gamma(array([2 * lam]))[0])
      norm = exp(log_norm)
    for i in range(x.size):
      C[n_c, i] = C[n_c, i] / norm

  res = empty((ret_ord_num, x.size), dtype=float64)
  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = C[N - ret_ord_num + j + 1, i]
  return res


def poly_der_Ultraspherical(M : int,
                            N : int,
                            lam : float,
                            x : NDArray[Any],
                            ret_der_num : int=1,
                            ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of Ultraspherical polynomial derivatives (via Gegenbauer polynomials) [1].

  Parameters
  ----------
  M : int
    Degree of polynomial derivative.

  N : int
    Order of polynomial to stop recursion.

  lam : float
    Lambda parameter (lam > -1/2).

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
  >>> from numpy import linspace
  >>> M, N, lam = 2, 4, 2.5
  >>> x = linspace(-1, 1, num=4)
  >>> dCU = poly_der_Ultraspherical(M, N, lam, x, ret_ord_num=2, ret_der_num=3)
  >>> dCU.shape
      (3, 2, 4)
  '''
  of = 1.

  # Precompute C_n^(lam)(1) = binom(n + 2*lam - 1, n) for n = 0..N
  # Units: P_n^(lam)(x) = C_n^(lam)(x) / C_n^(lam)(1)
  norm_C = zeros(N + 1, dtype=float64)
  norm_C[0] = of
  if N > 0:
    ns = arange(1, N + 1)
    log_2lam = special_log_abs_gamma(array([2 * lam]))[0]
    norm_C[1:] = exp(special_log_abs_gamma(ns + 2 * lam)
                     - special_log_abs_gamma(ns + of)
                     - log_2lam)

  res = zeros((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for m in range(M, M - ret_der_num, -1):
    m_ind = -(M - m + 1)
    rf = of
    for k in range(0, m):
      rf = rf * (lam + k)
    pref = (2 ** m) * rf

    # DLMF 18.9.19: d^m/dx^m C_n^(lam) = 2^m*(lam)_m * C_{n-m}^(lam+m)
    # For unit-normalized: d^m/dx^m P_n^(lam) = pref *
    #   C_{n-m}^(lam+m) / C_n^(lam)(1)
    max_shifted_order = N - m
    if max_shifted_order >= 0:
      C_shifted = poly_Gegenbauer(max_shifted_order, lam + m, x,
                                   ret_ord_num=max_shifted_order + 1)

      for n_c in range(N + 1 - ret_ord_num, N + 1):
        n_ind = n_c - (N + 1 - ret_ord_num)
        if n_c >= m:
          for i in range(x.size):
            res[m_ind, n_ind, i] = pref * C_shifted[n_c - m, i] / norm_C[n_c]

  return res

###############################################################################
# Direct Chebyshev recursion (not via Jacobi)
###############################################################################

def poly_ChebyshevT_direct(N : int,
                           x : NDArray[Any],
                           ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Direct three-term recursive computation of Chebyshev polynomials of the
  first kind [1].

  Uses the recurrence :math:`T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)` directly,
  avoiding the Jacobi-polynomial intermediary.  Faster for standalone
  Chebyshev evaluation.

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
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevT_direct(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-1. ,  1. ,  0. , -1. ,  1. ],
             [ 1. , -0.5, -1. , -0.5,  1. ]])
  '''
  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  pnm1 = ones(x.shape)
  pn = x.copy()

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1
  if N >= 1 and n_min <= 1:
    poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    pTmp = 2 * x * pn - pnm1
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr


def poly_ChebyshevU_direct(N : int,
                           x : NDArray[Any],
                           ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Direct three-term recursive computation of Chebyshev polynomials of the
  second kind [1].

  Uses the recurrence :math:`U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)` directly,
  with :math:`U_0(x) = 1`, :math:`U_1(x) = 2x`.

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
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_ChebyshevU_direct(10, linspace(-1, 1, num=5), ret_ord_num=2)
      array([[-10.,   1.,   0.,  -1.,  10.],
             [ 11.,  -1.,  -1.,  -1.,  11.]])
  '''
  two = 2.

  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  pnm1 = ones(x.shape)
  pn = two * x

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1
  if N >= 1 and n_min <= 1:
    poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    pTmp = two * x * pn - pnm1
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr

###############################################################################
# Trigonometric polynomial evaluation
###############################################################################

def poly_sin(N : int,
             x : NDArray[Any],
             ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of sin(n*x) polynomials.

  Uses the Chebyshev-like recurrence:

  .. math::
     \sin((n+1)x) = 2\cos(x)\sin(nx) - \sin((n-1)x)

  This is equivalent to the standard angle-addition formula
  :math:`\sin((n+1)x) = \sin(nx)\cos(x) + \cos(nx)\sin(x)`.

  Parameters
  ----------
  N : int
    Order to compute up to.

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result: sin(n*x) for n = N-ret_ord_num+1 ... N.

  Usage
  -----
  >>> from numpy import allclose, array, linspace, pi
  >>> N = 4
  >>> x = linspace(0, pi, num=5)
  >>> allclose(
  ...   poly_sin(N, x, ret_ord_num=2),
  ...   array([[ 0.00000000e+00,  7.07106781e-01, -1.00000000e+00,
  ...            7.07106781e-01,  3.67394040e-16],
  ...          [ 0.00000000e+00,  2.22044605e-16, -2.44929360e-16,
  ...            3.33066907e-16, -4.89858720e-16]])
  ... )
  True
  '''
  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  sn = sin(x)
  cs = cos(x)

  # sin(0*x) = 0
  pnm1 = zeros(x.shape)
  pn = sn.copy()

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1
  if N >= 1 and n_min <= 1:
    poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    # sin((n+1)x) = 2*cos(x)*sin(nx) - sin((n-1)x)
    pTmp = 2 * cs * pn - pnm1
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr


def poly_cos(N : int,
             x : NDArray[Any],
             ret_ord_num : int=1) -> NDArray[float64]:
  r'''
  Recursive computation of cos(n*x) polynomials.

  Uses the recurrence:

  .. math::
     \cos((n+1)x) = 2\cos(x)\cos(nx) - \cos((n-1)x)

  Parameters
  ----------
  N : int
    Order to compute up to.

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result: cos(n*x) for n = N-ret_ord_num+1 ... N.

  Usage
  -----
  >>> from numpy import allclose, array, linspace, pi
  >>> N = 4
  >>> x = linspace(0, pi, num=5)
  >>> allclose(
  ...   poly_cos(N, x, ret_ord_num=2),
  ...   array([[ 1.00000000e+00, -7.07106781e-01, -1.83697020e-16,
  ...            7.07106781e-01, -1.00000000e+00],
  ...          [ 1.00000000e+00, -1.00000000e+00,  1.00000000e+00,
  ...           -1.00000000e+00,  1.00000000e+00]])
  ... )
  True
  '''
  poly_arr = zeros((ret_ord_num, x.size))

  nrdiff = N - ret_ord_num + 1
  n_min = int(max(nrdiff, 0))
  ind_0 = int(abs(nrdiff) if nrdiff < 0 else 0)

  cs = cos(x)

  # cos(0*x) = 1
  pnm1 = ones(x.shape)
  pn = cs.copy()

  if n_min == 0:
    poly_arr[ind_0, :] = pnm1
  if N >= 1 and n_min <= 1:
    poly_arr[max(ind_0, int(1 - nrdiff)), :] = pn

  for n_it in range(2, N + 1):
    pTmp = 2 * cs * pn - pnm1
    pnm1 = pn
    pn = pTmp

    if n_it >= n_min:
      poly_arr[n_it - nrdiff, :] = pn

  return poly_arr

###############################################################################
# Legendre polynomial definitions
###############################################################################

def poly_LegendreP(N : int,
                   x : NDArray[Any],
                   ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of Legendre polynomials via Jacobi polynomials [1].
  (Delegates to poly_JacobiP(a=0,b=0); not a direct recurrence.)

  Parameters
  ----------
  N : int
    Order of polynomial to stop recursion.

  x : NDArray[Any]
    Evaluation points.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  Returns
  -------
  poly_arr : NDArray[float64]
    Result of recursion.

  Usage
  -----
  >>> from numpy import linspace
  >>> N = 10
  >>> poly_LegendreP(10, linspace(-1, 1, num=4), ret_ord_num=2)
      array([[-1.        ,  0.02433572, -0.02433572,  1.        ],
             [ 1.        ,  0.23026639,  0.23026639,  1.        ]])
  '''
  res = empty((ret_ord_num, x.size), dtype=float64)
  Jac_pol = poly_JacobiP(N, 0, 0, x, ret_ord_num=N + 1)

  for j in range(0, ret_ord_num):
    for i in range(x.size):
      res[j, i] = Jac_pol[N - ret_ord_num + j + 1, i]
  return res

def poly_der_LegendreP(M : int,
                       N : int,
                       x : NDArray[Any],
                       ret_der_num : int=1,
                       ret_ord_num : int=1) -> NDArray[float64]:
  '''
  Evaluation of derivatives of Legendre polynomials via Jacobi
  polynomials [1].  (Delegates to poly_der_JacobiP(a=0,b=0);
  not a direct recurrence.)

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
  >>> from numpy import linspace
  >>> M, N = 2, 10
  >>> x = linspace(-1, 1, num=4)
  >>> dPT = poly_der_LegendreP(M, N, x, ret_ord_num=2, ret_der_num=3)
  >>> dPT.shape
      (3, 2, 4)
  >>> dPT[-1,-1,:]
      array([1485.       ,  -29.3484225,  -29.3484225, 1485.       ])
  '''
  Jac_pol = poly_der_JacobiP(M, N, 0, 0, x,
                             ret_der_num=M+1, ret_ord_num=N+1)

  res = empty((ret_der_num, ret_ord_num, x.size), dtype=float64)

  for mix in range(0, ret_der_num):
    for j in range(0, ret_ord_num):
      for i in range(x.size):
        _m = M - ret_der_num + mix + 1
        _j = N - ret_ord_num + j + 1
        res[mix, j, i] = Jac_pol[_m, _j, i]
  return res

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  poly_JacobiP = JIT(poly_JacobiP)
  poly_der_JacobiP = JIT(poly_der_JacobiP)
  poly_ChebyshevT = JIT(poly_ChebyshevT)
  poly_der_ChebyshevT = JIT(poly_der_ChebyshevT)
  poly_ChebyshevU = JIT(poly_ChebyshevU)
  poly_der_ChebyshevU = JIT(poly_der_ChebyshevU)
  poly_ChebyshevV = JIT(poly_ChebyshevV)
  poly_der_ChebyshevV = JIT(poly_der_ChebyshevV)
  poly_ChebyshevW = JIT(poly_ChebyshevW)
  poly_der_ChebyshevW = JIT(poly_der_ChebyshevW)
  poly_LegendreP = JIT(poly_LegendreP)
  poly_der_LegendreP = JIT(poly_der_LegendreP)
  poly_Gegenbauer = JIT(poly_Gegenbauer)
  poly_der_Gegenbauer = JIT(poly_der_Gegenbauer)
  poly_ChebyshevT_direct = JIT(poly_ChebyshevT_direct)
  poly_ChebyshevU_direct = JIT(poly_ChebyshevU_direct)
  poly_sin = JIT(poly_sin)
  poly_cos = JIT(poly_cos)
  poly_Ultraspherical = JIT_NC(poly_Ultraspherical)
  poly_der_Ultraspherical = JIT_NC(poly_der_Ultraspherical)

#
# :D
#
