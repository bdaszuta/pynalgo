"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Differentiation using (pseudo)spectral methods

Refs:
[1]: Spectral methods: Algorithms, Analysis and Applications
     Shen, Tang and Wang; 2011
[2]: Chebyshev and Fourier Spectral Methods, 2E:
     Boyd, John P.; 2001
"""
from numpy import (cos, dot, empty, exp, ones, pi, power, rot90, sin, tan,
                   zeros)
from numpy.linalg import matrix_power

from pynalgo.common_tools import (NDArray, float64, JIT, JIT_NC, TYPE_CHECKING)
from pynalgo.special_functions import (grid_ChebyshevT, poly_der_ChebyshevT,
                                       grid_JacobiP, poly_der_JacobiP,
                                       special_log_abs_gamma)

###############################################################################
# Chebyshev first kind (ChebyshevT) definitions
###############################################################################

def diff_mat_nodal_ChebyshevT(N : int,
                              D : int=1,
                              variety : str="GL") -> NDArray[float64]:
  """
  Nodal differentiation matrices adapted to grids associated with Chebyshev
  polynomials of the first kind [1].

  Parameters
  ----------
  N : int
    Polynomial order.

  D : int = 1
    Derivative degree to construct.

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  Returns
  -------
  d_mat : NDArray[float64]
    Differentiation matrix.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_ChebyshevT
  >>> N = 8

  >>> x_G = grid_ChebyshevT(N, variety="G")
  >>> D_G = diff_mat_nodal_ChebyshevT(N, variety="G")
  >>> allclose(dot(D_G, x_G ** 2), 2 * x_G)
      True

  >>> x_GL = grid_ChebyshevT(N, variety="GL")
  >>> D_GL = diff_mat_nodal_ChebyshevT(N, variety="GL")
  >>> allclose(dot(D_GL, x_GL ** 2), 2 * x_GL)
      True

  >>> x_RL = grid_ChebyshevT(N, variety="RL")
  >>> D_RL = diff_mat_nodal_ChebyshevT(N, variety="RL")
  >>> allclose(dot(D_RL, x_RL ** 2), 2 * x_RL)
      True

  >>> x_RR = grid_ChebyshevT(N, variety="RR")
  >>> D_RR = diff_mat_nodal_ChebyshevT(N, variety="RR")
  >>> allclose(dot(D_RR, x_RR ** 2), 2 * x_RR)
      True

  >>> # second degree
  >>> x_GL = grid_ChebyshevT(N, variety="GL")
  >>> D_GL = diff_mat_nodal_ChebyshevT(N, D=2, variety="GL")
  >>> allclose(dot(D_GL, x_GL ** 2), 2)
      True

  Notes
  -----
  These matrices satisfy a power property where higher degrees can be computed
  through forming the matrix product of lower orders.
  """
  d_mat = empty((N + 1, N + 1), dtype=float64)

  if variety == "GL":
    x = grid_ChebyshevT(N, variety=variety)

    c = ones(N+1, dtype=float64)
    c[0] = c[-1] = 2

    for j in range(N+1):
      for i in range(0, j):
        d_mat[i, j] = c[i] / c[j] * (-1) ** (i + j) / (x[i] - x[j])
      for i in range(j+1,N+1):
        d_mat[i, j] = c[i] / c[j] * (-1) ** (i + j) / (x[i] - x[j])

    for i in range(1, N):
      d_mat[i, i] = -x[i] / (2 * (1 - x[i] ** 2))

    d_mat[0, 0] = - (2 * N ** 2 + 1.) / 6.
    d_mat[N, N] = (2 * N ** 2 + 1.) / 6.

    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  if variety == "G":
    x = grid_ChebyshevT(N, variety=variety)

    TNp1 = poly_der_ChebyshevT(1, N+1, x, ret_ord_num=1, ret_der_num=1)

    for j in range(N+1):
      for i in range(0, j):
        d_mat[i, j] = TNp1[0, 0, i] / (TNp1[0, 0, j] * (x[i] - x[j]))
      for i in range(j+1, N+1):
        d_mat[i, j] = TNp1[0, 0, i] / (TNp1[0, 0, j] * (x[i] - x[j]))

    for i in range(N+1):
      d_mat[i, i] = x[i] / (2 * (1 - x[i] ** 2))

    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  if variety in ("RL", "RR"):
    # use the RL for both cases and rotate if RR required

    x = grid_ChebyshevT(N, variety="RL")

    T = poly_der_ChebyshevT(1, N+1, x, ret_ord_num=2, ret_der_num=2)
    TNp1, TN = T[0, 1, :], T[0, 0, :]
    dTNp1, dTN = T[1, 1, :], T[1, 0, :]
    dQ = dTNp1 + dTN

    d_mat[0, 0] = - N * (N + 1) / 3

    for k in range(1, N+1):
      d_mat[k, k] = x[k] / (2 * (1 - x[k] ** 2))
      d_mat[k, k] += (2 * N + 1) * TN[k] / (2 * (1 - x[k] ** 2) * dQ[k])

      d_mat[k, 0] = dQ[k] / dQ[0] / (x[k] - x[0])

      # relabel j->k and use the same loop
      d_mat[0, k] = dQ[0] / dQ[k] / (x[0] - x[k])

    for k in range(1, N+1):
      for j in range(1, k):
        d_mat[k,j] = dQ[k] / dQ[j] / (x[k] - x[j])
      for j in range(k+1, N+1):
        d_mat[k,j] = dQ[k] / dQ[j] / (x[k] - x[j])

    if variety == "RR":
      d_mat = -rot90(d_mat, k=2)

    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  raise ValueError("Function arguments malformed.")

###############################################################################
# Fourier definitions
###############################################################################

# use split to partition logic and avoid segfault with recursion

def _diff_mat_nodal_Fourier_S1_CL(N : int,
                                  D : int=1) -> NDArray[float64]:
  d_mat = zeros((N, N), dtype=float64)

  if D == 1:
    for k in range(N):
      for j in range(0, k):
        trg = 1 / tan((k - j) * pi / N)
        d_mat[k, j] = power(-1, k + j) / 2 * trg

      for j in range(k + 1, N):
        trg = 1 / tan((k - j) * pi / N)
        d_mat[k, j] = power(-1, k + j) / 2 * trg

    return d_mat

  if D == 2:
    for k in range(N):
      for j in range(0, k):
        trg = 1 / sin((j - k) * pi / N) ** 2
        d_mat[k, j] = -power(-1, j + k) / 2 * trg

      for j in range(k + 1, N):
        trg = 1 / sin((j - k) * pi / N) ** 2
        d_mat[k, j] = -power(-1, j + k) / 2 * trg

      d_mat[k, k] = -(N ** 2 + 2) / 12

    return d_mat

  # Fourier matrices do not satisfy a power property;
  # higher derivatives (D > 2) require explicit formulation.
  raise NotImplementedError("Derivative degree not implemented.")

def _diff_mat_nodal_Fourier_H1_C_cos(N : int) -> NDArray[float64]:
  d_mat = zeros((N + 1, N + 1), dtype=float64)

  for k in range(1, N):
    xk = k * pi / N

    for j in range(0, k):
      xj = j * pi / N
      trg = sin(xk) / (cos(xj) - cos(xk))
      p_j = 1 + (j == 0) + (j == N)   # 2 if j == 0 or j == N, else 1
      d_mat[k, j] = 1 / p_j * power(-1, k + j) * trg

    for j in range(k + 1, N + 1):
      xj = j * pi / N
      trg = sin(xk) / (cos(xj) - cos(xk))
      p_j = 1 + (j == 0) + (j == N)   # 2 if j == 0 or j == N, else 1
      d_mat[k, j] = 1 / p_j * power(-1, k + j) * trg

    d_mat[k, k] = 1 / (2 * tan(xk))

  # Rows 0 and N are identically zero:
  # d/dx(cos(n*x)) at x=0 and x=pi = 0 for all n.
  return d_mat

def _diff_mat_nodal_Fourier_H1_C_sin(N : int) -> NDArray[float64]:
  d_mat = zeros((N + 1, N + 1), dtype=float64)

  for k in range(0, N+1):
    xk = k * pi / N

    for j in range(1, k):
      xj = j * pi / N
      trg = sin(xj) / (cos(xk) - cos(xj))
      d_mat[k, j] = power(-1, k + j + 1) * trg

    for j in range(k + 1, N):
      xj = j * pi / N
      trg = sin(xj) / (cos(xk) - cos(xj))
      d_mat[k, j] = power(-1, k + j + 1) * trg

  for k in range(1, N):
    xk = k * pi / N
    d_mat[k, k] = -1 / (2 * tan(xk))

  # Rows 0 and N are identically zero:
  # d/dx(sin(n*x)) at x=0 and x=pi = 0 for all n.
  return d_mat

def _diff_mat_nodal_Fourier_H1_I_sin(N : int) -> NDArray[float64]:
  d_mat = zeros((N, N), dtype=float64)

  for k in range(1, N+1):
    fack = (1 - 2 * k) * pi / (2 * N)

    for j in range(1, k):
      facj = (1 - 2 * j) * pi / (2 * N)
      trg_N = (sin((k - j) * pi / N) + sin((j + k - 1) * pi / N) +
              sin(2 * fack))
      trg_D = 1 / (2 * (cos(facj) - cos(fack)) ** 2)
      d_mat[k-1, j-1] = power(-1, j + k) * trg_N * trg_D

    for j in range(k + 1, N+1):
      facj = (1 - 2 * j) * pi / (2 * N)
      trg_N = (sin((k - j) * pi / N) + sin((j + k - 1) * pi / N) +
              sin(2 * fack))
      trg_D = 1 / (2 * (cos(facj) - cos(fack)) ** 2)
      d_mat[k-1, j-1] = power(-1, j + k) * trg_N * trg_D


    d_mat[k-1, k-1] = -1 / (2 * tan(fack))

  return d_mat

def _diff_mat_nodal_Fourier_H1_I_cos(N : int) -> NDArray[float64]:
  d_mat = zeros((N, N), dtype=float64)

  for k in range(1, N+1):
    fack = (1 - 2 * k) * pi / (2 * N)

    for j in range(1, k):
      facj = (1 - 2 * j) * pi / (2 * N)
      trg_N = (sin((j - k) * pi / N) + sin((j + k - 1) * pi / N) +
              sin(2 * facj))
      trg_D = 1 / (2 * (cos(facj) - cos(fack)) ** 2)
      d_mat[k-1, j-1] = -power(-1, j + k) * trg_N * trg_D

    for j in range(k + 1, N+1):
      facj = (1 - 2 * j) * pi / (2 * N)
      trg_N = (sin((j - k) * pi / N) + sin((j + k - 1) * pi / N) +
              sin(2 * facj))
      trg_D = 1 / (2 * (cos(facj) - cos(fack)) ** 2)
      d_mat[k-1, j-1] = -power(-1, j + k) * trg_N * trg_D


    d_mat[k-1, k-1] = 1 / (2 * tan(fack))

  return d_mat

def diff_mat_nodal_Fourier(N : int,
                           D : int=1,
                           variety : str="S1_CL") -> NDArray[float64]:
  """
  Nodal differentiation matrices adapted to grids associated with trigonometric
  functions. Matrix elements follow from differentiation of cardinal
  functions [2].

  Parameters
  ----------
  N : int
    Number of equispaced Fourier nodes.

  D : int = 1
    Derivative degree to construct.

  variety : str = "S1_CL"
    Type of grid to construct.
    May be one of {S1_CL, S1_CR, S1_I,
                   H1_C_cos, H1_C_sin, H1_I_cos, H1_I_sin, H1_S_sin}.
    S1_CL:     Interval closed-left  [0, 2 * pi)
    S1_CR:     Interval closed-right (0, 2 * pi]
    S1_I:      Interval interior     (0, 2 * pi)
    H1_C_cos:  Interval closed       [0, pi] - for cos series
    H1_C_sin:  Interval closed       [0, pi] - for sin series
    H1_I_cos:  Interval interior     (0, pi) - for cos series
    H1_I_sin:  Interval interior     (0, pi) - for sin series
    H1_S_sin:  Interval interior     (0, pi) - for sin series
               Alternate choice of sampling

  Returns
  -------
  d_mat : NDArray[float64]
    Differentiation matrix.

  Usage
  -----
  >>> from numpy import (allclose, cos, dot, sin)
  >>> from pynalgo import grid_Fourier
  >>> N = 16
  >>> gr = grid_Fourier(N, variety="S1_CL")
  >>> D1 = diff_mat_nodal_Fourier(N, D=1, variety="S1_CL")
  >>> N1der = dot(D1, sin(gr) + cos(2 * gr))
  >>> A1der = cos(gr) - 2 * sin(2 * gr)
  >>> allclose(N1der, A1der)
      True

  >>> gr = grid_Fourier(N, variety="H1_I")
  >>> D2 = diff_mat_nodal_Fourier(N, D=2, variety="H1_I_cos")
  >>> N1der = dot(D2, cos(2 * gr))
  >>> A1der = -4 * cos(2 * gr)
  >>> allclose(N1der, A1der)
      True

  Notes
  -----
  These matrices do not satisfy a power property. Higher degrees require
  explicit formulation.
  """
  if variety in ("S1_CL", "S1_CR", "S1_I"):
    return _diff_mat_nodal_Fourier_S1_CL(N, D=D)

  if variety == "H1_C_cos":
    if D == 1:
      return _diff_mat_nodal_Fourier_H1_C_cos(N)

    if D == 2:
      D1_sin = _diff_mat_nodal_Fourier_H1_C_sin(N)
      D1_cos = _diff_mat_nodal_Fourier_H1_C_cos(N)
      return dot(D1_sin, D1_cos)

    raise NotImplementedError("Derivative degree not implemented.")

  if variety == "H1_C_sin":
    if D == 1:
      return _diff_mat_nodal_Fourier_H1_C_sin(N)

    if D == 2:
      D1_sin = _diff_mat_nodal_Fourier_H1_C_sin(N)
      D1_cos = _diff_mat_nodal_Fourier_H1_C_cos(N)
      return dot(D1_cos, D1_sin)

    raise NotImplementedError("Derivative degree not implemented.")

  if variety == "H1_S_sin":
    if D == 1:
      # copy to avoid layout issue
      return _diff_mat_nodal_Fourier_H1_C_sin(N)[1:-1,1:-1].copy()

    if D == 2:
      D1_sin = _diff_mat_nodal_Fourier_H1_C_sin(N)
      D1_cos = _diff_mat_nodal_Fourier_H1_C_cos(N)
      # copy to avoid layout issue
      return dot(D1_cos, D1_sin)[1:-1,1:-1].copy()

    raise NotImplementedError("Derivative degree not implemented.")

  if variety == "H1_I_sin":
    if D == 1:
      return _diff_mat_nodal_Fourier_H1_I_sin(N)

    if D == 2:
      D1_sin = _diff_mat_nodal_Fourier_H1_I_sin(N)
      D1_cos = _diff_mat_nodal_Fourier_H1_I_cos(N)
      return dot(D1_cos, D1_sin)

    raise NotImplementedError("Derivative degree not implemented.")

  if variety == "H1_I_cos":
    if D == 1:
      return _diff_mat_nodal_Fourier_H1_I_cos(N)

    if D == 2:
      D1_sin = _diff_mat_nodal_Fourier_H1_I_sin(N)
      D1_cos = _diff_mat_nodal_Fourier_H1_I_cos(N)
      return dot(D1_sin, D1_cos)

    raise NotImplementedError("Derivative degree not implemented.")

  raise ValueError("Function arguments malformed.")

###############################################################################
# JacobiP definitions
###############################################################################

def _diff_mat_nodal_JacobiP_GL(N, a, b, root_polish=True):
  if N < 2:
    raise ValueError("N must be >= 2 for GL variety")
  gr = grid_JacobiP(N, a, b, variety='GL', root_polish=root_polish)
  dP = poly_der_JacobiP(1, N - 1, a + 1, b + 1, gr,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  d_mat = empty((N + 1, N + 1), dtype=float64)

  # first column
  d_mat[0, 0] = (a - N * (N + a + b + 1)) / (2 * (b + 2))

  for k in range(1, N):
    fac = power(-1, N - 1) * (1 - gr[k]) * dP[k] / 2
    num = special_log_abs_gamma(N) + special_log_abs_gamma(b + 2)
    den = special_log_abs_gamma(N + b + 1)

    d_mat[k, 0] = fac * exp(num - den)

  fac = power(-1, N) / 2
  num = special_log_abs_gamma(b + 2) + special_log_abs_gamma(N + a + 1)
  den = special_log_abs_gamma(a + 2) + special_log_abs_gamma(N + b + 1)
  d_mat[N, 0] = fac * exp(num - den)

  # interior
  for j in range(1, N):
    fac = 2 * power(-1, N) / ((1 - gr[j]) * (1 + gr[j]) ** 2 * dP[j])
    num = special_log_abs_gamma(N + b + 1)
    den = (
      special_log_abs_gamma(N) +
      special_log_abs_gamma(b + 2)
    )

    d_mat[0, j] = fac * exp(num - den)

    for k in range(1, j):
      num = (1 - gr[k] ** 2) * dP[k]
      den = (1 - gr[j] ** 2) * dP[j] * (gr[k] - gr[j])
      d_mat[k, j] = num / den

    for k in range(j + 1, N):
      num = (1 - gr[k] ** 2) * dP[k]
      den = (1 - gr[j] ** 2) * dP[j] * (gr[k] - gr[j])
      d_mat[k, j] = num / den

    fac = -2 / ((1 - gr[j]) ** 2 * (1 + gr[j]) * dP[j])
    num = special_log_abs_gamma(N + a + 1)
    den = (
      special_log_abs_gamma(N) +
      special_log_abs_gamma(a + 2)
    )

    d_mat[N, j] = fac * exp(num - den)

    d_mat[j, j] = (a - b + (a + b) * gr[j]) / (2 * (1 - gr[j] ** 2))

  # last column
  fac = power(-1, N + 1) / 2
  num = special_log_abs_gamma(a + 2) + special_log_abs_gamma(N + b + 1)
  den = special_log_abs_gamma(b + 2) + special_log_abs_gamma(N + a + 1)
  d_mat[0, N] = fac * exp(num - den)

  for k in range(1, N):
    fac = (1 + gr[k]) * dP[k] / 2
    num = special_log_abs_gamma(N) + special_log_abs_gamma(a + 2)
    den = special_log_abs_gamma(N + a + 1)

    d_mat[k, N] = fac * exp(num - den)

  d_mat[N, N] = (N * (N + a + b + 1) - b) / (2 * (a + 2))

  return d_mat

def _diff_mat_nodal_JacobiP_G(N, a, b, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for G variety")
  gr = grid_JacobiP(N, a, b, variety='G', root_polish=root_polish)
  dP = poly_der_JacobiP(1, N + 1, a, b, gr,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  d_mat = empty((N + 1, N + 1), dtype=float64)

  for k in range(N + 1):
    for j in range(0, k):
      d_mat[k, j] = dP[k] / (dP[j] * (gr[k] - gr[j]))

    for j in range(k + 1, N + 1):
      d_mat[k, j] = dP[k] / (dP[j] * (gr[k] - gr[j]))

    d_mat[k, k] = (a - b + (a + b + 2) * gr[k]) / (2 * (1 - gr[k] ** 2))

  return d_mat

def _diff_mat_nodal_JacobiP_RL(N, a, b, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for RL variety")
  gr = grid_JacobiP(N, a, b, variety='RL', root_polish=root_polish)
  dP = poly_der_JacobiP(1, N, a, b + 1, gr,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  d_mat = empty((N + 1, N + 1), dtype=float64)

  d_mat[0, 0] = -N * (N + a + b + 2) / (2 * (b + 2))

  for k in range(1, N + 1):
    fac = dP[k] / power(-1, N)
    num = special_log_abs_gamma(N + 1) + special_log_abs_gamma(b + 2)
    den = special_log_abs_gamma(N + b + 2)
    d_mat[k, 0] = fac * exp(num - den)

  for j in range(1, N + 1):
    fac = power(-1, N + 1) / ((1 + gr[j]) ** 2 * dP[j])
    num = special_log_abs_gamma(N + b + 2)
    den = special_log_abs_gamma(N + 1) + special_log_abs_gamma(b + 2)
    d_mat[0, j] = fac * exp(num - den)

  for k in range(1, N + 1):
    for j in range(1, k):
      num = (1 + gr[k]) * dP[k]
      den = (1 + gr[j]) * dP[j] * (gr[k] - gr[j])
      d_mat[k, j] = num / den

    for j in range(k + 1, N + 1):
      num = (1 + gr[k]) * dP[k]
      den = (1 + gr[j]) * dP[j] * (gr[k] - gr[j])
      d_mat[k, j] = num / den

  for k in range(1, N + 1):
    num = a - b + 1 + (a + b + 1) * gr[k]
    den = 2 * (1 - gr[k] ** 2)
    d_mat[k, k] = num / den

  return d_mat

def _diff_mat_nodal_JacobiP_RR(N, a, b, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for RR variety")
  d_mat = _diff_mat_nodal_JacobiP_RL(N, b, a, root_polish=root_polish)
  return -rot90(d_mat, k=2)

def diff_mat_nodal_JacobiP(N : int,
                           a : float,
                           b : float,
                           D : int=1,
                           variety : str="GL",
                           root_polish : bool=True) -> NDArray[float64]:
  """
  Nodal differentiation matrices adapted to grids associated with Jacobi
  polynomials [1].

  Parameters
  ----------
  N : int
    Polynomial order.

  a : float
    Jacobi parameter a (left weight exponent).

  b : float
    Jacobi parameter b (right weight exponent).

  D : int = 1
    Derivative degree to construct.

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  root_polish : bool, optional
    If True, apply Newton polishing to the eigenvalue-based grid.
    Default is True.

  Returns
  -------
  d_mat : NDArray[float64]
    Differentiation matrix.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_JacobiP
  >>> N = 8
  >>> a = 1.3
  >>> b = 1.3

  >>> x_G = grid_JacobiP(N, a, b, variety="G")
  >>> D_G = diff_mat_nodal_JacobiP(N, a, b, variety="G")
  >>> allclose(dot(D_G, x_G ** 2), 2 * x_G)
      True

  >>> x_GL = grid_JacobiP(N, a, b, variety="GL")
  >>> D_GL = diff_mat_nodal_JacobiP(N, a, b, variety="GL")
  >>> allclose(dot(D_GL, x_GL ** 2), 2 * x_GL)
      True

  >>> x_RL = grid_JacobiP(N, a, b, variety="RL")
  >>> D_RL = diff_mat_nodal_JacobiP(N, a, b, variety="RL")
  >>> allclose(dot(D_RL, x_RL ** 2), 2 * x_RL)
      True

  >>> x_RR = grid_JacobiP(N, a, b, variety="RR")
  >>> D_RR = diff_mat_nodal_JacobiP(N, a, b, variety="RR")
  >>> allclose(dot(D_RR, x_RR ** 2), 2 * x_RR)
      True

  >>> # second degree
  >>> x_GL = grid_JacobiP(N, a, b, variety="GL")
  >>> D_GL = diff_mat_nodal_JacobiP(N, a, b, D=2, variety="GL")
  >>> allclose(dot(D_GL, x_GL ** 2), 2)
      True

  Notes
  -----
  These matrices satisfy a power property where higher degrees can be computed
  through forming the matrix product of lower orders.
  """
  if variety == "GL":
    d_mat = _diff_mat_nodal_JacobiP_GL(N, a, b, root_polish=root_polish)

    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  if variety == "G":
    d_mat = _diff_mat_nodal_JacobiP_G(N, a, b, root_polish=root_polish)

    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  if variety == "RL":
    d_mat = _diff_mat_nodal_JacobiP_RL(N, a, b, root_polish=root_polish)
    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  if variety == "RR":
    d_mat = _diff_mat_nodal_JacobiP_RR(N, a, b, root_polish=root_polish)
    if D > 1:
      d_mat = matrix_power(d_mat, D)
    return d_mat

  raise ValueError("Function arguments malformed.")


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  diff_mat_nodal_ChebyshevT = JIT(diff_mat_nodal_ChebyshevT)

  _diff_mat_nodal_Fourier_S1_CL = JIT(_diff_mat_nodal_Fourier_S1_CL)
  _diff_mat_nodal_Fourier_H1_C_sin = JIT(_diff_mat_nodal_Fourier_H1_C_sin)
  _diff_mat_nodal_Fourier_H1_C_cos = JIT(_diff_mat_nodal_Fourier_H1_C_cos)
  _diff_mat_nodal_Fourier_H1_I_sin = JIT(_diff_mat_nodal_Fourier_H1_I_sin)
  _diff_mat_nodal_Fourier_H1_I_cos = JIT(_diff_mat_nodal_Fourier_H1_I_cos)

  diff_mat_nodal_Fourier = JIT(diff_mat_nodal_Fourier)

  _diff_mat_nodal_JacobiP_GL = JIT_NC(_diff_mat_nodal_JacobiP_GL)
  _diff_mat_nodal_JacobiP_G = JIT_NC(_diff_mat_nodal_JacobiP_G)
  _diff_mat_nodal_JacobiP_RL = JIT_NC(_diff_mat_nodal_JacobiP_RL)
  _diff_mat_nodal_JacobiP_RR = JIT_NC(_diff_mat_nodal_JacobiP_RR)

  diff_mat_nodal_JacobiP = JIT_NC(diff_mat_nodal_JacobiP)

#
# :D
#
