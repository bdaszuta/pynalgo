"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Grid generation related to special functions.

Refs:
[1]: Spectral methods for Time-Dependent Problems,
     Hesthaven & Gottlieb & Gottlieb; 2007
[2]: Numerical Recipes in C, 2E
     Press, et al., 1997
[3]: Chebyshev and Fourier Spectral Methods, 2E:
     Boyd, John P.; 2001
"""
from numpy import (abs,
                   arange, cos, diag, empty, pi, sort, sqrt, zeros)
from numpy.linalg import eigvalsh

from pynalgo.common_tools import (NDArray, float64, JITI, JIT, TYPE_CHECKING)
from pynalgo.special_functions.recursion_polynomial import (
  poly_JacobiP,
  poly_der_JacobiP
)

###############################################################################
# Chebyshev first kind (ChebyshevT) definitions
###############################################################################

def grid_ChebyshevT(N : int=16, variety : str="GL") -> NDArray[float64]:
  """
  Generate grids based on Chebyshev polynomials of the first kind [1].

  Parameters
  ----------
  N : int=16
    Polynomial order (meaning depends on extrema or roots grid).

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  Returns
  -------
  grid : NDArray[float64]
    Generated grid.

  Usage
  -----
  >>> grid_ChebyshevT(N=4, variety="G")
  array([-9.5106e-01, -5.8779e-01, -6.1232e-17,  5.8779e-01,  9.5106e-01])

  >>> grid_ChebyshevT(N=4, variety="GL")
  array([-1.0000e+00, -7.0711e-01,  6.1232e-17,  7.0711e-01,  1.0000e+00])

  >>> grid_ChebyshevT(N=3, variety="RL")
  array([-1.        , -0.6234898 ,  0.22252093,  0.90096887])

  >>> grid_ChebyshevT(N=3, variety="RR")
  array([-0.90096887, -0.22252093,  0.6234898 ,  1.        ])
  """
  if N == 0 and variety == "GL":
    raise ValueError("N must be >= 1 for GL variety")

  k = arange(0, N + 1)

  if variety == "GL":
    return cos(pi / N * k)[::-1]

  if variety == "G":
    return -cos(pi / (2*(N + 1)) * (2*k + 1))

  if variety == "RL":
    xk = -cos(2 * pi / (2*N + 1) * k)
    return xk

  if variety == "RR":
    xk = -cos(2 * pi / (2*N + 1) * k)
    return -xk[::-1]

  raise ValueError("Function arguments malformed.")

###############################################################################
# Jacobi polynomials and derivatives
###############################################################################

def _impl_grid_JacobiP_Gauss(N : int,
                             a : float,
                             b : float,
                             root_polish : bool=True)->NDArray[float64]:
  """Compute Gauss-grid roots for JacobiP(a,b) via eigenvalue method.

  Returns grid points in ascending order.
  """
  # roots (Gauss grid)

  j = arange(0, N+1)

  if abs(a) == abs(b):
    _a_fac = zeros(N+1)
    # can get a contribution from the zeroth term
    _a_fac[0] = (-a + b) / (2 + a + b)
  else:
    _a_fac = (b * b - a * a) / (
      (2 * j + a + b) * (2 * j + a + b + 2)
    )

  _b_fac = 2 / (2 * j[1:] + a + b) * sqrt(
    j[1:] * (j[1:] + a) * (j[1:] + b) * (j[1:] + a + b) / (
   (2 * j[1:] + a + b - 1) * (2 * j[1:] + a + b + 1))
  )

  # Use eigvalsh (dense) - scipy eigvalsh_tridiagonal not usable under numba
  A = diag(_a_fac) + diag(_b_fac, k=-1) + diag(_b_fac, k=+1)
  gr = sort(eigvalsh(A))

  if root_polish:
    last_max = max(abs(poly_JacobiP(N+1,a,b,gr)).flatten())
    MAX_ITER = 10
    NUM_ITER = 0
    DECR_FAC = 1e-5

    while True:
      fdf = poly_der_JacobiP(1, N+1, a, b, gr, ret_der_num=2)
      gr = gr - fdf[0,0,:] / fdf[1,0,:]
      cur_max = max(abs(poly_JacobiP(N+1,a,b,gr)).flatten())

      if abs(1-last_max/cur_max) < DECR_FAC:
        break

      last_max = cur_max

      if NUM_ITER == MAX_ITER:
        break

      NUM_ITER += 1

  return gr

def grid_JacobiP(N : int,
                 a : float,
                 b : float,
                 variety : str="GL",
                 root_polish : bool=True) -> NDArray[float64]:
  """
  Generate grids based on Jacobi polynomials [2].

  Parameters
  ----------
  N : int
    Polynomial order (meaning depends on extrema or roots grid).

  a : float
    Alpha parameter.

  b : float
    Beta parameter.

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  root_polish : bool=True
    Solver uses eigenvalue extraction; select to polish via root-finding.

  Returns
  -------
  grid : NDArray[float64]
    Generated grid.

  Usage
  -----
  >>> N, a, b = 4, -2/3, 2/3
  >>> grid_JacobiP(N, a, b, variety="G", root_polish=True)
      array([-0.81710718, -0.37284149,  0.18985254,  0.69224859,  0.9745142 ])

  >>> grid_JacobiP(N, a, b, variety="GL", root_polish=True)
      array([-1.        , -0.49660793,  0.20382686,  0.79278108,  1.        ])

  >>> grid_JacobiP(N, a, b, variety="RL", root_polish=True)
      array([-1.        , -0.58416278,  0.029843  ,  0.62291643,  0.96844038])

  >>> grid_JacobiP(N, a, b, variety="RR", root_polish=True)
      array([-0.7807962 , -0.26226128,  0.35526945,  0.83593618,  1.        ])
  """

  if variety == "GL":
    gr = empty(N+1, dtype=float64)
    gr[1:-1] = _impl_grid_JacobiP_Gauss(N-2, a+1, b+1,
                                        root_polish=root_polish)
    gr[0] = -1
    gr[-1] = 1
    return gr

  if variety == "G":
    return _impl_grid_JacobiP_Gauss(N, a, b, root_polish=root_polish)

  if variety == "RL":
    gr = empty(N+1, dtype=float64)
    gr[1:] = _impl_grid_JacobiP_Gauss(N-1, a, b+1,
                                      root_polish=root_polish)
    gr[0] = -1
    return gr

  if variety == "RR":
    gr = empty(N+1, dtype=float64)
    gr[:-1] = _impl_grid_JacobiP_Gauss(N-1, a+1, b,
                                       root_polish=root_polish)
    gr[-1] = 1
    return gr

  raise ValueError("Function arguments malformed.")

###############################################################################
# Legendre polynomials
###############################################################################

def grid_LegendreP(N : int,
                   variety : str="GL",
                   root_polish : bool=True) -> NDArray[float64]:
  """
  Generate grids based on Legendre polynomials [2].

  Parameters
  ----------
  N : int
    Polynomial order (meaning depends on extrema or roots grid).

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  root_polish : bool=True
    Solver uses eigenvalue extraction; select to polish via root-finding.

  Returns
  -------
  grid : NDArray[float64]
    Generated grid.

  Usage
  -----
  >>> N = 4
  >>> grid_LegendreP(N, variety="G", root_polish=True)
      array([-0.90617985, -0.53846931,  0.        ,  0.53846931,  0.90617985])

  >>> grid_LegendreP(N, variety="GL", root_polish=True)
      array([-1.        , -0.65465367,  0.        ,  0.65465367,  1.        ])

  >>> grid_LegendreP(N, variety="RL", root_polish=True)
      array([-1.        , -0.72048027, -0.16718086,  0.44631397,  0.88579161])

  >>> grid_LegendreP(N, variety="RR", root_polish=True)
      array([-0.88579161, -0.44631397,  0.16718086,  0.72048027,  1.        ])
  """
  return grid_JacobiP(N, 0., 0., variety=variety, root_polish=root_polish)


###############################################################################
# Fourier definitions
###############################################################################

def grid_Fourier(N : int=16, variety : str="S1_CL") -> NDArray[float64]:
  """
  Generate grids associated with standard trigonometric functions [3].

  Parameters
  ----------
  N : int=16
    Function order.

  variety : str = "S1_CL"
    Type of grid to construct.
    May be one of {S1_CL, S1_CR, S1_I, H1_C, H1_I, H1_S}.
    S1_CL: Interval closed-left  [0, 2 * pi)
    S1_CR: Interval closed-right (0, 2 * pi]
    S1_I:  Interval interior     (0, 2 * pi)
    H1_C:  Interval closed       [0, pi]
    H1_I:  Interval interior     (0, pi)
    H1_S:  Interval interior     (0, pi)
           Alternate choice of sampling

  Returns
  -------
  grid : NDArray[float64]
    Generated grid.

  Usage
  -----
  >>> N = 4
  >>> grid_Fourier(N, variety="S1_CL")
      array([0.        , 1.57079633, 3.14159265, 4.71238898])
  >>> grid_Fourier(N, variety="S1_CR")
      array([1.57079633, 3.14159265, 4.71238898, 6.28318531])
  >>> grid_Fourier(N, variety="S1_I")
      array([0.78539816, 2.35619449, 3.92699082, 5.49778714])
  >>> grid_Fourier(N, variety="H1_C")
      array([0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265])
  >>> grid_Fourier(N, variety="H1_I")
      array([0.39269908, 1.17809725, 1.96349541, 2.74889357])
  >>> grid_Fourier(N, variety="H1_S")
      array([0.78539816, 1.57079633, 2.35619449])
  """
  if variety in ("S1_CL", "S1_I"):
    gr = 2 * pi / N * arange(N)

    if variety == "S1_I":
      gr = gr + (gr[1] - gr[0]) / 2
    return gr

  if variety == "S1_CR":
    return 2 * pi / N * arange(1, N + 1)

  if variety == "H1_C":
    return pi / N * arange(N + 1)

  if variety == "H1_I":
    return (2 * arange(1, N + 1) - 1) * pi / (2 * N)

  if variety == "H1_S":
    return pi / N * arange(1, N)

  raise ValueError("Function arguments malformed.")

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  grid_ChebyshevT = JITI(grid_ChebyshevT)

  _impl_grid_JacobiP_Gauss = JIT(_impl_grid_JacobiP_Gauss)
  grid_JacobiP = JIT(grid_JacobiP)
  grid_LegendreP = JIT(grid_LegendreP)
  grid_Fourier = JIT(grid_Fourier)

#
# :D
#
