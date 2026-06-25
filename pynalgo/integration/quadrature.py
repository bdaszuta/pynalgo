"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Quadrature weight computation for Chebyshev, Jacobi, Legendre, Fourier, and Clenshaw-Curtis rules.

Refs:
[1]: Spectral methods for Time-Dependent Problems,
     Hesthaven & Gottlieb & Gottlieb; 2007
[2]: Spectral Methods in Matlab,
     Trefethen, L. N., 2001
[3]: Chebyshev and Fourier Spectral Methods, 2E:
     Boyd, John P.; 2001
[4]: Spectral methods: Algorithms, Analysis and Applications
     Shen, Tang and Wang; 2011
"""
from numpy import (arange, cos, empty, exp, ones, pi, power, sin, zeros)

from pynalgo.common_tools import (NDArray, float64,
                                  JIT_NC, JITI, JITI_NC, TYPE_CHECKING)
from pynalgo.special_functions import (grid_JacobiP,
                                       poly_der_JacobiP,
                                       special_log_abs_gamma)

###############################################################################
# Chebyshev first kind (ChebyshevT) definitions
###############################################################################

def quad_ChebyshevT(N : int=16, variety : str="GL") -> NDArray[float64]:
  r'''
  Quadrature weights for integration on fundamental grids related to Chebyshev
  polynomials of the first kind [1].

  These weights are adapted to integrals of the form:

  .. math::

      \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} \,\mathrm{d}x

  Parameters
  ----------
  N : int = 16
    Polynomial order (meaning depends on extrema or roots grid).

  variety : str = "GL"
    Type of grid to construct. May be one of {GL, G, RL, RR} where:
    GL: Gauss-Lobatto  (extrema)
    G:  Gauss (roots)
    RL: Radau-left (includes left end-point)
    RR: Radau-right (includes right end-point)

  Returns
  -------
  wei : NDArray[float64]
    Quadrature weights.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_ChebyshevT
  >>> N = 4
  >>> wei = quad_ChebyshevT(N, variety="GL")
  >>> wei
      array([0.39269908, 0.78539816, 0.78539816, 0.78539816, 0.39269908])
  >>> gr  = grid_ChebyshevT(N, variety="GL")
  >>> allclose(dot((gr ** 2), wei), pi/2)    # compare analytical
      True

  >>> allclose(dot((grid_ChebyshevT(N, variety="G") ** 2),
  ...               quad_ChebyshevT(N, variety="G")),
  ...         pi/2)
      True
  >>> allclose(dot((grid_ChebyshevT(N, variety="RL") ** 2),
  ...               quad_ChebyshevT(N, variety="RL")),
  ...         pi/2)
      True
  >>> allclose(dot((grid_ChebyshevT(N, variety="RR") ** 2),
  ...               quad_ChebyshevT(N, variety="RR")),
  ...         pi/2)
      True
  '''

  if variety == "GL":
    wei = pi / N * ones(N + 1, dtype=float64)
    wei[0] /= 2
    wei[-1] /= 2
    return wei

  if variety == "G":
    return pi / (N + 1) * ones(N + 1, dtype=float64)

  if variety == "RL":
    wei = 2 * pi / (2 * N + 1) * ones(N + 1, dtype=float64)
    wei[0] /= 2
    return wei

  if variety == "RR":
    wei = 2 * pi / (2 * N + 1) * ones(N + 1, dtype=float64)
    wei[-1] /= 2
    return wei

  raise ValueError("Function arguments malformed.")

def quad_Clenshaw_Curtis(N : int=16) -> NDArray[float64]:
  r'''
  Clenshaw-Curtis quadrature weight calculation, adapted to Gauss-Lobatto
  Chebyshev polynomial grids of the first kind.

  These weights are adapted to integrals of the form:

  .. math::

      \int_{-1}^{1} f(x) \,\mathrm{d}x

  Parameters
  ----------
  N : int = 16
    Order of the scheme; should be >= 5. The total number of samples is N+1.

  Returns
  -------
  wei : NDArray[float64]
    Quadrature weights.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_ChebyshevT
  >>> N = 4
  >>> wei = quad_Clenshaw_Curtis(N)
  >>> gr  = grid_ChebyshevT(N, variety="GL")
  >>> allclose(dot(gr ** 4, wei), 4 / 10)
      True

  Notes
  -----
  Courtesy of JF.
  '''
  _C1 = 1.
  w = zeros(N + 1)

  ii = arange(0, N - 1) + 1
  t = ii * pi / N

  v = ones(N - 1)

  if N % 2 == 0:
    w[0] = _C1 / (N ** 2 - 1)
    w[N] = w[0]

    for k in arange(1, N / 2):
      v = v - 2 * cos(2 * k * t) / (4 * k * k - 1)
    v = v - cos(N * t) * w[0]
  else:
    w[0] = _C1 / N**2
    w[N] = w[0]
    for k in arange(1, (N - 1) / 2 + 1):
      v += -2 * cos(2 * k * t) / (4 * k * k - 1)

  w[1:-1] = (2 * v) / N
  return w

###############################################################################
# Fourier definitions
###############################################################################

def quad_Fourier(N : int=16, variety : str="S1_CL") -> NDArray[float64]:
  r"""
  Generate quadrature weights associated with standard trigonometric
  functions [3].

  The weights are adapted to integrals of the form

  .. math::

      \int_{0}^{2\pi} f(x) \,\mathrm{d}x
      \qquad
      \int_{0}^{\pi} f(x) \,\mathrm{d}x

  corresponding to the "S1" and "H1" varieties, respectively.


  Parameters
  ----------
  N : int = 16
    Function order.

  variety : str = "S1_CL"
    Type of grid to construct.
    May be one of {S1_CL, S1_CR, S1_I,
                   H1_C_cos, H1_C_sin, H1_I_cos, H1_I_sin}.
    S1_CL: Interval closed-left      [0, 2 * pi)
    S1_CR: Interval closed-right     (0, 2 * pi]
    S1_I:  Interval interior         (0, 2 * pi)
    H1_C_cos:  Interval closed       [0, pi] - for cos series
    H1_C_sin:  Interval closed       [0, pi] - for sin series
    H1_I_cos:  Interval interior     (0, pi) - for cos series
    H1_I_sin:  Interval interior     (0, pi) - for sin series

  Returns
  -------
  wei : NDArray[float64]
    Generated weights.

  Usage
  -----
  >>> from numpy import (allclose, cos, dot, pi, sin)
  >>> from pynalgo import grid_Fourier
  >>> N = 24
  >>> gr_S1_CL = grid_Fourier(N, variety="S1_CL")
  >>> wei_S1_CL = quad_Fourier(N, variety="S1_CL")
  >>> f_S1 = lambda gr: sin(2 * gr) ** 2 * cos(3* gr) ** 4
  >>> f_S1_CL = f_S1(gr_S1_CL)
  >>> # f_S1 integration over [0, 2 * pi] yields 3 * pi / 8
  >>> I_S1_CL = dot(f_S1_CL, wei_S1_CL)
  >>> allclose(I_S1_CL, 3 * pi / 8)
      True
  """
  if variety in ("S1_CL", "S1_CR", "S1_I"):
    return 2 * pi / N * ones(N)

  if variety == "H1_C_cos":
    wei = (pi / N) * ones(N + 1, dtype=float64)
    wei[0] *= 1 / 2
    wei[N] *= 1 / 2
    return wei

  if variety == "H1_I_cos":
    return (pi / N) * ones(N, dtype=float64)

  if variety == "H1_C_sin":
    wei = zeros(N + 1, dtype=float64)
    fac = 2 / N
    for i in range(1, N):
      for m in range(1, N):
        wei[i] += fac / m * sin(m * pi * i / N) * (1 - cos(m * pi))
    return wei

  if variety == "H1_I_sin":
    wei = zeros(N, dtype=float64)
    for i in range(1, N+1):
      xi = (2 * i - 1) * pi / (2 * N)

      wei[i-1] += 2 / N ** 2 * sin(N * xi) * sin(N * pi / 2) ** 2

      for m in range(1, N):
        wei[i-1] += 4 / (m * N) * sin(m * xi) * sin(m * pi / 2) ** 2
    return wei

  raise ValueError("Function arguments malformed.")

###############################################################################
# Jacobi polynomials
###############################################################################

def _impl_JacobiP_Gtil(N=None, a=None, b=None):
  return (power(2, a + b + 1) *
    exp(
        (special_log_abs_gamma(N + a + 2) +
         special_log_abs_gamma(N + b + 2)) -
        (special_log_abs_gamma(N + 2) +
         special_log_abs_gamma(N + a + b + 2))
    )
  )

def _impl_JacobiP_G(N=None, a=None, b=None, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for G variety")
  gr_G = grid_JacobiP(N, a, b, variety='G', root_polish=root_polish)

  dP = poly_der_JacobiP(1, N + 1, a, b, gr_G,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  Gtil = _impl_JacobiP_Gtil(N, a, b)
  return Gtil / ((1 - gr_G ** 2) * dP ** 2)

def _impl_JacobiP_RL(N=None, a=None, b=None, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for RL variety")
  wei = empty(N + 1, dtype=float64)
  gr_RL = grid_JacobiP(N, a, b, variety='RL', root_polish=root_polish)

  dP = poly_der_JacobiP(1, N, a, b + 1, gr_RL,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  Gtil = _impl_JacobiP_Gtil(N - 1, a, b + 1)

  # interior and right end-point
  wei[1:] = Gtil / ((1 - gr_RL[1:]) * (1 + gr_RL[1:]) ** 2 * dP[1:] ** 2)

  # left end-point
  fac = power(2, a + b + 1) * (b + 1)
  num = (special_log_abs_gamma(b + 1) +
         special_log_abs_gamma(b + 1) +
         special_log_abs_gamma(N + 1) +
         special_log_abs_gamma(N + a + 1)
  )
  den = (special_log_abs_gamma(N + b + 2) +
         special_log_abs_gamma(N + a + b + 2)
  )

  wei[0] = fac * exp(num - den)

  return wei

def _impl_JacobiP_RR(N=None, a=None, b=None, root_polish=True):
  if N < 1:
    raise ValueError("N must be >= 1 for RR variety")
  return _impl_JacobiP_RL(N, b, a, root_polish=root_polish)[::-1].copy()

def _impl_JacobiP_GL(N=None, a=None, b=None, root_polish=True):
  if N < 2:
    raise ValueError("N must be >= 2 for GL variety")
  wei = empty(N + 1, dtype=float64)
  gr_GL = grid_JacobiP(N, a, b, variety='GL', root_polish=root_polish)

  dP = poly_der_JacobiP(1, N - 1, a + 1, b + 1, gr_GL,
                        ret_der_num=1, ret_ord_num=1)[0, 0, :]

  Gtil = _impl_JacobiP_Gtil(N - 2, a + 1, b + 1)
  wei[1:-1] = Gtil / ((1 - gr_GL[1:-1] ** 2) ** 2 * dP[1:-1] ** 2)

  fac = power(2, a + b + 1) * (b + 1)
  num = (special_log_abs_gamma(b + 1) +
         special_log_abs_gamma(b + 1) +
         special_log_abs_gamma(N) +
         special_log_abs_gamma(N + a + 1)
  )
  den = (special_log_abs_gamma(N + b + 1) +
         special_log_abs_gamma(N + a + b + 2)
  )

  wei[0] = fac * exp(num - den)

  fac = power(2, a + b + 1) * (a + 1)
  num = (special_log_abs_gamma(a + 1) +
         special_log_abs_gamma(a + 1) +
         special_log_abs_gamma(N) +
         special_log_abs_gamma(N + b + 1)
  )
  den = (special_log_abs_gamma(N + a + 1) +
         special_log_abs_gamma(N + a + b + 2)
  )

  wei[-1] = fac * exp(num - den)
  return wei

def quad_JacobiP(N : int,
                 a : float,
                 b : float,
                 variety : str="GL",
                 root_polish : bool=True) -> NDArray[float64]:
  r'''
  Quadrature weights for integration on fundamental grids related to Jacobi
  polynomials [4].

  These weights are adapted to integrals of the form:

  .. math::

      \int_{-1}^{1} f(x) (1-x)^a (1+x)^b \,\mathrm{d}x

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
  wei : NDArray[float64]
    Quadrature weights.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_JacobiP
  >>> N = 16
  >>> a, b = 1.3, 1.3
  >>> wei = quad_JacobiP(N, a, b, variety="GL")
  >>> gr  = grid_JacobiP(N, a, b, variety="GL")
  >>> allclose(dot((gr ** 2), wei), 0.22026696481959854)  # compare analytical
      True

  >>> allclose(dot((grid_JacobiP(N, a, b, variety="G") ** 2),
  ...               quad_JacobiP(N, a, b, variety="G")),
  ...          0.22026696481959854)
      True
  >>> allclose(dot((grid_JacobiP(N, a, b, variety="RL") ** 2),
  ...               quad_JacobiP(N, a, b, variety="RL")),
  ...          0.22026696481959854)
      True
  >>> allclose(dot((grid_JacobiP(N, a, b, variety="RR") ** 2),
  ...               quad_JacobiP(N, a, b, variety="RR")),
  ...          0.22026696481959854)
      True
  '''

  if variety == "GL":
    return _impl_JacobiP_GL(N, a, b, root_polish=root_polish)

  if variety == "G":
    return _impl_JacobiP_G(N, a, b, root_polish=root_polish)

  if variety == "RL":
    return _impl_JacobiP_RL(N, a, b, root_polish=root_polish)

  if variety == "RR":
    return _impl_JacobiP_RR(N, a, b, root_polish=root_polish)

  raise ValueError("Function arguments malformed.")

###############################################################################
# Legendre polynomials
###############################################################################

def quad_LegendreP(N : int,
                   variety : str="GL",
                   root_polish : bool=True) -> NDArray[float64]:
  r'''
  Quadrature weights for integration on fundamental grids related to Legendre
  polynomials [4].

  These weights are adapted to integrals of the form:

  .. math::

      \int_{-1}^{1} f(x) \,\mathrm{d}x

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
  wei : NDArray[float64]
    Quadrature weights.

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import grid_LegendreP
  >>> N = 16
  >>> wei = quad_LegendreP(N, variety="GL")
  >>> gr  = grid_LegendreP(N, variety="GL")
  >>> allclose(dot((gr ** 2), wei), 0.6666666666666657)  # compare analytical
      True

  >>> allclose(dot((grid_LegendreP(N, variety="G") ** 2),
  ...               quad_LegendreP(N, variety="G")),
  ...          0.6666666666666657)
      True
  >>> allclose(dot((grid_LegendreP(N, variety="RL") ** 2),
  ...               quad_LegendreP(N, variety="RL")),
  ...          0.6666666666666657)
      True
  >>> allclose(dot((grid_LegendreP(N, variety="RR") ** 2),
  ...               quad_LegendreP(N, variety="RR")),
  ...          0.6666666666666657)
      True
  '''
  return quad_JacobiP(N, 0., 0., variety=variety, root_polish=root_polish)

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  quad_ChebyshevT = JITI(quad_ChebyshevT)
  quad_Clenshaw_Curtis = JITI(quad_Clenshaw_Curtis)
  quad_Fourier = JITI(quad_Fourier)

  _impl_JacobiP_Gtil = JITI_NC(_impl_JacobiP_Gtil)

  _impl_JacobiP_GL = JITI_NC(_impl_JacobiP_GL)
  _impl_JacobiP_G = JITI_NC(_impl_JacobiP_G)
  _impl_JacobiP_RL = JITI_NC(_impl_JacobiP_RL)
  _impl_JacobiP_RR = JITI_NC(_impl_JacobiP_RR)

  quad_JacobiP = JIT_NC(quad_JacobiP)

  quad_LegendreP = JIT_NC(quad_LegendreP)

#
# :D
#
