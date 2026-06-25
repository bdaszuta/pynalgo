"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Construction of bases adapted to spectral methods.
"""
from numpy import (cos, empty, exp, ones, pi, sin)

from pynalgo.common_tools import (NDArray, complex128, float64, JIT,
                                  TYPE_CHECKING)
from pynalgo.special_functions import (
  grid_ChebyshevT,
  poly_ChebyshevT,
  grid_Fourier
)
from pynalgo.integration import (
  quad_ChebyshevT,
  quad_Fourier
)

###############################################################################
# Chebyshev first kind (ChebyshevT) definitions
###############################################################################

def poly_basis_mat_ChebyshevT(N : int,
                              variety : str="GL",
                              quad_weight : bool=False) -> NDArray[float64]:
  """
  Convenience function for construction of Chebyshev polynomials sampled over
  fundamental grids. Result can be optionally masked with quadrature weights.

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

  quad_weight : bool = False
    Control whether functions are masked with appropriate quadrature weight.

  Returns
  -------
  bas : NDArray[float64]
    Constructed basis output is: function orders x grid

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import (array_dot_2d_at_axis, grid_ChebyshevT)
  >>> N = 32
  >>> var_gr = "GL"
  >>> gr = grid_ChebyshevT(N, variety=var_gr)
  >>> wbas = poly_basis_mat_ChebyshevT(N, variety=var_gr, quad_weight=True)
  >>> fun = gr ** 5
  >>> # extract coefficients
  >>> f_n = array_dot_2d_at_axis(wbas, fun)

  >>> # project
  >>> bas = poly_basis_mat_ChebyshevT(N, variety=var_gr, quad_weight=False)
  >>> rec_fun = dot(bas.T, f_n)
  >>> allclose(rec_fun, fun)
      True
  """
  if variety in ("GL", "G", "RL", "RR"):
    gr = grid_ChebyshevT(N, variety=variety)
    sz_gr = gr.size

    bas = poly_ChebyshevT(N, gr, ret_ord_num=N+1)

    if quad_weight is True:
      wei = quad_ChebyshevT(N, variety=variety)

      for j in range(sz_gr):
        bas[0, j] = wei[j] * bas[0, j] / pi
        for i in range(1, bas.shape[0]):
          bas[i, j] = wei[j] * bas[i, j] / (pi / 2)

    return bas

  raise ValueError("Function arguments malformed.")

###############################################################################
# Fourier definitions
###############################################################################

def poly_basis_mat_trig(N : int,
                        variety : str="H1_C_cos",
                        quad_weight : bool=False) -> NDArray[float64]:
  """
  Convenience function for construction of trigonometric polynomials sampled
  over fundamental grids. Result can be optionally masked with quadrature
  weights.

  Parameters
  ----------
  N : int
    Number of grid intervals / sampling resolution (determines grid size
    and maximum resolved frequency).

  variety : str = "H1_C_cos"
    Type of grid to construct.
    May be one of {H1_C_cos, H1_C_sin, H1_I_cos, H1_I_sin}.
    H1_C_cos:  Interval closed       [0, pi] - for cos series
    H1_C_sin:  Interval closed       [0, pi] - for sin series
    H1_I_cos:  Interval interior     (0, pi) - for cos series
    H1_I_sin:  Interval interior     (0, pi) - for sin series

  quad_weight : bool = False
    Control whether functions are masked with appropriate quadrature weight.

  Returns
  -------
  bas : NDArray[float64]
    Constructed basis output is: function orders x grid

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import (array_dot_2d_at_axis, grid_Fourier)
  >>> N = 32
  >>> gr = grid_Fourier(N, variety="H1_C")
  >>> wbas = poly_basis_mat_trig(N, variety="H1_C_cos", quad_weight=True)
  >>> fun = cos(4 * gr)
  >>> # extract coefficients
  >>> f_n = array_dot_2d_at_axis(wbas, fun)

  >>> # project
  >>> bas = poly_basis_mat_trig(N, variety="H1_C_cos", quad_weight=False)
  >>> rec_fun = dot(bas.T, f_n)
  >>> allclose(rec_fun, fun)
      True
  """
  if variety in ("H1_C_cos", "H1_C_sin"):
    gr = grid_Fourier(N, variety="H1_C")
    sz_gr = gr.size

    B = empty((sz_gr, sz_gr), dtype=float64)

    if quad_weight:
      wei = (2 / pi) * quad_Fourier(N, variety=variety)  # normalize
    else:
      wei = ones(sz_gr, dtype=float64)

    if variety == "H1_C_cos":
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = wei[j] * cos(i * gr[j])

    if variety == "H1_C_sin":
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = wei[j] * sin((i + 1) * gr[j])

    return B

  if variety in ("H1_I_cos", "H1_I_sin"):
    gr = grid_Fourier(N, variety="H1_I")
    sz_gr = gr.size

    B = empty((sz_gr, sz_gr), dtype=float64)

    if quad_weight:
      wei = (2 / pi) * quad_Fourier(N, variety=variety)  # normalize
    else:
      wei = ones(sz_gr, dtype=float64)

    if variety == "H1_I_cos":
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = wei[j] * cos(i * gr[j])

    if variety == "H1_I_sin":
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = wei[j] * sin((i + 1) * gr[j])

    return B

  raise ValueError("Function arguments malformed.")

def poly_basis_mat_trig_exp(N : int,
                            variety : str="S1_CL",
                            quad_weight : bool=False) -> NDArray[complex128]:
  """
  Convenience function for construction of trigonometric polynomials sampled
  over fundamental grids. Result can be optionally masked with quadrature
  weights.

  Parameters
  ----------
  N : int
    Number of grid points / sampling resolution (determines grid size
    and maximum resolved frequency).

  variety : str = "S1_CL"
    Type of grid to construct.
    May be one of {S1_CL, S1_CR, S1_I}.
    S1_CL: Interval closed-left  [0, 2 * pi)
    S1_CR: Interval closed-right (0, 2 * pi]
    S1_I:  Interval interior     (0, 2 * pi)

  quad_weight : bool = False
    Control whether functions are masked with appropriate quadrature weight.

  Returns
  -------
  bas : NDArray[complex128]
    Constructed basis output is: function orders x grid

  Usage
  -----
  >>> from numpy import (allclose, dot)
  >>> from pynalgo import (array_dot_2d_at_axis, grid_Fourier)
  >>> N = 32
  >>> gr = grid_Fourier(N, variety="S1_I")
  >>> wbas = poly_basis_mat_trig_exp(N, variety="S1_I", quad_weight=True)
  >>> fun = exp(-4j * gr)
  >>> # extract coefficients
  >>> f_n = array_dot_2d_at_axis(wbas, fun)

  >>> # project
  >>> bas = poly_basis_mat_trig_exp(N, variety="S1_I", quad_weight=False)
  >>> rec_fun = dot(bas.T, f_n)
  >>> allclose(rec_fun, fun)
      True
  """
  if variety in ("S1_CL", "S1_CR", "S1_I"):
    gr = grid_Fourier(N, variety=variety)
    sz_gr = gr.size

    B = empty((sz_gr, sz_gr), dtype=complex128)

    if quad_weight:
      wei = quad_Fourier(N, variety=variety) / (2 * pi)  # normalize
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = wei[j] * exp(-1j * (i - (N // 2)) * gr[j])

    else:
      for j in range(sz_gr):
        for i in range(sz_gr):
          B[i, j] = exp(1j * (i - (N // 2)) * gr[j])

    return B

  raise ValueError("Function arguments malformed.")

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  poly_basis_mat_ChebyshevT = JIT(poly_basis_mat_ChebyshevT)
  poly_basis_mat_trig = JIT(poly_basis_mat_trig)
  poly_basis_mat_trig_exp = JIT(poly_basis_mat_trig_exp)

#
# :D
#
