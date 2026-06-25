"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Modified Chebyshev basis functions with prescribed endpoint
           vanishing for spectral Galerkin methods.

The basis functions T_n^{(k)}(x) are linear combinations of standard Chebyshev
polynomials T_n(x) such that T_n^{(k)} and its first k derivatives
vanish at x = +-1.
"""
from numpy import (empty,)

from pynalgo.common_tools import (NDArray, Any, float64,
                                  JIT, TYPE_CHECKING)
from pynalgo.special_functions.recursion_polynomial import (
    poly_ChebyshevT_direct,
)

###############################################################################
# Modified Chebyshev basis
###############################################################################

def poly_ChebyshevT_mod(N : int,
                        x : NDArray[Any],
                        ret_ord_num : int=1,
                        mod_ord : int=0) -> NDArray[float64]:
  r'''
  Modified Chebyshev basis functions with endpoint vanishing.

  Constructs :math:`\mathrm{Tmod}_n(x)` as linear combinations of standard
  :math:`T_n(x)` such that
  :math:`\mathrm{Tmod}_n` and its first :math:`\mathrm{mod\_ord}` derivatives
  vanish at :math:`x = \pm 1`:

  .. math::

     \mathrm{mod\_ord} = 0&: \quad \mathrm{Tmod}_n(\pm 1) = 0 \\
     \mathrm{mod\_ord} = 1&: \quad \mathrm{Tmod}_n(\pm 1) =
                                     \mathrm{Tmod}'_n(\pm 1) = 0 \\
     \mathrm{mod\_ord} = 2&: \quad \mathrm{Tmod}_n(\pm 1) =
                                     \mathrm{Tmod}'_n(\pm 1) =
                                     \mathrm{Tmod}''_n(\pm 1) = 0

  Index is shifted so :math:`\mathrm{Tmod}_0` is the lowest-order basis
  function.

  The construction uses analytically derived linear combination
  coefficients that cancel endpoint values and derivatives:

  .. math::
     \mathrm{Tmod}_k = T_{k+2(\mathrm{mod\_ord}+1)} - \sum_j c_j T_j

  Parameters
  ----------
  N : int
    Order of highest modified polynomial to compute.

  x : NDArray[Any]
    The domain over which to compute.

  ret_ord_num : int=1
    Number of orders to return (N as highest included).

  mod_ord : int=0
    Number of vanishing derivative conditions at endpoints (0, 1, or 2).

  Returns
  -------
  poly_arr : NDArray[float64]
    Values of modified Chebyshev basis functions.

  Usage
  -----
  >>> from numpy import linspace, array
  >>> N = 4
  >>> x = linspace(-1, 1, num=5)
  >>> Tm = poly_ChebyshevT_mod(N, x, ret_ord_num=3, mod_ord=0)
  >>> # Endpoints should be zero
  >>> Tm[:, array([0, -1])]
      array([[0., 0.],
             [0., 0.],
             [0., 0.]])
  '''
  ord_offset = (mod_ord + 1) * 2

  # Compute all required standard T_n polynomials
  T_all = poly_ChebyshevT_direct(N + ord_offset, x,
                                 ret_ord_num=N + ord_offset + 1)

  # The lowest ord_offset polynomials form the LC basis
  T_LC = T_all[:ord_offset, :]  # T_0 through T_{ord_offset-1}

  n_pts = x.size
  res = empty((ret_ord_num, n_pts), dtype=float64)

  for k_out in range(ret_ord_num):
    k = N - ret_ord_num + k_out + 1  # Actual Tmod_ index (0..N)
    n_mapped = ord_offset + k        # Actual T_n order

    for i in range(n_pts):
      val = T_all[ord_offset + k, i]  # T_{n_mapped}(x)

      if mod_ord == 0:
        # Tmod__k = T_{k+2} - T_0 (even parity) or T_{k+2} - T_1 (odd)
        lc_idx = 0 if n_mapped % 2 == 0 else 1
        val = val - T_LC[lc_idx, i]

      elif mod_ord == 1:
        if n_mapped % 2 == 0:
          nh = n_mapped / 2.0
          a = 1.0 - nh * nh
          b = nh * nh
          val = val - a * T_LC[0, i] - b * T_LC[2, i]
        else:
          n2 = n_mapped * n_mapped
          c = (n2 - 9.0) / 8.0
          d = (n2 - 1.0) / 8.0
          val = val + c * T_LC[1, i] - d * T_LC[3, i]

      elif mod_ord == 2:
        if n_mapped % 2 == 0:
          n2 = n_mapped * n_mapped
          n4 = n2 * n2
          a0 = (64.0 - 20.0 * n2 + n4) / 64.0
          a2 = n2 * (n2 - 16.0) / 48.0
          a4 = n2 * (n2 - 4.0) / 192.0
          val = val - a0 * T_LC[0, i] + a2 * T_LC[2, i] - a4 * T_LC[4, i]
        else:
          n2 = n_mapped * n_mapped
          n4 = n2 * n2
          a1 = (225.0 - 34.0 * n2 + n4) / 192.0
          a3 = (25.0 - 26.0 * n2 + n4) / 128.0
          a5 = (9.0 - 10.0 * n2 + n4) / 384.0
          val = val - a1 * T_LC[1, i] + a3 * T_LC[3, i] - a5 * T_LC[5, i]

      res[k_out, i] = val

  return res

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  poly_ChebyshevT_mod = JIT(poly_ChebyshevT_mod)

#
# :D
#
