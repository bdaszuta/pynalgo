"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Neville polynomial interpolation via triangular tableau.
"""
from numpy import zeros

from pynalgo.common_tools import (NDArray, JIT, TYPE_CHECKING)

###############################################################################
# Neville interpolation
###############################################################################

def interp_Neville(xx : NDArray,
                   pp : NDArray,
                   x  : NDArray) -> NDArray:
  '''
  Polynomial interpolation based on Neville's algorithm.

  Given n+1 (distinct) samples `pp` at `xx` the unique polynomial of
  degree `n` is evaluated over `x`.

  Parameters
  ----------
  xx : NDArray
    Source grid coordinates.

  pp : NDArray
    Function values at source grid.

  x : NDArray
    Interpolation target coordinates.

  Returns
  -------
  result : NDArray
    Interpolated values at `x`.

  Usage
  -----
  >>> from numpy import array
  >>> pp_f = lambda x: x ** 4 - 2 * x ** 2 + 0.2 * x
  >>> xx = array([0.1, 0.2, 0.3, 1, 3])
  >>> pp = pp_f(xx)
  >>> x = array([0.125, 2.5])
  >>> interp_Neville(xx, pp, x) - pp_f(x)
      array([8.67361738e-19, 7.10542736e-15])
  '''
  if xx.size > 1:
    for i in range(xx.size - 1):
      if xx[i + 1] <= xx[i]:
        raise ValueError("Source grid must be strictly increasing")
  return _interp_Neville_jit(xx, pp, x)

def _interp_Neville_jit(xx, pp, x):
  N = xx.size
  P = zeros((N, N, x.size), dtype=pp.dtype)
  n = N - 1

  for i in range(n + 1):
    P[i, i] = pp[i]

  for K in range(1, n + 1):
    for i in range(n + 1 - K):
      j = K + i
      P[i, j] = ((x[:] - xx[j]) * P[i, j - 1] -
                 (x[:] - xx[i]) * P[i + 1, j]) / (xx[i] - xx[j])

  return P[0, -1, :]

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _interp_Neville_jit = JIT(_interp_Neville_jit)

#
# :D
#
