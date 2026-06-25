"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Lebesgue function for interpolation nodes.

Given a set of source (interpolation) nodes, compute the Lebesgue function
on a set of target (evaluation) nodes.  The Lebesgue function is defined as
the sum of absolute values of the Lagrange basis polynomials at each
target node.

JIT-compiled at module load via pynalgo.common_tools.JITI.
"""
from numpy import (empty, ones, )

from pynalgo.common_tools import (NDArray, Any, JITI, TYPE_CHECKING)

###############################################################################

def Lebesgue_func(x_s : NDArray[Any],
                  x_t : NDArray[Any]) -> NDArray[Any]:
  '''
  Compute the Lebesgue function via explicit loops (numba nopython compatible).

  Parameters
  ----------
  x_s : ndarray
    Source (interpolation) nodes.
  x_t : ndarray
    Target (evaluation) nodes.

  Returns
  -------
  lam : ndarray
    Lebesgue function values at each x_t.
  '''
  if x_s.size > 1:
    for i in range(x_s.size - 1):
      if x_s[i + 1] <= x_s[i]:
        raise ValueError("Source grid must be strictly increasing")

  n = x_s.size
  m = x_t.size

  dx = empty((n, n))
  for j in range(n):
    for i in range(n):
      dx[j, i] = x_s[j] - x_s[i] + (1.0 if j == i else 0.0)

  odx = 1.0 / dx
  for j in range(n):
    odx[j, j] = 0.0  # subtract identity

  pr2 = empty((m, n, n))
  for k in range(m):
    for j in range(n):
      for i in range(n):
        pr2[k, j, i] = (x_t[k] - x_s[i]) * odx[j, i]
      pr2[k, j, j] += 1.0  # add identity along diagonal

  lj = ones((m, n))
  for k in range(m):
    for j in range(n):
      prod = 1.0
      for i in range(n):
        prod *= pr2[k, j, i]
      lj[k, j] = prod

  lam = empty(m)
  for k in range(m):
    s = 0.0
    for j in range(n):
      s += abs(lj[k, j])
    lam[k] = s

  return lam

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  Lebesgue_func = JITI(Lebesgue_func)

#
# :D
#
