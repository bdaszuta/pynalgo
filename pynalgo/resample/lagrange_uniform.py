"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Lagrange interpolation on uniform grids.

Uses fixed-width stencils with direct Lagrange basis computation.
O(order^2) per target point but order <= 10, so competitively fast.
"""
import numpy as np

from pynalgo.common_tools import (JIT, TYPE_CHECKING)
from pynalgo.resample.barycentric import _interp_barycentric_1d_jit


###############################################################################
# Uniformity check
###############################################################################

def _check_uniform_jit(X):
  """Verify X is uniformly spaced.  Returns spacing h."""
  n = X.size
  if n < 2:
    return 1.0
  h = X[1] - X[0]
  if h == 0.0:
    raise ValueError("Grid spacing must be positive")
  for i in range(2, n):
    dh = abs(X[i] - X[i - 1] - h)
    if dh > abs(h) * 1e-12 + 1e-15:
      return -1.0
  return h


###############################################################################
# Uniform-grid Lagrange
###############################################################################

def interp_lagrange_uniform(X, F, X_I, order):
  """Lagrange interpolation on a uniform grid.

  Uses a fixed-width stencil of order+1 points around each target.
  For interior targets with odd polynomial order the stencil is
  symmetric; even orders bias toward the closer node. Near boundaries
  the stencil is biased toward the interior.

  Parameters
  ----------
  X : NDArray, shape (n,)
      Uniformly-spaced source grid.
  F : NDArray, shape (n,)
      Function values at source grid.
  X_I : NDArray, shape (m,)
      Target evaluation points.
  order : int
      Polynomial order (1 = linear, 2 = quadratic, ...).

  When ``order >= X.size``, falls back to full barycentric
  interpolation (Berrut-Trefethen) using all source points.

  Returns
  -------
  F_I : NDArray, shape (m,)
  """
  h = _check_uniform_jit(X)
  if h < 0:
    raise ValueError("X must be uniformly spaced")

  if order >= X.size:
    # Fall back to full BT (all points)
    return _interp_barycentric_1d_jit(X, F, X_I, 0, 'BT')

  return _interp_lagrange_uniform_jit(X, F, X_I, order)


def _interp_lagrange_uniform_jit(X, F, X_I, order):
  """JIT kernel: uniform-grid Lagrange via direct weights."""
  n = X.size
  m = X_I.size
  out = np.empty(m, dtype=F.dtype)
  n_stencil = order + 1
  half = order // 2

  for i in range(m):
    xt = X_I[i]

    # Coincident node shortcut
    found = False
    for k in range(n):
      if xt == X[k]:
        out[i] = F[k]
        found = True
        break
    if found:
      continue

    # Find bracket index
    lo = 0
    hi = n - 1
    while hi - lo > 1:
      mid = (lo + hi) // 2
      if X[mid] <= xt:
        lo = mid
      else:
        hi = mid

    # Stencil range: initially symmetric around xt
    # (adjusted below for even orders)
    st_lo = lo - half
    if order % 2 == 0:
      # Even order: bias toward closest
      if abs(X[lo] - xt) >= abs(X[lo + 1] - xt) and lo + 1 < n:
        st_lo = lo + 1 - half

    # Clamp
    if st_lo < 0:
      st_lo = 0
    st_hi = st_lo + n_stencil - 1
    if st_hi >= n:
      st_hi = n - 1
      st_lo = st_hi - n_stencil + 1
    if st_lo < 0:
      st_lo = 0

    # Compute Lagrange weights
    val = 0.0
    for j in range(st_lo, st_hi + 1):
      wj = 1.0
      for k in range(st_lo, st_hi + 1):
        if k == j:
          continue
        wj *= (xt - X[k]) / (X[j] - X[k])
      val += wj * F[j]
    out[i] = val

  return out


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _check_uniform_jit = JIT(_check_uniform_jit)
  _interp_lagrange_uniform_jit = JIT(_interp_lagrange_uniform_jit)
  interp_lagrange_uniform = JIT(interp_lagrange_uniform)

  # Warm-up
  _wx = np.linspace(0.0, 1.0, 5)
  _wf = _wx ** 2
  _wt = np.array([0.3, 0.7])
  _interp_lagrange_uniform_jit(_wx, _wf, _wt, 2)
  interp_lagrange_uniform(_wx, _wf, _wt, 2)

#
# :D
#
