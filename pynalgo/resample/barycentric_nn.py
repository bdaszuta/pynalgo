"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Nearest-neighbor Floater-Hormann / Berrut-Trefethen interpolation.

For each target point, selects the W nearest source nodes and builds an
FH/BT interpolant on that subset.  Reduces O(N) per target to O(W).
"""
import numpy as np

from pynalgo.common_tools import (JIT, TYPE_CHECKING)
from pynalgo.resample.barycentric import _interp_barycentric_1d_jit


###############################################################################
# Nearest-neighbor interpolation
###############################################################################

def interp_nn_1d(X, F, X_I, d, method, W):
  """1D nearest-neighbor FH/BT interpolation.

  Parameters
  ----------
  X : NDArray, shape (n,)
      Source grid (sorted).
  F : NDArray, shape (n,)
      Function values at source grid.
  X_I : NDArray, shape (m,)
      Target evaluation points.
  d : int
      FH blending parameter.
  method : int
      2 = FH, 3 = BT.
  W : int
      Number of nearest neighbors to use.

  Returns
  -------
  F_I : NDArray, shape (m,)
  """
  if W < 2:
    raise ValueError("W must be >= 2")
  if X.size < 2:
    out = np.empty(X_I.size, dtype=F.dtype)
    out[:] = F[0]
    return out

  return _interp_nn_1d_jit(X, F, X_I, d, method, W)


def _interp_nn_1d_jit(X, F, X_I, d, method, W):
  """JIT kernel: NN FH/BT on arbitrary grids."""
  n = X.size
  m = X_I.size
  out = np.empty(m, dtype=F.dtype)

  method_str = 'FH' if method == 2 else 'BT'

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

    # Find nearest source index via binary search
    lo = 0
    hi = n - 1
    while hi - lo > 1:
      mid = (lo + hi) // 2
      if X[mid] <= xt:
        lo = mid
      else:
        hi = mid
    # lo is the largest index with X[lo] <= xt
    # Nearest is lo or lo+1
    if xt - X[lo] <= X[min(lo + 1, n - 1)] - xt:
      nearest = lo
    else:
      nearest = lo + 1

    # Build stencil of W points around nearest
    half = W // 2
    st_lo = nearest - half
    st_hi = st_lo + W - 1
    if st_lo < 0:
      st_hi -= st_lo
      st_lo = 0
    if st_hi >= n:
      st_lo -= (st_hi - (n - 1))
      st_hi = n - 1
    if st_lo < 0:
      st_lo = 0

    # Extract sub-grid and interpolate
    X_sub = X[st_lo:st_hi + 1]
    F_sub = F[st_lo:st_hi + 1]
    d_use = min(d, st_hi - st_lo)

    out[i] = _interp_barycentric_1d_jit(
        X_sub, F_sub, np.array([xt]), d_use, method_str)[0]

  return out


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _interp_nn_1d_jit = JIT(_interp_nn_1d_jit)
  interp_nn_1d = JIT(interp_nn_1d)

  # Warm-up
  _wx = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
  _wf = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
  _wt = np.array([0.5, 2.5], dtype=np.float64)
  _interp_nn_1d_jit(_wx, _wf, _wt, 2, 2, 3)
  interp_nn_1d(_wx, _wf, _wt, 2, 2, 3)

#
# :D
#
