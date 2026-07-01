"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Column-wise interpolation and smoothing for 2D tabular data.

All functions JIT-compiled.  No scipy dependencies.
"""
import numpy as np

from pynalgo.common_tools import (JIT, TYPE_CHECKING)
from pynalgo.resample.barycentric import _interp_barycentric_1d_jit


###############################################################################
# Public kernels
###############################################################################

def lerp_1d(X_src, F_src, X_tgt):
  """Piecewise linear interpolation via binary search.

  Given source points (X_src, F_src), evaluates the piecewise linear
  interpolant at target points X_tgt.  X_src must be sorted ascending.

  Parameters
  ----------
  X_src : NDArray, shape (n,)
      Source grid coordinates (sorted).
  F_src : NDArray, shape (n,)
      Function values at source grid.
  X_tgt : NDArray, shape (m,)
      Target evaluation points.

  Returns
  -------
  F_tgt : NDArray, shape (m,)
      Interpolated values.
  """
  n_src = X_src.size
  m = X_tgt.size
  F_tgt = np.empty(m, dtype=F_src.dtype)

  for i in range(m):
    xt = X_tgt[i]
    # Clamp outside source domain
    if xt <= X_src[0]:
      F_tgt[i] = F_src[0]
      continue
    if xt >= X_src[-1]:
      F_tgt[i] = F_src[-1]
      continue
    # Binary search for bracket
    j_left = 0
    j_right = n_src - 1
    while j_right - j_left > 1:
      j_mid = (j_left + j_right) // 2
      if X_src[j_mid] <= xt:
        j_left = j_mid
      else:
        j_right = j_mid

    if X_src[j_left + 1] == X_src[j_left]:
      F_tgt[i] = F_src[j_left]
    else:
      t = (xt - X_src[j_left]) / (X_src[j_left + 1] - X_src[j_left])
      F_tgt[i] = F_src[j_left] + t * (F_src[j_left + 1] - F_src[j_left])

  return F_tgt


###############################################################################
# Filter kernels
###############################################################################

def uniform_filter_1d(data, size):
  """Boxcar moving-average filter along axis=0.

  Each output row is the equal-weighted average of data rows in a
  centred window of width ``size``.  Boundary rows use truncated windows.

  Parameters
  ----------
  data : NDArray, shape (n_rows, n_cols)
      2D tabular data.
  size : int
      Filter window width (should be odd for symmetric centering;
      even values produce an asymmetric window).

  Returns
  -------
  out : NDArray, shape (n_rows, n_cols)
      Filtered data.
  """
  n_rows = data.shape[0]
  n_cols = data.shape[1]
  out = np.empty_like(data)

  half = size // 2
  for c in range(n_cols):
    for r in range(n_rows):
      lo = max(0, r - half)
      hi = min(n_rows - 1, r + half)
      s = data[lo, c]
      for k in range(lo + 1, hi + 1):
        s += data[k, c]
      out[r, c] = s / (hi - lo + 1)

  return out


def gaussian_filter_1d(data, sigma, truncate=4.0):
  """Gaussian filter along axis=0.

  Convolves each column with a Gaussian kernel of width ``sigma``,
  truncated at ``truncate * sigma``.

  Parameters
  ----------
  data : NDArray, shape (n_rows, n_cols)
      2D tabular data.
  sigma : float
      Gaussian standard deviation.
  truncate : float
      Kernel truncation radius in units of sigma (default 4.0).

  Returns
  -------
  out : NDArray, shape (n_rows, n_cols)
      Filtered data.
  """
  n_rows = data.shape[0]
  n_cols = data.shape[1]
  out = np.empty_like(data)

  radius = int(truncate * sigma + 0.5)
  n_kern = 2 * radius + 1
  kern = np.empty(n_kern)
  s2 = 2.0 * sigma * sigma
  denom = 0.0
  for k in range(n_kern):
    dx = k - radius
    kern[k] = np.exp(-dx * dx / s2)
    denom += kern[k]

  for c in range(n_cols):
    for r in range(n_rows):
      s = 0.0
      w_sum = 0.0
      for k in range(n_kern):
        rr = r + k - radius
        if 0 <= rr < n_rows:
          s += kern[k] * data[rr, c]
          w_sum += kern[k]
      out[r, c] = s / w_sum

  return out


###############################################################################
# columns_interpolate
###############################################################################

def columns_interpolate(data, X, idx_independent, method):
  """Interpolate all columns of 2D data.

  Parameters
  ----------
  data : NDArray, shape (n_rows, n_cols)
  X : NDArray, shape (n_tgt,)
      Target grid coordinates.
  idx_independent : int
      Column index of the independent variable.
  method : int
      1 = linear, 2 = FH barycentric, 3 = BT barycentric.

  Returns
  -------
  out : NDArray, shape (n_tgt, n_cols)
  """
  n_cols = data.shape[1]
  n_tgt = X.size
  out = np.empty((n_tgt, n_cols), dtype=data.dtype)

  X_src = data[:, idx_independent]

  if method == 1:
    # linear
    for cix in range(n_cols):
      if cix == idx_independent:
        out[:, cix] = X
      else:
        out[:, cix] = lerp_1d(X_src, data[:, cix], X)
  elif method == 2:
    # FH
    for cix in range(n_cols):
      if cix == idx_independent:
        out[:, cix] = X
      else:
        out[:, cix] = _interp_barycentric_1d_jit(
            X_src, data[:, cix], X, 4, 'FH')
  elif method == 3:
    # BT
    for cix in range(n_cols):
      if cix == idx_independent:
        out[:, cix] = X
      else:
        out[:, cix] = _interp_barycentric_1d_jit(
            X_src, data[:, cix], X, 0, 'BT')
  else:
    raise ValueError("method must be 1 (linear), 2 (FH), or 3 (BT)")

  return out


###############################################################################
# columns_smooth
###############################################################################

def columns_smooth(data, idx_independent,
                   num_neighbors, delta, sigma):
  """Smooth columns of 2D data.

  Parameters
  ----------
  data : NDArray, shape (n_rows, n_cols)
  idx_independent : int
      Column index of the independent variable.
  num_neighbors : int
      Boxcar filter width.  Set 0 to disable.
  delta : float
      Filter scale for interpolate-filter-restrict pipeline.
      Filter window width = delta / min_grid_spacing.
      When 0.0, produces a single-sample filter (no effective smoothing
      for boxcar; for Gaussian, truncate=0 skips filtering).
  sigma : float
      Gaussian sigma.  Set 0.0 for boxcar, > 0 for Gaussian.

  Returns
  -------
  out : NDArray, shape (n_rows, n_cols)
  """
  if num_neighbors > 0:
    out = uniform_filter_1d(data, num_neighbors)
    out[:, idx_independent] = data[:, idx_independent]
    return out

  # Interpolate to uniform grid
  X_ = data[:, idx_independent].copy()
  # Manual diff: Numba np.diff fails on non-contiguous slices
  n_x = X_.size
  if n_x < 2:
    return data.copy()
  dX_min = X_[1] - X_[0]
  for i in range(2, n_x):
    d = X_[i] - X_[i - 1]
    if d < dX_min:
      dX_min = d
  min_dX_ = dX_min
  if min_dX_ == 0.0:
    return data.copy()
  N_I = int((X_[-1] - X_[0]) / min_dX_)
  X_I = X_[0] + min_dX_ * np.arange(0, N_I + 1)

  data_I = columns_interpolate(data, X_I, idx_independent, 1)

  if sigma > 0.0:
    sz_window = int(delta / min_dX_)
    data_F = gaussian_filter_1d(data_I, sigma, truncate=float(sz_window))
  else:
    sz_window = max(1, int(delta / min_dX_))
    data_F = uniform_filter_1d(data_I, sz_window)

  data_F[:, idx_independent] = X_I

  return columns_interpolate(data_F, X_, idx_independent, 1)


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  lerp_1d = JIT(lerp_1d)
  uniform_filter_1d = JIT(uniform_filter_1d)
  gaussian_filter_1d = JIT(gaussian_filter_1d)
  columns_interpolate = JIT(columns_interpolate)
  columns_smooth = JIT(columns_smooth)

  # Warm-up
  _wx = np.array([0.0, 1.0, 2.0], dtype=np.float64)
  _wf = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  _wt = np.array([0.5], dtype=np.float64)
  lerp_1d(_wx, _wf, _wt)
  _d2 = np.column_stack([_wx, _wf])
  columns_interpolate(_d2, _wx, 0, 1)

#
# :D
#
