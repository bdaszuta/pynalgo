"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Barycentric interpolation (Floater-Hormann / Berrut-Trefethen).

1D and ND (2D, 3D) tensor-product barycentric interpolation with
dynamically computed weights. Supports Floater-Hormann rational
interpolation (pole-free) and Berrut-Trefethen Lagrange interpolation.

References
----------
[1]: Berrut, J.P. and Trefethen, L.N., 2004.
     Barycentric Lagrange interpolation. SIAM review, 46(3), pp.501-517.
[2]: Floater, M.S. and Hormann, K., 2007.
     Barycentric rational interpolation with no poles and high rates of
     approximation. Numerische Mathematik, 107(2), pp.315-331.
"""
from numpy import (empty, zeros, zeros_like, float64, ndarray, )

from pynalgo.common_tools import (JIT, JITI, TYPE_CHECKING)

###############################################################################
# Internal helpers
###############################################################################

def _set_J_FH(alpha, d, n):
  r'''
  Index set for Floater-Hormann weight computation
  (parameter ``alpha`` plays the role of :math:`k` in the
  standard notation):

    :math:`J_k = \{i : \max(k-d,0) \le i \le \min(k, n-d)\}`
  '''
  return range(max(alpha - d, 0), min(alpha, n - d) + 1)

def _wei_bary_FH(k, d, n, X):
  r'''
  Floater-Hormann weight:

  .. math::

      w_k = \sum_{i \in J_k} (-1)^i \prod_{j=i}^{i+d} \frac{1}{x_k - x_j}

  where the :math:`j=k` term in the product is replaced by 1.
  '''
  wei = 0
  for i in _set_J_FH(k, d, n):
    wei_pr = (-1.) ** i
    for j in range(i, i + d + 1):
      cur_val = 1. / (X[k] - X[j]) if j != k else 1.
      wei_pr *= cur_val
    wei += wei_pr
  return wei

def _wei_bary_BT(k, n, X):
  r'''
  Berrut-Trefethen weight:

    :math:`w_k = 1 / \prod_{j \ne k} (x_k - x_j)`
  '''
  wei = 1.
  for j in range(n + 1):
    wei *= (X[k] - X[j]) if j != k else 1.
  return 1. / wei

def _compute_weights(X, n, d, method):
  '''
  Compute barycentric weights for a 1D grid.
  '''
  wei = empty((n + 1), dtype=float64)
  if method == 'FH':
    for k in range(0, n + 1):
      wei[k] = _wei_bary_FH(k, d, n, X)
  elif method == 'BT':
    for k in range(0, n + 1):
      wei[k] = _wei_bary_BT(k, n, X)
  else:
    raise ValueError("method must be 'FH' or 'BT'.")
  return wei

###############################################################################
# 1D interpolation
###############################################################################

def interp_barycentric_1d(X      : ndarray,
                          F      : ndarray,
                          X_I    : ndarray,
                          d      : int = 4,
                          method : str = 'FH') -> ndarray:
  """
  1D barycentric interpolation. Input grid values must be in ascending order.

  Parameters
  ----------
  X : ndarray
    Source grid coordinates.
  F : ndarray
    Function values at source grid.
  X_I : ndarray
    Target grid coordinates.
  d : int = 4
    FH blending parameter (d+1 polynomial degree combinations).
  method : str = 'FH'
    'FH' for Floater-Hormann rational interpolation,
    'BT' for Berrut-Trefethen Lagrange interpolation.

  Returns
  -------
  F_I : ndarray
    Interpolated values at X_I.
  """
  if method == 'FH' and d > X.size - 1:
    raise ValueError(
        "FH method requires d <= X.size - 1, got d={} for {} points".format(
            d, X.size))
  return _interp_barycentric_1d_jit(X, F, X_I, d, method)

def _interp_barycentric_1d_jit(X, F, X_I, d, method):
  n = X.size - 1
  wei = _compute_weights(X, n, d, method)
  F_I = zeros_like(X_I)

  for ii in range(0, X_I.size):
    num, den = 0., 0.
    for i in range(0, n + 1):
      diff = X_I[ii] - X[i]
      if diff != 0.:
        fac = wei[i] / diff
        num += fac * F[i]
        den += fac
      else:
        num = F[i]
        den = 1.
        break
    F_I[ii] = num / den
  return F_I

###############################################################################
# ND interpolation (2D, 3D)
###############################################################################

def interp_barycentric_nd(*args,
                          d      : int = 4,
                          method : str = 'FH',
                          ordinates : bool = False) -> ndarray:
  '''
  ND tensor-product barycentric interpolation (N = 2 or 3).

  Parameters
  ----------
  *args : ndarray
    Source grids, source data, then target grids -- in order.
    For 2D tensor-product: (X, Y, F, X_I, Y_I)
    For 3D tensor-product: (X, Y, Z, F, X_I, Y_I, Z_I)
    For ordinates mode: all target-grid arrays must have equal length.
    No runtime validation is performed -- mismatched sizes produce
    wrong results silently.
  d : int = 4
    FH blending parameter (d+1 polynomial degree combinations).
  method : str = 'FH'
    'FH' or 'BT'.
  ordinates : bool = False
    If True, target grids are matching-size ordinate arrays
    (diagonal interpolation), output length = |X_I|.

  Returns
  -------
  F_I : ndarray
    Interpolated values.
  '''
  N_target = len(args) // 2
  if N_target not in (2, 3):
    raise ValueError(
      f"Expected 2 or 3 source grids, got {len(args)} arguments total."
    )
  grids = args[:N_target]
  F = args[N_target]
  grids_I = args[N_target + 1:]

  ndim = len(grids)
  if ndim == 2:
    if ordinates:
      return _interp_barycentric_2d_ordinates_jit(
        grids[0], grids[1], F, grids_I[0], grids_I[1], d, method
      )
    else:
      return _interp_barycentric_2d_jit(
        grids[0], grids[1], F, grids_I[0], grids_I[1], d, method
      )
  else:  # ndim == 3
    if ordinates:
      return _interp_barycentric_3d_ordinates_jit(
        grids[0], grids[1], grids[2], F,
        grids_I[0], grids_I[1], grids_I[2], d, method
      )
    else:
      return _interp_barycentric_3d_jit(
        grids[0], grids[1], grids[2], F,
        grids_I[0], grids_I[1], grids_I[2], d, method
      )

###############################################################################
# 2D tensor-product
###############################################################################

def _interp_barycentric_2d_jit(X, Y, F, X_I, Y_I, d, method):
  n_x = X.size - 1
  n_y = Y.size - 1

  wei_x = _compute_weights(X, n_x, d, method)
  wei_y = _compute_weights(Y, n_y, d, method)

  F_I = zeros((X_I.size, Y_I.size), dtype=F.dtype)

  r_wei_x = zeros_like(wei_x, dtype=X.dtype)
  r_wei_y = zeros_like(wei_y, dtype=Y.dtype)

  for ii in range(0, X_I.size):
    # regularize x weights
    for i in range(0, n_x + 1):
      diff_x = X_I[ii] - X[i]
      if diff_x != 0.:
        r_wei_x[i] = wei_x[i] / diff_x
      else:
        for i_r in range(0, n_x + 1):
          r_wei_x[i_r] = 0.
        r_wei_x[i] = 1.
        break

    for J in range(0, Y_I.size):
      num, den = 0., 0.

      # regularize y weights
      for j in range(0, n_y + 1):
        diff_y = Y_I[J] - Y[j]
        if diff_y != 0.:
          r_wei_y[j] = wei_y[j] / diff_y
        else:
          for j_r in range(0, n_y + 1):
            r_wei_y[j_r] = 0.
          r_wei_y[j] = 1.
          break

      for i in range(0, n_x + 1):
        for j in range(0, n_y + 1):
          fac = r_wei_x[i] * r_wei_y[j]
          num += fac * F[i, j]
          den += fac

      F_I[ii, J] = num / den

  return F_I

###############################################################################
# 2D diagonal (ordinates)
###############################################################################

def _interp_barycentric_2d_ordinates_jit(X, Y, F, X_I, Y_I, d, method):
  n_x = X.size - 1
  n_y = Y.size - 1

  wei_x = _compute_weights(X, n_x, d, method)
  wei_y = _compute_weights(Y, n_y, d, method)

  F_I = zeros((X_I.size,), dtype=F.dtype)

  r_wei_x = zeros_like(wei_x, dtype=X.dtype)
  r_wei_y = zeros_like(wei_y, dtype=Y.dtype)

  for ii in range(0, X_I.size):
    num, den = 0., 0.

    # regularize x weights
    for i in range(0, n_x + 1):
      diff_x = X_I[ii] - X[i]
      if diff_x != 0.:
        r_wei_x[i] = wei_x[i] / diff_x
      else:
        for i_r in range(0, n_x + 1):
          r_wei_x[i_r] = 0.
        r_wei_x[i] = 1.
        break

    # regularize y weights
    for j in range(0, n_y + 1):
      diff_y = Y_I[ii] - Y[j]
      if diff_y != 0.:
        r_wei_y[j] = wei_y[j] / diff_y
      else:
        for j_r in range(0, n_y + 1):
          r_wei_y[j_r] = 0.
        r_wei_y[j] = 1.
        break

    for i in range(0, n_x + 1):
      for j in range(0, n_y + 1):
        fac = r_wei_x[i] * r_wei_y[j]
        num += fac * F[i, j]
        den += fac

    F_I[ii] = num / den

  return F_I

###############################################################################
# 3D tensor-product
###############################################################################

def _interp_barycentric_3d_jit(X, Y, Z, F, X_I, Y_I, Z_I, d, method):
  n_x = X.size - 1
  n_y = Y.size - 1
  n_z = Z.size - 1

  wei_x = _compute_weights(X, n_x, d, method)
  wei_y = _compute_weights(Y, n_y, d, method)
  wei_z = _compute_weights(Z, n_z, d, method)

  F_I = zeros((X_I.size, Y_I.size, Z_I.size), dtype=F.dtype)

  r_wei_x = zeros_like(wei_x, dtype=X.dtype)
  r_wei_y = zeros_like(wei_y, dtype=Y.dtype)
  r_wei_z = zeros_like(wei_z, dtype=Z.dtype)

  for ii in range(0, X_I.size):
    # regularize x
    for i in range(0, n_x + 1):
      diff_x = X_I[ii] - X[i]
      if diff_x != 0.:
        r_wei_x[i] = wei_x[i] / diff_x
      else:
        for i_r in range(0, n_x + 1):
          r_wei_x[i_r] = 0.
        r_wei_x[i] = 1.
        break

    for J in range(0, Y_I.size):
      # regularize y
      for j in range(0, n_y + 1):
        diff_y = Y_I[J] - Y[j]
        if diff_y != 0.:
          r_wei_y[j] = wei_y[j] / diff_y
        else:
          for j_r in range(0, n_y + 1):
            r_wei_y[j_r] = 0.
          r_wei_y[j] = 1.
          break

      for K in range(0, Z_I.size):
        num, den = 0., 0.
        # regularize z
        for k in range(0, n_z + 1):
          diff_z = Z_I[K] - Z[k]
          if diff_z != 0.:
            r_wei_z[k] = wei_z[k] / diff_z
          else:
            for k_r in range(0, n_z + 1):
              r_wei_z[k_r] = 0.
            r_wei_z[k] = 1.
            break

        for i in range(0, n_x + 1):
          for j in range(0, n_y + 1):
            for k in range(0, n_z + 1):
              fac = r_wei_x[i] * r_wei_y[j] * r_wei_z[k]
              num += fac * F[i, j, k]
              den += fac

        F_I[ii, J, K] = num / den

  return F_I

###############################################################################
# 3D diagonal (ordinates)
###############################################################################

def _interp_barycentric_3d_ordinates_jit(X, Y, Z, F, X_I, Y_I, Z_I, d, method):
  n_x = X.size - 1
  n_y = Y.size - 1
  n_z = Z.size - 1

  wei_x = _compute_weights(X, n_x, d, method)
  wei_y = _compute_weights(Y, n_y, d, method)
  wei_z = _compute_weights(Z, n_z, d, method)

  F_I = zeros((X_I.size,), dtype=F.dtype)

  r_wei_x = zeros_like(wei_x, dtype=X.dtype)
  r_wei_y = zeros_like(wei_y, dtype=Y.dtype)
  r_wei_z = zeros_like(wei_z, dtype=Z.dtype)

  for ii in range(0, X_I.size):
    # regularize x
    for i in range(0, n_x + 1):
      diff_x = X_I[ii] - X[i]
      if diff_x != 0.:
        r_wei_x[i] = wei_x[i] / diff_x
      else:
        for i_r in range(0, n_x + 1):
          r_wei_x[i_r] = 0.
        r_wei_x[i] = 1.
        break

    # regularize y
    for j in range(0, n_y + 1):
      diff_y = Y_I[ii] - Y[j]
      if diff_y != 0.:
        r_wei_y[j] = wei_y[j] / diff_y
      else:
        for j_r in range(0, n_y + 1):
          r_wei_y[j_r] = 0.
        r_wei_y[j] = 1.
        break

    num, den = 0., 0.
    # regularize z
    for k in range(0, n_z + 1):
      diff_z = Z_I[ii] - Z[k]
      if diff_z != 0.:
        r_wei_z[k] = wei_z[k] / diff_z
      else:
        for k_r in range(0, n_z + 1):
          r_wei_z[k_r] = 0.
        r_wei_z[k] = 1.
        break

    for i in range(0, n_x + 1):
      for j in range(0, n_y + 1):
        for k in range(0, n_z + 1):
          fac = r_wei_x[i] * r_wei_y[j] * r_wei_z[k]
          num += fac * F[i, j, k]
          den += fac

    F_I[ii] = num / den

  return F_I

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _set_J_FH = JITI(_set_J_FH)
  _wei_bary_FH = JITI(_wei_bary_FH)
  _wei_bary_BT = JITI(_wei_bary_BT)
  _interp_barycentric_1d_jit = JIT(_interp_barycentric_1d_jit)
  _interp_barycentric_2d_jit = JIT(_interp_barycentric_2d_jit)
  _interp_barycentric_2d_ordinates_jit = \
    JIT(_interp_barycentric_2d_ordinates_jit)
  _interp_barycentric_3d_jit = JIT(_interp_barycentric_3d_jit)
  _interp_barycentric_3d_ordinates_jit = \
    JIT(_interp_barycentric_3d_ordinates_jit)
  _compute_weights = JIT(_compute_weights)

#
# :D
#
