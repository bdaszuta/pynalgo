"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ADI-like dimension-split rational interpolation.

For a 2D function F(x,y) on a tensor-product grid, iteratively
interpolates using 1D FH/BT: first along x for each y-slice,
then along y for each x-slice of the intermediate result.
"""
import numpy as np

from pynalgo.resample.barycentric import _interp_barycentric_1d_jit


def interp_barycentric_2d_split(X, Y, F, X_I, Y_I,
                                 d=4, method='FH'):
  """ADI-like dimension-split rational interpolation.

  Parameters
  ----------
  X, Y : NDArray, shape (nx,), (ny,)
      Source grid coordinates.
  F : NDArray, shape (nx, ny)
      Function values on tensor-product grid (indexing='ij').
  X_I, Y_I : NDArray, shape (mx,), (my,)
      Target grid coordinates.
  d : int
      FH blending parameter.
  method : str
      'FH' or 'BT'.

  Returns
  -------
  F_I : NDArray, shape (mx, my)
  """
  nx, ny = F.shape
  mx, my = X_I.size, Y_I.size

  # Pass 1: along X for each Y-slice
  F_x = np.empty((mx, ny), dtype=F.dtype)
  for j in range(ny):
    F_x[:, j] = _interp_barycentric_1d_jit(X, F[:, j], X_I, d, method)

  # Pass 2: along Y for each X-slice
  F_xy = np.empty((mx, my), dtype=F.dtype)
  for i in range(mx):
    F_xy[i, :] = _interp_barycentric_1d_jit(Y, F_x[i, :], Y_I, d, method)

  return F_xy

#
# :D
#
