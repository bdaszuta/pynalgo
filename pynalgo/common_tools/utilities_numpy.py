"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Array utilities: banded matrix/dot ops, extrema search,
           mirroring, padding, roll, identity mask, deduplicate/sort,
           interval intersection, dimension-order exchange,
           common type, window functions.
"""
from numpy import (abs, asarray,
                   array, concatenate, diag, dtype, dot, empty, exp, eye,
                   hstack, ones, piecewise, roll,
                   swapaxes, unique, zeros, zeros_like)

from pynalgo.common_tools.utilities_primitives import (is_odd, )
from pynalgo.common_tools.utilities_numba import (JITI, )
from pynalgo.common_tools.utilities_typing import (NDArray,
                                                   Any,
                                                   Optional,
                                                   int64,
                                                   TYPE_CHECKING)

###############################################################################
# array manipulations
###############################################################################

def array_common_type(A : NDArray[Any], B : NDArray[Any]) -> dtype:
  '''
  Deduce common dtype from input arrays.

  Parameters
  ----------
  A : NDArray[Any]
    One array to deduce from.
  B : NDArray[Any]
    Other array to deduce from.

  Returns
  -------
  common_dtype : dtype
    Reduced, common dtype.

  Usage
  -----
  >>> from numpy import (array, float64, int64)
  >>> A = array([1, 2, 3], dtype=int64)
  >>> B = array([2.2, 0],  dtype=float64)
  >>> A.dtype
      dtype('int64')
  >>> B.dtype
      dtype('float64')
  >>> array_common_type(A, B)
      dtype('float64')
  '''
  darr = array(A.flat[0] + B.flat[0])
  common_dtype = darr.dtype
  return common_dtype


def array_extend_mirror(arr : NDArray[Any],
                        axis : int=0) -> NDArray[Any]:
  """
  Extend array by mirroring along an axis.

  Parameters
  ----------
  arr : NDArray[Any]
    Array to mirror.

  axis : int = 0
    Axis to mirror over.

  Returns
  -------
  mirrored : NDArray[Any]
    Mirror array.

  Usage
  -----
  >>> from numpy import array
  >>> arr = array([[1, 2, 3], [3, 2, 1], [4, 5, 6]])
  >>> array_extend_mirror(arr, axis=0)
      array([[1, 2, 3],
             [3, 2, 1],
             [4, 5, 6],
             [3, 2, 1],
             [1, 2, 3]])

  >>> array_extend_mirror(arr, axis=1)
      array([[1, 2, 3, 2, 1],
             [3, 2, 1, 2, 3],
             [4, 5, 6, 5, 4]])
  """
  if axis > 0:
    arr = swapaxes(arr, axis, 0)
    arr = concatenate((arr, arr[1::-1]))
    return swapaxes(arr, axis, 0)

  mirrored = concatenate((arr, arr[1::-1]))

  return mirrored

def array_extend_uniform_range(
  ran : NDArray[Any],
  NGHOST : int,
  extend_left : bool=True,
  extend_right : bool=True
) -> NDArray[Any]:
  """
  Given a uniform range extend to ghost zones.
  Useful for e.g. construction of finite difference schemes.

  Parameters
  ----------
  ran : NDArray[Any]
    Uniform range to extend.
  NGHOST : int
    Number of nodes to extend by.
  extend_left : bool=True
    Control whether extension is performed on left.
  extend_right : bool=True
    Control whether extension is performed on right.

  Returns
  -------
  ran_ext : NDArray[Any]
    Extended range.

  Usage
  -----
  >>> from numpy import linspace
  >>> NGHOST = 2
  >>> s = linspace(-2, 1, num=3)
  >>> s
      array([-2. , -0.5,  1. ])
  >>> array_extend_uniform_range(s, NGHOST)
      array([-5. , -3.5, -2. , -0.5,  1. ,  2.5,  4. ])
  >>> array_extend_uniform_range(s, NGHOST, extend_left=False)
      array([-2. , -0.5,  1. ,  2.5,  4. ])
  >>> array_extend_uniform_range(s, NGHOST, extend_right=False)
      array([-5. , -3.5, -2. , -0.5,  1. ])
  """
  ds = ran[1] - ran[0]
  ran_ext = empty(ran.size +
                  extend_left * NGHOST +
                  extend_right * NGHOST,
                  dtype=ran.dtype)

  if extend_left:
    for i in range(NGHOST):
      ran_ext[i] = ran[0] - ds * (NGHOST - i)
    si = NGHOST
  else:
    si = 0

  if extend_right:
    for i in range(NGHOST):
      ran_ext[si+ran.size+i] = ran[ran.size-1] + ds * (i+1)

  se = si + ran.size
  ran_ext[si:se] = ran[:]

  return ran_ext

def array_dense_to_banded(arr : NDArray[Any],
                          width : int = 1) -> NDArray[Any]:
  """
  Extract / convert dense matrix to banded.

  Parameters
  ----------
  arr : NDArray[Any]
    Array to extract from.
  width : int = 1
    Width of band to extract (should be an odd number).

  Returns
  -------
  bands : NDArray[Any]
    Extracted bands of array.

  Usage
  -----
  >>> from numpy import arange
  >>> snake = arange(1, 26).reshape((5, 5))
  >>> snake
      array([[ 1,  2,  3,  4,  5],
             [ 6,  7,  8,  9, 10],
             [11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20],
             [21, 22, 23, 24, 25]])
  >>> array_dense_to_banded(snake, width=3)
      array([[ 0,  1,  2],
             [ 6,  7,  8],
             [12, 13, 14],
             [18, 19, 20],
             [24, 25,  0]])
  """
  n_rows, n_cols = arr.shape

  N = max([n_rows, n_cols])
  bands = zeros((N, width), dtype=arr.dtype)

  s = width // 2

  for ix in range(n_rows):
    for jx in range(max([ix-s, 0]), min([ix+s+1, n_cols])):
      bands[ix, jx-ix+s] = arr[ix, jx]

  return bands

def array_dense_from_band_uniform(
  banded : NDArray[Any], N : int
) -> NDArray[Any]:
  """
  Construct a dense array from a row vector.

  Parameters
  ----------
  banded : NDArray[Any]
    1-D array of band coefficients (odd length) used to construct
    the dense matrix.
  N : int
    Resultant (N x N) shape.

  Returns
  -------
  dense : NDArray[Any]
    Dense array constructed from bands.

  Usage
  -----
  >>> from numpy import array
  >>> banded = array([1, 2, 3])
  >>> array_dense_from_band_uniform(banded, 4)
      array([[2., 3., 0., 0.],
             [1., 2., 3., 0.],
             [0., 1., 2., 3.],
             [0., 0., 1., 2.]])
  """


  ban_ones_N = ones(N)

  h_sz = banded.size // 2

  # populate diagonal
  dense = diag(banded[h_sz] * ban_ones_N)

  # off-diagonals
  for ix in range(h_sz):
    # sub
    dense += diag(banded[ix] * ban_ones_N[:-(h_sz - ix)], -(h_sz - ix))
    # sup
    dense += diag(banded[h_sz+ix+1] * ban_ones_N[:-(ix + 1)], ix + 1)
  return dense

def array_dense_from_bands(
  bands : NDArray[Any], N : int, idx_center_fill : int=0
) -> NDArray[Any]:
  """
  Construct a dense array from stacked bands and specify fill.

  Parameters
  ----------
  bands : NDArray[Any]
    Array of stacked bands to use for construction of dense array.
  N : int
    Resultant (N x N) shape.
  idx_center_fill : int = 0
    Band to use for filling the center rows.

  Returns
  -------
  dense : NDArray[Any]
    Dense array constructed from bands.

  Usage
  -----
  >>> from numpy import array
  >>> bands = array([[0, 2, 3],
  ...                [1, 4, 1],
  ...                [3, 2, 3],
  ...                [3, 2, 0]])
  >>> N = 6
  >>> array_dense_from_bands(bands, N, idx_center_fill=1)
      array([[2, 3, 0, 0, 0, 0],
             [1, 4, 1, 0, 0, 0],
             [0, 1, 4, 1, 0, 0],
             [0, 0, 1, 4, 1, 0],
             [0, 0, 0, 3, 2, 3],
             [0, 0, 0, 0, 3, 2]])
  """

  band_width = bands.shape[1]
  band_hzw = band_width // 2

  dense = eye(N, dtype=bands.dtype)
  Z = zeros(N + 2 * band_hzw, dtype=bands.dtype)

  iis = idx_center_fill
  iie = N - (bands.shape[0] - idx_center_fill) + 1

  # bands prior to fill
  for ix in range(0, iis):
    Z[ix:(ix + band_width)] = bands[ix, :]
    dense[ix, :] = Z[band_hzw:-band_hzw]
    Z[:] = 0

  # band for center fill
  bx = idx_center_fill
  for ix in range(iis, iie):
    Z[ix:(ix + band_width)] = bands[bx, :]
    dense[ix, :] = Z[band_hzw:-band_hzw]
    Z[:] = 0

  # bands after center fill
  bx = 0
  for ix in range(iie, N):
    Z[(iie + bx):(iie + bx + band_width)] = bands[iis+bx+1, :]
    dense[ix, :] = Z[band_hzw:-band_hzw]
    Z[:] = 0
    bx += 1

  return dense

def array_identity_mask(ix_a : int,
                        ix_b : int,
                        N : int,
                        is_complement : bool=False) -> NDArray[int64]:
  """
  Construct identity matrix mask over an indicial range.

  Parameters
  ----------
  ix_a : int
    Start index.
  ix_b : int
    End index (excluded).
  N : int
    Control final shape (NxN)
  is_complement : bool=False
    Control whether unit is set on interior or exterior.

  Returns
  -------
  mask : NDArray[int64]
    Mask constructed over range.

  Usage
  -----
  >>> array_identity_mask(1, 5, 6)
      array([[0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]])
  >>> array_identity_mask(1, 5, 6, is_complement=True)
      array([[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]])
  """
  if ix_a < 0 or ix_b > N:
    raise ValueError("Index range out of bounds for mask size")

  mask = zeros((N, N), dtype=int64)

  if is_complement:
    for i in range(0, ix_a):
      mask[i, i] = 1

    for i in range(ix_b, N):
      mask[i, i] = 1

  else:
    for i in range(ix_a, ix_b):
      mask[i, i] = 1

  return mask

def array_pad_biased(vec : NDArray[Any], c_node : int) -> NDArray[Any]:
  """
  Zero-pad a 'biased' vector so that it is centered.

  Parameters
  ----------
  vec : NDArray[Any]
    Array to pad.
  c_node : int
    Node taken as central (left edge corresponds to 0).

  Returns
  -------
  p_vec : NDArray[Any]
    Padded array.

  Usage
  -----
  >>> from numpy import array
  >>> vec = array([1, 2, 3])
  >>> c_node = 0
  >>> array_pad_biased(vec, c_node)
      array([0, 0, 1, 2, 3])
  >>> c_node = 1
  >>> array_pad_biased(vec, c_node)
      array([1, 2, 3])
  >>> c_node = 2
  >>> array_pad_biased(vec, c_node)
      array([1, 2, 3, 0, 0])
  """
  sz_v = vec.size
  sz_l, sz_r = c_node, sz_v - c_node - 1
  w = max(abs(array([sz_l, sz_r])))

  sz = 2 * (w) + 1
  r_vec = zeros(sz, dtype=vec.dtype)
  r_vec[w-c_node:w+sz_v-c_node] = vec[:]
  return r_vec

def array_roll_row_to_edges(A : NDArray[Any],
                            row : int,
                            num_skipped : int=0,
                            roll_top : bool=True,
                            roll_bottom : bool=True) -> None:
  """
  Roll (inplace) a row towards top and bottom of a two dimensional array.
  Amount of rolling increases by distance from selected row.

  Useful (e.g.) for imposition of periodic boundary conditions in a banded
  array representing a finite difference stencil.

  Parameters
  ----------
  A : NDArray[Any]
    Array to roll (inplace).

  row : int
    Row to fix for rolling.

  num_skipped : int = 0
    Number of elements skipped during roll.

  roll_top : bool = True
    Control whether rolling is done towards the top.

  roll_bottom : bool = True
    Control whether rolling is done towards the bottom.

  Returns
  -------
  None
    Inplace operation.

  Usage
  -----
  >>> from numpy import array
  >>> A = array([[ 1,  2,  3,  0,  0,  0,  0],
  ...            [-2,  1,  2,  3,  0,  0,  0],
  ...            [-3, -2,  1,  2,  3,  0,  0],
  ...            [ 0, -3, -2,  1,  2,  3,  0],
  ...            [ 0,  0, -3, -2,  1,  2,  3],
  ...            [ 0,  0,  0, -3, -2,  1,  2],
  ...            [ 0,  0,  0,  0, -3, -2,  1]])
  >>> array_roll_row_to_edges(A, 2, num_skipped=1)
  >>> A
      array([[ 1,  2,  3,  0, -3, -2,  0],
             [-2,  1,  2,  3,  0, -3,  0],
             [-3, -2,  1,  2,  3,  0,  0],
             [ 0, -3, -2,  1,  2,  3,  0],
             [ 0,  0, -3, -2,  1,  2,  3],
             [ 0,  3,  0, -3, -2,  1,  2],
             [ 0,  2,  3,  0, -3, -2,  1]])

  Here the rolling structure is:
  array([[ ... ],  # roll: -2
         [ ... ],  # <- roll: -1
         [ ... ],  # roll = 0
         [ ... ],  # interior not rolled
         [ ... ],  # roll = 0
         [ ... ],  # -> roll: +1
         [ ... ]]) # -> roll: +2
  """
  if num_skipped == 0:
    # to roll (exclude one node for repeated vertex)
    tr_A_L = A[row, :]
    tr_A_R = A[-row-1, :]

    for ix in range(row):
      if roll_top:
        A[ix, :] = roll(tr_A_L, -(row-ix))
      if roll_bottom:
        A[-(ix+1), :] = roll(tr_A_R, row-ix)
  else:
    # to roll (exclude one node for repeated vertex)
    tr_A_L = A[row, :-num_skipped]
    tr_A_R = A[-row-1, num_skipped:]

    for ix in range(row):
      if roll_top:
        r_A_L = roll(tr_A_L, -(row-ix))
        A[ix, :] = hstack((r_A_L, zeros(num_skipped)))

      if roll_bottom:
        r_A_R = roll(tr_A_R, row-ix)
        A[-(ix+1), :] = hstack((zeros(num_skipped), r_A_R))

def array_dot_band_uniform(
  band : NDArray[Any],
  arr : NDArray[Any],
  axis : int=0,
  output : Optional[NDArray[Any]]=None
) -> NDArray[Any]:
  """
  Dot product between a uniform-band matrix and another array.

  Bands must be of _odd_ size.

  Parameters
  ----------
  band : NDArray[Any]
    Band to dot, shape should be (band_width, ).

  arr : NDArray[Any]
    Array to dot.

  axis : int=0
    Axis over which to take dot.

  output : Optional[NDArray[Any]]=None
    Preallocated space for result.

  Returns
  -------
  output : NDArray[Any]
    Result of dot.

  Usage
  -----
  >>> from numpy import (arange, array, ones, dot)
  >>> N = 8
  >>> A_band = array([2, 4.1, 2, 3, 8.1])
  >>> x = arange(2, 2 + N).reshape((N, 1))
  >>> array_dot_band_uniform(A_band, x)
      array([[ 45.4],
             [ 66.7],
             [ 87.9],
             [107.1],
             [126.3],
             [145.5],
             [ 83.7],
             [ 64.8]])
  >>> M = 6
  >>> y = arange(1, 1 + M).reshape((1, M))
  >>> xy = x + y
  >>> array_dot_band_uniform(A_band, xy, axis=1)
      array([[ 58.5,  83.9, 107.1, 126.3,  72.6,  56.7],
             [ 71.6, 101.1, 126.3, 145.5,  83.7,  64.8],
             [ 84.7, 118.3, 145.5, 164.7,  94.8,  72.9],
             [ 97.8, 135.5, 164.7, 183.9, 105.9,  81. ],
             [110.9, 152.7, 183.9, 203.1, 117. ,  89.1],
             [124. , 169.9, 203.1, 222.3, 128.1,  97.2],
             [137.1, 187.1, 222.3, 241.5, 139.2, 105.3],
             [150.2, 204.3, 241.5, 260.7, 150.3, 113.4]])
  >>> # compare numpy
  >>> A_bands = array_dense_from_band_uniform(A_band, M)
  >>> dot(A_bands, xy.T).T
      array([[ 58.5,  83.9, 107.1, 126.3,  72.6,  56.7],
             [ 71.6, 101.1, 126.3, 145.5,  83.7,  64.8],
             [ 84.7, 118.3, 145.5, 164.7,  94.8,  72.9],
             [ 97.8, 135.5, 164.7, 183.9, 105.9,  81. ],
             [110.9, 152.7, 183.9, 203.1, 117. ,  89.1],
             [124. , 169.9, 203.1, 222.3, 128.1,  97.2],
             [137.1, 187.1, 222.3, 241.5, 139.2, 105.3],
             [150.2, 204.3, 241.5, 260.7, 150.3, 113.4]])

  See also
  --------
  array_dot_bands
  array_pad_biased
  """


  if axis != 0:
    arr = swapaxes(arr, 0, axis)

  sz_band, hsz_band = band.size, band.size // 2
  sz_arr = arr.shape[0]

  if output is None:
    output = zeros_like(arr, dtype=array_common_type(band, arr))

  if is_odd(sz_band):
    for ix in range(0, hsz_band):
      for bx in range(hsz_band-ix, sz_band):
        # low-edge
        ail = bx - (hsz_band - ix)
        output[ix, ...] += band[bx] * arr[ail, ...]

    for ix in range(sz_arr - hsz_band, sz_arr):
      ix0 = ix - (sz_arr - hsz_band)

      for bx in range(0, sz_band-ix0-1):
        # high edge
        aiu = sz_arr - sz_band + 1 + bx + ix0
        output[ix, ...] += band[bx] * arr[aiu]

    for ix in range(hsz_band, sz_arr - hsz_band):  # interior
      for bx in range(0, hsz_band):
        # band and array edge indices resp.
        bis, bie = bx, sz_band-bx-1
        ais, aie = ix-hsz_band+bx, ix-hsz_band+sz_band-bx-1

        # symmetrized op
        output[ix, ...] += (band[bis] * arr[ais, ...] +
                            band[bie] * arr[aie, ...])

      output[ix, ...] += band[hsz_band] * arr[ix]

  else:
    raise NotImplementedError("Individual band size must be _odd_.")

  if axis != 0:
    return swapaxes(asarray(output), 0, axis)

  return output

def array_dot_bands(
  bands : NDArray[Any],
  arr : NDArray[Any],
  axis : int=0,
  output : Optional[NDArray[Any]]=None
) -> NDArray[Any]:
  """
  Dot product between a banded matrix and another array.

  Bands must be of _odd_ size.

  Parameters
  ----------
  bands : NDArray[Any]
    Band matrix to dot with, shape should be (N, band_width)
    where N is arr.shape[axis].

  arr : NDArray[Any]
    Array to dot.

  axis : int=0
    Axis over which to take dot.

  output : Optional[NDArray[Any]]=None
    Preallocated space for result.

  Returns
  -------
  output : NDArray[Any]
    Result of dot.

  Usage
  -----
  >>> from numpy import (arange, array, ones, dot)
  >>> N = 8
  >>> A_band = array([2, 4.1, 2, 3, 8.1])
  >>> A_bands = A_band[None,:] * ones((N, 1))
  >>> x = arange(2, 2 + N).reshape((N, 1))
  >>> array_dot_bands(A_bands, x)
      array([[ 45.4],
             [ 66.7],
             [ 87.9],
             [107.1],
             [126.3],
             [145.5],
             [ 83.7],
             [ 64.8]])
  >>> M = 6
  >>> y = arange(1, 1 + M).reshape((1, M))
  >>> xy = x + y
  >>> A_bands = A_band[None,:] * ones((M, 1))
  >>> array_dot_bands(A_bands, xy, axis=1)
      array([[ 58.5,  83.9, 107.1, 126.3,  72.6,  56.7],
             [ 71.6, 101.1, 126.3, 145.5,  83.7,  64.8],
             [ 84.7, 118.3, 145.5, 164.7,  94.8,  72.9],
             [ 97.8, 135.5, 164.7, 183.9, 105.9,  81. ],
             [110.9, 152.7, 183.9, 203.1, 117. ,  89.1],
             [124. , 169.9, 203.1, 222.3, 128.1,  97.2],
             [137.1, 187.1, 222.3, 241.5, 139.2, 105.3],
             [150.2, 204.3, 241.5, 260.7, 150.3, 113.4]])

  See also
  --------
  array_dot_band_uniform
  array_pad_biased
  """
  if bands.shape[0] != arr.shape[axis]:
    raise ValueError("Number of provided bands not sufficient.")


  if axis != 0:
    arr = swapaxes(arr, 0, axis)

  sz_band, hsz_band = bands.shape[-1], bands.shape[-1] // 2
  sz_arr = arr.shape[0]

  if output is None:
    output = zeros_like(arr, dtype=array_common_type(bands, arr))

  if is_odd(sz_band):
    for ix in range(0, hsz_band):
      for bx in range(hsz_band-ix, sz_band):
        # low-edge
        ail = bx - (hsz_band - ix)
        output[ix, ...] += bands[ix, bx] * arr[ail, ...]

    for ix in range(sz_arr - hsz_band, sz_arr):
      ix0 = ix - (sz_arr - hsz_band)

      for bx in range(0, sz_band-ix0-1):
        # high edge
        aiu = sz_arr - sz_band + 1 + bx + ix0
        output[ix, ...] += bands[ix, bx] * arr[aiu]

    for ix in range(hsz_band, sz_arr - hsz_band):  # interior
      for bx in range(0, hsz_band):
        # band and array edge indices resp.
        bis, bie = bx, sz_band-bx-1
        ais, aie = ix-hsz_band+bx, ix-hsz_band+sz_band-bx-1

        # symmetrized op
        output[ix, ...] += (bands[ix, bis] * arr[ais, ...] +
                            bands[ix, bie] * arr[aie, ...])

      output[ix, ...] += bands[ix, hsz_band] * arr[ix]

  else:
    raise NotImplementedError("Individual band size must be _odd_.")

  if axis != 0:
    return swapaxes(asarray(output), 0, axis)

  return output

def array_dot_2d_at_axis(mat : NDArray[Any],
                         fcn : NDArray[Any],
                         axis : int=0) -> NDArray[Any]:
  """
  Convenience function for dot product of a 2d array with another array over a
  given axis preserving order.

  Parameters
  ----------
    mat : NDArray[Any]
      Matrix-like array to dot with.

    fcn : NDArray[Any]
      Array to dot with.

    axis : int=0
      Axis to apply dot on.

  Returns
  -------
  result : NDArray[Any]
    Result of calculation.

  Usage
  -----
  >>> from numpy import (dot, empty)
  >>> mat = empty((4, 5))
  >>> fcn = empty((7, 5))
  >>> array_dot_2d_at_axis(mat, fcn, axis=1).shape
      (7, 4)
  """
  ndim = fcn.ndim

  if ndim == 1:
    return dot(mat, fcn)

  if ndim == 2:
    if axis == 0:
      return dot(mat, fcn)

    if axis == 1:
      return dot(mat, fcn.T).T

  raise NotImplementedError("fcn.ndim > 2 not implemented")

###############################################################################
# search / sort / deduplicate
###############################################################################

def ndarray_get_sorted_argmin(val : float, arr : NDArray[Any]) -> int:
  '''
  Given a value and a sorted input array, return index ix such that
  abs(arr[ix] - val) is minimized. Early exit when distance does not
  strictly decrease.

  Parameters
  ----------
  val : float
    Value to minimize against.
  arr : NDArray[Any]
    Array to search (must be sorted).

  Returns
  -------
  ix : int
    Found index.

  Usage
  -----
  >>> from numpy import array
  >>> arr = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
  >>> ndarray_get_sorted_argmin(2.3, arr)
      2
  '''
  ix = 0
  cval = abs(arr[ix] - val)
  for ix in range(1, arr.size):
    ccval = abs(arr[ix] - val)
    if ccval < cval:
      cval = ccval
    else:
      return ix - 1
  return ix

def interval_is_intersection_empty(I_a : int, I_b : int,
                                   J_a : int, J_b : int) -> bool:
  '''
  Return True if [I_a, I_b] n [J_a, J_b] is empty.

  Parameters
  ----------
  I_a : int
    Start of first interval.
  I_b : int
    End of first interval.
  J_a : int
    Start of second interval.
  J_b : int
    End of second interval.

  Returns
  -------
  bool
    True if intersection is empty, False otherwise.

  Usage
  -----
  >>> interval_is_intersection_empty(0, 5, 6, 10)
      True
  >>> interval_is_intersection_empty(0, 5, 3, 10)
      False
  '''
  return ((J_a > I_b) or (I_a > J_b))

def scalar_data_deduplicate_sort(data : NDArray[Any],
                                 column         : int = 0,
                                 return_indices : bool = False,
                                 use_reversed   : bool = False):
  '''
  Sort and deduplicate N x M data along axis=0 using np.unique on
  a selected column.

  Parameters
  ----------
  data : NDArray[Any]
    Data to deduplicate / sort.
  column : int = 0
    Column to use for deduplication and sorting.
  return_indices : bool = False
    Return unique/sorting indices alongside data.
  use_reversed : bool = False
    If True, reverse before unique (later entries selected).

  Returns
  -------
  data : NDArray[Any]
      Sorted, deduplicated data.
  indices : NDArray[Any], optional
      Sorting/deduplication indices, returned only when ``return_indices``
      is True. In that case, the return value is ``(data, indices)``.

  Usage
  -----
  >>> from numpy import array
  >>> data = array([[0., 2.], [1., 1.], [1., 3.], [3., 0.]])
  >>> scalar_data_deduplicate_sort(data, column=0)
      array([[0., 2.],
             [1., 1.],
             [3., 0.]])
  '''
  if data.ndim == 1:
    data = data[:, None]
  if use_reversed:
    data = data[::-1, ...]
  to_parse = data[:, column]
  _, ix_parsed = unique(to_parse, return_index=True)
  if return_indices:
    return (data[ix_parsed, :], ix_parsed)
  return data[ix_parsed, :]

###############################################################################
# extrema within geometric constraints
###############################################################################

def arg_extremum_interval(x_I : NDArray[Any], F_I : NDArray[Any],
                          x_0 : float, r_0 : float,
                          extrema_type : int = +1) -> int:
  '''
  Argmax/argmin within interval :math:`|x - x_0| < r_0`.

  Parameters
  ----------
  x_I : NDArray[Any]
    x-ordinates for sampled F_I.
  F_I : NDArray[Any]
    Sampled function.
  x_0 : float
    Centre of bracketing interval.
  r_0 : float
    Radius of bracketing interval.
  extrema_type : int = +1
    +1 for maximum, -1 for minimum.

  Returns
  -------
  ix : int
    Index of extremum.

  Usage
  -----
  >>> from numpy import array
  >>> x_I = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
  >>> F_I = array([0.1, 0.5, 0.9, 0.3, 0.2, 0.0])
  >>> arg_extremum_interval(x_I, F_I, 2.0, 2.0)
      2
  '''
  old_max = extrema_type * F_I[0]
  ix_max = 0
  for ix in range(x_I.size):
    if (x_I[ix] - x_0) ** 2 < r_0 ** 2:
      if (extrema_type * F_I[ix]) > old_max:
        old_max = extrema_type * F_I[ix]
        ix_max = ix
  return ix_max

def arg_extremum_disc(x_I : NDArray[Any], y_I : NDArray[Any],
                     F_I : NDArray[Any],
                     x_0 : float, y_0 : float, r_0 : float,
                     extrema_type : int = +1):
  '''
  Argmax/argmin within disc :math:`(x-x_0)^2 + (y-y_0)^2 < r_0^2`.

  Parameters
  ----------
  x_I : NDArray[Any]
    x-ordinates.
  y_I : NDArray[Any]
    y-ordinates.
  F_I : NDArray[Any]
    Sampled function, shape (x_I.size, y_I.size).
  x_0 : float
    x-centre of bracketing disc.
  y_0 : float
    y-centre of bracketing disc.
  r_0 : float
    Radius of bracketing disc.
  extrema_type : int = +1
    +1 for maximum, -1 for minimum.

  Returns
  -------
  (ix, jx) : Tuple[int, int]
    Index of extremum.

  Usage
  -----
  >>> from numpy import array, zeros
  >>> x_I = array([0.0, 1.0, 2.0, 3.0])
  >>> y_I = array([0.0, 1.0, 2.0, 3.0])
  >>> F_I = zeros((4, 4))
  >>> F_I[1, 2] = 1.0
  >>> arg_extremum_disc(x_I, y_I, F_I, 2.0, 1.0, 2.0)
      (1, 2)
  '''
  old_max = extrema_type * F_I[0, 0]
  ix_max, jx_max = 0, 0
  for jx in range(y_I.size):
    for ix in range(x_I.size):
      if (x_I[ix] - x_0) ** 2 + (y_I[jx] - y_0) ** 2 < r_0 ** 2:
        if (extrema_type * F_I[ix, jx]) > old_max:
          old_max = extrema_type * F_I[ix, jx]
          ix_max, jx_max = ix, jx
  return ix_max, jx_max

###############################################################################
# dimension order
###############################################################################

def ndarray_dim_order_exchange(arr : NDArray[Any],
                              num_skip_dim : int = 0) -> NDArray[Any]:
  '''
  Reverse-index permutation of trailing dimensions (C <-> Fortran order swap),
  skipping the first num_skip_dim axes.

  Parameters
  ----------
  arr : NDArray[Any]
    Array to transpose.
  num_skip_dim : int = 0
    Number of leading dimensions to preserve.

  Returns
  -------
  NDArray[Any]
    Appropriately transposed array.

  Usage
  -----
  >>> from numpy import zeros
  >>> arr = zeros((2, 3, 4, 5))
  >>> ndarray_dim_order_exchange(arr).shape
      (5, 4, 3, 2)
  >>> ndarray_dim_order_exchange(arr, num_skip_dim=1).shape
      (2, 5, 4, 3)
  >>> ndarray_dim_order_exchange(arr, num_skip_dim=2).shape
      (2, 3, 5, 4)
  '''
  if arr.ndim < num_skip_dim:
    raise ValueError(
      f"num_skip_dim = {num_skip_dim} > arr.ndim = {arr.ndim}"
    )
  suffix = tuple(reversed(range(num_skip_dim, arr.ndim)))
  axes = tuple(range(num_skip_dim)) + suffix
  return arr.transpose(axes)

###############################################################################
# windowing
###############################################################################

def map_window_exp(x : NDArray[Any], delta : float = 0.) -> NDArray[Any]:
  '''
  Double-sided exponential window on [0,1] with C-infinity transitions.

  Parameters
  ----------
  x : NDArray[Any]
    Points in [0, 1] at which to evaluate the window.
  delta : float = 0
    Width of transition region at each boundary.

  Returns
  -------
  w : NDArray[Any]
    Window values w(x).

  Usage
  -----
  >>> from numpy import array
  >>> x = array([0.0, 0.25, 0.5, 0.75, 1.0])
  >>> map_window_exp(x, delta=0.2)
      array([0., 1., 1., 1., 0.])
  '''
  return piecewise(x,
    [(x == 0),
     (x > 0) & (x < delta),
     (x >= delta) & (x <= 1. - delta),
     (x > 1. - delta) & (x < 1),
     (x == 1)],
    [lambda x: 0.,
     lambda x: exp(-1. / (1. - ((x - delta) / delta) ** 2) + 1.),
     lambda x: 1.,
     lambda x: exp(-1. / (1. - ((x - (1. - delta)) / delta) ** 2) + 1.),
     lambda x: 0.]
  )

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  array_common_type = JITI(array_common_type)
  array_extend_mirror = JITI(array_extend_mirror)
  array_extend_uniform_range = JITI(array_extend_uniform_range)
  array_dense_to_banded = JITI(array_dense_to_banded)
  array_dense_from_band_uniform = JITI(array_dense_from_band_uniform)
  array_dense_from_bands = JITI(array_dense_from_bands)
  array_identity_mask = JITI(array_identity_mask)
  array_pad_biased = JITI(array_pad_biased)
  array_roll_row_to_edges = JITI(array_roll_row_to_edges)
  array_dot_band_uniform = JITI(array_dot_band_uniform)
  array_dot_bands = JITI(array_dot_bands)
  array_dot_2d_at_axis = JITI(array_dot_2d_at_axis)
  ndarray_get_sorted_argmin = JITI(ndarray_get_sorted_argmin)
  interval_is_intersection_empty = JITI(interval_is_intersection_empty)
  arg_extremum_interval = JITI(arg_extremum_interval)
  arg_extremum_disc = JITI(arg_extremum_disc)

#
# :D
#
