"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Interpolation and extrapolation: barycentric, Lagrange, Neville, Richardson, smoothing.
"""


from pynalgo.resample.barycentric import (
  interp_barycentric_1d,
  interp_barycentric_nd,
)
from pynalgo.resample.barycentric_generalized import (
  interp_barycentric_1d_generalized,
)
from pynalgo.resample.barycentric_nn import (
  interp_nn_1d,
)
from pynalgo.resample.lagrange_uniform import (
  interp_lagrange_uniform,
)
from pynalgo.resample.barycentric_2d_split import (
  interp_barycentric_2d_split,
)
from pynalgo.resample.lebesgue import (
  Lebesgue_func,
)
from pynalgo.resample.column import (
  columns_interpolate,
  columns_smooth,
  lerp_1d,
  uniform_filter_1d,
  gaussian_filter_1d,
)
from pynalgo.resample.extrapolation import (
  extrap_Richardson,
  extrap_Richardson_err,
)
from pynalgo.resample.neville import (
  interp_Neville,
)

__all__ = [
  'interp_barycentric_1d',
  'interp_barycentric_nd',
  'interp_barycentric_1d_generalized',
  'interp_nn_1d',
  'interp_lagrange_uniform',
  'interp_barycentric_2d_split',
  'Lebesgue_func',
  'columns_interpolate',
  'columns_smooth',
  'lerp_1d',
  'uniform_filter_1d',
  'gaussian_filter_1d',
  'extrap_Richardson',
  'extrap_Richardson_err',
  'interp_Neville',
]

#
# :D
#
