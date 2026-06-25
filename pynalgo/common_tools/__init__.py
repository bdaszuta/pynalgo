"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Common tool re-exports: JIT decorators, array utilities, primitives.
"""



from pynalgo.common_tools.utilities_primitives import (is_even, is_odd)
from pynalgo.common_tools.utilities_numba import (JITG, JITI, JIT,
                                                  JITI_NC, JIT_NC,
                                                  idx_interchange)
from pynalgo.common_tools.utilities_typing import *  # noqa: F403
from pynalgo.common_tools.utilities_numpy import (
  arg_extremum_interval,
  arg_extremum_disc,
  array_common_type,
  array_dense_from_band_uniform,
  array_dense_from_bands,
  array_dense_to_banded,
  array_dot_2d_at_axis,
  array_dot_band_uniform,
  array_dot_bands,
  array_extend_mirror,
  array_extend_uniform_range,
  array_identity_mask,
  array_pad_biased,
  array_roll_row_to_edges,
  interval_is_intersection_empty,
  map_window_exp,
  ndarray_dim_order_exchange,
  ndarray_get_sorted_argmin,
  scalar_data_deduplicate_sort,
)

__all__ = [
  'JIT',
  'JITG',
  'JITI',
  'JITI_NC',
  'JIT_NC',
  'arg_extremum_interval',
  'arg_extremum_disc',
  'array_common_type',
  'array_dense_from_band_uniform',
  'array_dense_from_bands',
  'array_dense_to_banded',
  'array_dot_2d_at_axis',
  'array_dot_band_uniform',
  'array_dot_bands',
  'array_extend_mirror',
  'array_extend_uniform_range',
  'array_identity_mask',
  'array_pad_biased',
  'array_roll_row_to_edges',
  'idx_interchange',
  'interval_is_intersection_empty',
  'is_even',
  'is_odd',
  'map_window_exp',
  'ndarray_dim_order_exchange',
  'ndarray_get_sorted_argmin',
  'scalar_data_deduplicate_sort',
]

#
# :D
#
