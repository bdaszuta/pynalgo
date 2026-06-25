"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Apply finite difference stencils via banded matrix utilities
"""
from numpy import (array, linspace)
from pynalgo.common_tools.utilities_numpy import (
  array_dense_from_bands,
  array_dense_to_banded,
  array_dot_bands,
  array_extend_uniform_range)

NGHOST = 4
N_ng = 11
gr_ng = linspace(1, 4, num=N_ng)
gr = array_extend_uniform_range(gr_ng, NGHOST=NGHOST)
N = gr.size
ds = gr[1] - gr[0]

bands = array([
  [-1 / 2, 0, 1 / 2]
]) / ds

bands = array([
  [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
]) / ds

width = bands.shape[1]
bands_full = array_dense_to_banded(
    array_dense_from_bands(bands, N), width=width)

N0f = gr ** 2
A1f = 2 * gr

N1f = array_dot_bands(bands_full, N0f, axis=0)

#
# :D
#
