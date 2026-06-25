"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.special_functions.grids
"""

from pynalgo import grid_ChebyshevT
from numpy import (array, allclose)

def test_grid_ChebyshevT():
  gr_G  = grid_ChebyshevT(N=4, variety="G")

  # expected Gauss grid values
  e_gr_G = array([
    -9.51056516e-01, -5.87785252e-01, -6.12323400e-17,  5.87785252e-01,
     9.51056516e-01
  ])

  assert allclose(gr_G, e_gr_G)


#
# :D
#
