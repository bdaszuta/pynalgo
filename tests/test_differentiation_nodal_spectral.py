"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.differentiation.nodal_spectral
"""

from pynalgo import (grid_ChebyshevT, grid_Fourier,
                     diff_mat_nodal_ChebyshevT, diff_mat_nodal_Fourier)
from numpy import (dot, cos, sin, allclose)

def test_diff_mat_nodal_Fourier():
  # prepare all grids
  N = 6
  gr_S1_CL = grid_Fourier(N, variety="S1_CL")
  gr_S1_CR = grid_Fourier(N, variety="S1_CR")
  gr_S1_I  = grid_Fourier(N, variety="S1_I")
  gr_H1_C  = grid_Fourier(N, variety="H1_C")
  gr_H1_I  = grid_Fourier(N, variety="H1_I")
  gr_H1_S  = grid_Fourier(N, variety="H1_S")

  # prepare matrices [S1 grids]
  D1_S1_CL = diff_mat_nodal_Fourier(N, D=1, variety="S1_CL")
  D2_S1_CL = diff_mat_nodal_Fourier(N, D=2, variety="S1_CL")

  D1_S1_CR = diff_mat_nodal_Fourier(N, D=1, variety="S1_CR")
  D2_S1_CR = diff_mat_nodal_Fourier(N, D=2, variety="S1_CR")

  D1_S1_I = diff_mat_nodal_Fourier(N, D=1, variety="S1_I")
  D2_S1_I = diff_mat_nodal_Fourier(N, D=2, variety="S1_I")

  # prepare matrices [H1 grids]
  D1_H1_C_cos = diff_mat_nodal_Fourier(N, D=1, variety="H1_C_cos")
  D2_H1_C_cos = diff_mat_nodal_Fourier(N, D=2, variety="H1_C_cos")

  D1_H1_C_sin = diff_mat_nodal_Fourier(N, D=1, variety="H1_C_sin")
  D2_H1_C_sin = diff_mat_nodal_Fourier(N, D=2, variety="H1_C_sin")

  D1_H1_I_cos = diff_mat_nodal_Fourier(N, D=1, variety="H1_I_cos")
  D2_H1_I_cos = diff_mat_nodal_Fourier(N, D=2, variety="H1_I_cos")

  D1_H1_I_sin = diff_mat_nodal_Fourier(N, D=1, variety="H1_I_sin")
  D2_H1_I_sin = diff_mat_nodal_Fourier(N, D=2, variety="H1_I_sin")

  D1_H1_S_sin = diff_mat_nodal_Fourier(N, D=1, variety="H1_S_sin")
  D2_H1_S_sin = diff_mat_nodal_Fourier(N, D=2, variety="H1_S_sin")


  # ---------------------------------------------------------------------------
  # S1: close left
  A1der = cos(gr_S1_CL) - 2 * sin(2 * gr_S1_CL)
  A2der = -sin(gr_S1_CL) - 4 * cos(2 * gr_S1_CL)

  N1der = dot(D1_S1_CL, sin(gr_S1_CL) + cos(2 * gr_S1_CL))
  assert allclose(N1der, A1der)

  N2der = dot(D2_S1_CL, sin(gr_S1_CL) + cos(2 * gr_S1_CL))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # S1: close right
  A1der = cos(gr_S1_CR) - 2 * sin(2 * gr_S1_CR)
  A2der = -sin(gr_S1_CR) - 4 * cos(2 * gr_S1_CR)

  N1der = dot(D1_S1_CR, sin(gr_S1_CR) + cos(2 * gr_S1_CR))
  assert allclose(N1der, A1der)

  N2der = dot(D2_S1_CR, sin(gr_S1_CR) + cos(2 * gr_S1_CR))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # S1: internal
  A1der = cos(gr_S1_I) - 2 * sin(2 * gr_S1_I)
  A2der = -sin(gr_S1_I) - 4 * cos(2 * gr_S1_I)

  N1der = dot(D1_S1_I, sin(gr_S1_I) + cos(2 * gr_S1_I))
  assert allclose(N1der, A1der)

  N2der = dot(D2_S1_I, sin(gr_S1_I) + cos(2 * gr_S1_I))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # H1: closed (cos)
  A1der = -sin(gr_H1_C)
  A2der = -cos(gr_H1_C)

  N1der = dot(D1_H1_C_cos, cos(gr_H1_C))
  assert allclose(N1der, A1der)

  N2der = dot(D2_H1_C_cos, cos(gr_H1_C))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # H1: closed (sin)
  A1der = cos(gr_H1_C)
  A2der = -sin(gr_H1_C)

  N1der = dot(D1_H1_C_sin, sin(gr_H1_C))
  assert allclose(N1der, A1der)

  N2der = dot(D2_H1_C_sin, sin(gr_H1_C))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # H1: internal (cos)
  A1der = -sin(gr_H1_I)
  A2der = -cos(gr_H1_I)

  N1der = dot(D1_H1_I_cos, cos(gr_H1_I))
  assert allclose(N1der, A1der)

  N2der = dot(D2_H1_I_cos, cos(gr_H1_I))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # H1: internal (sin)
  A1der = cos(gr_H1_I)
  A2der = -sin(gr_H1_I)

  N1der = dot(D1_H1_I_sin, sin(gr_H1_I))
  assert allclose(N1der, A1der)

  N2der = dot(D2_H1_I_sin, sin(gr_H1_I))
  assert allclose(N2der, A2der)

  # ---------------------------------------------------------------------------
  # H1: internal - alternate (sin)
  A1der = cos(gr_H1_S)
  A2der = -sin(gr_H1_S)

  N1der = dot(D1_H1_S_sin, sin(gr_H1_S))
  assert allclose(N1der, A1der)

  N2der = dot(D2_H1_S_sin, sin(gr_H1_S))
  assert allclose(N2der, A2der)


def test_diff_mat_nodal_ChebyshevT_all_varieties():
  """D1*grid^2 = 2*grid, D2*grid^2 = 2 for all varieties."""
  N = 8
  for variety in ["GL", "G", "RL", "RR"]:
    gr = grid_ChebyshevT(N, variety=variety)
    D1 = diff_mat_nodal_ChebyshevT(N, D=1, variety=variety)
    D2 = diff_mat_nodal_ChebyshevT(N, D=2, variety=variety)
    assert allclose(dot(D1, gr ** 2), 2 * gr)
    assert allclose(dot(D2, gr ** 2), 2)


#
# :D
#
