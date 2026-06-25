"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.integration.quadrature
"""

from pynalgo import (grid_Fourier, grid_ChebyshevT,
                     grid_JacobiP, grid_LegendreP,
                     quad_Fourier, quad_ChebyshevT,
                     quad_Clenshaw_Curtis, quad_JacobiP,
                     quad_LegendreP)
from numpy import (sin, cos, dot, pi, allclose)

def test_quad_Fourier():
  # prepare all grids
  N = 24
  gr_S1_CL = grid_Fourier(N, variety="S1_CL")
  gr_S1_CR = grid_Fourier(N, variety="S1_CR")
  gr_S1_I  = grid_Fourier(N, variety="S1_I")
  gr_H1_C  = grid_Fourier(N, variety="H1_C")
  gr_H1_I  = grid_Fourier(N, variety="H1_I")

  # prepare all quadrature weights
  wei_S1_CL = quad_Fourier(N, variety="S1_CL")
  wei_S1_CR = quad_Fourier(N, variety="S1_CR")
  wei_S1_I  = quad_Fourier(N, variety="S1_I")
  wei_gr_H1_C_cos = quad_Fourier(N, variety="H1_C_cos")
  wei_gr_H1_I_cos = quad_Fourier(N, variety="H1_I_cos")
  wei_gr_H1_C_sin = quad_Fourier(N, variety="H1_C_sin")
  wei_gr_H1_I_sin = quad_Fourier(N, variety="H1_I_sin")

  # provide functions for testing
  f_S1 = lambda gr: sin(2 * gr) ** 2 * cos(3* gr) ** 4
  f_S1_CL = f_S1(gr_S1_CL)
  f_S1_CR = f_S1(gr_S1_CR)
  f_S1_I  = f_S1(gr_S1_I)

  f_H1_cos = lambda gr: cos(gr) ** 4
  f_H1_sin = lambda gr: sin(gr) ** 3

  # eval
  f_H1_C_cos = f_H1_cos(gr_H1_C)
  f_H1_I_cos = f_H1_cos(gr_H1_I)

  f_H1_C_sin = f_H1_sin(gr_H1_C)
  f_H1_I_sin = f_H1_sin(gr_H1_I)

  # f_S1 integration over [0, 2 * pi] yields 3 * pi / 8
  integral_S1_CL = dot(f_S1_CL, wei_S1_CL)
  integral_S1_CR = dot(f_S1_CR, wei_S1_CR)
  integral_S1_I = dot(f_S1_I, wei_S1_I)

  assert allclose(integral_S1_CL, 3 * pi / 8)
  assert allclose(integral_S1_CR, 3 * pi / 8)
  assert allclose(integral_S1_I, 3 * pi / 8)

  # f_H1_cos integration over [0, pi] yields 3 * pi / 8
  integral_H1_C_cos = dot(wei_gr_H1_C_cos, f_H1_C_cos)
  integral_H1_I_cos = dot(wei_gr_H1_I_cos, f_H1_I_cos)

  assert allclose(integral_H1_C_cos, 3 * pi / 8)
  assert allclose(integral_H1_I_cos, 3 * pi / 8)

  # f_H1_sin integration over [0, pi] yields 4 / 3
  integral_H1_C_sin = dot(wei_gr_H1_C_sin, f_H1_C_sin)
  integral_H1_I_sin = dot(wei_gr_H1_I_sin, f_H1_I_sin)

  assert allclose(integral_H1_C_sin, 4 / 3)
  assert allclose(integral_H1_I_sin, 4 / 3)


def test_quad_ChebyshevT_all_varieties():
  """Integrate x^2 / sqrt(1-x^2) = pi/2 for all varieties."""
  N = 16
  for variety in ["GL", "G", "RL", "RR"]:
    gr = grid_ChebyshevT(N, variety=variety)
    wei = quad_ChebyshevT(N, variety=variety)
    integral = dot(gr ** 2, wei)
    assert allclose(integral, pi / 2)


def test_quad_Clenshaw_Curtis():
  """Integrate x^4 = 2/5. N should be >= 5 for accuracy."""
  N = 8
  gr = grid_ChebyshevT(N, variety="GL")
  wei = quad_Clenshaw_Curtis(N)
  integral = dot(gr ** 4, wei)
  assert allclose(integral, 2.0 / 5.0)


def test_quad_LegendreP_all_varieties():
  """Integrate x^2 = 2/3. N=3, no polish."""
  N = 3
  gr = grid_LegendreP(N, variety="GL", root_polish=False)
  wei = quad_LegendreP(N, variety="GL", root_polish=False)
  integral = dot(gr ** 2, wei)
  assert allclose(integral, 2.0 / 3.0)


def test_quad_JacobiP_all_varieties():
  """Integrate x^2 with a=b=0 = 2/3. N=3, no polish."""
  N = 3
  a, b = 0.0, 0.0
  gr = grid_JacobiP(N, a, b, variety="GL", root_polish=False)
  wei = quad_JacobiP(N, a, b, variety="GL", root_polish=False)
  integral = dot(gr ** 2, wei)
  assert allclose(integral, 2.0 / 3.0)


#
# :D
#