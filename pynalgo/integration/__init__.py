"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Quadrature weight computation: Chebyshev, Clenshaw-Curtis,
           Fourier, Jacobi, Legendre.
"""



from pynalgo.integration.quadrature import (
  quad_ChebyshevT,
  quad_Clenshaw_Curtis,
  quad_Fourier,
  quad_JacobiP,
  quad_LegendreP,
)


__all__ = [
  'quad_ChebyshevT',
  'quad_Clenshaw_Curtis',
  'quad_Fourier',
  'quad_JacobiP',
  'quad_LegendreP',
]

#
# :D
#
