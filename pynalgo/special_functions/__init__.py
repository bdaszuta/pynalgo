"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Special functions and orthogonal polynomials: grids, polynomial bases, gamma functions.
"""



from pynalgo.special_functions.factorial import (
  special_exp_log_frac_sum,
  special_log_abs_gamma,
)

from pynalgo.special_functions.grids import (
  grid_ChebyshevT,
  grid_JacobiP,
  grid_LegendreP,
  grid_Fourier,
)

from pynalgo.special_functions.recursion_polynomial import (
  poly_JacobiP,
  poly_der_JacobiP,
  poly_ChebyshevT,
  poly_der_ChebyshevT,
  poly_ChebyshevU,
  poly_der_ChebyshevU,
  poly_ChebyshevV,
  poly_der_ChebyshevV,
  poly_ChebyshevW,
  poly_der_ChebyshevW,
  poly_Gegenbauer,
  poly_der_Gegenbauer,
  poly_Ultraspherical,
  poly_der_Ultraspherical,
  poly_ChebyshevT_direct,
  poly_ChebyshevU_direct,
  poly_sin,
  poly_cos,
  poly_LegendreP,
  poly_der_LegendreP,
)

from pynalgo.special_functions.recursion_polynomial_laguerre import (
  poly_Laguerre,
  poly_Laguerre_lambda,
  poly_der_Laguerre,
  poly_der_Laguerre_lambda,
)

from pynalgo.special_functions.recursion_polynomial_hermite import (
  poly_Hermite_psi,
  poly_Hermite_H,
  poly_der_Hermite_H,
)

from pynalgo.special_functions.recursion_polynomial_chebyshev_mod import (
  poly_ChebyshevT_mod,
)

__all__ = [
  'grid_ChebyshevT',
  'grid_Fourier',
  'grid_JacobiP',
  'grid_LegendreP',
  'poly_ChebyshevT',
  'poly_ChebyshevT_direct',
  'poly_ChebyshevT_mod',
  'poly_ChebyshevU',
  'poly_ChebyshevU_direct',
  'poly_ChebyshevV',
  'poly_ChebyshevW',
  'poly_Gegenbauer',
  'poly_Hermite_H',
  'poly_Hermite_psi',
  'poly_JacobiP',
  'poly_Laguerre',
  'poly_Laguerre_lambda',
  'poly_LegendreP',
  'poly_Ultraspherical',
  'poly_cos',
  'poly_der_ChebyshevT',
  'poly_der_ChebyshevU',
  'poly_der_ChebyshevV',
  'poly_der_ChebyshevW',
  'poly_der_Gegenbauer',
  'poly_der_Hermite_H',
  'poly_der_JacobiP',
  'poly_der_Laguerre',
  'poly_der_Laguerre_lambda',
  'poly_der_LegendreP',
  'poly_der_Ultraspherical',
  'poly_sin',
  'special_exp_log_frac_sum',
  'special_log_abs_gamma',
]

#
# :D
#
