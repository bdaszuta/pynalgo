"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Computation of function related to the factorial / gamma function.
"""
# scipy interfacing
# see: https://github.com/numba/numba/issues/3086
import ctypes
from numba.extending import get_cython_function_address
from numba import vectorize

from numpy import (arange, cumsum, exp, log)

from pynalgo.common_tools import (NDArray, float64, JITI, JITI_NC,
                                  TYPE_CHECKING)


_addr_gammaln = get_cython_function_address("scipy.special.cython_special",
                                            "gammaln")
_gammaln_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_gammaln_fn = _gammaln_functype(_addr_gammaln)

###############################################################################
# Miscellaneous definitions
###############################################################################

def special_exp_log_frac_sum(N : int,
                             f1 : float,
                             f2 : float,
                             f3 : float,
                             f4 : float,
                             g : float=1) -> NDArray[float64]:
  r'''
  Convenience function for calculation of:

  .. math::
     \exp\left[g \sum_{k=0}^{j} \ln f(k) \right], \quad j = 0, \ldots, N-1

  where:

  .. math::
     f(j) = \frac{f_1 j + f_2}{f_3 j + f_4}

  Parameters
  ----------
  N : int
    Limit of summation index (see above).

  f1, f2, f3, f4 : float
    Eqn. factors (see above).

  g : float, default=1
    Eqn. factor (see above.)

  Returns
  -------
  result : NDArray[float64]
    Result with cumulative summation provided.

  Usage
  -----
  >>> special_exp_log_frac_sum(3, 1., 1., 1., 3./2., g=1)
      array([0.66666667, 0.53333333, 0.45714286])
  '''
  j_arr = arange(0, N)
  g_ln_f_j = log((j_arr*f1 + f2)/(j_arr*f3 + f4)) * g

  result = exp(cumsum(g_ln_f_j))
  return result

###############################################################################
# From scipy
###############################################################################

@vectorize('float64(float64)')
def _vec_special_log_abs_gamma(y):
  """Vectorized log-abs-gamma function via scipy gammaln bridge.

  Used by quadrature and polynomial weight computations.
  """
  return _gammaln_fn(y)

def special_log_abs_gamma(x : NDArray[float64]) -> NDArray[float64]:
  """
  Log of absolute value of the gamma function.
  Interface to scipy.special.gammaln.

  Parameters
  ----------
  x : NDArray[float64]
    Real argument.

  Returns
  -------
  result : NDArray[float64]
    Result of calculation.

  Usage
  -----
  >>> from numpy import linspace
  >>> special_log_abs_gamma(linspace(1.2,4, num=4))
      array([-0.08537409,  0.06195055,  0.75553625,  1.79175947])
  """
  return _vec_special_log_abs_gamma(x)

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  special_exp_log_frac_sum = JITI(special_exp_log_frac_sum)
  special_log_abs_gamma = JITI_NC(special_log_abs_gamma)

#
# :D
#
