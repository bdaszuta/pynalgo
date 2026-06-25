"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Richardson extrapolation and convergence-order estimation.
"""

from numpy import (abs, inf, log, )

from pynalgo.common_tools import (JIT, TYPE_CHECKING)

###############################################################################
# Richardson extrapolation
###############################################################################

def extrap_Richardson(k, n0, Ac, Af):
  r'''
  Richardson extrapolation.

  Suppose

  .. math::

      A^* = A(h) + a_0 h^{n_0} + a_1 h^{n_1} + \ldots
      = A(h) + a_0 h^{n_0} + \mathcal{O}(h^{n_1})

  then with grid refinement factor :math:`k`,

  .. math::

      A^* = A(h/k) + a_0 (h/k)^{n_0} + \mathcal{O}(h^{n_1})

  Combining yields

  .. math::

      A^* = \frac{k^{n_0} A(h/k) - A(h)}{k^{n_0} - 1}
            + \mathcal{O}(h^{n_1})

  Parameters
  ----------
  k : float
    Grid refinement factor.
  n0 : float
    Leading-order convergence exponent.
  Ac : float or array_like
    A(h) -- value on the coarse grid.
  Af : float or array_like
    A(h/k) -- value on the fine grid.

  Returns
  -------
  A_star : float or array_like
    Extrapolated value.

  Note
  ----
  Ac and Af must be the same shape.
  '''
  if k == 1.0:
    raise ValueError("k must not be 1 (division by zero in k^n0 - 1)")

  kn0 = k ** n0
  return (kn0 * Af - Ac) / (kn0 - 1)

# Richardson error / convergence-order estimation (Newton iteration)
###############################################################################

def extrap_Richardson_err(k, ell, Ak, A, Al,
                          n0=1, ntol=1e-10, maxiter=100):
  r'''
  Estimate the convergence exponent n0 via Newton root-finding
  on the Richardson extrapolation equations.

  Suppose

  .. math::

      A^* = A(h) + a_0 h^{n_0} + a_1 h^{n_1} + \ldots

  Define

  .. math::

      A^* &= A_k + a_0 (h/k)^{n_0} + \\mathcal{O}(h^{n_1})
      \\qquad (A_k \\equiv A(h/k)) \\\\
      A^* &= A   + a_0 h^{n_0}     + \\mathcal{O}(h^{n_1})
      \\qquad (A   \\equiv A(h)) \\\\
      A^* &= A_\\ell + a_0 (h/\\ell)^{n_0} + \\mathcal{O}(h^{n_1})
      \\qquad (A_\\ell \\equiv A(h/\\ell))

  This function estimates n0 based on Newton root-finding.

  Parameters
  ----------
  k : float
    Refinement factor for the fine grid.
  ell : float
    Refinement factor for a second fine grid.
  Ak : float or array_like
    A(h/k) -- value on grid refined by factor k.
  A : float or array_like
    A(h) -- value on the base grid.
  Al : float or array_like
    A(h/ell) -- value on grid refined by factor ell.
  n0 : float, optional
    Initial guess for the convergence exponent (default 1).
  ntol : float, optional
    Absolute tolerance for the Newton iteration (default 1e-10).
  maxiter : int, optional
    Maximum number of Newton iterations (default 100).

  Returns
  -------
  n0 : float
    Estimated convergence exponent.
  '''
  if k == 1.0 or ell == 1.0:
    raise ValueError("k and ell must not be 1 (division by zero)")

  iter_count = 0
  diff = inf

  while (diff > ntol) and (iter_count < maxiter):
    diff = n0

    T1 = Ak + (Ak - A) / (k ** n0 - 1)
    T2 = Al + (Al - A) / (ell ** n0 - 1)

    dT1 = log(k) * k ** n0 * (A - Ak) / (k ** n0 - 1) ** 2
    dT2 = log(ell) * ell ** n0 * (A - Al) / (ell ** n0 - 1) ** 2

    f = T1 - T2
    df = dT1 - dT2

    n0 = n0 - f / df
    if abs(df) < ntol:
      break
    diff = abs(n0 - diff)
    iter_count = iter_count + 1

  return n0

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  extrap_Richardson = JIT(extrap_Richardson)
  extrap_Richardson_err = JIT(extrap_Richardson_err)

#
# :D
#
