"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Generalized Floater-Hormann barycentric interpolation.

Introduces a rational-power parameter ``gam`` that deforms the standard
FH barycentric form.  Standard FH is recovered at gam=1.

Refs:
  Themistoclakis, W., and Van Barel, M. (2023).
  "A Generalization of Floater-Hormann Interpolants."
  arXiv:2307.05345.  https://doi.org/10.48550/arXiv.2307.05345.
"""
import numpy as np

from pynalgo.common_tools import (JIT, TYPE_CHECKING)
from pynalgo.resample.barycentric import _interp_barycentric_1d_jit


###############################################################################
# Weight computation
###############################################################################

def _compute_gen_fh_weights_jit(X_src, d, gam):
  r"""Compute the target-independent part of generalized FH weights.

  The full weight for target point ``x`` and source node ``x_k`` is

  .. math::

      w_k(x) = w_k^0 \cdot |x - x_k|^{-\gamma}

  where the target-independent prefactor :math:`w_k^0` is

  .. math::

      w_k^0 = \sum_{i=i_l}^{i_u}
              (-1)^{i\gamma}
              \bigg/ \prod_{\substack{s=i\\s\neq k}}^{i+d}
              (x_k - x_s)

  with :math:`i_l = \max(k-d, 0)` and :math:`i_u = \min(k, N_s-d)`.
  The sign factor :math:`(-1)^{i\gamma}` is evaluated as
  :math:`\cos(\pi i \gamma)`.

  This function returns only :math:`w_k^0`; the target-dependent
  :math:`|x-x_k|^{-\gamma}` factor is folded in during evaluation.
  """
  n = X_src.size
  Ns = n - 1
  w = np.empty(n, dtype=X_src.dtype)

  for k in range(n):
    il = max(k - d, 0)
    iu = min(k, Ns - d)
    wei = 0.0
    for i in range(il, iu + 1):
      # (-1)^{i*gam}: use cos(pi * i * gam) for the sign
      sign = np.cos(np.pi * i * gam)
      prod = 1.0
      for s in range(i, i + d + 1):
        if s == k:
          continue
        prod *= (X_src[k] - X_src[s])
      wei += sign / prod
    w[k] = wei

  return w


###############################################################################
# Generalized FH evaluation
###############################################################################

def interp_barycentric_1d_generalized(X, F, X_I, d, gam):
  """Generalized Floater-Hormann interpolation.

  Parameters
  ----------
  X : NDArray, shape (n,)
      Source grid coordinates (sorted).
  F : NDArray, shape (n,)
      Function values at source grid.
  X_I : NDArray, shape (m,)
      Target evaluation points.
  d : int
      FH blending parameter. When gam=1: d=0 gives BT-style weights.
      When gam != 1, weights do not reduce to BT.
  gam : float
      Rational power.  gam=1 recovers standard FH.
      Must be > 0.

  Returns
  -------
  F_I : NDArray, shape (m,)
      Interpolated function values at target points X_I.
  """
  if gam <= 0.0:
    raise ValueError("gam must be > 0")
  if X.size == 0:
    return np.zeros_like(X_I)
  if X.size == 1:
    out = np.empty(X_I.size, dtype=X.dtype)
    out[:] = F[0]
    return out

  if abs(gam - 1.0) < 1e-15:
    # Dispatch to standard FH/BT for gam=1
    method = 'FH' if d > 0 else 'BT'
    return _interp_barycentric_1d_jit(X, F, X_I, d, method)

  return _interp_gen_fh_jit(X, F, X_I, d, gam)


def _interp_gen_fh_jit(X, F, X_I, d, gam):
  """JIT kernel: generalized FH on arbitrary grids."""
  n = X.size
  m = X_I.size
  out = np.empty(m, dtype=X.dtype)

  if n == 1:
    out[:] = F[0]
    return out

  # Precompute weights w_k (target-independent part)
  w = _compute_gen_fh_weights_jit(X, d, gam)

  for i in range(m):
    xt = X_I[i]

    # Coincident node shortcut
    for k in range(n):
      if xt == X[k]:
        out[i] = F[k]
        break
    else:
      num = 0.0
      den = 0.0
      for k in range(n):
        dx = xt - X[k]
        if dx == 0.0:
          dx = 1e-300  # guard against division by zero
        # rw_k = w_k * |dx|^{-gam}
        rw = w[k] * np.abs(dx) ** (-gam)
        num += rw * F[k]
        den += rw
      out[i] = num / den if den != 0.0 else 0.0

  return out


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _compute_gen_fh_weights_jit = JIT(_compute_gen_fh_weights_jit)
  _interp_gen_fh_jit = JIT(_interp_gen_fh_jit)
  interp_barycentric_1d_generalized = JIT(interp_barycentric_1d_generalized)

  # Warm-up
  _wx = np.array([0.0, 1.0, 2.0], dtype=np.float64)
  _wf = np.array([0.0, 1.0, 4.0], dtype=np.float64)
  _wt = np.array([0.5, 1.5], dtype=np.float64)
  _interp_gen_fh_jit(_wx, _wf, _wt, 2, 1.5)
  interp_barycentric_1d_generalized(_wx, _wf, _wt, 2, 0.5)

#
# :D
#
