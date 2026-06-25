"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Derivatives of barycentric rational functions.

First and second derivatives via switchable methods:
  'sw'       -- Schneider-Werner (1986) divided differences
  'matrix'   -- Differentiation matrix      (requires x == z)
  'quotient' -- Quotient rule on r = N/D

Refs:
  Schneider, C., and Werner, W. (1986).
  "Some New Aspects of Rational Interpolation."
  Math. Comp. 47(175), 285-299.
  https://doi.org/10.1090/S0025-5718-1986-0842136-8
"""
import numpy as np

from pynalgo.common_tools import (JIT, TYPE_CHECKING)


###############################################################################
# D1: first derivatives
###############################################################################


def _rat_D1_quotient(z, f, w, x):
  """D1 via quotient rule on barycentric form r = N/D."""
  c_ = (z.flat[0] + f.flat[0] + w.flat[0] + x.flat[0])
  common_dtype = np.array(c_).dtype

  n = z.size
  m = x.size
  df = np.empty(m, dtype=common_dtype)

  if n == 1:
    df[:] = common_dtype.type(0.0)
    return df

  for i in range(m):
    xi = x[i]
    k_match = -1
    for k in range(n):
      if xi == z[k]:
        k_match = k
        break

    if k_match >= 0:
      s = common_dtype.type(0.0)
      fk = f[k_match]
      zk = z[k_match]
      for j in range(n):
        if j == k_match:
          continue
        dd = (fk - f[j]) / (zk - z[j])
        s += w[j] * dd
      df[i] = -s / w[k_match]
    else:
      N  = common_dtype.type(0.0)
      N1 = common_dtype.type(0.0)
      D  = common_dtype.type(0.0)
      D1 = common_dtype.type(0.0)
      for j in range(n):
        C0 = common_dtype.type(1.0) / (xi - z[j])
        C1 = -C0 * C0
        N  += w[j] * f[j] * C0
        N1 += w[j] * f[j] * C1
        D  += w[j] * C0
        D1 += w[j] * C1
      df[i] = (N1 * D - N * D1) / (D * D)

  return df


def rat_D1(z, f, w, x, method='sw'):
  """First derivative of barycentric rational function.

  Convenience wrapper for rat_D(k=1).

  Parameters
  ----------
  z : NDArray, shape (n,)
      Support points.
  f : NDArray, shape (n,)
      Function values at support points.
  w : NDArray, shape (n,)
      Barycentric weights.
  x : NDArray, shape (m,)
      Evaluation points.
  method : {'sw', 'matrix', 'quotient'}, optional
      Differentiation method. 'sw' (default) uses the Schneider-Werner
      divided-difference approach. 'matrix' uses the differentiation
      matrix. 'quotient' uses the quotient-rule formula.

  Returns
  -------
  df : NDArray, shape (m,)
      First derivative values at the evaluation points x.
  """
  if method == 'quotient':
    return _rat_D1_quotient(z, f, w, x)
  return rat_D(z, f, w, x, k=1, method=method)


###############################################################################
# D2: second derivatives
###############################################################################


def _rat_D2_quotient(z, f, w, x):
  r"""D2 via quotient rule:

  .. math::

      r'' = \frac{(N'' D - N D'') D - 2(N' D - N D') D'}{D^3}
  """
  c_ = (z.flat[0] + f.flat[0] + w.flat[0] + x.flat[0])
  common_dtype = np.array(c_).dtype

  n = z.size
  m = x.size
  d2f = np.empty(m, dtype=common_dtype)

  if n == 1:
    d2f[:] = common_dtype.type(0.0)
    return d2f

  for i in range(m):
    xi = x[i]
    k_match = -1
    for k in range(n):
      if xi == z[k]:
        k_match = k
        break

    if k_match >= 0:
      fk = f[k_match]
      zk = z[k_match]
      s1 = common_dtype.type(0.0)
      for j in range(n):
        if j == k_match:
          continue
        dd1 = (fk - f[j]) / (zk - z[j])
        s1 += w[j] * dd1
      r1_k = -s1 / w[k_match]

      s2 = common_dtype.type(0.0)
      for j in range(n):
        if j == k_match:
          continue
        dd1_j = (fk - f[j]) / (zk - z[j])
        dd2_j = (dd1_j - r1_k) / (z[j] - zk)
        s2 += w[j] * dd2_j
      d2f[i] = common_dtype.type(-2.0) * s2 / w[k_match]
    else:
      N  = common_dtype.type(0.0)
      N1 = common_dtype.type(0.0)
      N2 = common_dtype.type(0.0)
      D  = common_dtype.type(0.0)
      D1 = common_dtype.type(0.0)
      D2 = common_dtype.type(0.0)
      for j in range(n):
        C0 = common_dtype.type(1.0) / (xi - z[j])
        C1 = -C0 * C0
        C2 = common_dtype.type(2.0) * C0 * C0 * C0
        wf = w[j] * f[j]
        w1 = w[j]
        N  += wf * C0
        N1 += wf * C1
        N2 += wf * C2
        D  += w1 * C0
        D1 += w1 * C1
        D2 += w1 * C2
      A = N1 * D - N * D1
      B = D * D
      two = common_dtype.type(2.0)
      d2f[i] = ((N2 * D - N * D2) * D - two * A * D1) / (B * D)

  return d2f


def rat_D2(z, f, w, x, method='sw'):
  """Second derivative of barycentric rational function.

  Convenience wrapper for rat_D(k=2).

  Parameters
  ----------
  z : NDArray, shape (n,)
      Support points.
  f : NDArray, shape (n,)
      Function values at support points.
  w : NDArray, shape (n,)
      Barycentric weights.
  x : NDArray, shape (m,)
      Evaluation points.
  method : {'sw', 'matrix', 'quotient'}, optional
      Differentiation method. 'sw' (default) uses the Schneider-Werner
      divided-difference approach. 'matrix' uses the differentiation
      matrix. 'quotient' uses the quotient-rule formula.

  Returns
  -------
  d2f : NDArray, shape (m,)
      Second derivative values at the evaluation points x.
  """
  if method == 'quotient':
    return _rat_D2_quotient(z, f, w, x)
  return rat_D(z, f, w, x, k=2, method=method)


###############################################################################
# Differentiation matrix (standalone)
###############################################################################

def diff_mat_nodal_rat(z, w, k=1):
  """Return k-th derivative matrix (n, n) at support nodes.

  Parameters
  ----------
  z : NDArray, shape (n,)
      Support points.
  w : NDArray, shape (n,)
      Barycentric weights.
  k : int
      Derivative order (k >= 1).

  Returns
  -------
  Dk : NDArray, shape (n, n)
      Differentiation matrix: :math:`r^{(k)} = D_k \\, f`.
  """
  n = z.size
  common_dtype = np.array(z.flat[0] + w.flat[0]).dtype

  if n == 1:
    return np.zeros((1, 1), dtype=common_dtype)

  # Build D1
  Dk = np.empty((n, n), dtype=common_dtype)
  for i in range(n):
    diag = common_dtype.type(0.0)
    zi = z[i]
    wi = w[i]
    for j in range(n):
      if i == j:
        continue
      dij = w[j] / wi / (zi - z[j])
      Dk[i, j] = dij
      diag += dij
    Dk[i, i] = -diag

  # Power up for k > 1
  D1 = Dk.copy()
  for p in range(1, k):
    Dk = Dk @ D1

  return Dk


###############################################################################
# Arbitrary-order derivative: rat_D
###############################################################################


def rat_D(z, f, w, x, k=1, method='sw'):
  """k-th derivative of barycentric rational function.

  Convenience: calls rat_eval and returns the k-th result.
  """
  result = rat_eval(z, f, w, x, max_deriv=k, method=method)
  return result[k]


###############################################################################
# Fused evaluation: rat_eval
###############################################################################

def _rat_eval_sw(z, f, w, x, max_deriv):
  """Evaluate r(x) and derivatives up to max_deriv via divided differences.

  Fused single-pass: computes r, D1, ..., Dk together.
  Returns a 2D array: result[p, i] = p-th derivative at x[i] (p=0 is r(x)).

  References
  ----------
  Schneider, C., and Werner, W. (1986). Math. Comp. 47(175), 285-299.
  """
  c_ = (z.flat[0] + f.flat[0] + w.flat[0] + x.flat[0])
  common_dtype = np.array(c_).dtype

  n = z.size
  m = x.size
  result = np.empty((max_deriv + 1, m), dtype=common_dtype)

  if n == 1:
    result[0, :] = f[0]
    for p in range(1, max_deriv + 1):
      result[p, :] = common_dtype.type(0.0)
    return result

  fac = np.empty(max_deriv + 1, dtype=common_dtype)
  fac[0] = common_dtype.type(1.0)
  for p in range(1, max_deriv + 1):
    fac[p] = fac[p - 1] * common_dtype.type(p)

  for i in range(m):
    xi = x[i]
    k_match = -1
    for j in range(n):
      if xi == z[j]:
        k_match = j
        break

    if k_match >= 0:
      # Node case
      fk = f[k_match]
      zk = z[k_match]

      result[0, i] = fk
      if max_deriv == 0:
        continue

      # p=1
      dd = np.empty(n, dtype=common_dtype)
      r_val = common_dtype.type(0.0)
      for j in range(n):
        if j == k_match:
          dd[j] = common_dtype.type(0.0)
          continue
        dd[j] = (fk - f[j]) / (zk - z[j])
        r_val -= w[j] * dd[j]
      r_val = r_val / w[k_match]
      result[1, i] = r_val

      dd_prev = dd
      for p in range(2, max_deriv + 1):
        dd_cur = np.empty(n, dtype=common_dtype)
        r_cur = common_dtype.type(0.0)
        for j in range(n):
          if j == k_match:
            dd_cur[j] = common_dtype.type(0.0)
            continue
          dd_cur[j] = (dd_prev[j] - r_val / fac[p - 1]) / (z[j] - zk)
          r_cur -= w[j] * dd_cur[j]
        r_val = r_cur * fac[p] / w[k_match]
        result[p, i] = r_val
        dd_prev = dd_cur

    else:
      # Non-node case
      inv = np.empty(n, dtype=common_dtype)
      for j in range(n):
        inv[j] = w[j] / (xi - z[j])

      num = common_dtype.type(0.0)
      den = common_dtype.type(0.0)
      for j in range(n):
        num += inv[j] * f[j]
        den += inv[j]
      rx = num / den
      result[0, i] = rx

      if max_deriv == 0:
        continue

      # p=1
      dd = np.empty(n, dtype=common_dtype)
      num1 = common_dtype.type(0.0)
      den1 = common_dtype.type(0.0)
      for j in range(n):
        dd[j] = (rx - f[j]) / (xi - z[j])
        num1 += inv[j] * dd[j]
        den1 += inv[j]
      r_val = num1 / den1
      result[1, i] = r_val

      dd_prev = dd
      for p in range(2, max_deriv + 1):
        dd_cur = np.empty(n, dtype=common_dtype)
        num_cur = common_dtype.type(0.0)
        den_cur = common_dtype.type(0.0)
        for j in range(n):
          dd_cur[j] = (r_val / fac[p - 1] - dd_prev[j]) / (xi - z[j])
          num_cur += inv[j] * dd_cur[j]
          den_cur += inv[j]
        r_val = fac[p] * num_cur / den_cur
        result[p, i] = r_val
        dd_prev = dd_cur

  return result


def rat_eval(z, f, w, x, max_deriv=0, method='sw'):
  """Evaluate r(x) and derivatives up to order max_deriv in one pass.

  Parameters
  ----------
  z, f, w : NDArray, shape (n,)
  x : NDArray, shape (m,)
  max_deriv : int
      Maximum derivative order. 0 = function only, 1 = +D1, 2 = +D2, etc.
  method : {'sw', 'matrix', 'quotient'}

  Returns
  -------
  list of NDArray
      [r]           if max_deriv == 0
      [r, dr]        if max_deriv == 1
      [r, dr, d2r]   if max_deriv == 2
      ...
  """
  if method == 'sw':
    res2d = _rat_eval_sw(z, f, w, x, max_deriv)
    out = []
    for p in range(max_deriv + 1):
      out.append(res2d[p, :])
    return out
  elif method == 'matrix':
    if not np.array_equal(z, x):
      raise ValueError("method='matrix' requires x == z")
    # Build results iteratively
    out = [f.copy()]
    D1 = diff_mat_nodal_rat(z, w, k=1)
    for p in range(1, max_deriv + 1):
      out.append(D1 @ out[-1])
    return out
  elif method == 'quotient':
    if max_deriv > 2:
      raise NotImplementedError("quotient only supports max_deriv <= 2")
    # Fall back to separate calls with inline eval
    # Compute r(x) via barycentric sum
    c_ = (z.flat[0] + f.flat[0] + w.flat[0] + x.flat[0])
    common_dtype = np.array(c_).dtype
    n = z.size
    m = x.size
    r = np.empty(m, dtype=common_dtype)
    for i in range(m):
      num = common_dtype.type(0.0)
      den = common_dtype.type(0.0)
      for j in range(n):
        inv = w[j] / (x[i] - z[j])
        num += inv * f[j]
        den += inv
      r[i] = num / den
    parts = [r]
    if max_deriv >= 1:
      parts.append(_rat_D1_quotient(z, f, w, x))
    if max_deriv >= 2:
      parts.append(_rat_D2_quotient(z, f, w, x))
    return parts
  else:
    raise ValueError("Unknown method: " + repr(method))


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  rat_D1 = JIT(rat_D1)
  _rat_D1_quotient = JIT(_rat_D1_quotient)
  rat_D2 = JIT(rat_D2)
  _rat_D2_quotient = JIT(_rat_D2_quotient)
  diff_mat_nodal_rat = JIT(diff_mat_nodal_rat)
  rat_D = JIT(rat_D)
  rat_eval = JIT(rat_eval)
  _rat_eval_sw = JIT(_rat_eval_sw)

  from numpy import array as _wa
  _wx = _wa([0.0, 1.0], dtype=np.float64)
  _wf = _wa([1.0, 2.0], dtype=np.float64)
  _rat_D1_quotient(_wx, _wf, _wf, _wx)
  _rat_D2_quotient(_wx, _wf, _wf, _wx)
  diff_mat_nodal_rat(_wx, _wf, k=1)
  _rat_eval_sw(_wx, _wf, _wf, _wx, 0)

#
# :D
#
