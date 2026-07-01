"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: AAA rational function approximation.

Functional-style implementation of the AAA algorithm.

Refs:
  Nakatsukasa, Y., Sete, O., and Trefethen, L. N. (2018).
  "The AAA Algorithm for Rational Approximation."
  SIAM J. Sci. Comput., 40(3), A1494-A1522.
  https://doi.org/10.1137/16M1106122
"""
import numpy as np

from pynalgo.common_tools import (JIT, JITI, TYPE_CHECKING)
from pynalgo.resample.lebesgue import Lebesgue_func as _lebesgue_jit

###############################################################################
# Barycentric rational evaluation
###############################################################################

def eval_rat(z_support, f_support, w, z_target,
             f_target=None, idx_target=None):
  r"""Evaluate barycentric rational function r(x) = N(x)/D(x) at target points.

  Parameters
  ----------
  z_support : NDArray
      Support nodes (n,).
  f_support : NDArray
      Function values at support nodes (n,).
  w : NDArray
      Barycentric weights (n,).
  z_target : NDArray
      Evaluation points (m,).
  f_target : NDArray, optional
      Pre-allocated output array. If None, allocated internally.
  idx_target : NDArray[int64], optional
      If provided, only evaluate at z_target[idx_target] and scatter into
      f_target. If None, evaluate at all z_target points.

  Returns
  -------
  f_target : NDArray
      Rational function values at target points.
  """
  c_ = (z_support.flat[0] + f_support.flat[0] +
        w.flat[0] + z_target.flat[0])
  common_dtype = np.array(c_).dtype

  if idx_target is not None:
    z_t = z_target[idx_target]
    m = idx_target.size
  else:
    z_t = z_target
    m = z_target.size

  n = z_support.size

  if f_target is None:
    f_target = np.empty(z_target.size, dtype=common_dtype)

  r = np.empty(m, dtype=common_dtype)
  for i in range(m):
    num = common_dtype.type(0.0)
    den = common_dtype.type(0.0)
    for j in range(n):
      diff = z_t[i] - z_support[j]
      if diff != 0:
        fac = w[j] / diff
        num += fac * f_support[j]
        den += fac
      else:
        num = f_support[j]
        den = 1
        break
    r[i] = num / den

  if idx_target is not None:
    for i in range(idx_target.size):
      f_target[idx_target[i]] = r[i]
  else:
    f_target[:] = r[:]

  return f_target


###############################################################################
# Pole and residue computation
###############################################################################

def poles_residues(z, f, w):
  r"""Return poles and residues of barycentric rational function.

  Constructs a rank-1 perturbed diagonal matrix (related to the Loewner
  matrix) and finds its ordinary eigenvalues. Filters infinite
  eigenvalues, removes the pole-at-infinity (smallest magnitude),
  and cleans up tiny imaginary parts.

  Parameters
  ----------
  z : NDArray
      Support nodes.
  f : NDArray
      Function values at support nodes.
  w : NDArray
      Barycentric weights.

  Returns
  -------
  poles : NDArray
      Poles of the rational function (real-valued).
  residues : NDArray
      Residues at each pole.
  """
  ak = w / w.sum()
  M = np.diag(z) - np.outer(ak, z)
  lam = np.linalg.eigvals(M + 0j)
  pol = lam[np.isfinite(lam)]
  tiny_imag = (np.abs(pol.imag) <
               np.finfo(pol.dtype).eps * (1.0 + np.abs(pol.real)))
  for i in range(pol.size):
    if tiny_imag[i]:
      pol[i] = pol[i].real
  sz_pol = pol.size
  ix_min = np.argmin(np.abs(pol))
  for j in range(ix_min, sz_pol - 1):
    pol[j] = pol[j + 1]
  pol = pol[:-1]
  C_pol = 1.0 / (pol[:, None] - z[None, :])
  N_pol = C_pol.dot(f * w)
  Ddiff_pol = (-C_pol ** 2).dot(w)
  res = N_pol / Ddiff_pol
  return pol, res


###############################################################################
# JIT array utility
###############################################################################

def _shift_delete_1d(arr, ix):
  """Delete element at index ix by shifting left."""
  sz = arr.size
  for j in range(ix, sz - 1):
    arr[j] = arr[j + 1]
  return arr[:-1]


###############################################################################
# AAA cores
###############################################################################

def _aaa_real_core(z, f, tol, max_terms, z_required, use_lebesgue):
  """AAA core: real weights, no cleanup."""
  _dtype = np.float64
  sz_rn = 0
  if z_required is None:
    z_req = np.empty(0, dtype=_dtype)
  else:
    z_req = np.zeros(z_required.size, dtype=_dtype)
    z_req[:] = z_required[:]
  max_terms = min(z.size, max(max_terms, z_req.size))
  sz_rn += z_req.size

  z_ = np.zeros(int(max_terms), dtype=_dtype)
  f_ = np.zeros(int(max_terms), dtype=_dtype)
  w_ = np.zeros(int(max_terms), dtype=_dtype)
  s_ = np.zeros(int(max_terms), dtype=np.int32)
  r_ = np.arange(z.size, dtype=np.int32)

  for j in range(sz_rn):
    ix = np.argmin(np.abs(z - z_req[j]))
    s_[j] = ix
    r_[ix] = -1
  if sz_rn > 0:
    f_[:sz_rn] = f[s_[:sz_rn]]
    z_[:sz_rn] = z[s_[:sz_rn]]

  sz_cur = sz_rn
  r_tol = tol
  R = np.mean(f) * np.ones_like(f)

  for m in range(max_terms - sz_rn):
    if use_lebesgue:
      lam = _lebesgue_jit(z_[:sz_cur].real, z[r_ >= 0].real)
      err = lam * np.abs(f[r_ >= 0] - R[r_ >= 0])
    else:
      err = np.abs(f[r_ >= 0] - R[r_ >= 0])
    n_ = np.argmax(err)
    ix = r_[r_ >= 0][n_]
    s_[sz_cur] = ix
    z_[sz_cur] = z[ix]
    f_[sz_cur] = f[ix]
    r_[ix] = -1
    sz_cur += 1

    oo_C = z[r_ >= 0, None] - z_[None, :sz_cur]
    C = 1.0 / oo_C
    L = (f[r_ >= 0, None] - f_[None, :sz_cur]) * C
    if L.shape[0] >= L.shape[1]:
      U, S, Vh = np.linalg.svd(L)
      w_[:sz_cur] = Vh[-1, :]
    else:
      UT, ST, VhT = np.linalg.svd(L.T)
      w_[:sz_cur] = UT.T[-1, :]

    R[:] = f[:]
    eval_rat(z_[:sz_cur], f_[:sz_cur], w_[:sz_cur], z,
             f_target=R, idx_target=r_[r_ >= 0])
    scale = max(1.0, np.linalg.norm(f, np.inf))
    if np.linalg.norm(f - R, np.inf) / scale <= r_tol:
      break

  s_ix = np.argsort(s_[:sz_cur])
  z_[:sz_cur] = z_[:sz_cur][s_ix]
  f_[:sz_cur] = f_[:sz_cur][s_ix]
  w_[:sz_cur] = w_[:sz_cur][s_ix]
  return z_[:sz_cur], f_[:sz_cur], w_[:sz_cur]


def _aaa_core(z, f, tol, max_terms, z_required, use_lebesgue,
              complex_weights, kappa):
  """AAA core: complex128 dtype, no cleanup. Complex weight perturbation
  is gated on kappa.
  Note: complex_weights parameter is accepted but unused (always complex128)."""
  _dtype = np.complex128
  sz_rn = 0
  if z_required is None:
    z_req = np.empty(0, dtype=_dtype)
  else:
    z_req = np.zeros(z_required.size, dtype=_dtype)
    z_req[:] = z_required[:]
  max_terms = min(z.size, max(max_terms, z_req.size))
  sz_rn += z_req.size

  z_ = np.zeros(int(max_terms), dtype=_dtype)
  f_ = np.zeros(int(max_terms), dtype=_dtype)
  w_ = np.zeros(int(max_terms), dtype=_dtype)
  s_ = np.zeros(int(max_terms), dtype=np.int32)
  r_ = np.arange(z.size, dtype=np.int32)

  for j in range(sz_rn):
    ix = np.argmin(np.abs(z - z_req[j]))
    s_[j] = ix
    r_[ix] = -1
  if sz_rn > 0:
    f_[:sz_rn] = f[s_[:sz_rn]]
    z_[:sz_rn] = z[s_[:sz_rn]]

  sz_cur = sz_rn
  r_tol = tol
  R = np.complex128(np.mean(f)) * np.ones(z.size, dtype=_dtype)

  for m in range(max_terms - sz_rn):
    if use_lebesgue:
      lam = _lebesgue_jit(z_[:sz_cur].real, z[r_ >= 0].real)
      err = lam * np.abs(f[r_ >= 0] - R[r_ >= 0])
    else:
      err = np.abs(f[r_ >= 0] - R[r_ >= 0])
    n_ = np.argmax(err)
    ix = r_[r_ >= 0][n_]
    s_[sz_cur] = ix
    z_[sz_cur] = z[ix]
    f_[sz_cur] = f[ix]
    r_[ix] = -1
    sz_cur += 1

    oo_C = z[r_ >= 0, None] - z_[None, :sz_cur]
    C = 1.0 / oo_C
    L = (f[r_ >= 0, None] - f_[None, :sz_cur]) * C
    if L.shape[0] >= L.shape[1]:
      U, S, Vh = np.linalg.svd(L)
      if Vh.shape[0] >= 2 and kappa is not None:
        w_[:sz_cur] = Vh[-1, :]
        ratio = max(S[-1], 0.0) / max(S[-2], 1e-300)
        w_[:sz_cur] += ratio ** kappa * 1j * Vh[-2, :]
      else:
        w_[:sz_cur] = Vh[-1, :]
    else:
      UT, ST, VhT = np.linalg.svd(L.T)
      if UT.T.shape[0] >= 2 and kappa is not None:
        w_[:sz_cur] = UT.T[-1, :]
        ratio = max(ST[-1], 0.0) / max(ST[-2], 1e-300)
        w_[:sz_cur] += ratio ** kappa * 1j * UT.T[-2, :]
      else:
        w_[:sz_cur] = UT.T[-1, :]

    R[:] = f[:] + 0j
    eval_rat(z_[:sz_cur], f_[:sz_cur], w_[:sz_cur], z,
             f_target=R, idx_target=r_[r_ >= 0])
    scale = max(1.0, np.linalg.norm(f, np.inf))
    if np.linalg.norm(f - R, np.inf) / scale <= r_tol:
      break

  s_ix = np.argsort(s_[:sz_cur])
  z_[:sz_cur] = z_[:sz_cur][s_ix]
  f_[:sz_cur] = f_[:sz_cur][s_ix]
  w_[:sz_cur] = w_[:sz_cur][s_ix]
  return z_[:sz_cur], f_[:sz_cur], w_[:sz_cur]


###############################################################################
# Public API with cleanup
###############################################################################

def _drop_small_weights(z_sup, f_sup, w_sup, tol_w):
  """Drop support nodes with |w| <= tol_w.  Returns cleaned triple."""
  ix_rel = np.abs(w_sup) > tol_w
  n_keep = 0
  for i in range(z_sup.size):
    if ix_rel[i]:
      n_keep += 1
  if n_keep == z_sup.size:
    return z_sup, f_sup, w_sup

  z_k = np.empty(n_keep, dtype=z_sup.dtype)
  f_k = np.empty(n_keep, dtype=f_sup.dtype)
  w_k = np.empty(n_keep, dtype=w_sup.dtype)
  k = 0
  for i in range(z_sup.size):
    if ix_rel[i]:
      z_k[k] = z_sup[i]
      f_k[k] = f_sup[i]
      w_k[k] = w_sup[i]
      k += 1
  return z_k, f_k, w_k


def _cleanup_loop_real(z_sup, f_sup, w_sup, z_all, f_all,
                        tol, max_t, z_required, use_lebesgue,
                        tol_F, tol_w):
  """Cleanup loop: iteratively remove Froissart doublets."""
  if tol_w > 0.0:
    z_sup, f_sup, w_sup = _drop_small_weights(
        z_sup, f_sup, w_sup, tol_w)

  z_all_c = z_all.copy()
  f_all_c = f_all.copy()

  for _ in range(10):
    pol, res = poles_residues(z_sup + 0j, f_sup + 0j, w_sup + 0j)
    n_bad = 0
    for k in range(pol.size):
      if np.abs(res[k]) < tol_F or np.isnan(res[k]):
        n_bad += 1
    if n_bad == 0:
      return z_sup, f_sup, w_sup

    for k in range(pol.size):
      if np.abs(res[k]) < tol_F or np.isnan(res[k]):
        ix_p = 0
        d_min = np.abs(z_all_c[0] - pol[k].real)
        for j in range(1, z_all_c.size):
          d = np.abs(z_all_c[j] - pol[k].real)
          if d < d_min:
            d_min = d
            ix_p = j
        z_all_c = _shift_delete_1d(z_all_c, ix_p)
        f_all_c = _shift_delete_1d(f_all_c, ix_p)

    z_sup, f_sup, w_sup = _aaa_real_core(
        z_all_c, f_all_c, tol, max_t, z_required, use_lebesgue)

  return z_sup, f_sup, w_sup


def _cleanup_loop_complex(z_sup, f_sup, w_sup, z_all, f_all,
                           tol, max_t, z_required, use_lebesgue,
                           complex_weights, kappa, tol_F, tol_w):
  """Cleanup loop for complex AAA."""
  if tol_w > 0.0:
    z_sup, f_sup, w_sup = _drop_small_weights(
        z_sup, f_sup, w_sup, tol_w)

  z_all_c = z_all.copy()
  f_all_c = f_all.copy()

  for _ in range(10):
    pol, res = poles_residues(z_sup, f_sup, w_sup)
    n_bad = 0
    for k in range(pol.size):
      if np.abs(res[k]) < tol_F or np.isnan(res[k]):
        n_bad += 1
    if n_bad == 0:
      return z_sup, f_sup, w_sup

    for k in range(pol.size):
      if np.abs(res[k]) < tol_F or np.isnan(res[k]):
        ix_p = 0
        d_min = np.abs(z_all_c[0] - pol[k].real)
        for j in range(1, z_all_c.size):
          d = np.abs(z_all_c[j] - pol[k].real)
          if d < d_min:
            d_min = d
            ix_p = j
        z_all_c = _shift_delete_1d(z_all_c, ix_p)
        f_all_c = _shift_delete_1d(f_all_c, ix_p)

    z_sup, f_sup, w_sup = _aaa_core(
        z_all_c, f_all_c, tol, max_t, z_required, use_lebesgue,
        complex_weights, kappa)

  return z_sup, f_sup, w_sup


def aaa_real(z, f, tol=1e-13, max_terms=None, z_required=None,
             use_lebesgue=False, clean_up=False, tol_F=1e-10,
             tol_w=0.0):
  r"""AAA rational approximation (real weights), with optional cleanup.

  Parameters
  ----------
  z : NDArray
      Sample points.
  f : NDArray
      Function values at sample points.
  tol : float, optional
      Convergence tolerance. Default 1e-13.
  max_terms : int, optional
      Maximum number of support points. Default z.size - 1.
  z_required : NDArray, optional
      Points that must be included as support points. Default None.
  use_lebesgue : bool, optional
      If True, use Lebesgue function weighting for error estimation.
      Default False.
  clean_up : bool, optional
      If True, iteratively remove Froissart doublets. Default False.
  tol_F : float, optional
      Residue threshold for Froissart doublet detection. Default 1e-10.
  tol_w : float, optional
      Weight threshold: drop support nodes with |w| <= tol_w. Default 0.0.

  Returns
  -------
  z_sup : NDArray
      Selected support points.
  f_sup : NDArray
      Function values at support points.
  w_sup : NDArray
      Barycentric weights.
  """
  if max_terms is None:
    max_terms = z.size - 1
  max_t = max_terms

  z_sup, f_sup, w_sup = _aaa_real_core(
      z, f, tol, max_t, z_required, use_lebesgue)
  if not clean_up:
    return z_sup, f_sup, w_sup

  return _cleanup_loop_real(
      z_sup, f_sup, w_sup, z, f,
      tol, max_t, z_required, use_lebesgue,
      tol_F, tol_w)


def aaa(z, f, tol=1e-13, max_terms=None, z_required=None,
        use_lebesgue=False, complex_weights=False, kappa=None,
        clean_up=False, tol_F=1e-10, tol_w=0.0):
  r"""AAA rational approximation (complex dtype), with optional cleanup.

  Parameters
  ----------
  z : NDArray
      Sample points.
  f : NDArray
      Function values at sample points.
  tol : float, optional
      Convergence tolerance. Default 1e-13.
  max_terms : int, optional
      Maximum number of support points. Default z.size - 1.
  z_required : NDArray, optional
      Points that must be included as support points. Default None.
  use_lebesgue : bool, optional
      If True, use Lebesgue function weighting for error estimation.
      Default False.
  complex_weights : bool, optional
      Unused. Accepted for API compatibility; all weights are complex128
      dtype. Default False.
  kappa : float, optional
      Complex weight perturbation exponent. When set, adds a small
      imaginary component to the weight vector for improved stability
      on real-valued functions with near-singular behavior.
      Default None (no perturbation).
  clean_up : bool, optional
      If True, iteratively remove Froissart doublets. Default False.
  tol_F : float, optional
      Residue threshold for Froissart doublet detection. Default 1e-10.
  tol_w : float, optional
      Weight threshold: drop support nodes with |w| <= tol_w. Default 0.0.

  Returns
  -------
  z_sup : NDArray
      Selected support points.
  f_sup : NDArray
      Function values at support points.
  w_sup : NDArray
      Barycentric weights.
  """
  if max_terms is None:
    max_terms = z.size - 1
  max_t = max_terms

  z_sup, f_sup, w_sup = _aaa_core(
      z, f, tol, max_t, z_required, use_lebesgue,
      complex_weights, kappa)
  if not clean_up:
    return z_sup, f_sup, w_sup

  return _cleanup_loop_complex(
      z_sup, f_sup, w_sup, z, f,
      tol, max_t, z_required, use_lebesgue,
      complex_weights, kappa, tol_F, tol_w)


###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  eval_rat = JITI(eval_rat)
  poles_residues = JIT(poles_residues)
  _shift_delete_1d = JIT(_shift_delete_1d)
  _aaa_real_core = JIT(_aaa_real_core)
  _aaa_core = JIT(_aaa_core)
  _drop_small_weights = JIT(_drop_small_weights)
  _cleanup_loop_real = JIT(_cleanup_loop_real)
  _cleanup_loop_complex = JIT(_cleanup_loop_complex)
  aaa_real = JIT(aaa_real)
  aaa = JIT(aaa)

  # Warm-up
  _wx = np.array([0.0, 1.0, 2.0], dtype=np.float64)
  _wf = np.array([0.0, 1.0, 4.0], dtype=np.float64)
  _aaa_real_core(_wx, _wf, 1.0, 2, None, False)
  aaa_real(_wx, _wf, tol=1.0, max_terms=2)

#
# :D
#
