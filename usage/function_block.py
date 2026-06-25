"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Piecewise AAA rational approximation with automatic domain fission.

Usage example -- not part of core pynalgo.  Demonstrates FunctionBlock.
"""
import numpy as np
import matplotlib.pyplot as plt
from pynalgo.rational import aaa_real, eval_rat


def _lin_nodes(N, a, b):
  return np.linspace(a, b, N)


class FunctionBlock:
  """Piecewise AAA rational function approximation with domain fission.

  Parameters
  ----------
  T : callable
      Function T(x_array) -> array to approximate.
  x : NDArray
      Domain grid (sorted).
  tol_res : float
      Absolute error tolerance triggering subdivision.
  tol_aaa : float
      AAA tolerance passed to aaa_real.
  max_terms : int
      Max AAA support nodes per block.
  min_terms : int or None
      If set, subdivide when block exceeds this many support nodes.
  clean_up : bool
      Enable Froissart doublet removal in AAA.
  tol_F : float
      Doublet residue threshold.
  tol_w : float
      Weight threshold for doublet removal.
  max_subdiv : int
      Maximum recursion depth for subdivision.
  extend_nodes : int
      Extend subdomain grids by N overlap nodes.
  fission_nodes : NDArray or None
      Explicit break points for domain splitting.
  """

  def __init__(self, T, x, tol_res=1e-13, tol_aaa=1e-13,
               max_terms=32, min_terms=None,
               clean_up=False, tol_F=1e-10, tol_w=0.0,
               max_subdiv=10, extend_nodes=0,
               fission_nodes=None,
               _cur_subdiv=0, _parent=None,
               _is_edge_L=True, _is_edge_R=True):
    if fission_nodes is not None:
      fission_nodes = np.sort(np.asarray(fission_nodes).ravel())

    self._parent = _parent
    self._extend_nodes = extend_nodes
    self._is_edge_L = _is_edge_L
    self._is_edge_R = _is_edge_R

    T_x = T(x)
    xx = _lin_nodes(x.size, x[0], x[-1])
    xx_E = xx.copy()
    if extend_nodes > 0:
      dx = (xx[-1] - xx[0]) / (xx.size - 1)
      ext_left = xx[0] - dx * np.arange(extend_nodes, 0, -1)
      ext_right = xx[-1] + dx * np.arange(1, extend_nodes + 1)
      xx_E = np.concatenate([ext_left, xx, ext_right])

      if fission_nodes is not None:
        for fn in fission_nodes:
          if ext_left.size > 0 and ext_left[0] <= fn <= xx[0]:
            xx_E = xx_E[extend_nodes:]
          if ext_right.size > 0 and xx[-1] <= fn <= ext_right[-1]:
            xx_E = xx_E[:-extend_nodes]

    self._xx_E = xx_E
    T_xx_E = T(xx_E)

    req = np.array([xx[0], xx[-1]])
    m_terms = min(max_terms, xx_E.size - 1)
    z_sup, f_sup, w_sup = aaa_real(
        xx_E, T_xx_E, tol=tol_aaa, max_terms=m_terms,
        z_required=req, clean_up=clean_up,
        tol_F=tol_F, tol_w=tol_w)

    self._z_sup = z_sup
    self._f_sup = f_sup
    self._w_sup = w_sup
    self._x = x
    self._T_x = T_x

    self._r_a = eval_rat(z_sup, f_sup, w_sup, x)
    self._norm = np.linalg.norm(T_x, np.inf)
    rel_tol = tol_res * max(1.0, self._norm)

    self._rel_err = np.abs(T_x - self._r_a)
    max_err = np.max(self._rel_err)

    n_sup = z_sup.size
    should_split = (max_err > rel_tol)
    if min_terms is not None and n_sup > min_terms:
      should_split = True
    if should_split and _cur_subdiv < max_subdiv:
      N_x = x.size
      ix_half = N_x // 2

      x_L_a = x[0]
      x_L_b = x[ix_half]
      x_R_a = x[ix_half]
      x_R_b = x[-1]

      if fission_nodes is not None:
        for fn in fission_nodes:
          if x_L_a < fn < x_R_b:
            x_L_b = fn
            x_R_a = fn
            break

      x_L = _lin_nodes(N_x, x_L_a, x_L_b)
      x_R = _lin_nodes(N_x, x_R_a, x_R_b)

      self._child_L = FunctionBlock(
          T, x_L, tol_res=tol_res, tol_aaa=tol_aaa,
          max_terms=max_terms, min_terms=min_terms,
          clean_up=clean_up, tol_F=tol_F, tol_w=tol_w,
          max_subdiv=max_subdiv, extend_nodes=extend_nodes,
          fission_nodes=fission_nodes,
          _cur_subdiv=_cur_subdiv + 1, _parent=self,
          _is_edge_L=_is_edge_L, _is_edge_R=False)
      self._child_R = FunctionBlock(
          T, x_R, tol_res=tol_res, tol_aaa=tol_aaa,
          max_terms=max_terms, min_terms=min_terms,
          clean_up=clean_up, tol_F=tol_F, tol_w=tol_w,
          max_subdiv=max_subdiv, extend_nodes=extend_nodes,
          fission_nodes=fission_nodes,
          _cur_subdiv=_cur_subdiv + 1, _parent=self,
          _is_edge_L=False, _is_edge_R=_is_edge_R)
    else:
      self._child_L = None
      self._child_R = None

  def tree_num_leaves(self):
    if self._child_L is None:
      return 1
    return self._child_L.tree_num_leaves() + self._child_R.tree_num_leaves()

  def tree_support_nodes(self):
    if self._child_L is None:
      return self._z_sup
    return np.unique(np.concatenate([
        self._child_L.tree_support_nodes(),
        self._child_R.tree_support_nodes()]))

  def tree_refined_grid(self):
    if self._child_L is None:
      return self._x
    return np.unique(np.concatenate([
        self._child_L.tree_refined_grid(),
        self._child_R.tree_refined_grid()]))

  def tree_block_boundaries(self):
    if self._child_L is None:
      return np.array([self._x[0], self._x[-1]])
    return np.unique(np.concatenate([
        self._child_L.tree_block_boundaries(),
        self._child_R.tree_block_boundaries()]))

  def _mask_in_range(self, x):
    x_mi = self._x[0]
    x_ma = self._x[-1]
    if self._is_edge_L:
      x_mi = x[0]
    if self._is_edge_R:
      x_ma = x[-1]
    return (x >= x_mi) & (x <= x_ma)

  def _no_child_blocks(self):
    if self._child_L is None:
      return [self]
    return (self._child_L._no_child_blocks()
            + self._child_R._no_child_blocks())

  def tree_max_err_block(self):
    blocks = self._no_child_blocks()
    errs = [np.max(b._rel_err) for b in blocks]
    return blocks[int(np.argmax(errs))]

  def __call__(self, x):
    x = np.asarray(x).ravel()
    if self._child_L is None:
      mask = self._mask_in_range(x)
      x_f = x[mask]
      if x_f.size == 0:
        return np.array([], dtype=self._T_x.dtype)
      return eval_rat(self._z_sup, self._f_sup, self._w_sup, x_f)

    mask_L = self._child_L._mask_in_range(x)
    mask_R = self._child_R._mask_in_range(x)
    result = np.empty(x.size, dtype=self._T_x.dtype)
    x_L = x[mask_L]
    if x_L.size > 0:
      result[mask_L] = self._child_L(x_L)
    mask_R_only = mask_R & ~mask_L
    x_R = x[mask_R_only]
    if x_R.size > 0:
      result[mask_R_only] = self._child_R(x_R)
    mask_none = ~(mask_L | mask_R)
    if np.any(mask_none):
      result[mask_none] = np.nan
    return result


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import os

  x = np.linspace(-5, 5, 300)
  fb = FunctionBlock(lambda x: np.tanh(30 * x), x,
                     tol_res=1e-12, max_subdiv=8)
  xt = np.linspace(-4, 4, 1000)
  err = np.abs(fb(xt) - np.tanh(30 * xt))
  print(f'leaves: {fb.tree_num_leaves()}')
  print(f'max error: {np.max(err):.2e}')

  plt.semilogy(xt, err)
  for b in fb.tree_block_boundaries():
    plt.axvline(b, color='red', lw=0.5)
  plt.xlabel('x')
  plt.ylabel('|error|')
  plt.title('FunctionBlock: tanh(30x) piecewise AAA error')
  plt.tight_layout()
  out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
  os.makedirs(out_dir, exist_ok=True)
  plt.savefig(os.path.join(out_dir, 'function_block_demo.png'), dpi=150)
  plt.close()
