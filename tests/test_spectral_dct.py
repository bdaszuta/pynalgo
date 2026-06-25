"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.spectral DCT-I through DCT-IV against scipy.fftpack.
"""

import numpy as np
from numpy.testing import assert_allclose

from scipy.fftpack import dct as scipy_dct

from pynalgo.spectral import dct1, dct2, dct3, dct4

_EPS = 1e-12

# =============================================================================
# DCT-I  (N+1 points, type=1)
# =============================================================================

def test_dct1_vs_scipy():
  """Unscaled DCT-I matches scipy.fftpack."""
  for N in [1, 2, 3, 7, 8, 15, 16, 32]:
    x = np.arange(N + 1, dtype=float)
    y = dct1(x, norm=None)
    y_ref = scipy_dct(x, type=1, norm=None)
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct1_ortho_vs_scipy():
  """Orthonormal DCT-I matches scipy.fftpack."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N + 1, dtype=float)
    y = dct1(x, norm='ortho')
    y_ref = scipy_dct(x, type=1, norm='ortho')
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct1_dct1_roundtrip_unscaled():
  """DCT-I is self-inverse up to 2N scaling."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N + 1, dtype=float)
    y = dct1(dct1(x, norm=None), norm=None)
    assert_allclose(y, 2.0 * N * x, atol=_EPS)


def test_dct1_dct1_roundtrip_ortho():
  """DCT-I is self-inverse (orthonormal)."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N + 1, dtype=float)
    y = dct1(dct1(x, norm='ortho'), norm='ortho')
    assert_allclose(y, x, atol=_EPS)


# =============================================================================
# DCT-II  (type=2)
# =============================================================================

def test_dct2_vs_scipy():
  for N in [1, 2, 3, 7, 8, 15, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct2(x, norm=None)
    y_ref = scipy_dct(x, type=2, norm=None)
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct2_ortho_vs_scipy():
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct2(x, norm='ortho')
    y_ref = scipy_dct(x, type=2, norm='ortho')
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct2_constant():
  """DCT-II of constant vector: only DC term is nonzero."""
  for N in [4, 8, 16]:
    x = np.ones(N, dtype=float)
    y = dct2(x, norm=None)
    # DC component should be 2*sum(x) = 2*N
    assert_allclose(y[0], 2.0 * N, atol=_EPS)
    assert_allclose(y[1:], 0.0, atol=_EPS)


# =============================================================================
# DCT-III  (type=3)
# =============================================================================

def test_dct3_vs_scipy():
  for N in [1, 2, 3, 7, 8, 15, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct3(x, norm=None)
    y_ref = scipy_dct(x, type=3, norm=None)
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct3_ortho_vs_scipy():
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct3(x, norm='ortho')
    y_ref = scipy_dct(x, type=3, norm='ortho')
    assert_allclose(y, y_ref, atol=_EPS)


# =============================================================================
# DCT-II / DCT-III roundtrip
# =============================================================================

def test_dct2_dct3_roundtrip_unscaled():
  """dct3(dct2(x)) = 2*N * x (unscaled convention)."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct3(dct2(x, norm=None), norm=None)
    assert_allclose(y, 2.0 * N * x, atol=_EPS)


def test_dct2_dct3_roundtrip_ortho():
  """dct3(dct2(x)) = x (orthonormal convention)."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct3(dct2(x, norm='ortho'), norm='ortho')
    assert_allclose(y, x, atol=_EPS)


# =============================================================================
# DCT-IV  (type=4)
# =============================================================================

def test_dct4_vs_scipy():
  for N in [1, 2, 3, 7, 8, 15, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct4(x, norm=None)
    y_ref = scipy_dct(x, type=4, norm=None)
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct4_ortho_vs_scipy():
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct4(x, norm='ortho')
    y_ref = scipy_dct(x, type=4, norm='ortho')
    assert_allclose(y, y_ref, atol=_EPS)


def test_dct4_dct4_roundtrip_unscaled():
  """DCT-IV is self-inverse up to 2N scaling."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct4(dct4(x, norm=None), norm=None)
    assert_allclose(y, 2.0 * N * x, atol=_EPS)


def test_dct4_dct4_roundtrip_ortho():
  """DCT-IV is self-inverse (orthonormal)."""
  for N in [1, 2, 3, 7, 8, 16, 32]:
    x = np.arange(N, dtype=float)
    y = dct4(dct4(x, norm='ortho'), norm='ortho')
    assert_allclose(y, x, atol=_EPS)
