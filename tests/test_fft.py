"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.fft: JIT-compiled FFT against numpy.fft reference.
"""

import numpy as np
from numpy.testing import assert_allclose

from pynalgo.fft import fft, ifft

_EPS = 1e-8


# =============================================================================
# 1D -- power-of-2
# =============================================================================

def test_fft_pow2_small():
  for N in [2, 4, 8]:
    x = np.arange(N, dtype=complex)
    assert_allclose(fft(x), np.fft.fft(x), atol=_EPS)


def test_fft_pow2_large():
  for N in [16, 32, 64, 128, 256]:
    x = np.arange(N, dtype=complex)
    assert_allclose(fft(x), np.fft.fft(x), atol=_EPS)


# =============================================================================
# 1D -- unrolled (N=1, 3, 4)
# =============================================================================

def test_fft_unrolled():
  for N in [1, 3, 4]:
    x = np.arange(N, dtype=complex)
    assert_allclose(fft(x), np.fft.fft(x), atol=_EPS)


# =============================================================================
# 1D -- composite
# =============================================================================

def test_fft_composite():
  for N in [6, 10, 12, 20, 30, 60, 84, 100]:
    x = np.arange(N, dtype=complex)
    assert_allclose(fft(x), np.fft.fft(x), atol=_EPS)


# =============================================================================
# 1D -- prime (Bluestein)
# =============================================================================

def test_fft_prime():
  for N in [7, 11, 13, 17, 19, 23, 29, 37, 41, 43]:
    x = np.arange(N, dtype=complex)
    assert_allclose(fft(x), np.fft.fft(x), atol=_EPS)


# =============================================================================
# 1D -- identity
# =============================================================================

def test_fft_identity():
  for N in [2, 3, 4, 8, 12, 17, 32, 100]:
    x = np.arange(N, dtype=complex)
    assert_allclose(ifft(fft(x)), x, atol=_EPS)


# =============================================================================
# 1D -- linearity
# =============================================================================

def test_fft_linearity():
  N = 16
  a = 2.0 + 1j
  b = -1.0 + 3j
  x = np.arange(N, dtype=complex)
  y = np.ones(N, dtype=complex)
  lhs = fft(a * x + b * y)
  rhs = a * fft(x) + b * fft(y)
  assert_allclose(lhs, rhs, atol=_EPS)


# =============================================================================
# 2D axis
# =============================================================================

def test_fft_2d_axis0():
  x = np.arange(12, dtype=complex).reshape(3, 4)
  assert_allclose(fft(x, axis=0), np.fft.fft(x, axis=0), atol=_EPS)


def test_fft_2d_axis1():
  x = np.arange(12, dtype=complex).reshape(3, 4)
  assert_allclose(fft(x, axis=1), np.fft.fft(x, axis=1), atol=_EPS)


def test_fft_2d_negative_axis():
  x = np.arange(12, dtype=complex).reshape(3, 4)
  assert_allclose(fft(x, axis=-1), np.fft.fft(x, axis=-1), atol=_EPS)
  assert_allclose(fft(x, axis=-2), np.fft.fft(x, axis=-2), atol=_EPS)


# =============================================================================
# 3D axis
# =============================================================================

def test_fft_3d_axis0():
  x = np.arange(24, dtype=complex).reshape(2, 3, 4)
  assert_allclose(fft(x, axis=0), np.fft.fft(x, axis=0), atol=_EPS)


def test_fft_3d_axis1():
  x = np.arange(24, dtype=complex).reshape(2, 3, 4)
  assert_allclose(fft(x, axis=1), np.fft.fft(x, axis=1), atol=_EPS)


def test_fft_3d_axis2():
  x = np.arange(24, dtype=complex).reshape(2, 3, 4)
  assert_allclose(fft(x, axis=2), np.fft.fft(x, axis=2), atol=_EPS)


# =============================================================================
# 2D/3D identity
# =============================================================================

def test_fft_2d_identity():
  x = np.arange(12, dtype=complex).reshape(3, 4)
  assert_allclose(ifft(fft(x, axis=0), axis=0), x, atol=_EPS)
  assert_allclose(ifft(fft(x, axis=1), axis=1), x, atol=_EPS)


def test_fft_3d_identity():
  x = np.arange(24, dtype=complex).reshape(2, 3, 4)
  for ax in [0, 1, 2]:
    assert_allclose(ifft(fft(x, axis=ax), axis=ax), x, atol=_EPS)


# =============================================================================
# 4D axis (fallback path)
# =============================================================================

def test_fft_4d():
  x = np.arange(120, dtype=complex).reshape(2, 3, 4, 5)
  assert_allclose(fft(x, axis=0), np.fft.fft(x, axis=0), atol=_EPS)
  assert_allclose(fft(x, axis=1), np.fft.fft(x, axis=1), atol=_EPS)
  assert_allclose(fft(x, axis=2), np.fft.fft(x, axis=2), atol=_EPS)
  assert_allclose(fft(x, axis=3), np.fft.fft(x, axis=3), atol=_EPS)
  # Negative axis
  assert_allclose(fft(x, axis=-1), np.fft.fft(x, axis=-1), atol=_EPS)
  assert_allclose(fft(x, axis=-2), np.fft.fft(x, axis=-2), atol=_EPS)


def test_fft_4d_identity():
  x = np.arange(120, dtype=complex).reshape(2, 3, 4, 5)
  for ax in [0, 1, 2, 3]:
    assert_allclose(ifft(fft(x, axis=ax), axis=ax), x, atol=_EPS)
