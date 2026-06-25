"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: JIT-compiled DCT-I through DCT-IV kernels.
           FFT-based: each type is computed via symmetric
           extension + pynalgo.fft.  Supports unscaled and
           orthonormal normalization (matching scipy.fftpack).

All functions are Numba JIT-safe.
"""
from numpy import (empty, exp, float64, pi, sqrt, zeros)
from numpy import complex128 as c128

from pynalgo.common_tools import (JIT, TYPE_CHECKING)
from pynalgo.fft import fft

########################################################################
# DCT-I  (N+1 points, whole-sample even symmetry, 2N-point FFT)
########################################################################

def dct1(x, norm=None):
  """DCT-I via 2N-point FFT of even extension.

  Parameters
  ----------
  x : NDArray[float64]
    1D input array of length N+1.
  norm : str or None
    None (default): unscaled.
    'ortho': orthonormal.

  Returns
  -------
  y : NDArray[float64]
    DCT-I of x, same shape.
  """
  N = x.shape[0] - 1
  M = 2 * N
  ortho = (norm == 'ortho')

  z = zeros(M, dtype=c128)

  if ortho:
    # Pre-scale input: x_scaled[n] = x[n] * c_n / v_n
    # c_n = {1/sqrt(2), 1, ..., 1, 1/sqrt(2)}
    # v_n = {1, 2, ..., 2, 1}
    sf = 1.0 / sqrt(2.0)
    z[0] = x[0] * sf + 0j
    z[N] = x[N] * sf + 0j
    for n in range(1, N):
      z[n] = x[n] * 0.5 + 0j
    for n in range(N + 1, M):
      z[n] = x[M - n] * 0.5 + 0j
  else:
    for n in range(N + 1):
      z[n] = x[n] + 0j
    for n in range(N + 1, M):
      z[n] = x[M - n] + 0j

  F = fft(z)
  out = empty(N + 1, dtype=float64)
  for k in range(N + 1):
    out[k] = F[k].real

  if ortho:
    fN = float(N)
    s_edge = 1.0 / sqrt(fN)
    s_int = sqrt(2.0 / fN)
    out[0] *= s_edge
    out[N] *= s_edge
    for k in range(1, N):
      out[k] *= s_int

  return out


########################################################################
# DCT-II  (N points, 2N-point FFT + post-twiddle)
########################################################################

def dct2(x, norm=None):
  """DCT-II via 2N-point FFT with zero-padding and post-twiddle.

  Parameters
  ----------
  x : NDArray[float64]
    1D input array of length N.
  norm : str or None
    None (default): unscaled.
    'ortho': orthonormal.

  Returns
  -------
  y : NDArray[float64]
    DCT-II of x, same shape.
  """
  N = x.shape[0]
  M = 2 * N
  z = zeros(M, dtype=c128)

  for n in range(N):
    z[n] = x[n] + 0j

  F = fft(z)
  out = empty(N, dtype=float64)
  for k in range(N):
    tw = exp(-1j * pi * k / (2.0 * N))
    out[k] = 2.0 * (F[k] * tw).real

  if norm == 'ortho':
    fN = float(N)
    out[0] *= 1.0 / (2.0 * sqrt(fN))
    for k in range(1, N):
      out[k] *= 1.0 / sqrt(2.0 * fN)

  return out


########################################################################
# DCT-III  (N points, 4N-point FFT)
#
# Unscaled:  X[k] = 2*Re(F[2k+1]) - x[0]
#             where x is zero-padded to 4N.
# Ortho:     Input pre-scaled by DCT-II ortho weights d2.
#             The unscaled formula is then applied with a boundary
#             cancellation: -x0_scaled + extra = 0, giving
#             X[k] = 2*Re(F[2k+1]) where F is FFT of pre-scaled input.
########################################################################

def dct3(x, norm=None):
  r"""DCT-III via 4N-point FFT.

  .. math::

      X_k = 2\,\mathrm{Re}(F_{2k+1}) - x_0^{\mathrm{scaled}}
      \quad [+ d^{(2)}_0 x_0 \ \text{for ortho}]

  where :math:`F = \mathrm{fft}(x^{\mathrm{scaled}})` padded to :math:`4N`.

  Parameters
  ----------
  x : NDArray[float64]
    1D input array of length N.
  norm : str or None
    None (default): unscaled.
    'ortho': orthonormal.

  Returns
  -------
  y : NDArray[float64]
    DCT-III of x, same shape.
  """
  N = x.shape[0]
  M = 4 * N
  ortho = (norm == 'ortho')

  z = zeros(M, dtype=c128)

  if ortho:
    # Pre-scale input by DCT-II ortho weights d2
    fN = float(N)
    d2_0 = 1.0 / (2.0 * sqrt(fN))
    d2_k = 1.0 / sqrt(2.0 * fN)
    extra = d2_0 * x[0]            # boundary compensation term
    for n in range(N):
      s = d2_0 if n == 0 else d2_k
      z[n] = x[n] * s + 0j
    x0_scaled = x[0] * d2_0
  else:
    extra = 0.0
    for n in range(N):
      z[n] = x[n] + 0j
    x0_scaled = x[0]

  F = fft(z)
  out = empty(N, dtype=float64)
  for k in range(N):
    out[k] = 2.0 * F[2 * k + 1].real - x0_scaled + extra

  return out


########################################################################
# DCT-IV  (N points, 4N-point FFT + post-twiddle)
########################################################################

def dct4(x, norm=None):
  r"""DCT-IV via 4N-point FFT with post-twiddle.

  .. math::

      X_k = 2\,\mathrm{Re}\left(
          e^{-i\pi(k+0.5)/(2N)} F_{2k+1}
      \right)

  where :math:`F = \mathrm{fft}(x)` padded to :math:`4N`.

  Parameters
  ----------
  x : NDArray[float64]
    1D input array of length N.
  norm : str or None
    None (default): unscaled.
    'ortho': orthonormal.

  Returns
  -------
  y : NDArray[float64]
    DCT-IV of x, same shape.
  """
  N = x.shape[0]
  M = 4 * N
  z = zeros(M, dtype=c128)

  for n in range(N):
    z[n] = x[n] + 0j

  F = fft(z)
  out = empty(N, dtype=float64)
  for k in range(N):
    tw = exp(-1j * pi * (k + 0.5) / (2.0 * N))
    out[k] = 2.0 * (F[2 * k + 1] * tw).real

  if norm == 'ortho':
    s = 1.0 / sqrt(2.0 * float(N))
    for k in range(N):
      out[k] *= s

  return out


########################################################################
# JIT if not type-checking as required
########################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  dct1 = JIT(dct1)
  dct2 = JIT(dct2)
  dct3 = JIT(dct3)
  dct4 = JIT(dct4)

#
# :D
#
