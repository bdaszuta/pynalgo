"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: JIT-compiled FFT: Cooley-Tukey, radix-2 DIT, Bluestein chirp-Z.
"""



from numpy import (empty, int64)
from numpy import complex128 as c128

from pynalgo.common_tools import (JITG, TYPE_CHECKING)
from pynalgo.fft._kernels import (_build_plan, _fft_exec)


# ---------------------------------------------------------------------------
# fft / ifft via JITG (per-ndim specialization at JIT-compilation time).
#
# JITG uses @overload + @njit under the hood, providing per-ndim
# specialization callable from both Python and nopython contexts.
# ---------------------------------------------------------------------------


if TYPE_CHECKING:
  def fft(x, axis=-1):
    """
    Discrete Fourier Transform via Cooley-Tukey / radix-2 /
    Bluestein decomposition.  JIT-compiled, fully nopython.

    Parameters
    ----------
    x : NDArray[c128]
      Input array.
    axis : int
      Axis along which to compute the FFT (default -1).

    Returns
    -------
    y : NDArray[c128]
      Transformed array.
    """
  def ifft(x, axis=-1):
    """
    Inverse FFT: ifft(x) = conj(fft(conj(x))) / N.

    Parameters
    ----------
    x : NDArray[c128]
      Input array.
    axis : int
      Axis along which to compute the iFFT (default -1).

    Returns
    -------
    y : NDArray[c128]
      Inverse transformed array.
    """
  reveal_locals()  # noqa: F821

else:
  @JITG
  def fft(x, axis=-1):
    """
    Discrete Fourier Transform via Cooley-Tukey / radix-2 /
    Bluestein decomposition.  JIT-compiled, fully nopython.

    Parameters
    ----------
    x : NDArray[c128]
      Input array.
    axis : int
      Axis along which to compute the FFT (default -1).

    Returns
    -------
    y : NDArray[c128]
      Transformed array.
    """
    nd = x.ndim

    if nd == 1:
      def _impl(x, axis=-1):
        n = x.shape[0]
        plan = _build_plan(n)
        x_w = empty((1, n), dtype=c128)
        for i in range(n):
          x_w[0, i] = x[i]
        _fft_exec(x_w, plan)
        out = empty(n, dtype=c128)
        for i in range(n):
          out[i] = x_w[0, i]
        return out
      return _impl

    if nd == 2:
      def _impl(x, axis=-1):
        nd2 = 2
        ax = axis if axis >= 0 else nd2 + axis
        m, n_orig = x.shape
        if ax == 0:
          batch = n_orig
          n_fft = m
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(m):
            for j in range(n_orig):
              x_2d[j, i] = x[i, j]
        else:
          batch = m
          n_fft = n_orig
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(m):
            for j in range(n_orig):
              x_2d[i, j] = x[i, j]
        plan = _build_plan(n_fft)
        _fft_exec(x_2d, plan)
        if ax == 0:
          out = empty((m, n_orig), dtype=c128)
          for i in range(m):
            for j in range(n_orig):
              out[i, j] = x_2d[j, i]
          return out
        return x_2d
      return _impl

    if nd == 3:
      def _impl(x, axis=-1):
        nd3 = 3
        ax = axis if axis >= 0 else nd3 + axis
        s0, s1, s2 = x.shape
        if ax == 0:
          batch = s1 * s2
          n_fft = s0
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                x_2d[j * s2 + k, i] = x[i, j, k]
        elif ax == 1:
          batch = s0 * s2
          n_fft = s1
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                x_2d[i * s2 + k, j] = x[i, j, k]
        else:
          batch = s0 * s1
          n_fft = s2
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                x_2d[i * s1 + j, k] = x[i, j, k]
        plan = _build_plan(n_fft)
        _fft_exec(x_2d, plan)
        if ax == 0:
          out = empty((s0, s1, s2), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                out[i, j, k] = x_2d[j * s2 + k, i]
          return out
        elif ax == 1:
          out = empty((s0, s1, s2), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                out[i, j, k] = x_2d[i * s2 + k, j]
          return out
        else:
          out = empty((s0, s1, s2), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                out[i, j, k] = x_2d[i * s1 + j, k]
          return out
      return _impl

    if nd == 4:
      def _impl(x, axis=-1):
        nd4 = 4
        ax = axis if axis >= 0 else nd4 + axis
        s0, s1, s2, s3 = x.shape
        if ax == 0:
          batch = s1 * s2 * s3
          n_fft = s0
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  x_2d[j*s2*s3 + k*s3 + ell, i] = x[i, j, k, ell]
        elif ax == 1:
          batch = s0 * s2 * s3
          n_fft = s1
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  x_2d[i*s2*s3 + k*s3 + ell, j] = x[i, j, k, ell]
        elif ax == 2:
          batch = s0 * s1 * s3
          n_fft = s2
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  x_2d[i*s1*s3 + j*s3 + ell, k] = x[i, j, k, ell]
        else:
          batch = s0 * s1 * s2
          n_fft = s3
          x_2d = empty((batch, n_fft), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  x_2d[i*s1*s2 + j*s2 + k, ell] = x[i, j, k, ell]
        plan = _build_plan(n_fft)
        _fft_exec(x_2d, plan)
        if ax == 0:
          out = empty((s0, s1, s2, s3), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  out[i, j, k, ell] = x_2d[j*s2*s3 + k*s3 + ell, i]
          return out
        elif ax == 1:
          out = empty((s0, s1, s2, s3), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  out[i, j, k, ell] = x_2d[i*s2*s3 + k*s3 + ell, j]
          return out
        elif ax == 2:
          out = empty((s0, s1, s2, s3), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  out[i, j, k, ell] = x_2d[i*s1*s3 + j*s3 + ell, k]
          return out
        else:
          out = empty((s0, s1, s2, s3), dtype=c128)
          for i in range(s0):
            for j in range(s1):
              for k in range(s2):
                for ell in range(s3):
                  out[i, j, k, ell] = x_2d[i*s1*s2 + j*s2 + k, ell]
          return out
      return _impl

    # nd >= 5: compute strides and use flat indexing
    def _impl(x, axis=-1):
      ndn = len(x.shape)
      ax = axis if axis >= 0 else ndn + axis
      n_fft = x.shape[ax]

      src_stride = empty(ndn, dtype=int64)
      acc = int64(1)
      for d in range(ndn - 1, -1, -1):
        src_stride[d] = acc
        acc *= x.shape[d]

      batch = int64(1)
      for d in range(ndn):
        if d != ax:
          batch *= x.shape[d]

      x_2d = empty((batch, n_fft), dtype=c128)
      n_total = acc

      for flat in range(n_total):
        rem = flat
        src_multi = empty(ndn, dtype=int64)
        for d in range(ndn):
          src_multi[d] = rem // src_stride[d]
          rem = rem % src_stride[d]
        dst_batch = int64(0)
        dst_stride = int64(1)
        for d in range(ndn - 1, -1, -1):
          if d != ax:
            dst_batch += src_multi[d] * dst_stride
            dst_stride *= x.shape[d]
        dst_col = src_multi[ax]
        x_2d[dst_batch, dst_col] = x.flat[flat]

      plan = _build_plan(n_fft)
      _fft_exec(x_2d, plan)

      out = empty(x.shape, dtype=c128)
      for flat in range(n_total):
        rem = flat
        src_multi = empty(ndn, dtype=int64)
        for d in range(ndn):
          src_multi[d] = rem // src_stride[d]
          rem = rem % src_stride[d]
        dst_batch = int64(0)
        dst_stride = int64(1)
        for d in range(ndn - 1, -1, -1):
          if d != ax:
            dst_batch += src_multi[d] * dst_stride
            dst_stride *= x.shape[d]
        dst_col = src_multi[ax]
        out.flat[flat] = x_2d[dst_batch, dst_col]

      return out
    return _impl


  @JITG
  def ifft(x, axis=-1):
    """
    Inverse FFT: ifft(x) = conj(fft(conj(x))) / N.

    Parameters
    ----------
    x : NDArray[c128]
      Input array.
    axis : int
      Axis along which to compute the iFFT (default -1).

    Returns
    -------
    y : NDArray[c128]
      Inverse transformed array.
    """
    nd = x.ndim

    if nd == 1:
      def _impl(x, axis=-1):
        n = x.shape[0]
        xc = empty(n, dtype=c128)
        for i in range(n):
          xc[i] = x[i].conjugate()
        yc = fft(xc, axis)
        for i in range(n):
          yc[i] = yc[i].conjugate() / float(n)
        return yc
      return _impl

    if nd == 2:
      def _impl(x, axis=-1):
        nd2 = 2
        ax = axis if axis >= 0 else nd2 + axis
        n = x.shape[ax]
        xc = empty(x.shape, dtype=c128)
        for i in range(x.shape[0]):
          for j in range(x.shape[1]):
            xc[i, j] = x[i, j].conjugate()
        yc = fft(xc, axis)
        for i in range(yc.shape[0]):
          for j in range(yc.shape[1]):
            yc[i, j] = yc[i, j].conjugate() / float(n)
        return yc
      return _impl

    if nd == 3:
      def _impl(x, axis=-1):
        nd3 = 3
        ax = axis if axis >= 0 else nd3 + axis
        n = x.shape[ax]
        xc = empty(x.shape, dtype=c128)
        for i in range(x.shape[0]):
          for j in range(x.shape[1]):
            for k in range(x.shape[2]):
              xc[i, j, k] = x[i, j, k].conjugate()
        yc = fft(xc, axis)
        for i in range(yc.shape[0]):
          for j in range(yc.shape[1]):
            for k in range(yc.shape[2]):
              yc[i, j, k] = yc[i, j, k].conjugate() / float(n)
        return yc
      return _impl

    # nd >= 4
    def _impl(x, axis=-1):
      ndn = len(x.shape)
      ax = axis if axis >= 0 else ndn + axis
      n = x.shape[ax]
      xc = empty(x.shape, dtype=c128)
      n_flat = int64(1)
      for d in range(ndn):
        n_flat *= x.shape[d]
      for i in range(n_flat):
        xc.flat[i] = x.flat[i].conjugate()
      yc = fft(xc, axis)
      for i in range(n_flat):
        yc.flat[i] = yc.flat[i].conjugate() / float(n)
      return yc
    return _impl


def build_plan(N):
  """
  Build a FFT plan for transform of size N.
  Returns plan arrays for use with _fft_exec.

  Parameters
  ----------
  N : int
    Transform size.

  Returns
  -------
  plan : tuple of NDArray
    Plan arrays (type, N, p1, p2, c1, c2, tw_data, tw_start).
  """
  return _build_plan(N)


__all__ = [
  'build_plan',
  'fft',
  'ifft',
]

#
# :D
#
