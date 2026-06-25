"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: JIT-compiled FFT kernels: naive DFT, radix-2 DIT,
           Cooley-Tukey mixed-radix, Bluestein chirp-Z, plan
           construction, and stack-based plan execution.

All functions are Numba JIT-safe: no numpy.fft, no recursion,
no Python objects, no scipy.

Refs:
  [1] Cooley, J.W. & Tukey, J.W. (1965)
      "An algorithm for the machine calculation of complex
       Fourier series."
      Mathematics of Computation, 19(90):297-301.

  [2] Bluestein, L.I. (1970)
      "A linear filtering approach to the computation of
       discrete Fourier transform."
      IEEE Trans. Audio Electroacoustics, 18(4):451-455.

  [3] Chu, E. & George, A. (1999)
      "Inside the FFT Black Box: Serial and Parallel Fast
       Fourier Transform Algorithms." CRC Press.

  [4] Johnson, S.G. & Frigo, M. (2009)
      "Implementing FFTs in Practice." OpenStax CNX.

  [5] Frigo, M. & Johnson, S.G. (2005)
      "The design and implementation of FFTW3."
      Proc. IEEE, 93(2):216-231.

  [6] https://numericalrecipes.wordpress.com/
"""
from numpy import (arange, array, concatenate, copy, empty,
                   exp, int8, int64, ones, pi,
                   sqrt, zeros)
from numpy import complex128 as c128

from pynalgo.common_tools import (JIT, NDArray, TYPE_CHECKING)
from pynalgo.number_theory import (get_prime_factors,
                                   next_pow2, prime_sqrt_decomp)


_MAX_NODES = 64
_STACK_CAP = 128
_NAIVE_TRUNC = 32


###############################################################################
# Twiddle factor construction
###############################################################################

def _twiddle_radix2(N : int) -> NDArray[c128]:
  r"""
  Radix-2 twiddle: :math:`e^{-2\pi i k/N}` for :math:`k = 0, \ldots, N/2-1`.
  """
  k = arange(N // 2, dtype=int64)
  arg = (-2.0 * pi * 1j) * k / float(N)
  return exp(arg)


def _twiddle_ct(N1 : int, N2 : int) -> NDArray[c128]:
  r"""
  Cooley-Tukey twiddle: :math:`e^{-2\pi i n_1 k_2 / (N_1 N_2)}`.
  Returns shape (N1, N2).
  """
  n1 = arange(N1, dtype=int64).reshape(N1, 1)
  k2 = arange(N2, dtype=int64).reshape(1, N2)
  arg = (-2.0 * pi * 1j) * n1 * k2 / float(N1 * N2)
  return exp(arg)


def _twiddle_bluestein(N : int, M : int) -> NDArray[c128]:
  """
  Bluestein twiddle factors.
  Returns concatenated: tw_A (N,), tw_B (M,), radix2 twiddles (M//2,).
  """
  n_idx = arange(N, dtype=int64)

  # tw_A[n] = exp(-pi*i * n * (n mod 2N) / N)
  # = exp(-pi*i * n * (n % (2*N)) / N)
  arg_A = (-pi * 1j) * n_idx * (n_idx % (2 * N)) / float(N)
  tw_A = exp(arg_A)

  # tw_B: padded to M, values are conj(tw_A) mirrored
  m_idx = arange(M, dtype=int64)
  tw_B = zeros(M, dtype=c128)
  for i in range(M):
    d = m_idx[i]
    if d >= M - d:
      d = M - d
    if d < N:
      tw_B[i] = tw_A[d].conjugate()

  rad2_tw = _twiddle_radix2(M)
  return concatenate((tw_A, tw_B, rad2_tw))


###############################################################################
# Plan construction
###############################################################################

def _is_pow2(n : int) -> int64:
  """1 (as int64) if n is a power of 2; returns 0 for n <= 0."""
  _n = int64(n)
  if _n <= 0:
    return int64(0)
  return int64((_n & (_n - 1)) == 0)


def _is_prime(n : int) -> int64:
  """True if n is prime."""
  _n = int64(n)
  if _n < 2:
    return int64(0)
  # Check small primes directly (n < 32, these are leaf nodes anyway)
  for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
    if _n == p:
      return int64(1)
    if _n % p == 0:
      return int64(0)
  # For larger n, check divisibility up to sqrt(n)
  limit = int64(sqrt(float(_n))) + 1
  d = int64(5)
  while d <= limit:
    if _n % d == 0 or _n % (d + 2) == 0:
      return int64(0)
    d += 6
  return int64(1)


def _balanced_split(N : int) -> NDArray[int64]:
  """
  Split N = N1*N2 using prime_sqrt_decomp for balanced factors.
  Returns array [N1, N2] with N1 <= N2.
  """
  _N = N
  fac = get_prime_factors(_N)
  if fac.size == 0:
    return array([int64(1), _N], dtype=int64)
  A = prime_sqrt_decomp(_N)
  B = _N // A
  return array([A, B], dtype=int64)


def _build_plan(N : int) -> tuple:
  """
  Build a flat post-order execution plan for an N-point FFT.

  Returns
  -------
  nodes_type : NDArray[int8]   -- node type code
  nodes_N    : NDArray[int64]  -- transform size
  nodes_p1   : NDArray[int64]  -- extra param 1
  nodes_p2   : NDArray[int64]  -- extra param 2
  nodes_c1   : NDArray[int64]  -- left child node id (-1 = leaf)
  nodes_c2   : NDArray[int64]  -- right child node id (-1 = leaf)
  tw_data    : NDArray[c128]   -- concatenated twiddle factors
  tw_start   : NDArray[int64]  -- offset into tw_data per node
  """
  _N = N

  # Stack arrays for tree traversal
  bs_n = zeros(_MAX_NODES, dtype=int64)
  bs_pid = -ones(_MAX_NODES, dtype=int64)
  bs_cp = zeros(_MAX_NODES, dtype=int8)

  # ---- Pass 1: count nodes and twiddle elements ----
  n_nodes = 0
  tw_total = int64(0)
  bsp = 0
  bs_n[0] = _N
  bs_pid[0] = -2
  bsp = 1

  while bsp > 0:
    bsp -= 1
    n_val = bs_n[bsp]
    if n_val == 1:
      pass
    elif n_val == 3:
      pass
    elif n_val == 4:
      pass
    elif _is_pow2(n_val):
      tw_total += int64(n_val // 2)
    elif n_val < _NAIVE_TRUNC:
      pass
    elif _is_prime(n_val):
      M = int(1 << next_pow2(n_val))
      tw_total += int64(n_val + M + M // 2)
    else:
      spl = _balanced_split(n_val)
      N1 = spl[0]
      N2 = spl[1]
      tw_total += int64(N1 * N2)
      if bsp + 2 <= _MAX_NODES:
        bs_n[bsp] = N2
        bs_pid[bsp] = -2
        bs_cp[bsp] = 0
        bsp += 1
        bs_n[bsp] = N1
        bs_pid[bsp] = -2
        bs_cp[bsp] = 0
        bsp += 1
    n_nodes += 1

  nodes_type = zeros(_MAX_NODES, dtype=int8)
  nodes_N = zeros(_MAX_NODES, dtype=int64)
  nodes_p1 = zeros(_MAX_NODES, dtype=int64)
  nodes_p2 = zeros(_MAX_NODES, dtype=int64)
  nodes_c1 = -ones(_MAX_NODES, dtype=int64)
  nodes_c2 = -ones(_MAX_NODES, dtype=int64)
  if tw_total > 0:
    tw_data = zeros(tw_total, dtype=c128)
  else:
    tw_data = zeros(1, dtype=c128)
  tw_start = zeros(n_nodes, dtype=int64)
  tw_pos = int64(0)
  node_counter = int64(0)

  # ---- Pass 2: build plan ----
  bsp = 0
  bs_n[0] = _N
  bs_pid[0] = -1
  bs_cp[0] = 0
  bsp = 1

  while bsp > 0:
    bsp -= 1
    n_val = bs_n[bsp]
    pid = bs_pid[bsp]
    cp = bs_cp[bsp]

    # Determine node type
    if n_val == 1:
      tp = int8(0)
    elif n_val == 3:
      tp = int8(1)
    elif n_val == 4:
      tp = int8(2)
    elif _is_pow2(n_val):
      tp = int8(4)
    elif n_val < _NAIVE_TRUNC:
      tp = int8(3)
    elif _is_prime(n_val):
      tp = int8(6)
    else:
      tp = int8(5)

    nid = node_counter
    node_counter += 1
    nodes_type[nid] = tp
    nodes_N[nid] = n_val

    if tp == 4:  # radix2 leaf
      tw = _twiddle_radix2(n_val)
      tw_sz = int64(tw.size)
      tw_start[nid] = tw_pos
      tw_data[tw_pos : tw_pos + tw_sz] = tw
      tw_pos += tw_sz

    elif tp == 5:  # Cooley-Tukey
      spl = _balanced_split(n_val)
      N1 = spl[0]
      N2 = spl[1]
      nodes_p1[nid] = N1
      nodes_p2[nid] = N2
      tw = _twiddle_ct(N1, N2).ravel()
      tw_sz = int64(tw.size)
      tw_start[nid] = tw_pos
      tw_data[tw_pos : tw_pos + tw_sz] = tw
      tw_pos += tw_sz
      # Push children (N2 first = right, then N1 = left)
      if bsp + 2 <= _MAX_NODES:
        bs_n[bsp] = N2
        bs_pid[bsp] = nid
        bs_cp[bsp] = 1
        bsp += 1
        bs_n[bsp] = N1
        bs_pid[bsp] = nid
        bs_cp[bsp] = 0
        bsp += 1

    elif tp == 6:  # Bluestein leaf
      M = int(1 << next_pow2(n_val))
      nodes_p1[nid] = M
      tw = _twiddle_bluestein(n_val, M)
      tw_sz = int64(tw.size)
      tw_start[nid] = tw_pos
      tw_data[tw_pos : tw_pos + tw_sz] = tw
      tw_pos += tw_sz

    # Link child to parent
    if pid >= 0:
      if cp == 0:
        nodes_c1[pid] = nid
      else:
        nodes_c2[pid] = nid

  # Trim to actual size
  nodes_type = nodes_type[:n_nodes]
  nodes_N = nodes_N[:n_nodes]
  nodes_p1 = nodes_p1[:n_nodes]
  nodes_p2 = nodes_p2[:n_nodes]
  nodes_c1 = nodes_c1[:n_nodes]
  nodes_c2 = nodes_c2[:n_nodes]
  tw_data = tw_data[:tw_total] if tw_total > 0 else zeros(0, dtype=c128)

  return (nodes_type, nodes_N, nodes_p1, nodes_p2,
          nodes_c1, nodes_c2, tw_data, tw_start)


###############################################################################
# Leaf kernels
###############################################################################

def _fft_ident(x_2d : NDArray[c128], br : int,
               bc : int, st : int) -> None:
  """Identity kernel. No-op leaf dispatched only when transform size N == 1."""
  pass


def _fft_3(x_2d : NDArray[c128], br : int,
           bc : int, st : int) -> None:
  r"""
  Unrolled DFT for N=3.
  :math:`w = e^{-2\pi i/3} = -1/2 - i\sqrt{3}/2`.
  """
  re_mhalf = -0.5 + 0j
  im_sqrt3h = sqrt(3.0) / 2.0 * 1j
  w1 = re_mhalf - im_sqrt3h   # exp(-2*pi*i/3)
  w2 = re_mhalf + im_sqrt3h   # exp(-4*pi*i/3) = conj(w1)

  a = x_2d[br, bc]
  b = x_2d[br, bc + st]
  c = x_2d[br, bc + 2 * st]

  x_2d[br, bc] = a + b + c
  x_2d[br, bc + st] = a + w1 * b + w2 * c
  x_2d[br, bc + 2 * st] = a + w2 * b + w1 * c


def _fft_4(x_2d : NDArray[c128], br : int,
           bc : int, st : int) -> None:
  """
  Unrolled DFT for N=4.
  Twiddle factors = {1, -i, -1, i} -- zero real multiplies.
  """
  a = x_2d[br, bc]
  b = x_2d[br, bc + st]
  c = x_2d[br, bc + 2 * st]
  d = x_2d[br, bc + 3 * st]

  ib = 1j * b
  id_ = 1j * d

  x_2d[br, bc] = a + b + c + d
  x_2d[br, bc + st] = a - ib - c + id_
  x_2d[br, bc + 2 * st] = a - b + c - d
  x_2d[br, bc + 3 * st] = a + ib - c - id_


def _fft_naive(x_2d : NDArray[c128], br : int,
               bc : int, st : int, N : int) -> None:
  """
  Direct O(N^2) Vandermonde DFT.
  """
  _N = N
  out = empty(_N, dtype=c128)
  for k in range(_N):
    s = 0j
    for j in range(_N):
      arg = (-2.0 * pi * k * j / float(_N)) * 1j
      s += x_2d[br, bc + j * st] * exp(arg)
    out[k] = s
  for k in range(_N):
    x_2d[br, bc + k * st] = out[k]


def _fft_radix2(x_2d : NDArray[c128], br : int, bc : int,
                st : int, N : int,
                tw : NDArray[c128]) -> None:
  r"""
  Iterative Cooley-Tukey DIT radix-2 FFT for N = 2^p.
  Twiddle factors: :math:`e^{-2\pi i k / N}` for :math:`k = 0, \ldots, N/2-1`.
  """
  _N = N
  p = 0
  tmp_n = _N >> 1
  while tmp_n > 0:
    p += 1
    tmp_n >>= 1

  # Bit-reversal permutation
  for i in range(_N):
    j = 0
    m = i
    for _ in range(p):
      j = (j << 1) | (m & 1)
      m >>= 1
    if j > i:
      tmp = x_2d[br, bc + i * st]
      x_2d[br, bc + i * st] = x_2d[br, bc + j * st]
      x_2d[br, bc + j * st] = tmp

  # Butterfly stages
  sz = 2
  while sz <= _N:
    hsz = sz >> 1
    d_tw = _N // sz
    for i in range(0, _N, sz):
      for j_off in range(hsz):
        j = i + j_off
        k_tw = j_off * d_tw
        ev = x_2d[br, bc + j * st]
        od = x_2d[br, bc + (j + hsz) * st]
        t = tw[k_tw] * od
        x_2d[br, bc + j * st] = ev + t
        x_2d[br, bc + (j + hsz) * st] = ev - t
    sz <<= 1


def _fft_bluestein(x_2d : NDArray[c128], br : int, bc : int,
                   st : int, N : int, M : int,
                   tw_all : NDArray[c128]) -> None:
  """
  Bluestein chirp-Z transform for prime N.
  Pads to M = 2^p >= 2N+1, performs convolution.
  tw_all contains: tw_A(N) + tw_B(M) + radix2_tw(M/2).
  """
  _N = N
  _M = M

  tw_A = tw_all[:_N]
  tw_B = tw_all[_N:_N + _M]
  radix2_tw = tw_all[_N + _M:]

  # Extract signal and multiply by tw_A
  sig = empty(_N, dtype=c128)
  for n in range(_N):
    sig[n] = x_2d[br, bc + n * st] * tw_A[n]

  # Zero-pad to M
  y = zeros(_M, dtype=c128)
  for n in range(_N):
    y[n] = sig[n]

  # Forward FFT of y (radix-2 on size M)
  # y is contiguous, stride=1
  _fft_radix2_inplace(y, _M, radix2_tw)

  # tw_B is stored raw in the plan; FFT it here for convolution.
  z_B = copy(tw_B)
  _fft_radix2_inplace(z_B, _M, radix2_tw)

  # Convolution: Y .* FFT(tw_B)
  for m_idx in range(_M):
    y[m_idx] *= z_B[m_idx]

  # Inverse FFT: conj(FFT(conj(y))) / M
  for m_idx in range(_M):
    y[m_idx] = y[m_idx].conjugate()
  _fft_radix2_inplace(y, _M, radix2_tw)
  for m_idx in range(_M):
    y[m_idx] = y[m_idx].conjugate() / float(M)

  # Crop and multiply by tw_A
  for n in range(_N):
    x_2d[br, bc + n * st] = y[n] * tw_A[n]


###############################################################################
# Radix-2 helper operating on a contiguous 1D array (stride=1)
###############################################################################

def _fft_radix2_inplace(x : NDArray[c128], N : int,
                        tw : NDArray[c128]) -> None:
  """
  Radix-2 DIT FFT on a contiguous array x[0:N] with unit stride.
  Used internally by Bluestein.
  """
  _N = N
  p = 0
  tmp_n = _N >> 1
  while tmp_n > 0:
    p += 1
    tmp_n >>= 1

  # Bit-reversal
  for i in range(_N):
    j = 0
    m = i
    for _ in range(p):
      j = (j << 1) | (m & 1)
      m >>= 1
    if j > i:
      tmp_val = x[i]
      x[i] = x[j]
      x[j] = tmp_val

  # Butterfly
  sz = 2
  while sz <= _N:
    hsz = sz >> 1
    d_tw = _N // sz
    for i in range(0, _N, sz):
      for j_off in range(hsz):
        j = i + j_off
        k_tw = j_off * d_tw
        ev = x[j]
        od = x[j + hsz]
        t = tw[k_tw] * od
        x[j] = ev + t
        x[j + hsz] = ev - t
    sz <<= 1


###############################################################################
# Cooley-Tukey helpers
###############################################################################

def _ct_twiddle_multiply(x_2d : NDArray[c128], br : int,
                         bc : int, st : int, N1 : int,
                         N2 : int,
                         tw : NDArray[c128]) -> None:
  """
  Multiply (N1,N2) block by twiddle factors in-place.
  """
  _N1 = int64(N1)
  _N2 = int64(N2)
  for k1 in range(_N1):
    for k2 in range(_N2):
      idx = bc + (k1 * _N2 + k2) * st
      x_2d[br, idx] *= tw[k1, k2]


def _ct_transpose_output(x_2d : NDArray[c128], br : int,
                         bc : int, st : int, N1 : int,
                         N2 : int) -> None:
  """
  Transpose (N1,N2) block to (N2,N1) in-place.
  Reads row-major (N1,N2), writes row-major (N2,N1).
  """
  _N1 = int64(N1)
  _N2 = int64(N2)
  tmp = empty((_N1, _N2), dtype=c128)
  for k1 in range(_N1):
    for k2 in range(_N2):
      idx = bc + (k1 * _N2 + k2) * st
      tmp[k1, k2] = x_2d[br, idx]

  for k2 in range(_N2):
    for k1 in range(_N1):
      idx = bc + (k2 * _N1 + k1) * st
      x_2d[br, idx] = tmp[k1, k2]


###############################################################################
# Stack-based plan executor
###############################################################################

def _fft_exec(x_2d : NDArray[c128],
              plan : tuple) -> None:
  """
  Execute a FFT plan on x_2d shape (batch, N).

  The plan is 8 arrays as returned by _build_plan().
  Operates in-place.
  """
  (nodes_tp, nodes_n, nodes_p1, nodes_p2,
   nodes_c1, nodes_c2, tw_data, tw_start) = plan

  batch = x_2d.shape[0]
  root = int64(0)

  # Stack per batch row
  st_id = zeros(_STACK_CAP, dtype=int64)
  st_bc = zeros(_STACK_CAP, dtype=int64)
  st_st = zeros(_STACK_CAP, dtype=int64)
  st_ph = zeros(_STACK_CAP, dtype=int8)

  for br in range(batch):
    sp = 0
    st_id[0] = root
    st_bc[0] = int64(0)
    st_st[0] = int64(1)
    st_ph[0] = int8(0)
    sp = 1

    while sp > 0:
      sp -= 1
      nid = st_id[sp]
      bc = st_bc[sp]
      st = st_st[sp]
      ph = st_ph[sp]

      tp = nodes_tp[nid]
      Ns = nodes_n[nid]

      if tp == 0:  # identity
        pass

      elif tp == 1:  # unrolled N=3
        _fft_3(x_2d, br, bc, st)

      elif tp == 2:  # unrolled N=4
        _fft_4(x_2d, br, bc, st)

      elif tp == 3:  # naive
        _fft_naive(x_2d, br, bc, st, Ns)

      elif tp == 4:  # radix2
        tw = tw_data[tw_start[nid] : tw_start[nid] + Ns // 2]
        _fft_radix2(x_2d, br, bc, st, Ns, tw)

      elif tp == 5:  # Cooley-Tukey
        N1 = nodes_p1[nid]
        N2 = nodes_p2[nid]
        c1 = nodes_c1[nid]  # N1 child
        c2 = nodes_c2[nid]  # N2 child

        if ph == 0:
          # Phase 0: push N1 children (column DFTs, size N1, stride N2)
          st_id[sp] = nid
          st_bc[sp] = bc
          st_st[sp] = st
          st_ph[sp] = int8(1)
          sp += 1
          for k in range(int(N2)):
            st_id[sp] = c1
            st_bc[sp] = bc + int64(k) * st
            st_st[sp] = int64(N2) * st
            st_ph[sp] = int8(0)
            sp += 1

        elif ph == 1:
          # Phase 1: twiddle multiply (data still in (N1,N2) layout)
          tw = tw_data[tw_start[nid] :
                       tw_start[nid] + N1 * N2].reshape(N1, N2)
          _ct_twiddle_multiply(x_2d, br, bc, st,
                               N1, N2, tw)
          # Push N2 children (row DFTs, size N2, stride 1)
          st_id[sp] = nid
          st_bc[sp] = bc
          st_st[sp] = st
          st_ph[sp] = int8(2)
          sp += 1
          for k in range(int(N1)):
            st_id[sp] = c2
            st_bc[sp] = bc + int64(k) * N2 * st
            st_st[sp] = st
            st_ph[sp] = int8(0)
            sp += 1

        elif ph == 2:
          # Phase 2: transpose (N1,N2) -> (N2,N1) for output order
          _ct_transpose_output(x_2d, br, bc, st, N1, N2)

      elif tp == 6:  # Bluestein
        M_val = nodes_p1[nid]
        tw = tw_data[tw_start[nid] :
                     tw_start[nid] + Ns + M_val + M_val // 2]
        _fft_bluestein(x_2d, br, bc, st, Ns, M_val, tw)


###############################################################################
# High-level 2D FFT (batch, N) -- for use inside generated_jit
###############################################################################

def _fft_2d(x_2d : NDArray[c128]) -> NDArray[c128]:
  """
  FFT on (batch, N) array. Builds plan, executes in-place.
  """
  n = x_2d.shape[1]
  plan = _build_plan(n)
  _fft_exec(x_2d, plan)
  return x_2d


###############################################################################
# JIT compilation
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _twiddle_radix2 = JIT(_twiddle_radix2)
  _twiddle_ct = JIT(_twiddle_ct)
  _twiddle_bluestein = JIT(_twiddle_bluestein)
  _is_pow2 = JIT(_is_pow2)
  _is_prime = JIT(_is_prime)
  _balanced_split = JIT(_balanced_split)
  _build_plan = JIT(_build_plan)
  _fft_ident = JIT(_fft_ident)
  _fft_3 = JIT(_fft_3)
  _fft_4 = JIT(_fft_4)
  _fft_naive = JIT(_fft_naive)
  _fft_radix2 = JIT(_fft_radix2)
  _fft_radix2_inplace = JIT(_fft_radix2_inplace)
  _fft_bluestein = JIT(_fft_bluestein)
  _ct_twiddle_multiply = JIT(_ct_twiddle_multiply)
  _ct_transpose_output = JIT(_ct_transpose_output)
  _fft_exec = JIT(_fft_exec)
  _fft_2d = JIT(_fft_2d)

#
# :D
#
