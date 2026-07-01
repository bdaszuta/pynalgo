"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Spectral transforms and bases: DCT-I through DCT-IV,
           polynomial basis matrix construction.
"""



from pynalgo.spectral.bases import (
  poly_basis_mat_ChebyshevT,
  poly_basis_mat_trig,
  poly_basis_mat_trig_exp
)

from pynalgo.spectral._dct import (
  dct1,
  dct2,
  dct3,
  dct4,
)

__all__ = [
  'dct1',
  'dct2',
  'dct3',
  'dct4',
  'poly_basis_mat_ChebyshevT',
  'poly_basis_mat_trig',
  'poly_basis_mat_trig_exp',
]

#
# :D
#
