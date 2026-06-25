"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Rational function approximation (AAA algorithm).
"""
from pynalgo.rational.aaa import (
    aaa,
    aaa_real,
    eval_rat,
    poles_residues,
)

from pynalgo.rational.derivatives import (
    rat_D1,
    rat_D2,
    rat_D,
    rat_eval,
    diff_mat_nodal_rat,
)

__all__ = [
    'aaa',
    'aaa_real',
    'eval_rat',
    'poles_residues',
    'rat_D1',
    'rat_D2',
    'rat_D',
    'rat_eval',
    'diff_mat_nodal_rat',
]

#
# :D
#
