"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Inspect fundamental Chebyshev grids
"""
import os
from pynalgo import (grid_ChebyshevT, poly_ChebyshevT)
from numpy import (linspace)

import matplotlib.pyplot as plt

# settings --------------------------------------------------------------------

N = 4
gr_GL = grid_ChebyshevT(N, variety="GL")
gr_G  = grid_ChebyshevT(N, variety="G")

gr_RR  = grid_ChebyshevT(N, variety="RR")
gr_RL  = grid_ChebyshevT(N, variety="RL")

plt.plot(gr_RL, 0 * gr_RL, "sg",
         label="RL", ms=5.2)

plt.plot(gr_RR, 0 * gr_RR, "sk",
         label="RR", ms=5.2)

plt.plot(gr_GL, 0 * gr_GL, "ob",
         label="GL", ms=3.2)

plt.plot(gr_G, 0 * gr_G, "or",
         label="G", ms=3.2)

gr = linspace(-1, 1, num=100)

TNm1 = poly_ChebyshevT(N-1, gr).squeeze()
TN = poly_ChebyshevT(N, gr).squeeze()
TNp1 = poly_ChebyshevT(N+1, gr).squeeze()

plt.plot(gr, TNm1, "-c",
         label="T_N-1", ms=3.2)

plt.plot(gr, TN, "-k",
         label="T_N", ms=3.2)

# plt.plot(gr, TNp1, "-c",
#          label="T_N+1", ms=3.2)

plt.title("N=4")
plt.xlabel('x')
plt.ylabel('T_n(x), grid markers')

plt.legend()
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'chebyshev_grids.png'), dpi=150)
plt.close()

#
# :D
#
