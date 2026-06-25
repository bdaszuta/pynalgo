"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Cartesian-to-polar coordinate remap via AAA rational interpolation.

Maps a function defined on a planar Cartesian grid to polar coordinates
using 1D AAA along radial slices.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import linspace, meshgrid, pi

from pynalgo.rational import aaa_real, eval_rat

# Source: Cartesian grid
N = 80
x = linspace(-2, 2, N)
y = linspace(-2, 2, N)
X, Y = meshgrid(x, y, indexing='ij')
F_cart = np.exp(-(X**2 + Y**2)) * np.cos(3 * np.sqrt(X**2 + Y**2))

# Target: polar grid
Nr, Nth = 60, 90
r = linspace(0.1, 2.0, Nr)
th = linspace(0, 2*pi, Nth)
R, TH = meshgrid(r, th, indexing='ij')

# Remap: for each theta, build AAA along radial line
F_polar = np.empty((Nr, Nth))
for j in range(Nth):
    # Sample Cartesian function along ray at angle th[j]
    x_ray = r * np.cos(th[j])
    y_ray = r * np.sin(th[j])
    # Evaluate F_cart at these (x,y) points by interpolating the grid
    f_ray = np.empty(Nr)
    for k in range(Nr):
        xi, yi = x_ray[k], y_ray[k]
        # Find nearest grid cell and bilinearly interpolate
        ix = np.searchsorted(x, xi) - 1
        iy = np.searchsorted(y, yi) - 1
        ix = max(0, min(N-2, ix))
        iy = max(0, min(N-2, iy))
        tx = (xi - x[ix]) / (x[ix+1] - x[ix])
        ty = (yi - y[iy]) / (y[iy+1] - y[iy])
        f_ray[k] = ((1-tx)*(1-ty)*F_cart[ix, iy] +
                     tx*(1-ty)*F_cart[ix+1, iy] +
                     (1-tx)*ty*F_cart[ix, iy+1] +
                     tx*ty*F_cart[ix+1, iy+1])

    # AAA along radial line
    z_sup, f_sup, w_sup = aaa_real(r, f_ray, tol=1e-8, max_terms=20)
    F_polar[:, j] = eval_rat(z_sup, f_sup, w_sup, r)

print(f'Polar shape: {F_polar.shape}')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].pcolormesh(X, Y, F_cart, shading='auto')
axs[0].set_title('Cartesian')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_aspect('equal')
X_pol = R * np.cos(TH)
Y_pol = R * np.sin(TH)
axs[1].pcolormesh(X_pol, Y_pol, F_polar, shading='auto')
axs[1].set_title('Polar (AAA remap)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_aspect('equal')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'planar_remap.png'), dpi=150)
plt.close()
