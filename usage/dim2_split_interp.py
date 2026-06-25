"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: 2D dimension-split rational interpolation demo.

Demonstrates interp_barycentric_2d_split on a tensor-product grid.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import linspace, meshgrid

from pynalgo.resample import interp_barycentric_2d_split

# Source grid and function
x_src = linspace(-2, 2, 50)
y_src = linspace(-2, 2, 40)
X_src, Y_src = meshgrid(x_src, y_src, indexing='ij')
F_src = np.sin(3 * X_src) * np.cos(2 * Y_src) + 0.3 * X_src * Y_src

# Target grid (finer, shifted)
x_tgt = linspace(-1.8, 1.8, 80)
y_tgt = linspace(-1.8, 1.8, 60)

F_tgt = interp_barycentric_2d_split(
    x_src, y_src, F_src, x_tgt, y_tgt, d=4, method='FH')

X_tgt, Y_tgt = meshgrid(x_tgt, y_tgt, indexing='ij')
F_exact = np.sin(3 * X_tgt) * np.cos(2 * Y_tgt) + 0.3 * X_tgt * Y_tgt

err = np.abs(F_tgt - F_exact)
print(f'max error: {np.max(err):.2e}')
print(f'shape: {F_tgt.shape}')

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].pcolormesh(X_tgt, Y_tgt, F_tgt, shading='auto')
axs[0].set_title('Interpolated')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[1].pcolormesh(X_tgt, Y_tgt, F_exact, shading='auto')
axs[1].set_title('Exact')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
im = axs[2].pcolormesh(X_tgt, Y_tgt, np.log10(err + 1e-16), shading='auto')
axs[2].set_title('log10(error)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
plt.colorbar(im, ax=axs[2])
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
for ax in axs:
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'dim2_split_interp.png'), dpi=150)
plt.close()
