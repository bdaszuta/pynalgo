"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Grid clustering and domain mapping: S2 tangent map, spherical cap,
algebraic coordinate transform with Jacobian chain

"""

import matplotlib.pyplot as plt
import os
from pynalgo import (diff_mat_nodal_ChebyshevT,
  grid_ChebyshevT, grid_Fourier,
  poly_basis_mat_ChebyshevT,
  poly_basis_mat_trig_exp,
  quad_ChebyshevT)
from numpy import (
  arctan, arctan2, asarray,
  broadcast_arrays, ceil, cos, dot,
  empty, exp, int64, linspace,
  nan, pi, sin, sqrt, tan)

N_th, N_ph = 16, 16

gr_th = grid_Fourier(N_th, variety="H1_I")
gr_ph = grid_Fourier(N_ph, variety="S1_I")

L = 1 - 1 / 8
gr_th_cluster = 2 * arctan(L * tan((gr_th - 0) / 2))
gr_ph_cluster = 2 * arctan(L * tan((gr_ph - pi - 0) / 2)) + pi

T2 = tan((gr_ph - pi - 0) / 2) ** 2
dph = L * (1 + T2) / (1 + L ** 2 * T2)

wbas_ph = poly_basis_mat_trig_exp(N_ph, variety="S1_I", quad_weight=True)

r = 1

gth, gph = gr_th[:, None], gr_ph[None, :]

x = r * sin(gth) * cos(gph)
y = r * sin(gth) * sin(gph)
z = broadcast_arrays(r * cos(gth), gph)[0]

# At fixed z we have a circle of radius sin(th)
# In order to improve uniformity of points we could try scaling N_ph by this
# radius.

N_ph_th_dep = ceil(sin(gr_th) * N_ph)

gr_ph_var = empty((N_th, N_ph))
gr_ph_var[:] = nan

N_ph_th = asarray(ceil(sin(gr_th) * N_ph), dtype=int64)

for i in range(N_th):
  gr_ph_var[i, :N_ph_th[i]] = grid_Fourier(N_ph_th[i], variety="S1_CL")

### inspect

x = r * sin(gth) * cos(gr_ph)
y = r * sin(gth) * sin(gr_ph)
z = broadcast_arrays(r * cos(gth), gr_ph)[0]

x_sc = r * sin(gth) * cos(gr_ph_var)
y_sc = r * sin(gth) * sin(gr_ph_var)
z_sc = broadcast_arrays(r * cos(gth), gr_ph_var)[0]

ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z, s=2, c="b")
# ax.scatter3D(x_sc, y_sc, z_sc, s=5, c="r")
# ax.scatter3D(z_sc, x_sc, y_sc, s=5, c="g")

ax.scatter3D(x_sc, y_sc, z_sc, s=5, c="r")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1, 1, 1])
ax.set_title('S2 cap: variable-Nph grid (scaled by sin(th))')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'grid_interchange_variable_ph.png'), dpi=150)
plt.close()

###

N_th, N_ph = 5, 32
th_a, th_b = pi / 4, 3 * pi / 4
# ph_a, ph_b = -3 * pi / 4, 3 * pi / 4
ph_a, ph_b = 0, 2 * pi

f_gr_th = grid_ChebyshevT(N_th, variety="GL")
f_gr_th = linspace(-1, 1, num=N_th)
f_gr_ph = grid_ChebyshevT(N_ph, variety="GL")

a, b = th_a, th_b
yi_th = (b - a) / 2 * f_gr_th + (b + a) / 2
a, b = ph_a, ph_b
yi_ph = (b - a) / 2 * f_gr_ph + (b + a) / 2

yi_ph = grid_Fourier(N_ph, variety="S1_CL")

wbas = poly_basis_mat_ChebyshevT(N_th, variety="GL", quad_weight=True)

x_yi = r * sin(yi_th[:, None]) * cos(yi_ph[None, :])
y_yi = r * sin(yi_th[:, None]) * sin(yi_ph[None, :])
z_yi = broadcast_arrays(r * cos(yi_th[:, None]), yi_ph[None, :])[0]

ax = plt.axes(projection='3d')
ax.scatter3D(x_yi, y_yi, z_yi, s=5, c="r")
ax.scatter3D(-x_yi, z_yi, y_yi, s=5, c="b")
ax.scatter3D(z_yi, -y_yi, x_yi, s=5, c="g")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1, 1, 1])
ax.set_title('S2 spherical cap patch')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'grid_interchange_s2cap.png'), dpi=150)
plt.close()

#

L = 10
N_x = 32
f_gr = grid_ChebyshevT(N_x, variety="RL")
f_wei = quad_ChebyshevT(N_x, variety="RL")
f_wbas = poly_basis_mat_ChebyshevT(N_x, variety="RL", quad_weight=True)
f_bas = poly_basis_mat_ChebyshevT(N_x, variety="RL", quad_weight=False)

a = 0
y = a + L * (1 + f_gr) / (1 - f_gr)
t = 2 * arctan2(sqrt(y), sqrt(L))

D1_x = diff_mat_nodal_ChebyshevT(N_x, D=1, variety="RL")
D2_x = diff_mat_nodal_ChebyshevT(N_x, D=2, variety="RL")
J1_x = 2 * L / (y + L - a) ** 2
J2_x = 4 * L / (a - y - L) ** 3

D1_y = J1_x[:, None] * D1_x
D2_y = J1_x[:, None] ** 2 * D2_x + J2_x[:, None] * D1_x

# function
f_n = exp(-y)
f_m = dot(f_wbas, f_n)
r_n = dot(f_bas.T, f_m)

# derivs.
D1_n = -exp(-y)
D2_n = exp(-y)

N1_n = dot(D1_y, f_n)
N1_m = dot(f_wbas, N1_n)

N2_n = dot(D2_y, f_n)
N2_m = dot(f_wbas, N2_n)

# just use basis recursion and not this
# TL_1 = cos(1 * t)
# TL_2 = cos(2 * t)

plt.semilogx(y, f_n, '-ob', label='f(y)=exp(-y)')
plt.semilogx(y, r_n, '--r', label='Chebyshev reconstruction')
plt.xlabel('y')
plt.ylabel('f(y)')
plt.title('Algebraic map: exp(-y) on [0,inf) via RL Chebyshev')
plt.legend()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'grid_interchange_expmap.png'), dpi=150)
plt.close()

#
# :D
#
