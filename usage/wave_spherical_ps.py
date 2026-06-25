"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Multi-domain pseudospectral wave equation in spherical symmetry:
Chebyshev interior-exterior with exterior compactification (logarithmic
mapping r = a + L*arctanh((x+1)/2) compactifies r in [a, infinity) to
x in [-1,1]), domain matching (C0/C1/C2), Orszag filtering, RK4 evolution

Radial part of Laplacian in spherical coordinates (with spherical symmetry)
Lap[f] = D[r^2 D[f[r], r], r] / r ^ 2
       = D[f[r], {r, 2}] + 2 D[f[r], r] / r
"""
import os

from pynalgo import (
  diff_mat_nodal_ChebyshevT,
  grid_ChebyshevT,
  poly_basis_mat_ChebyshevT,
  quad_ChebyshevT)
from numpy import (
  arctan2, arctanh, cos, cosh,
  dot, exp, hstack, sin, sinh,
  sqrt, tanh, zeros_like)

import matplotlib.pyplot as plt

# settings --------------------------------------------------------------------

int_N = 32
int_a, int_b = 0, 2
int_var = "RR"

ext_N = 128
ext_L = 10
ext_a = int_b
ext_var = "RL"

# assets: interior ------------------------------------------------------------

int_f_gr = grid_ChebyshevT(int_N, variety=int_var)
int_f_wei = quad_ChebyshevT(int_N, variety=int_var)
int_f_D1 = diff_mat_nodal_ChebyshevT(int_N, D=1, variety=int_var)
int_f_D2 = diff_mat_nodal_ChebyshevT(int_N, D=2, variety=int_var)

int_f_wbas = poly_basis_mat_ChebyshevT(int_N, variety=int_var,
                                       quad_weight=True)
int_f_bas = poly_basis_mat_ChebyshevT(int_N, variety=int_var,
                                      quad_weight=False)

int_f_wbas_GL = poly_basis_mat_ChebyshevT(int_N, variety="GL",
                                          quad_weight=True)
int_f_bas_GL = poly_basis_mat_ChebyshevT(int_N, variety="GL",
                                         quad_weight=False)

int_gr = (int_b + int_a) / 2 + int_f_gr * (int_b - int_a) / 2
int_J1_x = 2 / (int_b - int_a)
int_J2_x = 4 / (int_b - int_a) ** 2
int_D1 = int_J1_x * int_f_D1
int_D2 = int_J2_x * int_f_D2

# assets: exterior ------------------------------------------------------------

ext_f_gr = grid_ChebyshevT(ext_N, variety=ext_var)
ext_f_wei = quad_ChebyshevT(ext_N, variety=ext_var)

ext_f_D1 = diff_mat_nodal_ChebyshevT(ext_N, D=1, variety=ext_var)
ext_f_D2 = diff_mat_nodal_ChebyshevT(ext_N, D=2, variety=ext_var)

ext_f_wbas = poly_basis_mat_ChebyshevT(ext_N, variety=ext_var,
                                       quad_weight=True)
ext_f_bas = poly_basis_mat_ChebyshevT(ext_N, variety=ext_var,
                                      quad_weight=False)

ext_gr = ext_a + ext_L * (1 + ext_f_gr) / (1 - ext_f_gr)
ext_t = 2 * arctan2(sqrt(ext_gr), sqrt(ext_L))

ext_J1_x = 2 * ext_L / (ext_gr + ext_L - ext_a) ** 2
ext_J2_x = 4 * ext_L / (ext_a - ext_gr - ext_L) ** 3
ext_D1 = ext_J1_x[:, None] * ext_f_D1
ext_D2 = ext_J1_x[:, None] ** 2 * ext_f_D2 + ext_J2_x[:, None] * ext_f_D1

# inspect test function -------------------------------------------------------

fcn = lambda r: (r * sin(r)) * exp(-r)
dfcn = lambda r: (r * cos(r) + sin(r) - r * sin(r)) * exp(-r)
ddfcn = lambda r: -2 * (sin(r) + (r - 1) * cos(r)) * exp(-r)

int_N_n = fcn(int_gr)
ext_N_n = fcn(ext_gr)

int_N_m = dot(int_f_wbas, int_N_n)
ext_N_m = dot(ext_f_wbas, ext_N_n)

int_N_r = dot(int_f_bas.T, int_N_m)
ext_N_r = dot(ext_f_bas.T, ext_N_m)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)

plt.semilogx(int_gr, int_N_r, 'ob', label='interior (reconstructed)')
plt.semilogx(int_gr, int_N_n, '-b', label='interior (exact)')
plt.semilogx(ext_gr, ext_N_r, 'og', label='exterior (reconstructed)')
plt.semilogx(ext_gr, ext_N_n, '-g', label='exterior (exact)')
plt.xlabel('r')
plt.ylabel('f(r)')
plt.title('Chebyshev reconstruction: test function')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'wave_spherical_reconstruct.png'), dpi=150)
plt.close()

# -----------------------------------------------------------------------------

int_u_n = int_N_n.copy()
ext_u_n = ext_N_n.copy()

# match C0
mc_C0 = (int_u_n[-1] + ext_u_n[0]) / 2
int_u_n[-1] = ext_u_n[0] = mc_C0

D1_i = dot(int_D1, int_u_n)
D1_e = dot(ext_D1, ext_u_n)

# condition at origin [RR -> GL -> impose -> RR]
D1_i_GL = dot(int_f_bas_GL.T, dot(int_f_wbas, D1_i))
D1_i_GL[0] = 0
D1_i = dot(int_f_bas.T, dot(int_f_wbas_GL, D1_i_GL))

# match C1
mc_C1 = (D1_i[-1] + D1_e[0]) / 2
D1_i[-1] = D1_e[0] = mc_C1

# second der.
D2_i = dot(int_D1, D1_i)
D2_e = dot(ext_D1, D1_e)

# match C2
mc_C2 = (D2_i[-1] + D2_e[0]) / 2
D2_i[-1] = D2_e[0] = mc_C2

# inspect: domain matching ----------------------------------------------------

plt.semilogx(int_gr, int_u_n, '-or', label='u (interior)')
plt.semilogx(ext_gr, ext_u_n, '-or', label='u (exterior)')
plt.semilogx(int_gr[-1], int_u_n[-1], 'sk', ms=8, label='matching node')

plt.semilogx(int_gr, D1_i, '-og', label="u' (interior)")
plt.semilogx(ext_gr, D1_e, '-og', label="u' (exterior)")
plt.semilogx(int_gr[-1], D1_i[-1], 'sk', ms=8)

plt.semilogx(int_gr, D2_i, '-ob', label="u'' (interior)")
plt.semilogx(ext_gr, D2_e, '-ob', label="u'' (exterior)")
plt.semilogx(int_gr[-1], D2_i[-1], 'sk', ms=8)

plt.xlabel('r')
plt.ylabel('amplitude')
plt.title('Domain matching: C0, C1, C2 continuity')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'wave_spherical_matching.png'), dpi=150)
plt.close()

# -----------------------------------------------------------------------------

# propagate ===================================================================

class PseudospectralSphericalSymmetry:
  def __init__(self, int_N, int_b, ext_N, ext_L, int_a=0):

    # grid parameters ---------------------------------------------------------
    self.int_N = int_N
    self.int_a, self.int_b = int_a, int_b
    self.int_var = "RR"

    self.ext_N = ext_N
    self.ext_L = ext_L
    self.ext_a = int_b
    self.ext_var = "RL"

    # interior assets --------------------------------------------------------
    self.int_f_gr = grid_ChebyshevT(int_N, variety=self.int_var)
    self.int_f_wei = quad_ChebyshevT(int_N, variety=self.int_var)
    self.int_f_D1 = diff_mat_nodal_ChebyshevT(int_N, D=1,
                                              variety=self.int_var)
    self.int_f_D2 = diff_mat_nodal_ChebyshevT(int_N, D=2,
                                              variety=self.int_var)

    self.int_f_wbas = poly_basis_mat_ChebyshevT(int_N, variety=self.int_var,
                                                quad_weight=True)
    self.int_f_bas = poly_basis_mat_ChebyshevT(int_N, variety=self.int_var,
                                               quad_weight=False)

    self.int_f_wbas_GL = poly_basis_mat_ChebyshevT(int_N, variety="GL",
                                                   quad_weight=True)
    self.int_f_bas_GL = poly_basis_mat_ChebyshevT(int_N, variety="GL",
                                                  quad_weight=False)

    self.int_gr = (int_b + int_a) / 2 + self.int_f_gr * (int_b - int_a) / 2
    self.int_J1_x = 2 / (int_b - int_a)
    self.int_J2_x = 4 / (int_b - int_a) ** 2
    self.int_D1 = self.int_J1_x * self.int_f_D1
    self.int_D2 = self.int_J2_x * self.int_f_D2

    # exterior assets ---------------------------------------------------------
    self.ext_f_gr = grid_ChebyshevT(ext_N, variety=self.ext_var)
    self.ext_f_wei = quad_ChebyshevT(ext_N, variety=self.ext_var)

    self.ext_f_D1 = diff_mat_nodal_ChebyshevT(ext_N, D=1, variety=self.ext_var)
    self.ext_f_D2 = diff_mat_nodal_ChebyshevT(ext_N, D=2, variety=self.ext_var)

    self.ext_f_wbas = poly_basis_mat_ChebyshevT(ext_N, variety=self.ext_var,
                                                quad_weight=True)
    self.ext_f_bas = poly_basis_mat_ChebyshevT(ext_N, variety=self.ext_var,
                                               quad_weight=False)

    # algebraic mapping
    self.ext_gr = (self.ext_a +
                   ext_L * (1 + self.ext_f_gr) / (1 - self.ext_f_gr))
    # self.ext_t = 2 * arctan2(sqrt(self.ext_gr), sqrt(ext_L))

    self.ext_J1_x = 2 * ext_L / (self.ext_gr + ext_L - self.ext_a) ** 2
    self.ext_J2_x = 4 * ext_L / (self.ext_a - self.ext_gr - ext_L) ** 3

    # exponential mapping
    self.ext_gr = self.ext_a + sinh(ext_L * (self.ext_f_gr + 1) / 2)
    rsqrt = 1 / sqrt(1 + (self.ext_a - self.ext_f_gr) ** 2)
    self.ext_J1_x = 2 / ext_L * rsqrt
    self.ext_J1_x = 2 / ext_L * (rsqrt ** 3) * (self.ext_a - self.ext_f_gr)

    # logarithmic mapping
    self.ext_gr = (self.ext_a +
                   ext_L * arctanh((self.ext_f_gr + 1) / 2))

    farg = (self.ext_gr - self.ext_a) / ext_L
    sech2 = 1 / cosh(farg) ** 2
    self.ext_J1_x = 2 / ext_L * sech2
    self.ext_J2_x = -(2 / ext_L) ** 2 * sech2 * tanh(farg)

    self.ext_D1 = self.ext_J1_x[:, None] * self.ext_f_D1
    self.ext_D2 = (self.ext_J1_x[:, None] ** 2 * self.ext_f_D2 +
                   self.ext_J2_x[:, None] * self.ext_f_D1)

  def impose_zero_origin(self, int_u_n):
    # Rake nodal data on interior domain and impose zero there
    #
    # Condition at origin [RR -> GL -> impose -> RR]

    int_u_n_GL = dot(self.int_f_bas_GL.T, dot(self.int_f_wbas, int_u_n))
    int_u_n_GL[0] = 0
    return dot(self.int_f_bas.T, dot(self.int_f_wbas_GL, int_u_n_GL))

  def match_domains(self, int_n, ext_n):
    # in-place operation; average of coincident nodes imposed
    mc = (int_n[-1] + ext_n[0]) / 2
    int_n[-1] = ext_n[0] = mc

  def Laplacian(self, data_n,
                impose_zero_origin=True,
                match_C1=True,
                match_C2=True):
    # Compute Laplacian

    int_n, ext_n = self.split_domains(data_n)
    tmp_int_n = int_n.copy()
    tmp_ext_n = ext_n.copy()

    # if impose_zero_origin:
    #   tmp_int_n = self.impose_zero_origin(tmp_int_n)

    # match C0
    self.match_domains(tmp_int_n, tmp_ext_n)

    D1_i = dot(self.int_D1, tmp_int_n)
    D1_e = dot(self.ext_D1, tmp_ext_n)

    if impose_zero_origin:
      D1_i = self.impose_zero_origin(D1_i)

    # match C1
    if match_C1:
      self.match_domains(D1_i, D1_e)

    D2_i = dot(self.int_D1, D1_i)
    D2_e = dot(self.ext_D1, D1_e)

    if match_C2:
      self.match_domains(D2_i, D2_e)

    # assemble
    int_Lap = D2_i + 2 * D1_i / self.int_gr
    ext_Lap = D2_e + 2 * D1_e / self.ext_gr

    return self.combine_domains(int_Lap, ext_Lap)

  def combine_domains(self, int_n, ext_n):
    return hstack([int_n, ext_n])

  def split_domains(self, data_n):
    ix = self.int_D1.shape[0]
    return (data_n[:ix], data_n[ix:])

  def state_vector_flatten_to(self, u, v):
    # flatten to state vector
    return hstack([u.flatten(), v.flatten()])

  def state_vector_reconstruct_uv(self, U):
    # reconstruct fields from state vector
    Ix = self.int_D1.shape[0]
    Jx = self.ext_D1.shape[0]

    u = U[:(Ix + Jx)]
    v = U[(Ix + Jx):]

    return u, v

  def inspect_modal(self, int_n, ext_n, plot=True, save_path=None):
    # look at coefficient space representation of functions
    int_m = dot(self.int_f_wbas, int_n)
    ext_m = dot(self.ext_f_wbas, ext_n)

    if plot:
      plt.semilogy(abs(int_m), "-g", label='interior coeffs')
      plt.semilogy(abs(ext_m), "-b", label='exterior coeffs')
      plt.xlabel('mode index')
      plt.ylabel('|coefficient|')
      plt.title('Spectral coefficients')
      plt.legend()
      plt.tight_layout()
      plt.savefig(save_path, dpi=150)
      plt.close()

    return int_m, ext_m

  def filter_orszag(self, data_n):

    int_n, ext_n = self.split_domains(data_n)

    int_m = dot(self.int_f_wbas, int_n)
    ext_m = dot(self.ext_f_wbas, ext_n)

    int_m[(2 * self.int_N) // 3:] = 0
    ext_m[(2 * self.ext_N) // 3:] = 0

    fint_n = dot(self.int_f_bas.T, int_m)
    fext_n = dot(self.ext_f_bas.T, ext_m)

    return self.combine_domains(fint_n, fext_n)

  def RHS(self, t, U):

    # given state vector evaluate RHS
    u, v = self.state_vector_reconstruct_uv(U)
    Lap = self.Laplacian(u, impose_zero_origin=True,
                         match_C1=True,
                         match_C2=False)

    # return self.state_vector_flatten_to(
    #   v,
    #   Lap
    # )

    return self.state_vector_flatten_to(
      self.filter_orszag(v),
      self.filter_orszag(Lap)
    )

  def _ev_step_RK4(self, t, z, dt, **kwargs):
    dt_2 = dt / 2
    k1 = self.RHS(t, z)
    k2 = self.RHS(t + dt_2, z + dt_2 * k1)
    k3 = self.RHS(t + dt_2, z + dt_2 * k2)
    k4 = self.RHS(t + dt, z + dt * k3)

    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

  def evolve_RK4(self, ti, tf, N_t, z_ini, sys_post_step_fcn=None, **kwargs):
    # solve using RK4
    dt = (tf - ti) / N_t
    t = ti
    z = z_ini

    for t_idx in range(int(N_t)):
      z = self._ev_step_RK4(t, z, dt, **kwargs)
      # if sys_post_step_fcn is not None:
      #     sys_post_step_fcn(t, z, t_idx=t_idx, N_t=N_t, **kwargs)
      t = t + dt
    return z

int_N, int_b = 256, 5
ext_N, ext_L = 128, 10

PSS = PseudospectralSphericalSymmetry(int_N, int_b, ext_N, ext_L)

fcn = lambda r: r * exp(-r)
fcn = lambda r: (r + r ** 2) * exp(-r)
fcn = lambda r: (r * sin(r)) * exp(-r)

# should be an even function to get regularity at origin
fcn = lambda r: (r ** 2) * exp(-r ** 2)
# fcn = lambda r: exp(-r ** 2)

U_ini = PSS.combine_domains(fcn(PSS.int_gr), fcn(PSS.ext_gr))
V_ini = zeros_like(U_ini)

Z_ini = PSS.state_vector_flatten_to(U_ini, V_ini)
U, V = PSS.state_vector_reconstruct_uv(Z_ini)

Lap = PSS.Laplacian(U_ini, impose_zero_origin=True)
PSS.inspect_modal(
    Lap[:int_N+1], Lap[int_N+1:], plot=True,
    save_path=os.path.join(out_dir, 'wave_spherical_lap_coeffs.png'))

# plt.semilogx(PSS.int_gr, fcn(PSS.int_gr), '-og')
# plt.semilogx(PSS.ext_gr, fcn(PSS.ext_gr), '-or')
# plt.show()

# plt.semilogx(PSS.combine_domains(PSS.int_gr, PSS.ext_gr),
#              PSS.Laplacian(PSS.combine_domains(int_u_n, ext_u_n)))

Z_fin = PSS.evolve_RK4(0, 2, 10000, Z_ini)
U_fin, V_fin = PSS.state_vector_reconstruct_uv(Z_fin)

PSS.inspect_modal(
    U_fin[:int_N+1], U_fin[int_N+1:], plot=True,
    save_path=os.path.join(out_dir, 'wave_spherical_u_fin_coeffs.png'))
PSS.inspect_modal(
    V_fin[:int_N+1], V_fin[int_N+1:], plot=True,
    save_path=os.path.join(out_dir, 'wave_spherical_v_fin_coeffs.png'))

# compare initial profile to final profile

plt.semilogx(PSS.int_gr, U_ini[:int_N+1], "-g", label='u initial (interior)')
plt.semilogx(PSS.ext_gr, U_ini[int_N+1:], "-g", label='u initial (exterior)')
plt.semilogx(PSS.int_gr, U_fin[:int_N+1], "-r", label='u final (interior)')
plt.semilogx(PSS.ext_gr, U_fin[int_N+1:], "-r", label='u final (exterior)')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Wave evolution: initial vs final')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'wave_spherical_evolution.png'), dpi=150)
plt.close()

#
# :D
#
