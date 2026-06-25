"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Pseudospectral Laplacian on S^2: parity-split vs domain-extension.
           Orszag filtering, RK4 wave evolution.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
from numpy import (asarray, complex128, cos, einsum, empty,
                   exp, float64, flipud, hstack, ones, pi,
                   power, roll, sin, tan, vstack)
from pynalgo import (diff_mat_nodal_Fourier, grid_Fourier, is_even,
                     quad_Fourier)
from pynalgo.common_tools.utilities_numba import JIT
from pynalgo.common_tools.utilities_numpy import array_dot_2d_at_axis


# ---------------------------------------------------------------------------
# Parity / extension helpers
# ---------------------------------------------------------------------------

def scalar_field_parity_factor(field, s=0):
    N_ph = field.shape[1]
    N_roll = N_ph // 2 if is_even(N_ph) else N_ph // 2 + 1
    rfield = flipud(power(-1, s) * flipud(roll(field, N_roll, axis=1)))
    f_e = (field + rfield) / 2
    f_o = (field - rfield) / 2
    return f_e, f_o


def scalar_field_extend(field, s=0):
    N_ph = field.shape[1]
    N_roll = N_ph // 2 if is_even(N_ph) else N_ph // 2 + 1
    return vstack([field, power(-1, s) * flipud(roll(field, N_roll, axis=1))])


# ---------------------------------------------------------------------------
# Nodal polynomial basis matrix builder (used by WaveEquation_S2)
# ---------------------------------------------------------------------------

@JIT
def poly_basis_mat_nodal(N, variety="S1_CL", quad_weight=False,
                         dtype=complex128):
    if variety in ("H1_C_cos", "H1_C_sin"):
        gr = grid_Fourier(N, variety="H1_C")
        sz_gr = gr.size
        B = empty((sz_gr, sz_gr), dtype=dtype)
        if quad_weight:
            wei = (2 / pi) * quad_Fourier(N, variety=variety)
        else:
            wei = ones(sz_gr, dtype=float64)
        if variety == "H1_C_cos":
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = wei[j] * cos(i * gr[j])
        if variety == "H1_C_sin":
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = wei[j] * sin((i + 1) * gr[j])
        return B

    if variety in ("H1_I_cos", "H1_I_sin"):
        gr = grid_Fourier(N, variety="H1_I")
        sz_gr = gr.size
        B = empty((sz_gr, sz_gr), dtype=dtype)
        if quad_weight:
            wei = (2 / pi) * quad_Fourier(N, variety="H1_I_cos")
        else:
            wei = ones(sz_gr, dtype=float64)
        if variety == "H1_I_cos":
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = wei[j] * cos(i * gr[j])
        if variety == "H1_I_sin":
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = wei[j] * sin((i + 1) * gr[j])
        return B

    if variety in ("S1_CL", "S1_CR", "S1_I"):
        gr = grid_Fourier(N, variety=variety)
        sz_gr = gr.size
        B = empty((sz_gr, sz_gr), dtype=complex128)
        if quad_weight:
            wei = quad_Fourier(N, variety=variety) / (2 * pi)
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = wei[j] * exp(-1j * (i - (N // 2)) * gr[j])
        else:
            for j in range(sz_gr):
                for i in range(sz_gr):
                    B[i, j] = exp(1j * (i - (N // 2)) * gr[j])
        return asarray(B, dtype=dtype)

    raise ValueError("Function arguments malformed.")


# ---------------------------------------------------------------------------
# WaveEquation_S2: parity-split vs domain-extension Laplacian
# ---------------------------------------------------------------------------

class WaveEquation_S2:
    def __init__(self, N_th, N_ph):
        self.N_th = N_th
        self.N_ph = N_ph

        if is_even(N_ph):
            self.lbl_ph = "S1_CL"
        else:
            self.lbl_ph = "S1_I"

        self.gr_th = grid_Fourier(N_th, variety="H1_I")
        self.gr_ph = grid_Fourier(N_ph, variety="S1_I")

        self.gr_th_ext = grid_Fourier(2 * N_th, variety="S1_I")

        self.D02 = diff_mat_nodal_Fourier(N_ph, D=2, variety=self.lbl_ph)

        self.D10_e = diff_mat_nodal_Fourier(N_th, D=1, variety="H1_I_cos")
        self.D10_o = diff_mat_nodal_Fourier(N_th, D=1, variety="H1_I_sin")
        self.D20_e = diff_mat_nodal_Fourier(N_th, D=2, variety="H1_I_cos")
        self.D20_o = diff_mat_nodal_Fourier(N_th, D=2, variety="H1_I_sin")

        self.D10_ext = diff_mat_nodal_Fourier(2 * N_th, D=1,
                                              variety="S1_I")[:N_th, :]
        self.D20_ext = diff_mat_nodal_Fourier(2 * N_th, D=2,
                                              variety="S1_I")[:N_th, :]

        self.cot_th = 1 / tan(self.gr_th[:, None])
        self.csc_th = 1 / sin(self.gr_th[:, None])

        self.wB_th_e = poly_basis_mat_nodal(
            N_th, variety="H1_I_cos", quad_weight=True, dtype=complex128)
        self.wB_th_o = poly_basis_mat_nodal(
            N_th, variety="H1_I_sin", quad_weight=True, dtype=complex128)
        self.wB_ph = poly_basis_mat_nodal(
            N_ph, variety=self.lbl_ph, quad_weight=True, dtype=complex128)
        self.wB_th_ext = poly_basis_mat_nodal(
            2 * N_th, variety="S1_I", quad_weight=True, dtype=complex128)

        self.B_th_e = poly_basis_mat_nodal(
            N_th, variety="H1_I_cos", quad_weight=False, dtype=complex128).T
        self.B_th_o = poly_basis_mat_nodal(
            N_th, variety="H1_I_sin", quad_weight=False, dtype=complex128).T
        self.B_ph = poly_basis_mat_nodal(
            N_ph, variety=self.lbl_ph, quad_weight=False, dtype=complex128).T
        self.B_th_ext = poly_basis_mat_nodal(
            2 * N_th, variety="S1_I", quad_weight=False, dtype=complex128).T

    def Laplacian(self, N00):
        N00_e, N00_o = scalar_field_parity_factor(N00)
        N02 = einsum("ij,kj->ki", self.D02, N00)
        N10_e = einsum("ij,jk->ik", self.D10_e, N00_e)
        N10_o = einsum("ij,jk->ik", self.D10_o, N00_o)
        N10 = N10_e + N10_o
        N20_e = einsum("ij,jk->ik", self.D20_e, N00_e)
        N20_o = einsum("ij,jk->ik", self.D20_o, N00_o)
        N20 = N20_e + N20_o
        return N20 + self.cot_th * N10 + self.csc_th ** 2 * N02

    def Laplacian_ext(self, N00):
        N00_ext = scalar_field_extend(N00)
        N02 = einsum("ij,kj->ki", self.D02, N00)
        N10 = einsum("ij,jk->ik", self.D10_ext, N00_ext)
        N20 = einsum("ij,jk->ik", self.D20_ext, N00_ext)
        return N20 + self.cot_th * N10 + self.csc_th ** 2 * N02

    def nodal2modal(self, N):
        N_e, N_o = scalar_field_parity_factor(N)
        pr_th_e = array_dot_2d_at_axis(self.wB_th_e, N_e, 0)
        pr_th_o = array_dot_2d_at_axis(self.wB_th_o, N_o, 0)
        return array_dot_2d_at_axis(self.wB_ph, pr_th_e + pr_th_o, 1)

    def nodal2modal_ext(self, N):
        N_ext = scalar_field_extend(N)
        pr_th_ext = array_dot_2d_at_axis(self.wB_th_ext, N_ext, 0)
        return array_dot_2d_at_axis(self.wB_ph, pr_th_ext, 1)

    def modal2nodal_ext(self, N):
        return array_dot_2d_at_axis(self.B_ph,
            array_dot_2d_at_axis(self.B_th_ext, N, 0), 1)[:self.N_th, :]

    def filter_orszag_ext(self, N):
        N_ext = scalar_field_extend(N)
        pr_th_ext = array_dot_2d_at_axis(self.wB_th_ext, N_ext, 0)
        pr = array_dot_2d_at_axis(self.wB_ph, pr_th_ext, 1)
        pr[:, -(self.N_ph // 6):] = 0
        pr[:, :self.N_ph // 6] = 0
        pr[-(self.N_th // 3):, :] = 0
        pr[:self.N_th // 3, :] = 0
        return array_dot_2d_at_axis(self.B_ph,
            array_dot_2d_at_axis(self.B_th_ext, pr, 0), 1)[:self.N_th, :]

    def state_vector_flatten_to(self, u, v):
        return hstack([u.flatten(), v.flatten()])

    def state_vector_reconstruct_uv(self, U):
        N_th, N_ph = self.N_th, self.N_ph
        u = U[:N_th * N_ph].reshape((N_th, N_ph))
        v = U[N_th * N_ph:].reshape((N_th, N_ph))
        return u, v

    def RHS(self, t, U):
        u, v = self.state_vector_reconstruct_uv(U)
        return self.state_vector_flatten_to(
            -self.filter_orszag_ext(v),
            -self.filter_orszag_ext(self.Laplacian_ext(u)))

    def _ev_step_RK4(self, t, z, dt):
        dt_2 = dt / 2
        k1 = self.RHS(t, z)
        k2 = self.RHS(t + dt_2, z + dt_2 * k1)
        k3 = self.RHS(t + dt_2, z + dt_2 * k2)
        k4 = self.RHS(t + dt, z + dt * k3)
        return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def evolve_RK4(self, ti, tf, N_t, z_ini, sys_post_step_fcn=None):
        dt = (tf - ti) / N_t
        t = ti
        z = z_ini
        for _t_idx in range(int(N_t)):
            z = self._ev_step_RK4(t, z, dt)
            t = t + dt
        return z


# ---------------------------------------------------------------------------
# Main: spherical harmonic test + wave evolution demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    N_th, N_ph = 32, 32

    gr_th = grid_Fourier(N_th, variety="H1_I")
    if is_even(N_ph):
        gr_ph = grid_Fourier(N_ph, variety="S1_CL")
    else:
        gr_ph = grid_Fourier(N_ph, variety="S1_I")

    # Spherical harmonics
    def scalar_spherical_harmonic(ell, m, th, ph):
        return sps.sph_harm(m, ell, ph, th)

    Y11 = scalar_spherical_harmonic(1, 1, gr_th[:, None], gr_ph[None, :])
    Y20 = scalar_spherical_harmonic(2, 0, gr_th[:, None], gr_ph[None, :])
    Y22 = scalar_spherical_harmonic(2, 2, gr_th[:, None], gr_ph[None, :])
    Y32 = scalar_spherical_harmonic(3, 2, gr_th[:, None], gr_ph[None, :])
    Y5n3 = scalar_spherical_harmonic(5, -3, gr_th[:, None], gr_ph[None, :])
    Y66 = scalar_spherical_harmonic(6, 6, gr_th[:, None], gr_ph[None, :])

    # Laplacian via parity-split
    D02 = diff_mat_nodal_Fourier(N_ph, D=2, variety="S1_CL")
    D10_e = diff_mat_nodal_Fourier(N_th, D=1, variety="H1_I_cos")
    D10_o = diff_mat_nodal_Fourier(N_th, D=1, variety="H1_I_sin")
    D20_e = diff_mat_nodal_Fourier(N_th, D=2, variety="H1_I_cos")
    D20_o = diff_mat_nodal_Fourier(N_th, D=2, variety="H1_I_sin")

    N00_f = Y11 + Y20 + Y32 + Y5n3 + Y66
    N00_f_e, N00_f_o = scalar_field_parity_factor(N00_f)

    N02_f = einsum("ij,kj->ki", D02, N00_f)
    N10_f_e = einsum("ij,jk->ik", D10_e, N00_f_e)
    N10_f_o = einsum("ij,jk->ik", D10_o, N00_f_o)
    N10_f = N10_f_e + N10_f_o
    N20_f_e = einsum("ij,jk->ik", D20_e, N00_f_e)
    N20_f_o = einsum("ij,jk->ik", D20_o, N00_f_o)
    N20_f = N20_f_e + N20_f_o

    cot_th = 1 / tan(gr_th[:, None])
    csc_th = 1 / sin(gr_th[:, None])

    N_Lap = N20_f + cot_th * N10_f + csc_th ** 2 * N02_f
    A_Lap = -2 * Y11 - 6 * Y20 - 12 * Y32 - 30 * Y5n3 - 42 * Y66

    # Laplacian via domain extension
    N00_f_ext = scalar_field_extend(N00_f)
    D10 = diff_mat_nodal_Fourier(N_th * 2, D=1, variety="S1_I")[:N_th, :]
    D20 = diff_mat_nodal_Fourier(N_th * 2, D=2, variety="S1_I")[:N_th, :]
    N10_f_ext = einsum("ij,jk->ik", D10, N00_f_ext)
    N20_f_ext = einsum("ij,jk->ik", D20, N00_f_ext)
    N_Lap_ext = N20_f_ext + cot_th * N10_f_ext + csc_th ** 2 * N02_f

    # Spectral coefficient diagnostic
    we = WaveEquation_S2(N_th, N_ph)
    N_Lap_e, N_Lap_o = scalar_field_parity_factor(N_Lap_ext)
    pr_th = (array_dot_2d_at_axis(we.wB_th_e, N_Lap_e, 0)
             + array_dot_2d_at_axis(we.wB_th_o, N_Lap_o, 0))
    pr = array_dot_2d_at_axis(we.wB_ph, pr_th, 1)

    # --- Plot: spectral coefficients ---
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(abs(pr), aspect='auto', origin='lower')
    ax1.set_title('Laplacian spectral coefficients (abs)')
    ax1.set_xlabel(r'$m$ mode')
    ax1.set_ylabel(r'$\ell$ mode')
    plt.colorbar(im1, ax=ax1)
    ax2.semilogy(abs(pr).ravel(), 'k.', ms=1)
    ax2.set_title('Laplacian spectral coefficients (semilogy)')
    ax2.set_xlabel('flattened mode index')
    ax2.set_ylabel('|coeff|')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'laplacian_s2_spectral.png'), dpi=150)
    plt.close()
    print(f"Max |Laplacian - analytic| = {np.max(np.abs(N_Lap - A_Lap)):.2e}")

    # --- Wave evolution ---
    N_th2, N_ph2 = 24, 24
    WE_S2 = WaveEquation_S2(N_th2, N_ph2)

    # Initial data: superposition of low-ell spherical harmonics
    gr_th2 = WE_S2.gr_th
    gr_ph2 = WE_S2.gr_ph
    N00_init = scalar_spherical_harmonic(
        2, 0, gr_th2[:, None], gr_ph2[None, :])
    N00_init += 0.5 * scalar_spherical_harmonic(
        3, 2, gr_th2[:, None], gr_ph2[None, :])

    z_ini = WE_S2.state_vector_flatten_to(N00_init, 0 * N00_init)
    res = WE_S2.evolve_RK4(0, 1, 1000, z_ini)
    u_fin, _v_fin = WE_S2.state_vector_reconstruct_uv(res)

    coeff = WE_S2.nodal2modal_ext(u_fin)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(abs(coeff), aspect='auto', origin='lower')
    ax.set_title('Wave evolution: final spectral coefficients (abs)')
    ax.set_xlabel(r'$m$ mode')
    ax.set_ylabel(r'$\ell$ mode')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'laplacian_s2_wave_coeffs.png'),
                dpi=150)
    plt.close()
    print("Saved: laplacian_s2_spectral.png, laplacian_s2_wave_coeffs.png")
