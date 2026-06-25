"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Demonstrate AAA rational approximation derivatives via
           rat_D1 and rat_D2.  Includes near-coincident stability
           check (SW vs quotient).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pynalgo import aaa_real, eval_rat, rat_D1, rat_D2


# ---------------------------------------------------------------------------
# Test function: dual-frequency oscillation
# ---------------------------------------------------------------------------

def f_analytic(x):
    return np.sin(5 * x) + 0.5 * np.sin(13 * x)


def f_prime_analytic(x):
    return 5 * np.cos(5 * x) + 6.5 * np.cos(13 * x)


def f_double_analytic(x):
    return -25 * np.sin(5 * x) - 84.5 * np.sin(13 * x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Sample
    x_sample = np.linspace(-3, 3, 500)
    f_sample = f_analytic(x_sample)

    # AAA (real weights)
    z, fv, w = aaa_real(x_sample, f_sample, tol=1e-13, max_terms=80)

    ix_good = np.abs(w) > 1e-14
    z = z[ix_good]
    fv = fv[ix_good]
    w = w[ix_good]
    print(f"AAA converged with {z.size} support points"
          f" (filtered {np.sum(~ix_good)} zero-weight)")

    # Fine evaluation grid
    x_fine = np.linspace(-2.9, 2.9, 600)

    # AAA evaluation
    f_aaa_vals = eval_rat(z, fv, w, x_fine)

    # D1
    df_sw = rat_D1(z, fv, w, x_fine, method='sw')
    df_q  = rat_D1(z, fv, w, x_fine, method='quotient')
    df_mat = rat_D1(z, fv, w, z, method='matrix')

    # D2
    d2f_sw = rat_D2(z, fv, w, x_fine, method='sw')
    d2f_q  = rat_D2(z, fv, w, x_fine, method='quotient')
    d2f_mat = rat_D2(z, fv, w, z, method='matrix')

    # Analytic
    f_exact = f_analytic(x_fine)
    df_exact = f_prime_analytic(x_fine)
    d2f_exact = f_double_analytic(x_fine)
    df_exact_z = f_prime_analytic(z)
    d2f_exact_z = f_double_analytic(z)

    # Error summary
    print(f"Max |f - AAA|      = {np.max(np.abs(f_aaa_vals - f_exact)):.2e}")
    print(f"Max |D1 - exact|    = {np.max(np.abs(df_sw - df_exact)):.2e}")
    print(f"Max |D2 - exact|    = {np.max(np.abs(d2f_sw - d2f_exact)):.2e}")
    print(f"Max |qu D1 - sw D1| = {np.max(np.abs(df_q - df_sw)):.2e}")
    print(f"Max |qu D2 - sw D2| = {np.max(np.abs(d2f_q - d2f_sw)):.2e}")

    # -------------------------------------------------------------------
    # Near-coincident stability: evaluate at z_k +/- h for tiny h
    # -------------------------------------------------------------------
    print("\n--- Near-coincident stability ---")
    h_vals = [1e-8, 1e-10, 1e-12, 1e-14]
    z_test = z[z.size // 2]  # pick a support node near the middle

    for h in h_vals:
        x_near = np.array([z_test + h])
        d1_sw = rat_D1(z, fv, w, x_near, method='sw')[0]
        d1_q  = rat_D1(z, fv, w, x_near, method='quotient')[0]
        d1_ex = f_prime_analytic(x_near)[0]
        d2_sw = rat_D2(z, fv, w, x_near, method='sw')[0]
        d2_q  = rat_D2(z, fv, w, x_near, method='quotient')[0]
        d2_ex = f_double_analytic(x_near)[0]

        print(f"  h={h:.0e}:")
        print(f"    D1  sw={d1_sw:.6e}  q={d1_q:.6e}  exact={d1_ex:.6e}"
              f"  err_sw={abs(d1_sw-d1_ex):.1e}  err_q={abs(d1_q-d1_ex):.1e}")
        print(f"    D2  sw={d2_sw:.6e}  q={d2_q:.6e}  exact={d2_ex:.6e}"
              f"  err_sw={abs(d2_sw-d2_ex):.1e}  err_q={abs(d2_q-d2_ex):.1e}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Function
    ax = axes[0, 0]
    ax.plot(x_sample, f_sample, '.', color='gray', ms=1, alpha=0.5,
            label='samples')
    ax.plot(z, fv, 'xr', ms=6, label=f'support ({z.size})')
    ax.plot(x_fine, f_exact, '-k', lw=1, label='analytic')
    ax.plot(x_fine, f_aaa_vals, '--b', lw=1, label='AAA')
    ax.set_title('Function: sin(5x) + 0.5 sin(13x)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: First derivative
    ax = axes[0, 1]
    ax.plot(x_fine, df_exact, '-k', lw=1.5, label='analytic')
    ax.plot(x_fine, df_sw, '--b', lw=1, label='SW')
    ax.plot(x_fine, df_q, ':g', lw=1, label='quotient')
    ax.plot(z, df_mat, 'xr', ms=5, label='matrix (at z)')
    ax.plot(z, df_exact_z, '.k', ms=3)
    ax.set_title("f'(x)")
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Second derivative
    ax = axes[1, 0]
    ax.plot(x_fine, d2f_exact, '-k', lw=1.5, label='analytic')
    ax.plot(x_fine, d2f_sw, '--b', lw=1, label='SW')
    ax.plot(x_fine, d2f_q, ':g', lw=1, label='quotient')
    ax.plot(z, d2f_mat, 'xr', ms=5, label='matrix (at z)')
    ax.plot(z, d2f_exact_z, '.k', ms=3)
    ax.set_title("f''(x)")
    ax.set_xlabel('x')
    ax.set_ylabel("f''(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Errors
    ax = axes[1, 1]
    ax.semilogy(x_fine, np.abs(f_aaa_vals - f_exact),
                '-k', lw=0.8, label='|AAA - exact|')
    ax.semilogy(x_fine, np.abs(df_sw - df_exact),
                '--b', lw=0.8, label='|SW D1 - exact|')
    ax.semilogy(x_fine, np.abs(d2f_sw - d2f_exact),
                '--r', lw=0.8, label='|SW D2 - exact|')
    ax.semilogy(x_fine, np.abs(df_q - df_sw),
                ':c', lw=0.8, label='|qu - sw| D1')
    ax.semilogy(x_fine, np.abs(d2f_q - d2f_sw),
                ':m', lw=0.8, label='|qu - sw| D2')
    ax.set_title('Error (semilogy)')
    ax.set_xlabel('x')
    ax.set_ylabel('absolute error')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rat_derivatives_demo.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()

#
# :D
#
