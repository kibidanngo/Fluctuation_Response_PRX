#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:21:26 2026

@author: ryota
"""


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp  

# ---------------- physical parameters ----------------
rho0     = 5.0     # mass density
eta_par  = 1.0     # longitudinal viscosity η‖
lam      = 1.0     # chemical relaxation rate λ
xi_mu    = 1.0     # chemo-mechanical coupling ξ_μ
kappaII  = 1.0     # chemical susceptibility
nu       = 0.6     # chemo-mechanical coupling
# -----------------------------------------------------

# =====================================================
# USER SWITCHES / RANGES
# =====================================================

# True: log omega axis & log omega grid, False: linear omega axis & grid
OMEGA_LOG = True

# number of omega points
N_OMEGA = 600

# ---- Separate omega ranges for G' and G'' ----
# (You can tune these freely)
OMEGA_MIN_PRIME      = 1e-2
OMEGA_MIN_DBLPRIME   = 1e-1

# For q<1
OMEGA_MAX_PRIME_SMALLQ     = 200.0
OMEGA_MAX_DBLPRIME_SMALLQ  = 3.0

# For q>=1
OMEGA_MAX_PRIME_LARGEQ     = 1000.0
OMEGA_MAX_DBLPRIME_LARGEQ  = 3.0

# ---------------- plots: compare many eps values at fixed q ----------------
#eps_values = [0.1, 0.5, 0.7, 1.0, 3.0, 5.0]
eps_values = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0]
q_targets  = [0.1, 0.5, 1.0, 10]

# gradient colors for eps (small -> large)
cmap = plt.cm.cool
norm = plt.Normalize(vmin=min(eps_values), vmax=max(eps_values))

marker_cycle = ['o','^','s','D','v','>','<','p','h','x','+','*']
linestyle_cycle = ['-','--','-.',':']

# ---- Normalization (optional) ----
G_norm = eta_par * lam

# ---- Output / style ----
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
plt.style.use(SCRIPT_DIR / "plot.mplstyle")
output_dir = SCRIPT_DIR / "output_fig"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# Kernels
# =====================================================
def zeta_hat_param(s, q, eps_val):
    num = (
        rho0 * (
            -eps_val * kappaII * nu**2 * s * (eta_par * q**2 + rho0 * s)
            + eta_par * rho0 * (lam + s) * (s + lam * q**2 * xi_mu**2)**2
        )
    )
    den = (
        eps_val * kappaII * nu**2 * q**2 * (eta_par * q**2 + rho0 * s)
        + rho0**2 * (lam + s) * (s + lam * q**2 * xi_mu**2)**2
    )
    return num / den

def Lambda_hat_param(s, q, eps_val):
    num = (
        nu * rho0 * (lam + s) * (eta_par * q**2 + rho0 * s)
        * (s + lam * q**2 * xi_mu**2)
    )
    den = (
        eps_val * kappaII * nu**2 * q**2 * (eta_par * q**2 + rho0 * s)
        + rho0**2 * (lam + s) * (s + lam * q**2 * xi_mu**2)**2
    )
    return num / den

# ---- Complex moduli G(ω) = i ω * kernel(i ω) ----
def G_zeta(omega, q, eps_val):
    s = 1j * omega
    return 1j * omega * zeta_hat_param(s, q, eps_val)

def G_Lambda(omega, q, eps_val):
    s = 1j * omega
    return 1j * omega * Lambda_hat_param(s, q, eps_val)

# =====================================================
# Omega grid helper (SEPARATE ranges for G' and G'')
# =====================================================
def omega_grid_for(q, which='zeta', part='prime'):
    """
    part: 'prime' for G', 'doubleprime' for G''
    """
    if part == 'prime':
        wmin = OMEGA_MIN_PRIME
        wmax = OMEGA_MAX_PRIME_SMALLQ if q < 1 else OMEGA_MAX_PRIME_LARGEQ
    elif part == 'doubleprime':
        wmin = OMEGA_MIN_DBLPRIME
        wmax = OMEGA_MAX_DBLPRIME_SMALLQ if q < 1 else OMEGA_MAX_DBLPRIME_LARGEQ
    else:
        raise ValueError("part must be 'prime' or 'doubleprime'")

    if OMEGA_LOG:
        wmin = max(wmin, 1e-12)  # avoid 0 for log
        return np.logspace(np.log10(wmin), np.log10(wmax), N_OMEGA)
    else:
        return np.linspace(wmin, wmax, N_OMEGA)

# =====================================================
# Plot functions
# =====================================================
def place_inside_legend(ax):
    """Use the largest tested legend font that clears the plotted curves."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths = [line.get_path().transformed(line.get_transform()) for line in ax.lines]
    bounds = ax.get_window_extent(renderer)
    clearance = 6 * fig.dpi / 72  # clearance for line widths and markers
    for size in (30, 28, 26, 24, 22, 20, 18):
        for columns in (2, 1, 3):
            for location in ('upper left', 'upper right', 'lower left', 'lower right',
                             'center left', 'center right'):
                legend = ax.legend(loc=location, ncol=columns, fontsize=size,
                                   markerscale=1.2, handlelength=1.5,
                                   handletextpad=0.4, columnspacing=0.8)
                box = legend.get_window_extent(renderer).padded(clearance)
                inside = (box.x0 >= bounds.x0 and box.x1 <= bounds.x1
                          and box.y0 >= bounds.y0 and box.y1 <= bounds.y1)
                if inside and not any(path.intersects_bbox(box, filled=False) for path in paths):
                    return legend
                legend.remove()
    # If no clear position exists, reserve headroom inside the axes.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.65 * (hi - lo))
    return ax.legend(loc='upper left', ncol=3, fontsize=24,
                     markerscale=1.2, handlelength=1.5,
                     handletextpad=0.4, columnspacing=0.8)

def plot_Gprime_many_eps_for_q(q, eps_list, which='zeta'):
    omega_vals = omega_grid_for(q, which=which, part='prime')

    plt.figure(figsize=(10, 7), dpi=300)

    # choose plotting function based on omega scaling
    plotx = plt.semilogx if OMEGA_LOG else plt.plot

    for i, epsi in enumerate(eps_list):
        mk = marker_cycle[i % len(marker_cycle)]
        ls = linestyle_cycle[(i // len(marker_cycle)) % len(linestyle_cycle)]
        color = cmap(norm(epsi))

        if which == 'zeta':
            G = np.array([G_zeta(om, q, epsi) for om in omega_vals], dtype=complex)
            ylabel = r"$\widetilde{G}^{\prime}(q,\omega)/(\eta^{\parallel}\lambda)$"
            fname  = output_dir / f"G1Zeta_q={q}_revision.pdf"
        else:
            G = np.array([G_Lambda(om, q, epsi) for om in omega_vals], dtype=complex)
            ylabel = r"$\widetilde{G}^{\prime}_{\Lambda}(q,\omega)/(\eta^{\parallel}\lambda)$"
            fname  = output_dir / f"G1Lambda_q={q}_revision.pdf"

        plotx(
            omega_vals, np.real(G) / G_norm,
            linestyle=ls, marker=mk, color=color,
            alpha=0.9, label=rf'$\epsilon={epsi}$',
            markevery=20, lw=3, ms=8
        )

    plt.xlabel(r'$\omega / \lambda$', fontsize=30, labelpad=16)
    plt.tick_params(axis='both', which='both', labelsize=30)
    plt.tick_params(axis='x', which='both', pad=6)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(rf"$q\xi_\mu={q}$", fontsize=30)
    plt.grid(True, alpha=0.35)
    plt.axhline(0, color='k', lw=1.0, ls='-', alpha=0.6)
    plt.tight_layout()
    place_inside_legend(plt.gca())
    plt.savefig(fname)
    plt.show()

def plot_Gdoubleprime_many_eps_for_q(q, eps_list, which='zeta'):
    omega_vals = omega_grid_for(q, which=which, part='doubleprime')

    plt.figure(figsize=(10, 7), dpi=300)

    # choose plotting function based on omega scaling
    plotx = plt.semilogx if OMEGA_LOG else plt.plot

    for i, epsi in enumerate(eps_list):
        mk = marker_cycle[i % len(marker_cycle)]
        ls = linestyle_cycle[(i // len(marker_cycle)) % len(linestyle_cycle)]
        color = cmap(norm(epsi))

        if which == 'zeta':
            G = np.array([G_zeta(om, q, epsi) for om in omega_vals], dtype=complex)
            ylabel = r"$\widetilde{G}^{\prime\prime}(q,\omega)/(\eta^{\parallel}\lambda)$"
            fname  = output_dir / f"G2Zeta_q={q}_revision.pdf"
        else:
            G = np.array([G_Lambda(om, q, epsi) for om in omega_vals], dtype=complex)
            ylabel = r"$\widetilde{G}^{\prime\prime}_{\Lambda}(q,\omega)/(\eta^{\parallel}\lambda)$"
            fname  = output_dir / f"G2Lambda_q={q}_revision.pdf"

        plotx(
            omega_vals, np.imag(G) / G_norm,
            linestyle=ls, marker=mk, color=color,
            alpha=0.9, label=rf'$\epsilon={epsi}$',
            markevery=20, lw=3, ms=8
        )

    plt.xlabel(r'$\omega / \lambda$', fontsize=30, labelpad=16)
    plt.tick_params(axis='both', which='both', labelsize=30)
    plt.tick_params(axis='x', which='both', pad=6)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(rf"$q\xi_\mu={q}$", fontsize=30)
    plt.axhline(0, color='k', lw=1.0, ls='-', alpha=0.6)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    place_inside_legend(plt.gca())
    plt.savefig(fname)
    plt.show()

# =====================================================
# Run
# =====================================================
for q in q_targets:
    plot_Gprime_many_eps_for_q(q, eps_values, which='zeta')
    plot_Gdoubleprime_many_eps_for_q(q, eps_values, which='zeta')

    # If we also want Lambda:
    # plot_Gprime_many_eps_for_q(q, eps_values, which='Lambda')
    # plot_Gdoubleprime_many_eps_for_q(q, eps_values, which='Lambda')
