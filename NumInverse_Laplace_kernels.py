#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:18:47 2026

@author: ryota
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


# ---------------- USER SWITCH ----------------
# If True: plot normalized curves
# If False: plot raw kernels
NORMALIZE = False

# If True: use log-spaced t grid + log x-axis
# If False: use linear-spaced t grid + linear x-axis
TIME_LOG = False
# --------------------------------------------

# ---------------- physical parameters ----------------
# All parameters are set to 1 by default
rho0     = 5.0    # mass density rho_0
eta_par  = 1.0    # longitudinal viscosity eta_parallel
lam      = 1.0    # chemical relaxation rate lambda
xi_mu    = 1.0    # chemo-mechanical coupling length xi_mu
kappaII  = 1.0    # susceptibility kappa_II
nu       = 0.6    # chemo-mechanical coupling, units 1/volume
eps      = 1.0    # reciprocity-breaking parameter
# -----------------------------------------------------


# =====================================================
# Revised Laplace-space expressions
# =====================================================

def den_common(s, q, eps_val):
    """
    Common denominator:
        eps*kappaII*nu^2*q^2*(eta_par*q^2 + rho0*s)
        + rho0^2*(lambda+s)*(s+lambda*q^2*xi_mu^2)^2
    """
    B = s + lam * q**2 * xi_mu**2
    return (
        eps_val * kappaII * nu**2 * q**2 * (eta_par * q**2 + rho0 * s)
        + rho0**2 * (lam + s) * B**2
    )


# ========== Laplace-space zeta ==========
def zeta_hat(s, q):
    B = s + lam * q**2 * xi_mu**2
    num = rho0 * (
        -eps * kappaII * nu**2 * s * (eta_par * q**2 + rho0 * s)
        + eta_par * rho0 * (lam + s) * B**2
    )
    den = den_common(s, q, eps)
    return num / den


# ========== Laplace-space Lambda ==========
def Lambda_hat(s, q):
    B = s + lam * q**2 * xi_mu**2
    num = (
        nu * rho0 * (lam + s)
        * (eta_par * q**2 + rho0 * s)
        * B
    )
    den = den_common(s, q, eps)
    return num / den


# ========== Laplace-space Xi ==========
def Xi_hat(s, q):
    B = s + lam * q**2 * xi_mu**2
    num = (
        -eps * kappaII * nu * rho0 * (lam + s)
        * (eta_par * q**2 + rho0 * s)
        * B
    )
    den = kappaII * den_common(s, q, eps)
    return num / den


# ========== Laplace-space Upsilon ==========
def Upsilon_hat(s, q):
    B = s + lam * q**2 * xi_mu**2
    num = (
        -eps * kappaII * nu**2 * q**2 * s * (eta_par * q**2 + rho0 * s)
        + lam * rho0**2 * (lam + s) * B**2
    )
    den = kappaII * den_common(s, q, eps)
    return num / den


# ========== numerical inversion via Talbot contour ==========
def zeta_num(t, q, eps0=1e-14):
    return mp.invertlaplace(lambda s: zeta_hat(s, q),
                            max(t, eps0),
                            method='talbot')

def Lambda_num(t, q, eps0=1e-14):
    return mp.invertlaplace(lambda s: Lambda_hat(s, q),
                            max(t, eps0),
                            method='talbot')

def Xi_num(t, q, eps0=1e-14):
    return mp.invertlaplace(lambda s: Xi_hat(s, q),
                            max(t, eps0),
                            method='talbot')

def Upsilon_num(t, q, eps0=1e-14):
    return mp.invertlaplace(lambda s: Upsilon_hat(s, q),
                            max(t, eps0),
                            method='talbot')


# precision settings
mp.mp.dps = 50
DEG = 20

# time grid
t_min, t_max = 0.001, 200.0
num_points = 100

t_vals = lam * np.logspace(np.log10(t_min), np.log10(t_max), num_points)
# t_vals = np.linspace(t_min, t_max, num_points)

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
plt.style.use(SCRIPT_DIR / "plot.mplstyle")
output_dir = SCRIPT_DIR / "output_fig"
output_dir.mkdir(parents=True, exist_ok=True)


# =====================================================
# Helper: revised Laplace-space with explicit eps
# =====================================================

def zeta_hat_param(s, q, eps_val):
    B = s + lam * q**2 * xi_mu**2
    num = rho0 * (
        -eps_val * kappaII * nu**2 * s * (eta_par * q**2 + rho0 * s)
        + eta_par * rho0 * (lam + s) * B**2
    )
    den = den_common(s, q, eps_val)
    return num / den


def Lambda_hat_param(s, q, eps_val):
    B = s + lam * q**2 * xi_mu**2
    num = (
        nu * rho0 * (lam + s)
        * (eta_par * q**2 + rho0 * s)
        * B
    )
    den = den_common(s, q, eps_val)
    return num / den


def Xi_hat_param(s, q, eps_val):
    B = s + lam * q**2 * xi_mu**2
    num = (
        -eps_val * kappaII * nu * rho0 * (lam + s)
        * (eta_par * q**2 + rho0 * s)
        * B
    )
    den = kappaII * den_common(s, q, eps_val)
    return num / den


def Upsilon_hat_param(s, q, eps_val):
    B = s + lam * q**2 * xi_mu**2
    num = (
        -eps_val * kappaII * nu**2 * q**2 * s * (eta_par * q**2 + rho0 * s)
        + lam * rho0**2 * (lam + s) * B**2
    )
    den = kappaII * den_common(s, q, eps_val)
    return num / den


# =====================================================
# Numerical inversion wrappers that pass eps explicitly
# =====================================================

def zeta_num_eps(t, q, eps_val, eps0=1e-14, method='talbot'):
    return mp.invertlaplace(lambda s: zeta_hat_param(s, q, eps_val),
                            max(t, eps0), method=method)

def Lambda_num_eps(t, q, eps_val, eps0=1e-14, method='talbot'):
    return mp.invertlaplace(lambda s: Lambda_hat_param(s, q, eps_val),
                            max(t, eps0), method=method)

def Xi_num_eps(t, q, eps_val, eps0=1e-14, method='talbot'):
    return mp.invertlaplace(lambda s: Xi_hat_param(s, q, eps_val),
                            max(t, eps0), method=method)

def Upsilon_num_eps(t, q, eps_val, eps0=1e-14, method='talbot'):
    return mp.invertlaplace(lambda s: Upsilon_hat_param(s, q, eps_val),
                            max(t, eps0), method=method)


# =====================================================
# Normalizations
# =====================================================
# These are only used if NORMALIZE = True.

def zeta_c_eps(eps_val):
    return -eps_val * kappaII * nu**2 / lam

def N_Lambda(q):
    return nu


# ---------------- plots: compare many eps values at fixed q ----------------
#eps_values = [0.1, 0.5, 0.7, 1.0, 3.0, 5.0]
eps_values = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0]
q_targets  = [0.1, 0.5, 1.0, 5.0, 10]

# gradient colors for eps
cmap = plt.cm.cool
norm = plt.Normalize(vmin=min(eps_values), vmax=max(eps_values))

marker_cycle = ['o', '^', 's', 'D', 'v', '>', '<', 'p', 'h', 'x', '+', '*']
linestyle_cycle = ['-', '--', '-.', ':']


for q in q_targets:
    if TIME_LOG:
        if q < 1:
            t_min, t_max = 0.0001, 1000.0
            num_points = 100
        else:
            t_min, t_max = 0.0001, 1000.0
            num_points = 100
    else:
        if q < 1:
            t_min, t_max = 0, 5.0
            num_points = 100
        else:
            t_min, t_max = 0, 5.0
            num_points = 100

    # --- choose time grid based on TIME_LOG ---
    if TIME_LOG:
        t_vals = np.logspace(np.log10(t_min), np.log10(t_max), num_points)
    else:
        t_vals = np.linspace(t_min, t_max, num_points)

    # =====================================================
    # zeta: many eps in one figure
    # =====================================================
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(0, color='k', lw=1.0, ls='-', alpha=0.6)

    for i, epsi in enumerate(eps_values):
        mk = marker_cycle[i % len(marker_cycle)]
        ls = linestyle_cycle[(i // len(marker_cycle)) % len(linestyle_cycle)]
        color = cmap(norm(epsi))

        z_arr = np.array([complex(zeta_num_eps(t, q, epsi)) for t in t_vals], dtype=complex)

        if NORMALIZE:
            y = np.real(z_arr) / zeta_c_eps(epsi)
            ylab = r'$\zeta(q,t)/\zeta_c(\varepsilon)$'
        else:
            y = -np.real(z_arr)
            ylab = r'$-\zeta(q,t)$'

        plt.plot(
            t_vals, y,
            linestyle=ls, marker=mk, color=color,
            alpha=0.9, label=rf'$\epsilon={epsi}$',
            markevery=3, lw=3, ms=8
        )

    if TIME_LOG:
        plt.xscale('log')

    plt.xlabel(r'$\lambda\,t$')
    plt.ylabel(ylab)
    plt.title(rf'$q\xi_\mu=${q}', size=35)

    plt.legend(markerscale=1.2, ncol=2, fontsize=25)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"Zeta_eps_compare_q={q}_revision.pdf")
    plt.show()

    # =====================================================
    # Lambda: many eps in one figure
    # =====================================================
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(0, color='k', lw=1.0, ls='-', alpha=0.6)

    for i, epsi in enumerate(eps_values):
        mk = marker_cycle[i % len(marker_cycle)]
        ls = linestyle_cycle[(i // len(marker_cycle)) % len(linestyle_cycle)]
        color = cmap(norm(epsi))

        Lam_arr = np.array([complex(Lambda_num_eps(t, q, epsi)) for t in t_vals], dtype=complex)

        if NORMALIZE:
            y = np.real(Lam_arr) / N_Lambda(q)
            ylab = r'$\Lambda(q,t)/\Lambda_c(q)$'
        else:
            y = -np.real(Lam_arr)
            ylab = r'$-\Lambda(q,t)$'

        plt.plot(
            t_vals, y,
            linestyle=ls, marker=mk, color=color,
            alpha=0.9, label=rf'$\epsilon={epsi}$',
            markevery=3, lw=3, ms=8
        )

    if TIME_LOG:
        plt.xscale('log')

    plt.xlabel(r'$\lambda\,t$')
    plt.ylabel(ylab)
    plt.title(rf'$q \xi_\mu=${q}', size=35)

    plt.legend(markerscale=1.2, ncol=2, fontsize=25)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"Lambda_eps_compare_q={q}_revision.pdf")
    plt.show()
