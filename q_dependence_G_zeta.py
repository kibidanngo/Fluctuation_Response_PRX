#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:17:06 2026

@author: ryota
"""


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# -----------------------------
# Parameters (fixed)
# -----------------------------
rho0 = 5.0
eta_par = 1.0
lam = 1.0
xi_mu = 1.0
kappaII = 1.0
nu = 0.6
eps = 0.1  # Onsager reciprocity parameter

mp.mp.dps = 50

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
plt.style.use(SCRIPT_DIR / "plot.mplstyle")
output_dir = SCRIPT_DIR / "output_fig"
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# USER CHOICES
# -----------------------------
QXI_MIN, QXI_MAX, N_QXI = 0.01, 100, 500

USE_DIMENSIONLESS_LABELS = True
QXI_LOG = True

# sample qxi
if USE_DIMENSIONLESS_LABELS:
    qxi_vals = np.logspace(np.log10(QXI_MIN), np.log10(QXI_MAX), N_QXI)
else:
    qxi_vals = np.linspace(QXI_MIN, QXI_MAX, N_QXI)

# focus values
t0 = 1.0       # lambda t = 1
omega0 = 1.0  # omega/lambda = 1

# -----------------------------
# Revised Laplace-space kernel zeta_hat(s,q)
# -----------------------------
def zeta_hat_param(s, q, eps_val):
    B = s + lam * q**2 * xi_mu**2

    num = rho0 * (
        -eps_val * kappaII * nu**2 * s * (eta_par * q**2 + rho0 * s)
        + eta_par * rho0 * (lam + s) * B**2
    )

    den = (
        eps_val * kappaII * nu**2 * q**2 * (eta_par * q**2 + rho0 * s)
        + rho0**2 * (lam + s) * B**2
    )

    return num / den

# -----------------------------
# Inverse Laplace for zeta(q,t) (Talbot)
# -----------------------------
def zeta_num_eps(t, q, eps_val, eps0=1e-14, method="talbot"):
    return mp.invertlaplace(lambda ss: zeta_hat_param(ss, q, eps_val),
                            max(t, eps0), method=method)

# -----------------------------
# Complex modulus: Gtilde(q,omega) = i omega zeta_hat(q,s=iomega)
# -----------------------------
def G_tilde(omega, q, eps_val):
    s = 1j * omega
    return 1j * omega * complex(zeta_hat_param(s, q, eps_val))

# =========================================================
# Compute zeta(q,t0) vs qxi
# =========================================================
def compute_zeta_vs_qxi(t):
    vals = np.empty_like(qxi_vals, dtype=float)
    for i, qxi in enumerate(qxi_vals):
        q = qxi / xi_mu
        z = complex(zeta_num_eps(t, q, eps))
        vals[i] = np.real(z)
    return vals

# =========================================================
# Compute G'(q,omega0), G''(q,omega0) vs qxi
# =========================================================
def compute_Gparts_vs_qxi(omega):
    gprime = np.empty_like(qxi_vals, dtype=float)
    g2     = np.empty_like(qxi_vals, dtype=float)
    for i, qxi in enumerate(qxi_vals):
        q = qxi / xi_mu
        G = G_tilde(omega, q, eps)
        gprime[i] = np.real(G)
        g2[i]     = np.imag(G)
    return gprime, g2

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

# =========================================================
# (A) zeta(q,t0) vs qxi
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

# highlight qxi = 1 (vertical line)
plt.axvline(1.0, color="k", lw=2.5, ls="--", alpha=0.9, zorder=10, label=r"$q\xi_\mu=1$")

y_zeta = compute_zeta_vs_qxi(t0)
lab_zeta = (rf"$\lambda t={t0}$" if USE_DIMENSIONLESS_LABELS else rf"$t={t0}$")

plt.plot(qxi_vals, -y_zeta, lw=4, alpha=0.9)

if QXI_LOG:
    plt.xscale("log")

plt.xlabel(r"$q\xi_\mu$", fontsize=30, labelpad=16)
plt.tick_params(axis="both", which="both", labelsize=30)
plt.tick_params(axis="x", which="both", pad=6)
plt.ylabel(r"$-\zeta(q,t)/(\eta^\parallel \lambda)$", fontsize=30)
plt.title(rf"$\epsilon={eps}$, {lab_zeta}")
plt.grid(alpha=0.3)
plt.tight_layout()
place_inside_legend(plt.gca())
plt.savefig(output_dir / f"zeta_vs_qxi_t={t0}_eps={eps}.pdf")
plt.show()

# =========================================================
# (B) G' and G'' in the same figure vs qxi at omega0
# =========================================================
plt.figure(figsize=(10, 7), dpi=300)

# highlight qxi = 1 (vertical line)
plt.axvline(1.0, color="k", lw=2.5, ls="--", alpha=0.9, zorder=10, label=r"$q\xi_\mu=1$")
plt.axhline(0, color='k', lw=1.0, ls='-', alpha=0.6)

gp, g2 = compute_Gparts_vs_qxi(omega0)
lab_w = (rf"$\omega/\lambda={omega0}$" if USE_DIMENSIONLESS_LABELS else rf"$\omega={omega0}$")

plt.plot(qxi_vals, gp, '-o', markevery=20, lw=4, color="tab:pink", alpha=0.9, label=rf"$\widetilde{{G}}'(q,\omega)$")
plt.plot(qxi_vals, g2, '-x', markevery=20, lw=4, color="tab:cyan", alpha=0.9, label=rf"$\widetilde{{G}}''(q,\omega)$")

if QXI_LOG:
    plt.xscale("log")

plt.xlabel(r"$q\xi_\mu$", fontsize=30, labelpad=16)
plt.tick_params(axis="both", which="both", labelsize=30)
plt.tick_params(axis="x", which="both", pad=6)
plt.ylabel(r"$\widetilde{G}(q,\omega)/(\eta^\parallel \lambda)$", fontsize=30)
plt.title(rf"$\epsilon={eps}$, {lab_w}")
plt.grid(alpha=0.3)
plt.tight_layout()
place_inside_legend(plt.gca())
plt.savefig(output_dir / f"Gprime_Gdoubleprime_vs_qxi_w={omega0}_eps={eps}.pdf")
plt.show()
