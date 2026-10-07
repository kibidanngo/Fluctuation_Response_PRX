#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


# %% Parameters: manuscript Fig. A1
rho0 = 5.0
eta_par = 1.0
lam = 1.0
xi_mu = 1.0
kappa_II = 1.0
nu = 0.6

# bar_rho = rho0 * g_mumu / (kappa_II * g_jj).
bar_rho = 1.0

# Choose ONE q*xi_mu for each run, as in the original code.
qxi = 10
omega_fixed = np.logspace(-3, 2, 600)
eps_grid = np.linspace(0.1, 2.0, 30)

# Original single-panel dimensions and font sizes.
FONT_SIZE = 35
TITLE_SIZE = 30
FIGURE_SIZE = (10.0, 6.0)
LINE_WIDTH = 2.0
SAVE_FIGURE = True
SHOW_FIGURE = True

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "output_fig"


# %% Correlation spectrum
def minimum_spectrum_eigenvalue(qxi, omega, eps):
    """Return lambda_min(S) for one q*xi_mu and epsilon, over omega.

    This is the raw eigenvalue of S, not the additional diagonally normalized
    positivity margin used in the original diagnostic script.
    """
    q = qxi / xi_mu
    s = 1j * np.asarray(omega, dtype=float)
    a = eta_par * q**2 / rho0
    b = lam * q**2 * xi_mu**2
    denominator = (s + lam) * (s + b)
    coupling = q * nu * np.sqrt(kappa_II / rho0)

    P = np.zeros(s.shape + (2, 2), dtype=complex)
    P[..., 0, 0] = 1.0 / (s + a)
    P[..., 1, 1] = 1.0 / (s + lam)
    P[..., 0, 1] = -1j * coupling * np.sqrt(bar_rho) / denominator
    P[..., 1, 0] = -1j * coupling * eps / (np.sqrt(bar_rho) * denominator)

    S = P + P.conj().swapaxes(-1, -2)
    return np.linalg.eigvalsh(S)[..., 0]


# %% Single fixed-q figure
def plot_fig_A1():
    """Create only the original raw-eigenvalue plot for the selected qxi."""
    if min(rho0, eta_par, lam, xi_mu, kappa_II, bar_rho) <= 0:
        raise ValueError("Material parameters and bar_rho must be positive.")
    if qxi <= 0:
        raise ValueError("qxi must be positive.")
    if len(omega_fixed) < 2 or np.any(np.asarray(omega_fixed) <= 0):
        raise ValueError("omega_fixed must contain at least two positive frequencies.")

    style = {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=300)
        cmap = plt.get_cmap("cool")
        norm = Normalize(vmin=np.min(eps_grid), vmax=np.max(eps_grid))
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)

        for eps in eps_grid:
            eigenvalues = minimum_spectrum_eigenvalue(qxi, omega_fixed, eps)
            ax.semilogx(omega_fixed, eigenvalues, color=cmap(norm(eps)),
                        linewidth=LINE_WIDTH, alpha=0.9)

        ax.set_xlabel(r"$\omega/\lambda$")
        ax.set_ylabel(r"$\Lambda_{\rm min}$")
        ax.set_title(rf"$q\xi_\mu={qxi:g},\ \bar{{\rho}}={bar_rho:g}$")
        ax.grid(True, which="both", alpha=0.3)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\epsilon$")
        fig.tight_layout()

        if SAVE_FIGURE:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            destination = OUTPUT_DIR / f"S_min_eigen_qxi={qxi:g}.pdf"
            fig.savefig(destination)
            print(f"Saved: {destination}")

        if SHOW_FIGURE:
            plt.show()

    return fig


if __name__ == "__main__":
    fig = plot_fig_A1()
