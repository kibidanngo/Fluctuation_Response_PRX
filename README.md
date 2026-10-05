# Fluctuation-Response Theory of Non-Equilibrium Complex Fluids

Python scripts for the numerical panels in Figs. 1, 2, and A1 of the manuscript by Ryota Takaki and Frank Jülicher. The scripts evaluate the chemically driven fluid example using analytical Laplace-space kernels, numerical inverse Laplace transforms, and correlation-spectrum eigenvalues. No input data files are required.


## Files and manuscript figures

| Script | Manuscript panels | Calculation |
| --- | --- | --- |
| `NumInverse_Laplace_kernels.py` | Fig. 1(a) | Time-dependent viscous kernel `-zeta(q,t)` at fixed wave number, comparing several epsilon values. Also generates additional `-Lambda(q,t)` plots. |
| `q_dependence_G_zeta.py` | Fig. 1(b) and Fig. 2(c) | Wave-number dependence of `-zeta(q,t)` at fixed time, and of the storage and loss moduli at fixed frequency. One epsilon value per run. |
| `complex_modulus.py` | Fig. 2(a,b) | Frequency-dependent storage and loss moduli at fixed wave number, comparing several epsilon values. |
| `plot_fig_A1_correlation_spectrum.py` | Fig. A1(a–d) | Smallest eigenvalue of the covariance-normalized two-sided correlation spectrum. One fixed wave number per run. |

## Reproduction notes

The saved epsilon lists for Figs. 1(a) and 2(a,b) both match the manuscript: `[0.1, 0.5, 1.0, 1.5, 1.8, 2.0]`. The first three scripts load the included `plot.mplstyle` from the same directory as the script. The Fig. A1 script defines its own style.

All scripts save to `output_fig` beside the script and create that directory if needed. The figure-specific instructions below describe which settings to change for the remaining panels and which default outputs are additional plots.

## Requirements and setup

Requires Python 3, NumPy, Matplotlib, and mpmath. 
The checked runs used Python 3.13.7, NumPy 2.5.2, Matplotlib 3.11.1, and mpmath 1.2.1. 

## Physical parameters and conventions

All four scripts use the following manuscript parameters, in arbitrary units:

| Symbol | Python name | Value |
| --- | --- | --- |
| Mass density, rho_0 | `rho0` | 5.0 |
| Longitudinal viscosity, eta_parallel | `eta_par` | 1.0 |
| Chemical relaxation rate, lambda | `lam` | 1.0 |
| Coupling length, xi_mu | `xi_mu` | 1.0 |
| Chemical susceptibility, kappa_II | `kappaII` or `kappa_II` | 1.0 |
| Chemo-mechanical coupling, nu | `nu` | 0.6 |

The Fig. A1 script additionally uses `bar_rho = 1.0`, where
`bar_rho = rho0 * g_mumu / (kappa_II * g_jj)`.

The complex modulus is evaluated with the convention

$$
\widetilde G(q,\omega)=i\omega\widehat\zeta(q,s=i\omega),
\qquad G'=\operatorname{Re}\widetilde G,
\qquad G''=\operatorname{Im}\widetilde G.
$$

The numerical inverse Laplace transforms use mpmath's Talbot method with 50 decimal digits. The time-domain plots show the regular part of the kernel; the instantaneous `eta_par * delta(t)` contribution is not displayed. A grid point at `t = 0` is evaluated at `1e-14`.

The plotted labels are consistent with the manuscript's unit choices `lam = xi_mu = eta_par = 1`. Some scripts plot numerical `t`, `omega`, or `q` while labelling them as scaled variables. Also, `q_dependence_G_zeta.py` labels the moduli as divided by `eta_par * lam` but plots their unscaled numerical values. If these unit parameters are changed, the plotting coordinates and normalizations must be updated as well; changing the parameter values alone is insufficient.

## Reproducing Fig. 1

### Panel (a): time-dependent viscous kernel

Run `NumInverse_Laplace_kernels.py` with:

```python
NORMALIZE = False
TIME_LOG = False
eps_values = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0]
q_targets = [0.1, 1.0, 10]
```

The time grid is 100 linearly spaced points from 0 to 5. The saved script also includes `q = 0.5` and `q = 5.0`; those are additional plots, not panels of Fig. 1.

Use `output_fig/Zeta_eps_compare_q=<q>_revision.pdf` for the three manuscript panels. The simultaneously generated `Lambda_eps_compare_q=<q>_revision.pdf` files are additional coupling-kernel plots.

### Panel (b): wave-number dependence at fixed time

Run `q_dependence_G_zeta.py` separately with `eps = 0.1`, `1.0`, and `2.0`. Keep `t0 = 1.0`, `QXI_MIN = 0.01`, `QXI_MAX = 100`, `N_QXI = 500`, `USE_DIMENSIONLESS_LABELS = True`, and `QXI_LOG = True`.

Use `output_fig/zeta_vs_qxi_t=1.0_eps=<eps>.pdf`. The saved default produces only the `eps = 2.0` panel. Each run also produces the corresponding Fig. 2(c) plot.

## Reproducing Fig. 2

### Panels (a,b): storage and loss moduli versus frequency

In `complex_modulus.py`, set:

```python
eps_values = [0.1, 0.5, 1.0, 1.5, 1.8, 2.0]
q_targets = [0.1, 1.0, 10]
```

Keep `OMEGA_LOG = True` and `N_OMEGA = 600`. The default frequency ranges are:

| Quantity | q < 1 | q >= 1 |
| --- | --- | --- |
| Storage modulus | 0.01 to 200 | 0.01 to 1000 |
| Loss modulus | 0.1 to 3 | 0.1 to 3 |

Use `output_fig/G1Zeta_q=<q>_revision.pdf` for the storage modulus and `output_fig/G2Zeta_q=<q>_revision.pdf` for the loss modulus. The saved script additionally runs `q = 0.5`.

### Panel (c): moduli versus wave number

Run `q_dependence_G_zeta.py` separately with `eps = 0.1`, `1.0`, and `2.0`, keeping `omega0 = 1.0` and the wave-number grid specified for Fig. 1(b).

Use `output_fig/Gprime_Gdoubleprime_vs_qxi_w=1.0_eps=<eps>.pdf`. Both moduli appear in each PDF; the dashed vertical line marks `q * xi_mu = 1`.

## Reproducing Fig. A1

Set `qxi` and `omega_fixed` for each panel:

| Panel | `qxi` | `omega_fixed` |
| --- | --- | --- |
| (a) | `0.1` | `np.logspace(-3, 1, 600)` |
| (b) | `0.5` | `np.logspace(-2, 2, 600)` |
| (c) | `1.0` | `np.logspace(-2, 2, 600)` |
| (d) | `10.0` | `np.logspace(-2, 2, 600)` |

Keep `bar_rho = 1.0` and `eps_grid = np.linspace(0.1, 2.0, 30)`. Each run saves `output_fig/S_min_eigen_qxi=<qxi>.pdf` beside the script. This script creates the output directory automatically. `SAVE_FIGURE` and `SHOW_FIGURE` control saving and display.

The plotted quantity is the smallest eigenvalue of

$$
\widetilde S(q,\omega)=\widehat P(q,i\omega)+\widehat P(q,i\omega)^\dagger,
\qquad
\widehat P=g^{-1/2}\widehat\psi\,g^{-1/2},
$$

using Eqs. (F47)–(F49). It is the eigenvalue of the covariance-normalized spectrum itself, without any further division by its diagonal entries.

## Realizability

For the parameters above and `bar_rho = 1`, the global realizability condition in Eq. (F52), which enforces the spectral condition in Eq. (F51) for all wave numbers and frequencies, reduces to

$$
0.09(1+\varepsilon)^2\leq1.
$$

All manuscript epsilon values from 0.1 through 2.0 satisfy this condition; its largest value over that interval is 0.81. If changing parameters, check the full realizability condition rather than only the eigenvalues at the displayed wave numbers.

