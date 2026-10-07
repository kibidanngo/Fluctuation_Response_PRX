# Fluctuation-Response Theory of Non-Equilibrium Complex Fluids

Python scripts for the numerical panels in Figs. 1, 2, and A1 of the manuscript by Ryota Takaki and Frank Jülicher. The scripts evaluate the chemically driven fluid example using analytical Laplace-space kernels, numerical inverse Laplace transforms, and correlation-spectrum eigenvalues. No input data files are required.


## Files and manuscript figures

| Script | Manuscript panels | Calculation |
| --- | --- | --- |
| `NumInverse_Laplace_kernels.py` | Fig. 1(a) | Dimensionless regular viscous kernel `-zeta(q,t)/(eta_par * lam)` versus `lam * t` at fixed wave number, comparing several epsilon values. Also generates additional `-Lambda(q,t)` plots. |
| `q_dependence_G_zeta.py` | Fig. 1(b) and Fig. 2(c) | Dimensionless viscous kernel at fixed scaled time and dimensionless storage and loss moduli at fixed scaled frequency, versus `q * xi_mu`. One epsilon value per run. |
| `complex_modulus.py` | Fig. 2(a,b) | Storage and loss moduli divided by `eta_par * lam`, versus `omega / lam` at fixed wave number, comparing several epsilon values. |
| `plot_fig_A1_correlation_spectrum.py` | Fig. A1(a–d) | Smallest eigenvalue of the dimensionless covariance-normalized two-sided spectrum `lam * S`, versus `omega / lam`. One fixed wave number per run. |

## Reproduction notes

The saved epsilon lists for Figs. 1(a) and 2(a,b) both match the manuscript: `[0.1, 0.5, 1.0, 1.5, 1.8, 2.0]`. The first three scripts load the included `plot.mplstyle` from the same directory as the script. The Fig. A1 script defines its own style.

All scripts save to `output_fig` beside the script and create that directory if needed. The figure-specific instructions below describe which settings to change for the remaining panels and which default outputs are additional plots. Individual panel PDFs are generated separately; the manuscript's combined layouts and panel letters are assembled separately.

## Requirements and setup

Requires Python 3, NumPy, Matplotlib, and mpmath. 
The checked runs used Python 3.13.7, NumPy 2.5.2, Matplotlib 3.11.1, and mpmath 1.2.1. 

## Dimensionless parameters and plotting conventions

The normalized mechanical responses in Figs. 1–2 depend on two fixed dimensionless material parameters,

$$
a=\frac{\eta^\parallel}{\rho_0\lambda\xi_\mu^2}=0.2,
\qquad
b=\frac{\kappa_{II}\nu^2}{\rho_0\lambda^2\xi_\mu^2}=0.072,
$$

together with the reciprocity-breaking parameter $\varepsilon$ and the scaled wave number, time, or frequency. 

Fig. A1 additionally depends on the dimensionless covariance ratio

$$
\bar\rho=\frac{\rho_0 g^{\mu\mu}}{\kappa_{II}g^{jj}}=1.
$$

The revised manuscript uses the following plotted quantities:

| Figure | Horizontal coordinate | Vertical quantity |
| --- | --- | --- |
| Fig. 1(a) | $\lambda t$ | $-\zeta(q,t)/(\eta^\parallel\lambda)$, regular part only |
| Fig. 1(b) | $q\xi_\mu$, at $\lambda t=1$ | $-\zeta(q,t)/(\eta^\parallel\lambda)$, regular part only |
| Fig. 2(a,b) | $\omega/\lambda$ | $\widetilde G'/(\eta^\parallel\lambda)$ and $\widetilde G''/(\eta^\parallel\lambda)$ |
| Fig. 2(c) | $q\xi_\mu$, at $\omega/\lambda=1$ | $\widetilde G'/(\eta^\parallel\lambda)$ and $\widetilde G''/(\eta^\parallel\lambda)$ |
| Fig. A1 | $\omega/\lambda$ | $\lambda_{\min}[\lambda\widetilde{\mathbf S}(q,\omega)]$ |

### Numerical parameter values

All four scripts realize the dimensionless parameters above with the following numerical values in arbitrary units:

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
\qquad \widetilde G'={\rm Re}\widetilde G,
\qquad \widetilde G''={\rm Im}\widetilde G.
$$

For $\varepsilon=0$, $\widehat\zeta=\eta^\parallel$, so $\widetilde G'=0$ and $\widetilde G''=\omega\eta^\parallel$. For finite $\varepsilon$, the storage modulus vanishes as $\omega\to0$ at fixed nonzero $q$.

The numerical inverse Laplace transforms use mpmath's Talbot method with 50 decimal digits. The time-domain plots show the regular part of the kernel; the instantaneous `eta_par * delta(t)` contribution is not displayed. A grid point at `t = 0` is evaluated at `1e-14`.

### Changing the unit parameters

The revised labels and unchanged numerical values are consistent because the manuscript calculations use `lam = xi_mu = eta_par = 1`. Some plotting operations rely on these unit choices: the time-domain scripts plot the raw regular kernel, `q_dependence_G_zeta.py` plots the raw moduli, and the Fig. A1 script plots the raw eigenvalue of `S`. In these units, they equal the dimensionless quantities in the table above. Likewise, some coordinates or titles use the numerical physical variables directly.

If changing the unit parameters, apply the scaling to the data and coordinates explicitly: plot `lam * t`, `omega / lam`, `q * xi_mu`, `-zeta / (eta_par * lam)`, the moduli divided by `eta_par * lam`, and `lam * lambda_min(S)`, as appropriate. Changing the parameter values or labels alone is insufficient.

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

Keep `NORMALIZE = False`: the `True` option applies a different, epsilon-dependent normalization by `zeta_c_eps`, not the manuscript's normalization by `eta_par * lam`. With the supplied unit parameters, the `False` branch gives the manuscript's dimensionless numerical values.

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

Use the common frequency grid `omega_fixed = np.logspace(-3, 2, 600)` for all four panels. With `lam = 1`, this covers $10^{-3}\leq\omega/\lambda\leq10^2$. Set `qxi` separately for each panel:

| Panel | `qxi` |
| --- | --- |
| (a) | `0.1` |
| (b) | `0.5` |
| (c) | `1.0` |
| (d) | `10.0` |

Keep `bar_rho = 1.0` and `eps_grid = np.linspace(0.1, 2.0, 30)`. Each run saves `output_fig/S_min_eigen_qxi=<qxi>.pdf` beside the script. This script creates the output directory automatically. `SAVE_FIGURE` and `SHOW_FIGURE` control saving and display.

The covariance-normalized two-sided spectrum is

$$
\widetilde{\mathbf S}(q,\omega)=\widehat{\mathbf P}(q,i\omega)+\widehat{\mathbf P}(q,i\omega)^\dagger,
\qquad
\widehat{\mathbf P}=\mathbf g^{-1/2}\widehat{\boldsymbol\psi}\mathbf g^{-1/2},
$$

using Eqs. (F47)–(F49). This spectrum has units of time. The plotted dimensionless quantity, denoted by `lambda_min` on the vertical axis, is

$$
\lambda_{\min}\left[\lambda\widetilde{\mathbf S}(q,\omega)\right]
=\lambda\times \lambda_{\min}\left[\widetilde{\mathbf S}(q,\omega)\right].
$$

Here $\lambda>0$ is the chemical relaxation rate, whereas $\lambda_{\min}$ denotes the smallest eigenvalue. The additional factor of $\lambda$ makes the spectrum dimensionless and does not change the signs of its eigenvalues. Its numerical value is unchanged for the supplied `lam = 1`.

## Realizability

For positive material parameters and the nonnegative epsilon values considered here, the global realizability condition in Eq. (F52) can be written in terms of the dimensionless parameters as

$$
\frac{b}{4a}\frac{(\bar\rho+\varepsilon)^2}{\bar\rho}
\max[1,a^2]\leq1.
$$

It enforces the spectral condition in Eq. (F51) for all wave numbers and frequencies. With the parameters above and `bar_rho = 1`, it reduces to

$$
0.09(1+\varepsilon)^2\leq1.
$$

All manuscript epsilon values from 0.1 through 2.0 satisfy this condition; its largest value over that interval is 0.81. If changing parameters, check the full realizability condition rather than only the eigenvalues at the displayed wave numbers.

