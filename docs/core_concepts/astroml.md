# astroML: periodograms, deconvolved densities and fits with errors

`pinneapple_physics.pde_environment.presets.astroml_bridge` wraps
[astroML](https://github.com/astroML/astroML) (BSD-2-Clause), the code companion of *Statistics, Data Mining, and
Machine Learning in Astronomy*. It sits next to the astropy bridge and returns NumPy arrays, small dataclasses and
`PhysicalSample` objects. The module imports without astroML; calling a function that needs it raises an
`ImportError` that says how to install it.

```bash
pip install astroML                # pulls astropy, scikit-learn, scipy, matplotlib
pip install "pinneapple[astroml]"  # same, through the extra
```

## Quick start

```python
import numpy as np
from pinneapple_physics.pde_environment.presets import astroml_bridge as ab

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 200, 150))                                   # days, irregular
y = 15 + 0.3 * np.sin(2 * np.pi * t / 0.613) + rng.normal(0, 0.15, 150)   # mag

ab.best_period(t, y, 0.15, period_range=(0.2, 10))      # 0.61297 d
pg = ab.lomb_scargle_periodogram(t, y, 0.15, period_range=(0.2, 10))
pg.false_alarm_levels       # {0.01: 0.138, 0.001: 0.169}  power thresholds (Scargle 1982)
pg.peak_power, pg.peak_false_alarm   # 0.633, ~0
pg.to_upd()                  # PhysicalSample: power on a frequency coordinate
ab.light_curve_sample(t, y, 0.15, band="V")   # the light curve itself, point domain

xd = ab.xd_fit(X, Xerr, n_components=2)        # X (N, D); Xerr (N, D, D) covariances or (N, D) sigmas
xd.means, xd.covariances, xd.weights           # the intrinsic mixture, errors deconvolved

ab.bayesian_block_edges(t, x, sigma, fitness="measures", p0=0.001)   # change points of a time series
counts, edges, width = ab.knuth_histogram(data)

ab.fit_line_with_errors(x, y, dy)          # WLS: slope, intercept, covariance, chi2, dof
ab.fit_line_tls(x, y, dx, dy)              # errors on both axes (Hogg, Bovy & Lang 2010)

ab.validate()                              # the synthetic checks below
```

Units: astroML is unit-agnostic. Time is in the unit you pass (usually days); `freq` is in cycles per that
unit, `omega = 2π freq`, periods in the unit of `t`. Periodogram power is the generalised Lomb-Scargle
normalisation (Zechmeister & Kürster 2009), dimensionless in [0, 1].

| function | astroML piece |
|---|---|
| `lomb_scargle_periodogram`, `best_period` | `time_series.lomb_scargle` (generalised, with `significance`) |
| `multiterm_periodogram`, `best_period(n_terms=k)` | `time_series.multiterm_periodogram` |
| `bootstrap_peak_distribution` | `time_series.lomb_scargle_bootstrap` |
| `xd_fit` | `density_estimation.XDGMM` (extreme deconvolution) |
| `bayesian_block_edges`, `knuth_histogram` | `density_estimation.bayesian_blocks`, `knuth_bin_width` |
| `fit_line_with_errors`, `fit_line_tls` | `linear_model.LinearRegression(y_error)`, `linear_model.TLS_logL` |
| `load_dataset(name)` | `datasets.fetch_*` (SDSS, LINEAR, RR Lyrae, ...; see `DATASETS`) |

`best_period` searches a uniform grid with step `1 / (oversample · baseline)` and then refines on 2001 points
around the peak, so the answer is not limited by the coarse step.

`load_dataset` passes `download_if_missing=False` unless told otherwise (astroML's own default downloads).
The datasets are a few MB to several hundred MB and are cached in `~/astroML_data`.

## Check before trusting

`validate()` runs four synthetic checks offline. Measured with astroML 1.0.2, astropy 8.0.1, scikit-learn 1.9.1,
seed 0, and over seeds 0-9:

| check | setup | seed 0 | seeds 0-9 | tolerance |
|---|---|---|---|---|
| period recovery | 150 irregular points over 200 d, P = 0.613 d, amplitude 0.3, noise 0.15 | 0.612967 d, rel. error 5.4e-5 | max 1.9e-4 | 1e-3 |
| XD means | 600 points, two 2-D Gaussians (sd 0.5) at (0, 0) and (4, 3), per-point errors 0.2-1.0 | max mean error 0.072 (0.14 sd) | max 0.19 sd | 0.25 sd |
| Bayesian-block step | 300 points over 10 s, step 1 → 2 at t = 6.3 s, noise 0.2, `p0 = 0.001` | one edge at 6.316 s | one edge each time, max error 0.025 s | 0.05 s |
| line fit, WLS and TLS | y = 2.5 x − 1, 80 points, dx = 0.05, dy = 0.3 | slopes 2.483 (WLS), 2.487 (TLS) | max rel. error 1.0e-2 | 2e-2 |

All four pass on all ten seeds; `validate()` takes about 2 s on one CPU core.

XD deconvolves the width as well as the means: on the seed-0 data the intrinsic standard deviations come out
0.41-0.54 (true 0.5), while a plain scikit-learn `GaussianMixture` on the same noisy points gives 0.75-0.84.

## Limits

- **Scargle false-alarm levels are optimistic on wide grids.** astroML's `significance` assumes M = 2 N
  independent frequencies. For 150 points over 200 d searched over 0.1-5 cycles/day there are about 1000, and
  12 of 200 pure-noise series (6 %) exceeded the nominal 1 % level. Use `bootstrap_peak_distribution` (or a
  narrower search) for a false-alarm probability you intend to quote.
- **Bayesian Blocks `p0` on measured series.** The `p0` prior is calibrated for event data. On the step series
  above, `p0 = 0.01` added a spurious edge in 2 of 10 noise seeds and the default `p0 = 0.05` in more;
  `p0 = 0.001` found only the step in all ten.
- **Aliases.** A periodogram peak can be an alias (1-day alias for ground-based data, harmonics for
  non-sinusoidal shapes). `best_period` returns the highest peak; check `multiterm_periodogram` and the
  phase-folded light curve.
- **Deprecated upstream.** In astroML 1.0 `lomb_scargle`, `multiterm_periodogram`, `bayesian_blocks` and
  `knuth_bin_width` delegate to astropy and emit deprecation warnings; the bridge silences them. If a future
  astroML removes them the bridge will raise, and the astropy equivalents are the replacement.
- **Not wrapped.** `LinearRegressionwithErrors` needs the old PyMC3 (< 3.11); `fit_line_tls`
  maximises `TLS_logL` instead and returns no covariance. XD's mixing matrix `R` is not implemented in astroML.
- XD costs O(N K D³) per EM step and converges to a local maximum; try a few `random_state` values for K > 2.
