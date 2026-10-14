# Spectral fidelity: what relative L2 does not see

Relative L2 error rewards smooth predictions. A blurred field can have a small L2 error and still miss the small
scales that carry heat transfer, mixing or loads, and neural operators are known to under-predict them (spectral
bias). `pinneapple_analysis.validation.spectral` measures what L2 hides. It is NumPy only and works on any field on a
regular grid, from any code.

```python
import numpy as np
from pinneapple_analysis.validation.spectral import spectral_fidelity

report = spectral_fidelity(pred, ref, ndim=2)          # arrays (..., ny, nx): leading axes are averaged
print(report.summary())                                # rel. L2, high-band energy ratio and bias, per-band table
print(report.passed)                                   # False when |bias| > tol (default 0.25)
```

## What is measured

| Quantity | Function | Meaning |
|---|---|---|
| Energy spectrum `E(k)` | `energy_spectrum(u, ndim, periodic=..., window="hann")` | Shell-summed, integer radial wavenumbers `k = rint(|m|)`, `sum E = mean(u^2)` for a periodic field. Non-periodic fields are mean-removed and Hann-windowed (unit mean square). |
| Energy ratio per band, **high-wavenumber bias** | `spectral_comparison`, `high_wavenumber_bias` | `E_pred / E_ref - 1` over the top third of the shells where the reference has energy: about `-1` for a smoothed prediction, `0` for the reference itself, positive for added noise. |
| Phase error and coherence per band | `spectral_comparison` | Energy-weighted mean phase difference [rad]; `|sum P R*|^2 / (sum |P|^2 sum |R|^2)` (1 = the same modes). |
| Structure functions | `structure_functions(u, ndim, orders=(2, 3))` | `S_p(r) = <|u(x + r) - u(x)|^p>` per axis, signed `S_3`, flatness `S_4 / S_2^2` (3 for Gaussian increments; larger at small `r` for intermittent fields). |
| Gradient errors | `gradient_metrics(pred, ref, ndim, spacing, vector_axis=..., walls=...)` | Error in `grad u` and `|grad u|`; for a velocity field, vorticity and strain-rate magnitude; for a scalar, the mean wall flux `-k du/dn` on chosen faces. |
| Power-law slope | `fit_power_law(k, E, kmin, kmax)` | Least-squares slope in log-log. |

## Where it is reported

- **`pinneapple check`**: rule [PC704 `spectral-bias`](physics_checks_in_ci.md#pc704) on every grid result with a
  grid reference of the same shape (tolerance `hf-bias`, default 0.25).
- **`pp.benchmark`**: field problems (Burgers, Darcy, Navier-Stokes) add an "HF energy bias" column next to the
  relative L2 error.
- **`pp.validate`**: part of the validation report (once it lands, #696).

## How it was checked

`tests/test_spectral_fidelity.py`:

- a synthetic field with a prescribed `k^-5/3` spectrum: every shell within 1 % (the construction makes it exact to
  round-off) and the fitted slope within 1 %;
- a 2-D decaying-turbulence velocity field from the pseudo-spectral solver (`pinneapple_physics.multiscale.ns2d`,
  64², `nu = 5e-4`): the field against itself has zero bias, zero phase error and unit coherence; a copy through a
  one-cell Gaussian filter has a relative L2 error below 5 % and a high-wavenumber bias below `-0.5` (flagged); 2 %
  white noise gives a positive bias above 0.25;
- closed forms: `S_2(r)` of a linear field, flatness 3 of Gaussian noise, vorticity and strain of an analytic
  velocity field, and the wall flux of `x^2 + y`.

## Limits

- The bias compares energies, not positions: a field shifted by a few cells keeps a zero bias. The phase error and
  coherence catch that.
- The band edges are fractions of the resolved range of the *reference*. A reference with almost no energy at high
  wavenumbers (smooth solutions) gives a short high band; the check is most informative for multi-scale fields.
- Non-periodic fields are windowed, which spreads energy across neighbouring shells (leakage). Ratios are robust to
  it because prediction and reference get the same window; absolute spectra of non-periodic fields are approximate.
