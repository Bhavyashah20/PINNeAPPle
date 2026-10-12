"""astroML-backed statistics for astronomical data: periodograms, deconvolved densities, fits with errors.

`astroML <https://github.com/astroML/astroML>`_ (BSD-2-Clause; VanderPlas et al. 2012, CIDU, 47) is the code
companion of *Statistics, Data Mining, and Machine Learning in Astronomy* (Ivezic et al. 2014/2020). This module
is a thin wrapper around it, next to ``astropy_bridge.py``. It returns NumPy arrays, small dataclasses and, where
it fits, PINNeAPPle :class:`PhysicalSample` objects. Nothing here reimplements astroML's algorithms.

What is wrapped
---------------
- ``lomb_scargle_periodogram`` / ``multiterm_periodogram`` / ``best_period``: periodograms of unevenly sampled
  time series (``astroML.time_series``), with Scargle (1982) false-alarm power levels and an optional bootstrap
  of the highest peak (``lomb_scargle_bootstrap``). In astroML 1.0 these functions delegate to
  ``astropy.timeseries.LombScargle`` and are marked deprecated; the deprecation warning is silenced here.
- ``xd_fit``: extreme deconvolution (``astroML.density_estimation.XDGMM``, Bovy et al. 2011): a Gaussian mixture
  for the *underlying* distribution when every point has its own measurement error covariance.
- ``bayesian_block_edges`` / ``knuth_histogram``: adaptive histogram / change-point bins (Scargle et al. 2013;
  Knuth 2006).
- ``fit_line_with_errors`` (weighted least squares, ``astroML.linear_model.LinearRegression`` with ``y_error``) and
  ``fit_line_tls`` (errors on both axes, maximising ``astroML.linear_model.TLS_logL``, Hogg et al. 2010).
- ``load_dataset``: the ``astroML.datasets`` fetchers (SDSS, LINEAR, RR Lyrae, ...). They download tens to hundreds
  of MB; ``download_if_missing`` defaults to ``False`` here, so nothing is downloaded unless asked for.
- ``validate``: synthetic, offline checks (period recovery, XD means, block edge, line fits) to run before
  trusting a setting on real data.

Units
-----
astroML is unit-agnostic. Time ``t`` is in whatever unit you pass (days for most survey light curves, MJD);
frequencies are returned in cycles per that unit (``freq``) and in radians per that unit (``omega = 2 pi freq``);
periods are in the unit of ``t``. Magnitudes / fluxes keep the unit of ``y``. Periodogram power is the
generalised Lomb-Scargle normalisation of Zechmeister & Kurster (2009), dimensionless in [0, 1].

Quick start
-----------
>>> import numpy as np
>>> from pinneapple_physics.pde_environment.presets.astroml_bridge import best_period, lomb_scargle_periodogram
>>> rng = np.random.default_rng(0)
>>> t = np.sort(rng.uniform(0, 100, 200))                       # days, irregular
>>> y = 15 + 0.3 * np.sin(2 * np.pi * t / 0.61) + rng.normal(0, 0.1, t.size)   # mag
>>> best_period(t, y, 0.1, period_range=(0.2, 5.0))              # ~0.61 d
>>> pg = lomb_scargle_periodogram(t, y, 0.1, period_range=(0.2, 5.0))
>>> pg.peak_false_alarm                                           # ~0: the peak is not noise
>>> sample = pg.to_upd()                                          # PhysicalSample over frequency

Install: ``pip install astroML`` (pulls astropy, scikit-learn, scipy, matplotlib) or ``pip install "pinneapple[astroml]"``.
"""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_INSTALL_HINT = (
    "astroML is not installed. Install it with `pip install astroML` "
    "(or `pip install \"pinneapple[astroml]\"`) to use pinneapple's astroML bridge."
)


def astroml_available() -> bool:
    """True if ``astroML`` can be imported (does not import the heavy sub-packages)."""
    import importlib.util

    try:
        return importlib.util.find_spec("astroML") is not None
    except (ImportError, ValueError):
        return False


def _require(submodule: str = "astroML"):
    """Import ``submodule`` of astroML lazily, with a clear message when astroML is missing."""
    import importlib

    try:
        with warnings.catch_warnings():
            # astroML.linear_model warns at import time that PyMC3 is missing (only needed by
            # LinearRegressionwithErrors, which is not wrapped here).
            warnings.simplefilter("ignore", UserWarning)
            return importlib.import_module(submodule)
    except ImportError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "astroML":
            raise                                   # a dependency of astroML is broken; show that error
        raise ImportError(_INSTALL_HINT) from exc


@contextmanager
def _quiet():
    """Silence astroML's own deprecation warnings (its periodograms delegate to astropy's LombScargle)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from astroML.utils.exceptions import AstroMLDeprecationWarning
            warnings.simplefilter("ignore", AstroMLDeprecationWarning)
        except ImportError:
            pass
        yield


def _as_1d(t, y, dy):
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dy = np.broadcast_to(np.asarray(dy, dtype=float), y.shape).copy()
    if t.shape != y.shape:
        raise ValueError(f"t and y must have the same length, got {t.size} and {y.size}")
    if t.size < 3:
        raise ValueError("need at least 3 observations")
    if np.any(dy <= 0):
        raise ValueError("dy must be strictly positive")
    return t, y, dy


# ===========================================================================
# PERIODOGRAMS
# ===========================================================================

@dataclass
class Periodogram:
    """A periodogram on a frequency grid. ``freq`` in cycles per unit of ``t``; ``power`` dimensionless."""

    freq: np.ndarray                   # (F,) [1 / unit of t]
    power: np.ndarray                  # (F,) generalised Lomb-Scargle power in [0, 1]
    method: str                        # "lomb_scargle" or "multiterm"
    n_terms: int = 1
    n_obs: int = 0
    false_alarm_levels: dict = field(default_factory=dict)   # {false-alarm probability: power threshold}
    meta: dict = field(default_factory=dict)

    @property
    def omega(self) -> np.ndarray:
        """Angular frequency [rad / unit of t], the grid astroML itself works on."""
        return 2.0 * np.pi * self.freq

    @property
    def peak_index(self) -> int:
        return int(np.argmax(self.power))

    @property
    def best_frequency(self) -> float:
        return float(self.freq[self.peak_index])

    @property
    def best_period(self) -> float:
        """Period of the highest peak [unit of t]."""
        return 1.0 / self.best_frequency

    @property
    def peak_power(self) -> float:
        return float(self.power[self.peak_index])

    @property
    def peak_false_alarm(self) -> float:
        """Scargle (1982) false-alarm probability of the highest peak, with M = 2 N independent frequencies.

        This is the same approximation astroML uses for ``significance=`` (single-term periodograms only); it is
        a rough guide, not a calibrated p-value. It is optimistic when the grid spans more than ~2 N independent
        frequencies: for 150 points over 200 d searched over 0.1-5 cycles/d, 12 of 200 pure-noise series (6 %)
        exceeded the nominal 1 % level. Use :func:`bootstrap_peak_distribution` for a data-driven estimate.
        """
        if self.method != "lomb_scargle":
            return float("nan")
        return float(_scargle_fap(self.peak_power, self.n_obs))

    def to_upd(self):
        """The periodogram as a :class:`PhysicalSample` with a frequency coordinate."""
        from pinneapple_data.physical_sample import PhysicalSample

        tu = self.meta.get("time_unit", "unit of t")
        return PhysicalSample(
            state={"power": self.power.astype(np.float32)},
            domain={"type": "grid", "coords": {"freq": self.freq.astype(np.float64)}},
            provenance={"version": "0.1", "source": "astroML", "method": f"astroML.time_series.{self.method}",
                        "n_terms": self.n_terms, "n_obs": self.n_obs, "best_period": self.best_period,
                        "false_alarm_levels": {str(k): float(v) for k, v in self.false_alarm_levels.items()},
                        **self.meta},
            schema={"units": {"power": "1", "freq": f"1/{tu}"}},
        )


def _scargle_fap(power, n_obs: int):
    """False-alarm probability of a peak of ``power`` (inverse of astroML's ``significance`` formula)."""
    M = 2 * n_obs
    p_single = np.exp(-0.5 * (n_obs - 1.0) * np.asarray(power, dtype=float))
    return 1.0 - (1.0 - p_single) ** M


def frequency_grid(t, period_range: tuple | None = None, f_min: float | None = None,
                   f_max: float | None = None, oversample: float = 10.0, n_max: int = 200_000) -> np.ndarray:
    """Uniform frequency grid [1 / unit of t] for a periodogram of samples at times ``t``.

    ``period_range=(P_min, P_max)`` sets ``f_min = 1/P_max`` and ``f_max = 1/P_min``. Otherwise ``f_min`` defaults
    to ``1 / baseline`` and ``f_max`` to the pseudo-Nyquist ``N / (2 baseline)``. The step is
    ``1 / (oversample * baseline)``, so a peak is resolved by about ``oversample`` grid points.
    """
    t = np.asarray(t, dtype=float)
    span = float(t.max() - t.min())
    if span <= 0:
        raise ValueError("t must span a positive time baseline")
    if period_range is not None:
        p_lo, p_hi = sorted(float(p) for p in period_range)
        f_min, f_max = 1.0 / p_hi, 1.0 / p_lo
    f_min = 1.0 / span if f_min is None else float(f_min)
    f_max = 0.5 * t.size / span if f_max is None else float(f_max)
    if not 0 < f_min < f_max:
        raise ValueError(f"need 0 < f_min < f_max, got {f_min}, {f_max}")
    n = int(np.ceil((f_max - f_min) * oversample * span)) + 1
    if n > n_max:
        raise ValueError(f"frequency grid would have {n} points (> n_max={n_max}); narrow the range or oversample")
    return np.linspace(f_min, f_max, max(n, 2))


def lomb_scargle_periodogram(t, y, dy, freq=None, *, period_range=None, oversample: float = 10.0,
                             generalized: bool = True, significance: Sequence[float] = (0.01, 0.001),
                             time_unit: str = "d") -> Periodogram:
    """Generalised (floating-mean) Lomb-Scargle periodogram, ``astroML.time_series.lomb_scargle``.

    ``t`` [time_unit], ``y`` and ``dy`` (scalar or per point) in the same unit. ``freq`` [1/time_unit] or a grid
    from :func:`frequency_grid`. ``significance``: false-alarm probabilities for which astroML returns the power
    threshold (Scargle 1982 with M = 2 N), stored in ``false_alarm_levels``.
    """
    ts = _require("astroML.time_series")
    t, y, dy = _as_1d(t, y, dy)
    freq = frequency_grid(t, period_range, oversample=oversample) if freq is None else np.asarray(freq, float)
    with _quiet():
        if significance:
            p, z = ts.lomb_scargle(t, y, dy, 2 * np.pi * freq, generalized=generalized,
                                   significance=np.asarray(significance, float))
            levels = {float(s): float(v) for s, v in zip(significance, np.atleast_1d(z), strict=False)}
        else:
            p, levels = ts.lomb_scargle(t, y, dy, 2 * np.pi * freq, generalized=generalized), {}
    return Periodogram(freq=freq, power=np.asarray(p, float), method="lomb_scargle", n_terms=1, n_obs=t.size,
                       false_alarm_levels=levels, meta={"generalized": generalized, "time_unit": time_unit})


def multiterm_periodogram(t, y, dy, freq=None, *, n_terms: int = 3, period_range=None, oversample: float = 10.0,
                          time_unit: str = "d") -> Periodogram:
    """Multi-harmonic periodogram, ``astroML.time_series.multiterm_periodogram``.

    Fits ``n_terms`` Fourier harmonics at every frequency; better than single-term Lomb-Scargle for strongly
    non-sinusoidal light curves (RR Lyrae, eclipsing binaries), at ``n_terms`` times the cost. Power is
    ``1 - chi2 / chi2_0``. No analytic false-alarm level is attached.
    """
    ts = _require("astroML.time_series")
    t, y, dy = _as_1d(t, y, dy)
    freq = frequency_grid(t, period_range, oversample=oversample) if freq is None else np.asarray(freq, float)
    with _quiet():
        p = ts.multiterm_periodogram(t, y, dy, 2 * np.pi * freq, n_terms=int(n_terms))
    return Periodogram(freq=freq, power=np.asarray(p, float), method="multiterm", n_terms=int(n_terms),
                       n_obs=t.size, meta={"time_unit": time_unit})


def bootstrap_peak_distribution(t, y, dy, freq, n_bootstraps: int = 100, random_state: int | None = 0,
                                generalized: bool = True) -> np.ndarray:
    """Highest periodogram peak of ``n_bootstraps`` resamplings of the data (``lomb_scargle_bootstrap``).

    The fraction of these values above an observed peak is a data-driven false-alarm probability; e.g. the
    99th percentile is the 1 % false-alarm power. Costs ``n_bootstraps`` periodograms.
    """
    ts = _require("astroML.time_series")
    t, y, dy = _as_1d(t, y, dy)
    with _quiet():
        return np.asarray(ts.lomb_scargle_bootstrap(t, y, dy, 2 * np.pi * np.asarray(freq, float),
                                                    generalized=generalized, N_bootstraps=int(n_bootstraps),
                                                    random_state=random_state), float)


def best_period(t, y, dy, *, period_range=None, n_terms: int = 1, oversample: float = 10.0,
                refine: bool = True) -> float:
    """Period [unit of t] of the highest periodogram peak.

    A coarse search on :func:`frequency_grid` is followed (``refine=True``) by a fine grid of 2001 points over
    +-2 coarse steps around the peak, so the result is not limited by the coarse step. ``n_terms > 1`` uses the
    multi-term periodogram. Beware aliases: for ground-based data the 1-day alias can beat the true peak.
    """
    fn: Callable = lomb_scargle_periodogram if n_terms == 1 else multiterm_periodogram
    kw: dict[str, Any] = {} if n_terms == 1 else {"n_terms": n_terms}
    if n_terms == 1:
        kw["significance"] = ()
    pg = fn(t, y, dy, period_range=period_range, oversample=oversample, **kw)
    if not refine or pg.freq.size < 3:
        return pg.best_period
    i, df = pg.peak_index, pg.freq[1] - pg.freq[0]
    fine = np.linspace(max(pg.freq[i] - 2 * df, 1e-12 * df), pg.freq[i] + 2 * df, 2001)
    return fn(t, y, dy, fine, **kw).best_period


def light_curve_sample(t, y, dy, *, band: str = "", time_unit: str = "d", value_unit: str = "mag", **provenance):
    """An irregularly sampled light curve as a point-domain :class:`PhysicalSample` (no astroML needed)."""
    from pinneapple_data.physical_sample import PhysicalSample

    t, y, dy = _as_1d(t, y, dy)
    return PhysicalSample(
        state={"y": y.astype(np.float32), "dy": dy.astype(np.float32)},
        domain={"type": "points", "coords": {"time": t}},
        provenance={"version": "0.1", "source": provenance.pop("source", "light_curve"), "band": band,
                    **provenance},
        schema={"units": {"y": value_unit, "dy": value_unit, "time": time_unit}},
    )


# ===========================================================================
# DENSITY ESTIMATION
# ===========================================================================

@dataclass
class XDResult:
    """Extreme-deconvolution Gaussian mixture of the underlying (error-free) distribution."""

    weights: np.ndarray      # (K,)
    means: np.ndarray        # (K, D)  [unit of X]
    covariances: np.ndarray  # (K, D, D)  [unit of X]^2, intrinsic (measurement errors removed)
    log_likelihood: float    # total log-likelihood of the noisy data
    model: Any = None        # the fitted astroML XDGMM (sample(), logprob_a(), ...)

    def sample(self, n: int, random_state: int | None = None) -> np.ndarray:
        """Draw ``n`` points from the deconvolved mixture."""
        return np.asarray(self.model.sample(n, random_state=random_state))


def _error_covariances(Xerr, n: int, d: int) -> np.ndarray:
    Xerr = np.asarray(Xerr, dtype=float)
    if Xerr.shape == (n, d, d):
        return Xerr
    sig = np.broadcast_to(Xerr, (n, d))
    out = np.zeros((n, d, d))
    idx = np.arange(d)
    out[:, idx, idx] = sig ** 2
    return out


def xd_fit(X, Xerr, n_components: int, *, max_iter: int = 200, tol: float = 1e-5,
           random_state: int | None = 0) -> XDResult:
    """Extreme deconvolution, ``astroML.density_estimation.XDGMM``.

    ``X``: (N, D) measured points. ``Xerr``: (N, D, D) per-point error covariances, or (N, D) / (D,) / scalar
    standard deviations (treated as independent per axis). Returns the intrinsic mixture: unlike a plain
    Gaussian mixture its covariances are not inflated by the measurement errors. EM cost is O(N K D^3) per
    iteration; astroML initialises with a few scikit-learn GaussianMixture steps.
    """
    de = _require("astroML.density_estimation")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    cov = _error_covariances(Xerr, n, d)
    xd = de.XDGMM(int(n_components), max_iter=int(max_iter), tol=float(tol), random_state=random_state)
    xd.fit(X, cov)
    return XDResult(weights=np.asarray(xd.alpha), means=np.asarray(xd.mu), covariances=np.asarray(xd.V),
                    log_likelihood=float(xd.logL(X, cov)), model=xd)


def bayesian_block_edges(t, x=None, sigma=None, *, fitness: str = "events", p0: float = 0.05,
                         gamma: float | None = None) -> np.ndarray:
    """Bin edges from Bayesian Blocks (Scargle et al. 2013), ``astroML.density_estimation.bayesian_blocks``.

    ``fitness="events"``: ``t`` are event times / values (an adaptive histogram). ``"regular_events"``: binary
    ``x`` on a regular grid ``t``. ``"measures"``: a time series ``x(t)`` with errors ``sigma``; the interior edges
    are then change points. ``p0`` is the false-alarm probability for adding a change point; ``gamma`` replaces it
    with the geometric prior when given.
    """
    de = _require("astroML.density_estimation")
    kw: dict[str, Any] = {"gamma": gamma} if gamma is not None else {"p0": p0}
    with _quiet():
        return np.asarray(de.bayesian_blocks(np.asarray(t, float), x, sigma, fitness=fitness, **kw), float)


def knuth_histogram(data) -> tuple[np.ndarray, np.ndarray, float]:
    """Histogram with Knuth's (2006) Bayesian optimal bin width: ``(counts, edges, width)``."""
    de = _require("astroML.density_estimation")
    data = np.asarray(data, dtype=float).ravel()
    with _quiet():
        width, edges = de.knuth_bin_width(data, return_bins=True, disp=False)
    counts, edges = np.histogram(data, bins=edges)
    return counts, np.asarray(edges, float), float(width)


# ===========================================================================
# LINEAR REGRESSION WITH ERRORS
# ===========================================================================

@dataclass
class LineFit:
    """``y = slope * x + intercept`` with a 2x2 covariance of ``(intercept, slope)``."""

    slope: float
    intercept: float
    covariance: np.ndarray     # [[var(b), cov(b, m)], [cov(b, m), var(m)]]; NaN when not available
    chi2: float
    dof: int
    method: str

    @property
    def slope_err(self) -> float:
        return float(np.sqrt(self.covariance[1, 1]))

    @property
    def intercept_err(self) -> float:
        return float(np.sqrt(self.covariance[0, 0]))


def fit_line_with_errors(x, y, dy) -> LineFit:
    """Weighted least squares with errors on ``y`` only, ``astroML.linear_model.LinearRegression(y_error=dy)``.

    The covariance is the textbook ``(A^T C^-1 A)^-1`` of the scaled design matrix (astroML returns only the
    coefficients). It assumes ``dy`` are correct; multiply it by ``chi2 / dof`` if they are not.
    """
    lm = _require("astroML.linear_model")
    x, y, dy = _as_1d(x, y, dy)
    reg = lm.LinearRegression(fit_intercept=True).fit(x[:, None], y, dy)
    b, m = (float(c) for c in reg.coef_)
    A = np.column_stack([np.ones_like(x), x]) / dy[:, None]
    cov = np.linalg.inv(A.T @ A)
    chi2 = float(np.sum(((y - (m * x + b)) / dy) ** 2))
    return LineFit(slope=m, intercept=b, covariance=cov, chi2=chi2, dof=x.size - 2, method="wls")


def fit_line_tls(x, y, dx, dy, initial: tuple | None = None) -> LineFit:
    """Line fit with errors on both axes: maximises ``astroML.linear_model.TLS_logL`` (Hogg, Bovy & Lang 2010).

    The line is parametrised by its normal vector ``v`` (points satisfy ``X . v/|v| = |v|``). ``dx``/``dy`` are
    standard deviations, converted to the variances ``TLS_logL`` expects. ``initial = (slope, intercept)``
    defaults to the WLS fit. The intercept is meaningless for a line through the origin direction
    (``|v| -> 0``); centre the data first if that is a concern. No covariance is returned (NaN).
    """
    lm = _require("astroML.linear_model")
    from scipy.optimize import minimize

    x, y, dy = _as_1d(x, y, dy)
    dx = np.broadcast_to(np.asarray(dx, float), x.shape)
    X = np.column_stack([x, y])
    var = np.column_stack([dx ** 2, dy ** 2])
    if initial is None:
        w = fit_line_with_errors(x, y, dy)
        initial = (w.slope, w.intercept)
    m0, b0 = initial
    v0 = np.array([-m0 * b0, b0]) / (1.0 + m0 ** 2)
    if np.linalg.norm(v0) == 0:
        v0 = np.array([-m0, 1.0]) * 1e-6
    res = minimize(lambda v: -lm.TLS_logL(v, X, var), v0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000})
    v = res.x
    m, b = -v[0] / v[1], float(v @ v) / v[1]
    chi2 = float(np.sum((y - m * x - b) ** 2 / (dy ** 2 + m ** 2 * dx ** 2)))
    return LineFit(slope=float(m), intercept=float(b), covariance=np.full((2, 2), np.nan), chi2=chi2,
                   dof=x.size - 2, method="tls")


# ===========================================================================
# DATASETS (download on demand; never used by the tests)
# ===========================================================================

DATASETS: dict[str, str] = {
    "sdss_specgals": "fetch_sdss_specgals",            # SDSS spectroscopic galaxies (~660k rows)
    "sdss_galaxy_colors": "fetch_sdss_galaxy_colors",
    "sdss_sspp": "fetch_sdss_sspp",                    # SEGUE stellar parameters
    "sdss_s82standards": "fetch_sdss_S82standards",    # Stripe 82 standard stars
    "dr7_quasar": "fetch_dr7_quasar",
    "linear_sample": "fetch_LINEAR_sample",            # ~7000 LINEAR variable-star light curves
    "linear_geneva": "fetch_LINEAR_geneva",
    "rrlyrae_mags": "fetch_rrlyrae_mags",
    "rrlyrae_combined": "fetch_rrlyrae_combined",
    "rrlyrae_templates": "fetch_rrlyrae_templates",
    "moving_objects": "fetch_moving_objects",          # SDSS asteroids
    "great_wall": "fetch_great_wall",
    "ligo_large": "fetch_LIGO_large",
    "hogg2010test": "fetch_hogg2010test",              # bundled, no download: the Hogg et al. line-fit data
}


def load_dataset(name: str, *, data_home: str | None = None, download_if_missing: bool = False, **kwargs):
    """Call an ``astroML.datasets`` fetcher by short name (see ``DATASETS``) and return what it returns.

    Downloads only if ``download_if_missing=True`` (astroML's own default is ``True``); otherwise a dataset that is
    not already in ``data_home`` (default ``~/astroML_data``) raises astroML's ``IOError``. Sizes range from a few
    MB to several hundred MB.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    ds = _require("astroML.datasets")
    fetch = getattr(ds, DATASETS[name])
    params = inspect.signature(fetch).parameters
    if "data_home" in params:
        kwargs["data_home"] = data_home
    if "download_if_missing" in params:
        kwargs["download_if_missing"] = download_if_missing
    return fetch(**kwargs)


# ===========================================================================
# VALIDATION (synthetic, offline)
# ===========================================================================

def check_period_recovery(seed: int = 0, period: float = 0.613, n: int = 150, noise: float = 0.15) -> dict:
    """Irregular sinusoid (amplitude 0.3, Gaussian noise) over 200 d: recovered period vs the true one."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 200.0, n))
    y = 15.0 + 0.3 * np.sin(2 * np.pi * t / period + 0.7) + rng.normal(0, noise, n)
    P = best_period(t, y, noise, period_range=(0.2, 10.0))
    return _row("period recovery", "period [d]", P, period, abs(P - period) / period, 1e-3)


def check_xd_means(seed: int = 0, n: int = 600) -> dict:
    """Two 2-D Gaussians, per-point errors drawn in [0.2, 1.0]: worst error of the XD means / component spread."""
    rng = np.random.default_rng(seed)
    mu = np.array([[0.0, 0.0], [4.0, 3.0]])
    k = rng.random(n) < 0.4
    true = np.where(k[:, None], mu[0], mu[1]) + rng.normal(0, 0.5, (n, 2))
    sig = rng.uniform(0.2, 1.0, (n, 2))
    X = true + rng.normal(0, 1, (n, 2)) * sig
    res = xd_fit(X, sig, 2, random_state=seed)
    order = np.argsort(res.means[:, 0])
    err = float(np.abs(res.means[order] - mu).max())
    return _row("XD means", "max |mu - mu_true| / 0.5", err / 0.5, 0.0, err / 0.5, 0.25,
                means=res.means[order].tolist(), intrinsic_sd=np.sqrt(res.covariances[order][:, [0, 1],
                                                                                            [0, 1]]).tolist())


def check_block_step(seed: int = 0, t_step: float = 6.3, p0: float = 0.001) -> dict:
    """Noisy series (300 points over 10 s, step 1 -> 2, noise 0.2): the interior Bayesian-Blocks edge vs the step.

    Passes only if there is exactly one interior edge and it lies within 0.05 s (about 1.5 sample spacings)
    of the step. astroML's ``p0`` prior is calibrated for event data; on this series ``p0 = 0.01`` added a
    spurious edge in 2 of 10 noise seeds and ``p0 = 0.05`` in more, while ``p0 = 0.001`` found only the step.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 10.0, 300))
    sigma = np.full(t.size, 0.2)
    x = np.where(t < t_step, 1.0, 2.0) + rng.normal(0, 0.2, t.size)
    inner = bayesian_block_edges(t, x, sigma, fitness="measures", p0=p0)[1:-1]
    found = float(inner[np.argmin(np.abs(inner - t_step))]) if inner.size else float("nan")
    err = abs(found - t_step) if inner.size == 1 else float("inf")
    return _row("Bayesian-block step", "change point [s]", found, t_step, err, 0.05, n_inner=int(inner.size))


def check_line_fits(seed: int = 0) -> dict:
    """y = 2.5 x - 1 with errors on both axes: worst relative slope error of the WLS and TLS fits."""
    rng = np.random.default_rng(seed)
    xt = np.linspace(0, 10, 80)
    dx, dy = 0.05, 0.3
    x = xt + rng.normal(0, dx, xt.size)
    y = 2.5 * xt - 1.0 + rng.normal(0, dy, xt.size)
    w = fit_line_with_errors(x, y, dy)
    tl = fit_line_tls(x, y, dx, dy)
    err = max(abs(w.slope - 2.5), abs(tl.slope - 2.5)) / 2.5
    return _row("line fit (WLS, TLS)", "slope", tl.slope, 2.5, err, 0.02, wls_slope=w.slope,
                wls_slope_err=w.slope_err, tls_intercept=tl.intercept, wls_intercept=w.intercept)


CASES: dict[str, Callable] = {f.__name__: f for f in
                               (check_period_recovery, check_xd_means, check_block_step, check_line_fits)}


def validate(cases: list[str] | None = None, log=print, **kw) -> list[dict]:
    """Run the synthetic checks; returns one row per case and logs a table. Needs astroML, no network."""
    rows = []
    for name in cases or list(CASES):
        r = CASES[name](**kw)
        rows.append(r)
        if log:
            log(f"{r['case']:<22} {r['quantity']:<26} got {r['measured']:.6g}  ref {r['reference']:.6g}  "
                f"err {r['error']:.3g}  tol {r['tolerance']:.3g}  {'ok' if r['passed'] else 'FAIL'}")
    return rows


def _row(case, quantity, measured, ref, err, tol, **extra) -> dict:
    return {"case": case, "quantity": quantity, "measured": float(measured), "reference": float(ref),
            "error": float(err), "tolerance": tol, "passed": bool(err <= tol), **extra}


__all__ = [
    "astroml_available", "Periodogram", "frequency_grid", "lomb_scargle_periodogram", "multiterm_periodogram",
    "bootstrap_peak_distribution", "best_period", "light_curve_sample", "XDResult", "xd_fit",
    "bayesian_block_edges", "knuth_histogram", "LineFit", "fit_line_with_errors", "fit_line_tls", "DATASETS",
    "load_dataset", "CASES", "validate",
]
