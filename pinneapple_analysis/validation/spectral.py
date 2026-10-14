"""Spectral fidelity of a predicted field: energy spectra, high-wavenumber bias, phase error, coherence, structure
functions and gradient-based errors (NumPy only, no torch).

Relative L2 error rewards smooth predictions: a blurred field can have a small L2 error and still miss the small
scales that carry heat transfer, mixing or loads. The metrics here measure what L2 hides.

``energy_spectrum``
    Radially binned spectrum ``E(k)`` of a field on a regular grid, integer shells ``k = rint(|m|)`` (``m`` the signed
    mode index), shell-summed so that ``sum_k E(k) = mean(u^2)`` (Parseval). Leading axes (samples, time, channels)
    are averaged. Non-periodic fields are mean-removed and multiplied by a Hann window normalised to unit mean square,
    so the windowed spectrum keeps the field's energy on average.
``spectral_comparison``
    Prediction against reference per shell and per band: energy ratio, phase error and coherence. The **high-wavenumber
    bias** is ``E_pred / E_ref - 1`` summed over the high band: about ``-1`` for a field with its small scales smoothed
    away, ``0`` for the reference against itself, positive for added noise. Bands are fractions of the *resolved*
    range: the shells where the reference has energy above ``floor * max(E_ref)``, so a dealiased reference (exact
    zeros above ``n / 3``) does not divide by zero.
``structure_functions``
    ``S_p(r) = <|u(x + r) - u(x)|^p>`` along each grid axis (averaged over the axes), the signed third-order function
    and the flatness ``S_4 / S_2^2`` (3 for a Gaussian field; larger at small ``r`` for an intermittent field).
``gradient_metrics``
    Errors in ``|grad u|``, in the full gradient, and for a velocity field in vorticity and strain-rate magnitude, plus
    the error of a wall flux ``-k du/dn`` on declared faces.
``spectral_fidelity``
    All of the above in one :class:`SpectralFidelityReport` with a pass/fail on the high-wavenumber bias, used by
    ``pinneapple check`` (PC704), by ``pp.benchmark`` field problems and by ``pp.validate``.

Layout: arrays are ``(..., *space)``: any leading axes followed by ``ndim`` spatial axes (as in
:mod:`pinneapple_analysis.validation.field_metrics`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "energy_spectrum", "fit_power_law", "spectral_comparison", "SpectralComparison", "high_wavenumber_bias",
    "structure_functions", "StructureFunctions", "gradient_metrics", "spectral_fidelity", "SpectralFidelityReport",
]


def _axes(ndim: int) -> tuple[int, ...]:
    return tuple(range(-ndim, 0))


def _check(u: np.ndarray, ndim: int) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if not 1 <= ndim <= u.ndim:
        raise ValueError(f"ndim must be between 1 and {u.ndim}")
    return u


def _window(shape: Sequence[int]) -> np.ndarray:
    w = np.ones(())
    for n in shape:
        h = np.hanning(n) if n > 2 else np.ones(n)
        h = h / np.sqrt(np.mean(h ** 2))
        w = np.multiply.outer(w, h)
    return w


def _prepare(u: np.ndarray, ndim: int, periodic: bool, window: str | None) -> np.ndarray:
    if periodic:
        return u
    u = u - u.mean(axis=_axes(ndim), keepdims=True)
    if window in (None, "none"):
        return u
    if window != "hann":
        raise ValueError(f"unknown window {window!r} (use 'hann' or None)")
    return u * _window(u.shape[-ndim:])


def _fft(u: np.ndarray, ndim: int, periodic: bool, window: str | None) -> np.ndarray:
    v = _prepare(u, ndim, periodic, window)
    n_total = int(np.prod(v.shape[-ndim:]))
    return np.fft.fftn(v, axes=_axes(ndim)) / n_total


def _shells(shape: Sequence[int]) -> np.ndarray:
    grids = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    return np.rint(np.sqrt(sum(g ** 2 for g in grids))).astype(int)


def _bin(values: np.ndarray, shell: np.ndarray, ndim: int, n_shells: int) -> np.ndarray:
    """Sum ``values`` (``(..., *space)``) per shell, then average the leading axes -> ``(n_shells,)``."""
    flat = values.reshape(-1, shell.size)
    return np.stack([np.bincount(shell.ravel(), weights=row, minlength=n_shells)[:n_shells] for row in flat]).mean(0)


def _nyquist(shape: Sequence[int]) -> int:
    return int(min(shape) // 2)


def energy_spectrum(u: np.ndarray, ndim: int | None = None, *, periodic: bool = True, window: str | None = "hann",
                    kmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(k, E)``: shell-summed spectrum of ``u`` (``(..., *space)``), ``k = 0 .. kmax`` in mode-index units
    (multiply by ``2 pi / L`` for a physical wavenumber), ``sum(E) == mean(u**2)`` for a periodic field.

    ``kmax`` defaults to the Nyquist shell of the smallest axis: shells beyond it are only partly filled (the corners
    of the box) and would bias a radial average."""
    u = _check(u, ndim or np.asarray(u).ndim)
    nd = ndim or u.ndim
    F = _fft(u, nd, periodic, window)
    shape = u.shape[-nd:]
    shell = _shells(shape)
    kmax = _nyquist(shape) if kmax is None else int(kmax)
    E = _bin(np.abs(F) ** 2, shell, nd, int(shell.max()) + 1)
    return np.arange(kmax + 1), E[:kmax + 1]


def fit_power_law(k: np.ndarray, E: np.ndarray, kmin: float, kmax: float) -> tuple[float, float]:
    """Least-squares fit ``E = C k^slope`` on ``kmin <= k <= kmax`` (log-log); returns ``(slope, C)``."""
    k, E = np.asarray(k, float), np.asarray(E, float)
    sel = (k >= kmin) & (k <= kmax) & (E > 0)
    if sel.sum() < 2:
        raise ValueError("fewer than two positive shells in the fit range")
    slope, logc = np.polyfit(np.log(k[sel]), np.log(E[sel]), 1)
    return float(slope), float(np.exp(logc))


@dataclass
class SpectralComparison:
    k: np.ndarray
    E_pred: np.ndarray
    E_ref: np.ndarray
    ratio: np.ndarray                       # E_pred / E_ref per shell (nan where the reference has no energy)
    phase_error: np.ndarray                 # energy-weighted mean |phase difference| per shell [rad]
    coherence: np.ndarray                   # |sum P R*|^2 / (sum |P|^2 sum |R|^2) per shell, 1 = same modes
    bands: dict[str, tuple[int, int]]       # band -> (first shell, last shell), inclusive
    band_ratio: dict[str, float]
    band_phase_error: dict[str, float]
    band_coherence: dict[str, float]
    hf_bias: float                          # band_ratio["high"] - 1

    def to_dict(self) -> dict[str, Any]:
        return {"bands": {b: list(v) for b, v in self.bands.items()}, "band_ratio": self.band_ratio,
                "band_phase_error": self.band_phase_error, "band_coherence": self.band_coherence,
                "hf_bias": self.hf_bias, "k": self.k.tolist(), "E_pred": self.E_pred.tolist(),
                "E_ref": self.E_ref.tolist()}


def _bands(E_ref: np.ndarray, fractions: Sequence[float], floor: float) -> dict[str, tuple[int, int]]:
    resolved = np.nonzero(E_ref[1:] > floor * max(E_ref[1:].max(initial=0.0), 1e-300))[0] + 1
    if resolved.size == 0:
        raise ValueError("the reference has no energy above the floor at k >= 1")
    top = int(resolved.max())
    a, b = (max(1, int(round(f * top))) for f in fractions)
    b = max(b, a + 1)
    return {"low": (1, a - 1) if a > 1 else (1, 1), "mid": (a, b - 1), "high": (min(b, top), top)}


def spectral_comparison(pred: np.ndarray, ref: np.ndarray, ndim: int, *, periodic: bool = True,
                        window: str | None = "hann", bands: Sequence[float] = (1 / 3, 2 / 3),
                        floor: float = 1e-10) -> SpectralComparison:
    """Compare the spectra of ``pred`` and ``ref`` (same shape).

    ``bands`` are fractions of the highest resolved shell: low ``[1, f1)``, mid ``[f1, f2)``, high ``[f2, top]``."""
    pred, ref = _check(pred, ndim), _check(ref, ndim)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    shape = pred.shape[-ndim:]
    P, R = _fft(pred, ndim, periodic, window), _fft(ref, ndim, periodic, window)
    shell = _shells(shape)
    ns = int(shell.max()) + 1
    kmax = _nyquist(shape)
    Ep = _bin(np.abs(P) ** 2, shell, ndim, ns)[:kmax + 1]
    Er = _bin(np.abs(R) ** 2, shell, ndim, ns)[:kmax + 1]
    cross = P * np.conj(R)
    cre = _bin(cross.real, shell, ndim, ns)[:kmax + 1]
    cim = _bin(cross.imag, shell, ndim, ns)[:kmax + 1]
    amp = _bin(np.abs(cross), shell, ndim, ns)[:kmax + 1]
    wphase = _bin(np.abs(cross) * np.abs(np.angle(cross)), shell, ndim, ns)[:kmax + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 1e-300
        ratio = np.where(Er > floor * Er[1:].max(initial=0.0), Ep / np.maximum(Er, eps), np.nan)
        phase = np.where(amp > 0, wphase / np.maximum(amp, eps), np.nan)
        coh = np.where(Ep * Er > 0, (cre ** 2 + cim ** 2) / np.maximum(Ep * Er, eps), np.nan)
    B = _bands(Er, bands, floor)

    def band(fn):
        return {b: fn(lo, hi) for b, (lo, hi) in B.items()}

    band_ratio = band(lambda lo, hi: float(Ep[lo:hi + 1].sum() / max(Er[lo:hi + 1].sum(), eps)))
    band_phase = band(lambda lo, hi: float(wphase[lo:hi + 1].sum() / max(amp[lo:hi + 1].sum(), eps)))
    band_coh = band(lambda lo, hi: float((cre[lo:hi + 1].sum() ** 2 + cim[lo:hi + 1].sum() ** 2)
                                         / max(Ep[lo:hi + 1].sum() * Er[lo:hi + 1].sum(), eps)))
    return SpectralComparison(np.arange(kmax + 1), Ep, Er, ratio, phase, coh, B, band_ratio, band_phase, band_coh,
                              band_ratio["high"] - 1.0)


def high_wavenumber_bias(pred: np.ndarray, ref: np.ndarray, ndim: int, **kw) -> float:
    """``E_pred / E_ref - 1`` over the high band (see :func:`spectral_comparison`)."""
    return spectral_comparison(pred, ref, ndim, **kw).hf_bias


@dataclass
class StructureFunctions:
    r: np.ndarray                           # lags in grid steps
    S: dict[int, np.ndarray]                # order -> <|du|^p>(r)
    S3_signed: np.ndarray                   # <du^3>(r)
    flatness: np.ndarray                    # S4 / S2^2

    def to_dict(self) -> dict[str, Any]:
        return {"r": self.r.tolist(), "S": {str(p): v.tolist() for p, v in self.S.items()},
                "S3_signed": self.S3_signed.tolist(), "flatness": self.flatness.tolist()}


def structure_functions(u: np.ndarray, ndim: int, *, orders: Sequence[int] = (2, 3), lags: Sequence[int] | None = None,
                        periodic: bool = True) -> StructureFunctions:
    """Structure functions along each spatial axis, averaged over the axes and all leading axes.

    ``lags`` (grid steps) default to ``1, 2, 4, ...`` up to a quarter of the smallest axis. Order 4 is always
    computed for the flatness."""
    u = _check(u, ndim)
    shape = u.shape[-ndim:]
    if lags is None:
        top = max(1, min(shape) // 4)
        lags = sorted({int(2 ** i) for i in range(int(np.log2(top)) + 1)})
    lags = [int(r) for r in lags]
    pset = sorted(set(int(p) for p in orders) | {2, 4})
    S = {p: np.zeros(len(lags)) for p in pset}
    S3s = np.zeros(len(lags))
    for i, r in enumerate(lags):
        acc = {p: [] for p in pset}
        acc3 = []
        for ax in _axes(ndim):
            if periodic:
                du = np.roll(u, -r, axis=ax) - u
            else:
                n = u.shape[ax]
                du = np.take(u, range(r, n), axis=ax) - np.take(u, range(0, n - r), axis=ax)
            a = np.abs(du)
            for p in pset:
                acc[p].append(np.mean(a ** p))
            acc3.append(np.mean(du ** 3))
        for p in pset:
            S[p][i] = float(np.mean(acc[p]))
        S3s[i] = float(np.mean(acc3))
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = S[4] / np.where(S[2] > 0, S[2] ** 2, np.nan)
    return StructureFunctions(np.array(lags), {p: S[p] for p in pset if p in orders or p in (2, 4)}, S3s, flat)


def _grad(u: np.ndarray, ndim: int, spacing: Sequence[float], periodic: bool) -> list[np.ndarray]:
    out = []
    for ax, h in zip(_axes(ndim), spacing):
        if periodic:
            out.append((np.roll(u, -1, axis=ax) - np.roll(u, 1, axis=ax)) / (2 * h))
        else:
            out.append(np.gradient(u, h, axis=ax, edge_order=2))
    return out


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.sqrt(np.mean((a - b) ** 2)))
    s = float(np.sqrt(np.mean(b ** 2)))
    return d / s if s > 0 else (0.0 if d == 0 else float("inf"))


def gradient_metrics(pred: np.ndarray, ref: np.ndarray, ndim: int, spacing: float | Sequence[float] = 1.0, *,
                     periodic: bool = False, vector_axis: int | None = None,
                     walls: Sequence[tuple[int, int]] = (), conductivity: float = 1.0) -> dict[str, float]:
    """Relative RMS errors of gradient-based quantities.

    ``vector_axis``: the axis of ``pred`` holding velocity components (``ndim`` of them, in the order of the spatial
    axes); adds vorticity (2-D: scalar, 3-D: magnitude) and strain-rate magnitude ``sqrt(2 S:S)``. Without it the
    field is a scalar and ``walls`` (``(spatial axis 0..ndim-1, side 0 or -1)``) give the error of the mean wall flux
    ``-conductivity du/dn`` (``n`` the outward normal)."""
    pred, ref = _check(pred, ndim), _check(ref, ndim)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    h = [float(spacing)] * ndim if np.isscalar(spacing) else [float(s) for s in spacing]
    out: dict[str, float] = {}
    if vector_axis is None:
        gp, gr = _grad(pred, ndim, h, periodic), _grad(ref, ndim, h, periodic)
        out["grad_rel_err"] = _rel(np.stack(gp), np.stack(gr))
        out["grad_mag_rel_err"] = _rel(np.sqrt(sum(g ** 2 for g in gp)), np.sqrt(sum(g ** 2 for g in gr)))
        for ax, side in walls:
            sign = -1.0 if side == 0 else 1.0                       # outward normal along -x on the min face
            idx = 0 if side == 0 else -1
            fp = -conductivity * sign * np.take(gp[ax], idx, axis=ax - ndim).mean()
            fr = -conductivity * sign * np.take(gr[ax], idx, axis=ax - ndim).mean()
            out[f"wall_flux_rel_err[{ax},{side}]"] = abs(fp - fr) / abs(fr) if fr else (0.0 if fp == fr else float("inf"))
        return out
    va = vector_axis % pred.ndim
    if va >= pred.ndim - ndim or pred.shape[va] != ndim:
        raise ValueError(f"vector_axis must be a leading axis with {ndim} components")
    P = [np.take(pred, i, axis=va) for i in range(ndim)]
    R = [np.take(ref, i, axis=va) for i in range(ndim)]
    Jp = [_grad(c, ndim, h, periodic) for c in P]                    # Jp[i][j] = d u_i / d x_j
    Jr = [_grad(c, ndim, h, periodic) for c in R]
    out["grad_rel_err"] = _rel(np.stack([g for row in Jp for g in row]), np.stack([g for row in Jr for g in row]))

    def vort(J):
        if ndim == 2:
            return J[1][0] - J[0][1]
        if ndim == 3:
            return np.sqrt((J[2][1] - J[1][2]) ** 2 + (J[0][2] - J[2][0]) ** 2 + (J[1][0] - J[0][1]) ** 2)
        return None

    def strain(J):
        return np.sqrt(2 * sum((0.5 * (J[i][j] + J[j][i])) ** 2 for i in range(ndim) for j in range(ndim)))

    if ndim in (2, 3):
        out["vorticity_rel_err"] = _rel(vort(Jp), vort(Jr))
    out["strain_rate_rel_err"] = _rel(strain(Jp), strain(Jr))
    return out


@dataclass
class SpectralFidelityReport:
    rel_l2: float
    spectrum: SpectralComparison
    structure_pred: StructureFunctions
    structure_ref: StructureFunctions
    gradients: dict[str, float]
    tol: float
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hf_bias(self) -> float:
        return self.spectrum.hf_bias

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks if c["passed"] is not None)

    def to_dict(self) -> dict[str, Any]:
        return {"rel_l2": self.rel_l2, "hf_bias": self.hf_bias, "tol": self.tol, "passed": self.passed,
                "spectrum": self.spectrum.to_dict(), "structure_pred": self.structure_pred.to_dict(),
                "structure_ref": self.structure_ref.to_dict(), "gradients": self.gradients, "checks": self.checks}

    def summary(self) -> str:
        s = self.spectrum
        lines = [f"rel. L2 {self.rel_l2:.3g}; high-wavenumber energy ratio {s.band_ratio['high']:.3g} "
                 f"(bias {self.hf_bias:+.3g}, tolerance {self.tol:g}), shells {s.bands['high'][0]}-{s.bands['high'][1]}",
                 "band   energy ratio  phase err [rad]  coherence"]
        for b in ("low", "mid", "high"):
            lines.append(f"{b:5s}  {s.band_ratio[b]:12.4g}  {s.band_phase_error[b]:15.3g}  {s.band_coherence[b]:9.3g}")
        lines += [f"{k}: {v:.3g}" for k, v in self.gradients.items()]
        return "\n".join(lines)


def spectral_fidelity(pred: np.ndarray, ref: np.ndarray, ndim: int, *, periodic: bool = True,
                      window: str | None = "hann", bands: Sequence[float] = (1 / 3, 2 / 3), tol: float = 0.25,
                      spacing: float | Sequence[float] = 1.0, vector_axis: int | None = None,
                      walls: Sequence[tuple[int, int]] = (), conductivity: float = 1.0) -> SpectralFidelityReport:
    """Spectral fidelity of ``pred`` against ``ref``: passes when ``|hf_bias| <= tol`` (high-band energy within
    ``tol`` of the reference), and reports the S2 ratio at the smallest lag and the gradient errors alongside."""
    pred, ref = _check(pred, ndim), _check(ref, ndim)
    sp = spectral_comparison(pred, ref, ndim, periodic=periodic, window=window, bands=bands)
    sf_p = structure_functions(pred, ndim, periodic=periodic)
    sf_r = structure_functions(ref, ndim, periodic=periodic)
    grads = gradient_metrics(pred, ref, ndim, spacing, periodic=periodic, vector_axis=vector_axis, walls=walls,
                             conductivity=conductivity)
    rel = _rel(pred, ref)
    s2 = float(sf_p.S[2][0] / sf_r.S[2][0]) if sf_r.S[2][0] > 0 else float("nan")
    checks = [
        {"id": "spectral.hf_bias", "passed": bool(abs(sp.hf_bias) <= tol), "value": sp.hf_bias, "threshold": tol,
         "detail": f"high-band energy ratio {sp.band_ratio['high']:.3g} over shells {sp.bands['high'][0]}-"
                   f"{sp.bands['high'][1]}"},
        {"id": "spectral.s2_small_scale", "passed": None, "value": s2, "threshold": None,
         "detail": f"S2 ratio at lag {int(sf_r.r[0])} (information)"},
    ]
    return SpectralFidelityReport(rel, sp, sf_p, sf_r, grads, tol, checks)
