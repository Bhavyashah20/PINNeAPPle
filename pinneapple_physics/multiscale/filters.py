"""Spatial filters for large-eddy simulation and multiscale checks on periodic grids.

A filter of width ``Delta`` turns a field ``u`` into ``u_bar = G_Delta * u`` (convolution). On a periodic domain it acts
on each Fourier mode by its transfer function ``G_hat(k)``, which is how it is applied here, so the transfer function
of the applied filter is exact (to round-off) and every filter commutes with derivatives as long as ``Delta`` is the
same everywhere. The three classical filters (Pope, *Turbulent Flows*, 2000, section 13.2, table 13.2):

=========  ===================================  =========================================================
kind       transfer function ``G_hat(k)``        notes
=========  ===================================  =========================================================
box        ``sin(k Delta / 2) / (k Delta / 2)``  top-hat average over ``Delta``; changes sign (zeros at
                                                  ``k = 2 pi n / Delta``)
gaussian   ``exp(-k^2 Delta^2 / 24)``            same second moment as the box filter; positive, smooth;
                                                  two Gaussians compose into one: ``Delta^2 = D1^2 + D2^2``
sharp      ``1`` if ``|k| <= pi / Delta`` else 0  spectral cut-off at ``k_c = pi / Delta``; idempotent
=========  ===================================  =========================================================

In several dimensions the filter is the tensor product of the 1-D filter in each direction (for the sharp filter this
is a square cut-off ``max(|k_x|, |k_y|) <= k_c``).

Also here: Fourier restriction / prolongation between grids (restriction keeps the mean exactly, so integrals are
preserved to round-off), the post-filter that turns a field already filtered at one width into one filtered at a
larger width (and how much of the requested filter cannot be reached from it), and filtering with a width that varies
in space, whose commutation error with derivatives is not zero (Ghosal & Moin 1995, J. Comput. Phys. 118:24).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["FILTER_KINDS", "Filter", "wavenumbers", "transfer", "apply_filter", "restrict", "prolong",
           "post_filter_transfer", "realizability_mismatch", "refilter", "filter_variable_width",
           "spectral_derivative", "commutation_error", "low_modes", "fourier_basis", "fourier_coefficients"]

FILTER_KINDS = ("box", "gaussian", "sharp")


def transfer(kind: str, k: np.ndarray, delta: float) -> np.ndarray:
    """1-D transfer function ``G_hat(k)`` of a filter of width ``delta`` (``delta = 0``: identity)."""
    k = np.asarray(k, dtype=float)
    if delta <= 0:
        return np.ones_like(k)
    if kind == "box":
        return np.sinc(k * delta / (2 * np.pi))                  # np.sinc(x) = sin(pi x) / (pi x)
    if kind == "gaussian":
        return np.exp(-(k * delta) ** 2 / 24.0)
    if kind == "sharp":
        return (np.abs(k) <= np.pi / delta * (1 + 1e-12)).astype(float)
    raise ValueError(f"unknown filter kind {kind!r}; use one of {FILTER_KINDS}")


@dataclass(frozen=True)
class Filter:
    """A filter of width ``delta`` (physical length) and ``kind`` in :data:`FILTER_KINDS`. ``delta = 0`` is the
    identity (a fully resolved field)."""
    kind: str = "gaussian"
    delta: float = 0.0

    def __post_init__(self):
        if self.kind not in FILTER_KINDS:
            raise ValueError(f"unknown filter kind {self.kind!r}; use one of {FILTER_KINDS}")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")

    @property
    def cutoff(self) -> float:
        """Wavenumber that separates resolved and subfilter scales, ``pi / delta`` (inf for the identity)."""
        return np.pi / self.delta if self.delta > 0 else np.inf

    def transfer(self, k: np.ndarray) -> np.ndarray:
        return transfer(self.kind, k, self.delta)

    def transfer_nd(self, ks: Sequence[np.ndarray]) -> np.ndarray:
        """Tensor-product transfer on a grid of wavenumbers (``ks``: one array per axis, broadcastable)."""
        out = np.ones(np.broadcast_shapes(*[np.shape(k) for k in ks]))
        for k in ks:
            out = out * self.transfer(k)
        return out

    def __call__(self, u: np.ndarray, length: float = 2 * np.pi, axes: Sequence[int] | None = None) -> np.ndarray:
        return apply_filter(u, self, length, axes)

    def label(self) -> str:
        return "unfiltered" if self.delta == 0 else f"{self.kind} {self.delta:.3g}"


def wavenumbers(n: int, length: float = 2 * np.pi) -> np.ndarray:
    """Angular wavenumbers of an ``n``-point periodic grid, in FFT order."""
    return 2 * np.pi * np.fft.fftfreq(n, length / n)


def _axes(u: np.ndarray, axes: Sequence[int] | None) -> list[int]:
    if axes is None:                                                # trailing two axes for 2-D or more, else the last
        return [-2, -1] if u.ndim >= 2 else [-1]
    return [int(a) for a in axes]


def _grid_k(u: np.ndarray, axes: list[int], length: float) -> list[np.ndarray]:
    ks = []
    for j, a in enumerate(axes):
        shape = [1] * len(axes)
        shape[j] = u.shape[a]
        ks.append(wavenumbers(u.shape[a], length).reshape(shape))
    return ks


def apply_filter(u: np.ndarray, filt: Filter, length: float = 2 * np.pi, axes: Sequence[int] | None = None
                 ) -> np.ndarray:
    """Filter a periodic field over ``axes`` (default: the last two, or the last one for 1-D data)."""
    u = np.asarray(u, dtype=float)
    if filt.delta == 0:
        return u.copy()
    ax = _axes(u, axes)
    G = filt.transfer_nd(_grid_k(u, ax, length))
    shape = [1] * u.ndim
    for j, a in enumerate(ax):
        shape[a] = G.shape[j]
    uh = np.fft.fftn(u, axes=ax)
    return np.fft.ifftn(uh * G.reshape(shape), axes=ax).real


def _resample(u: np.ndarray, n, axes, finer: bool) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    ax = _axes(u, axes)
    ns = [int(n)] * len(ax) if np.isscalar(n) else [int(m) for m in n]
    uh = np.fft.fftn(u, axes=ax)
    scale = 1.0
    for a, m in zip(ax, ns, strict=True):
        N = uh.shape[a]
        if (m < N) if finer else (m > N):
            raise ValueError("prolong goes to a finer grid, restrict to a coarser one")
        h = (min(N, m) - 1) // 2                                    # modes k = -h .. h (no Nyquist mode)
        shape = list(uh.shape)
        shape[a] = m
        out = np.zeros(shape, dtype=complex)
        src_pos = [slice(None)] * uh.ndim
        dst_pos = [slice(None)] * uh.ndim
        src_pos[a], dst_pos[a] = slice(0, h + 1), slice(0, h + 1)
        out[tuple(dst_pos)] = uh[tuple(src_pos)]
        if h > 0:
            src_pos[a], dst_pos[a] = slice(N - h, N), slice(m - h, m)
            out[tuple(dst_pos)] = uh[tuple(src_pos)]
        uh = out
        scale *= m / N
    return np.fft.ifftn(uh, axes=ax).real * scale


def restrict(u: np.ndarray, n: int | Sequence[int], axes: Sequence[int] | None = None) -> np.ndarray:
    """Fourier restriction to a coarser periodic grid of ``n`` points per axis: keeps the modes ``|k| < n / 2`` (the
    Nyquist mode of the coarse grid is dropped, so the result is real). The mean, hence the integral over the
    domain, is preserved to round-off."""
    return _resample(u, n, axes, finer=False)


def prolong(u: np.ndarray, n: int | Sequence[int], axes: Sequence[int] | None = None) -> np.ndarray:
    """Fourier interpolation (zero padding) to a finer periodic grid of ``n`` points per axis."""
    return _resample(u, n, axes, finer=True)


def post_filter_transfer(source: Filter, target: Filter, k: np.ndarray, max_gain: float = 2.0) -> np.ndarray:
    """1-D transfer of the post-filter that brings a field filtered by ``source`` to ``target``:
    ``G_target / G_source`` where that ratio is at most ``max_gain`` in magnitude, 0 elsewhere. A mild deconvolution
    (gain up to 2 by default, e.g. a sharp cut-off at k = 4 from a slightly damped Gaussian output) is allowed;
    beyond that the scales the source filter removed or damped are treated as lost: recovering them would only
    amplify the source's errors."""
    gs, gt = source.transfer(k), target.transfer(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(gs) > 1e-12, gt / np.where(np.abs(gs) > 1e-12, gs, 1.0), np.inf)
    return np.where(np.abs(r) <= max_gain, r, 0.0)


def realizability_mismatch(source: Filter, target: Filter, kmax: float, *, slope: float = -2.0, n: int = 4096,
                           max_gain: float = 2.0, kmin: float = 1.0) -> float:
    """Fraction (0 to 1, as an L2 ratio) of the ``target`` filter that cannot be obtained by post-filtering a field
    already filtered by ``source``. The transfer functions are compared on ``kmin <= k <= kmax`` (``kmin``: the
    fundamental wavenumber, 1 on a ``2 pi`` domain) weighted by a model
    spectrum of the field, ``|k| ** slope`` per mode (default -2: a vorticity field with an energy spectrum
    ``k^-3`` per shell in 2-D). 0 when the target is reachable exactly (e.g. a wider Gaussian from a Gaussian, any
    filter from an unfiltered field); 1 when nothing is (a finer target than the source cut-off)."""
    k = np.linspace(kmin, max(kmax, kmin), n)
    w = k ** slope
    gt = target.transfer(k)
    got = source.transfer(k) * post_filter_transfer(source, target, k, max_gain)
    den = float(np.sum(w * gt ** 2))
    return float(np.sqrt(np.sum(w * (gt - got) ** 2) / den)) if den > 0 else 0.0


def refilter(u: np.ndarray, source: Filter, target: Filter, length: float = 2 * np.pi,
             axes: Sequence[int] | None = None, max_gain: float = 2.0) -> np.ndarray:
    """Turn a field filtered by ``source`` into the ``target`` filter (as far as it can be, see
    :func:`post_filter_transfer`)."""
    u = np.asarray(u, dtype=float)
    ax = _axes(u, axes)
    ks = _grid_k(u, ax, length)
    G = np.ones(np.broadcast_shapes(*[k.shape for k in ks]))
    for k in ks:
        G = G * post_filter_transfer(source, target, k, max_gain)
    shape = [1] * u.ndim
    for j, a in enumerate(ax):
        shape[a] = G.shape[j]
    return np.fft.ifftn(np.fft.fftn(u, axes=ax) * G.reshape(shape), axes=ax).real


def filter_variable_width(u: np.ndarray, kind: str, delta: np.ndarray, length: float = 2 * np.pi,
                          axes: Sequence[int] | None = None, decimals: int = 10) -> np.ndarray:
    """Filter whose width varies in space: ``u_bar(x) = sum_k G_hat(k; delta(x)) u_hat(k) exp(i k x)``.

    ``delta`` has the shape of ``u``. Evaluated exactly by filtering the whole field once per distinct width and
    taking each point from the field of its own width. With a uniform ``delta`` this is :func:`apply_filter`."""
    u = np.asarray(u, dtype=float)
    d = np.round(np.broadcast_to(np.asarray(delta, dtype=float), u.shape), decimals)
    out = np.empty_like(u)
    for v in np.unique(d):
        m = d == v
        out[m] = apply_filter(u, Filter(kind, float(v)), length, axes)[m]
    return out


def spectral_derivative(u: np.ndarray, axis: int, length: float = 2 * np.pi, order: int = 1) -> np.ndarray:
    """Exact derivative of the trigonometric interpolant along ``axis`` (the Nyquist mode is dropped for odd
    orders)."""
    u = np.asarray(u, dtype=float)
    n = u.shape[axis]
    k = wavenumbers(n, length)
    if order % 2 == 1 and n % 2 == 0:
        k = k.copy()
        k[n // 2] = 0.0
    shape = [1] * u.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(u, axis=axis) * ((1j * k) ** order).reshape(shape), axis=axis).real


def commutation_error(u: np.ndarray, kind: str, delta: float | np.ndarray, axis: int = -1,
                      length: float = 2 * np.pi, axes: Sequence[int] | None = None) -> np.ndarray:
    """``d(u_bar)/dx - (du/dx)_bar`` for a filter of width ``delta`` (scalar or a field). Zero to round-off for a
    uniform width; for a width that varies in space it is the term that a residual of the filtered equations picks
    up from filtering alone, not from the model (to leading order ``(d Delta / dx) d u_bar / d Delta``)."""
    u = np.asarray(u, dtype=float)
    dfield = np.broadcast_to(np.asarray(delta, dtype=float), u.shape)
    ubar = filter_variable_width(u, kind, dfield, length, axes)
    dubar = spectral_derivative(ubar, axis, length)
    du_bar = filter_variable_width(spectral_derivative(u, axis, length), kind, dfield, length, axes)
    return dubar - du_bar


def low_modes(kmax: int) -> list[tuple[int, int]]:
    """Half-plane list of the 2-D integer wavenumbers with ``|kx|, |ky| <= kmax`` (sharp square cut-off), mean
    excluded: one of each pair ``k, -k``."""
    return [(kx, ky) for kx in range(0, kmax + 1) for ky in range(-kmax, kmax + 1) if (kx > 0 or ky > 0)]


def fourier_basis(modes: Sequence[tuple[int, int]], n: int, length: float = 2 * np.pi) -> np.ndarray:
    """``(2 len(modes), n * n)`` real basis ``cos, sin(kx x + ky y)`` on an ``n x n`` grid (``k`` in units of
    ``2 pi / length``)."""
    x = np.arange(n) * length / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = 2 * np.pi / length
    rows = []
    for kx, ky in modes:
        rows += [np.cos(c * (kx * X + ky * Y)).ravel(), np.sin(c * (kx * X + ky * Y)).ravel()]
    return np.array(rows)


def fourier_coefficients(w: np.ndarray, modes: Sequence[tuple[int, int]], length: float = 2 * np.pi) -> np.ndarray:
    """Coefficients of a periodic field on :func:`fourier_basis` (exact for modes below the grid's Nyquist)."""
    n = w.shape[-1]
    B = fourier_basis(modes, n, length)
    return np.asarray(w, float).reshape(*np.shape(w)[:-2], -1) @ B.T * (2.0 / (n * n))
