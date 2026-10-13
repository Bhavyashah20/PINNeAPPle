"""2-D incompressible Navier-Stokes in vorticity form on the periodic square ``[0, L)^2``, for resolved (DNS) and
large-eddy (LES) simulation, with the energy spectrum and the residual of the (filtered) vorticity equation.

    d(omega)/dt + u . grad(omega) = nu lap(omega) - mu omega + f + C(omega),
    lap(psi) = -omega,  u = d(psi)/dy,  v = -d(psi)/dx

``C`` is an optional subfilter closure (LES): any callable ``closure(omega, solver) -> tendency`` evaluated on the
resolved field, e.g. :class:`~.closures.Smagorinsky`. Pseudo-spectral in space with the 2/3 dealiasing rule (the
dealiased grid resolves ``|k| <= n / 3``, which is also the implicit sharp filter of an LES on that grid), ETDRK4 in
time (Cox & Matthews 2002; Kassam & Trefethen 2005, through
:class:`pinneapple_simulation.numerical_solvers.etdrk4.ETDRK4`). Arrays are ``(n, n)`` or batched ``(batch, n, n)``,
axis -2 is x and axis -1 is y. Written from the equations; no third-party solver code was used.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .filters import wavenumbers

__all__ = ["VorticityNS2D", "random_vorticity", "energy_spectrum", "velocity_from_vorticity",
           "exact_advection", "vorticity_residual"]


def _k2d(n: int, length: float):
    k = wavenumbers(n, length)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return KX, KY


def velocity_from_vorticity(w: np.ndarray, length: float = 2 * np.pi) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` of a periodic vorticity field (zero mean velocity)."""
    n = w.shape[-1]
    KX, KY = _k2d(n, length)
    K2 = KX ** 2 + KY ** 2
    inv = np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0)
    psih = np.fft.fft2(w) * inv                                     # lap psi = -w  ->  psi_hat = w_hat / k^2
    return np.fft.ifft2(1j * KY * psih).real, np.fft.ifft2(-1j * KX * psih).real


def energy_spectrum(w: np.ndarray, length: float = 2 * np.pi, kmax: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Shell-summed kinetic energy spectrum ``E(k)`` from vorticity, ``sum_k E(k) = mean(u^2 + v^2) / 2``. Returns
    integer shell wavenumbers (in units of ``2 pi / L``) and ``E`` (averaged over leading batch axes)."""
    w = np.asarray(w, dtype=float)
    n = w.shape[-1]
    KX, KY = _k2d(n, length)
    K2 = KX ** 2 + KY ** 2
    wh = np.fft.fft2(w) / n ** 2
    e = 0.5 * np.abs(wh) ** 2 * np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0)
    e = e.reshape(-1, n, n).mean(axis=0)
    shell = np.rint(np.sqrt(K2) * length / (2 * np.pi)).astype(int)
    kmax = int(kmax if kmax is not None else n // 2)
    E = np.bincount(shell.ravel(), weights=e.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), E


def random_vorticity(rng: np.random.Generator, n: int, k0: float = 4.0, energy: float = 0.5,
                     length: float = 2 * np.pi, batch: int | None = None) -> np.ndarray:
    """Random vorticity with energy spectrum ``E(k) ~ k^4 exp(-2 (k / k0)^2)`` (peak at ``k0``), random phases, total
    kinetic energy ``energy``."""
    shape = (n, n) if batch is None else (batch, n, n)
    KX, KY = _k2d(n, length)
    k = np.sqrt(KX ** 2 + KY ** 2) * length / (2 * np.pi)
    # energy per shell E ~ k^4 exp(..) -> per mode E / k -> vorticity per mode k^2 E / k = k^5 exp(..)
    amp = np.sqrt(k ** 5 * np.exp(-2 * (k / k0) ** 2))
    w = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * amp).real
    w -= w.mean(axis=(-2, -1), keepdims=True)
    u, v = velocity_from_vorticity(w, length)
    e = 0.5 * (u ** 2 + v ** 2).mean(axis=(-2, -1), keepdims=True)
    return w * np.sqrt(energy / e)


@dataclass
class VorticityNS2D:
    """Pseudo-spectral 2-D vorticity solver. ``closure(omega, solver) -> tendency`` (physical space, same shape) adds
    a subfilter model; ``forcing``: ``(n, n)`` time-independent vorticity forcing; ``drag``: linear friction ``mu``."""
    n: int
    nu: float
    length: float = 2 * np.pi
    forcing: np.ndarray | None = None
    drag: float = 0.0
    closure: Callable[[np.ndarray, VorticityNS2D], np.ndarray] | None = None
    dealias: bool = True

    def __post_init__(self):
        self.KX, self.KY = _k2d(self.n, self.length)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.inv_k2 = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        kint = np.abs(np.fft.fftfreq(self.n, 1.0 / self.n))
        KXi, KYi = np.meshgrid(kint, kint, indexing="ij")
        self.mask = ((KXi <= self.n / 3) & (KYi <= self.n / 3)).astype(float) if self.dealias else np.ones_like(self.K2)
        self.fh = np.fft.fft2(self.forcing) if self.forcing is not None else 0.0
        self.kmax = (self.n // 3 if self.dealias else self.n // 2) * 2 * np.pi / self.length

    @property
    def h(self) -> float:
        return self.length / self.n

    @property
    def implicit_filter_width(self) -> float:
        """Width of the implicit sharp filter of the dealiased grid, ``pi / k_max``."""
        return float(np.pi / self.kmax)

    # ----------------------------------------------------------------------------------------- operators
    def velocity(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psih = np.fft.fft2(w) * self.inv_k2
        return np.fft.ifft2(1j * self.KY * psih).real, np.fft.ifft2(-1j * self.KX * psih).real

    def ddx(self, f: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(1j * self.KX * np.fft.fft2(f)).real

    def ddy(self, f: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(1j * self.KY * np.fft.fft2(f)).real

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(-self.K2 * np.fft.fft2(f)).real

    def advection(self, w: np.ndarray) -> np.ndarray:
        """``u . grad(w)`` in conservative form ``div(u w)`` (equal for divergence-free ``u``), dealiased."""
        u, v = self.velocity(w)
        return np.fft.ifft2(self.mask * (1j * self.KX * np.fft.fft2(u * w) + 1j * self.KY * np.fft.fft2(v * w))).real

    def tendency(self, w: np.ndarray) -> np.ndarray:
        """Right-hand side ``d(omega)/dt`` of the resolved equations (closure included)."""
        out = -self.advection(w) + self.nu * self.laplacian(w) - self.drag * w
        if self.forcing is not None:
            out = out + self.forcing
        if self.closure is not None:
            out = out + self.closure(w, self)
        return out

    # ----------------------------------------------------------------------------------------- time stepping
    def simulate(self, w0: np.ndarray, times: Sequence[float], dt: float = 5e-3, check: bool = True) -> np.ndarray:
        """Integrate from ``w0`` (at t = 0) and return the vorticity at each of ``times`` (sorted, >= 0):
        ``(len(times), n, n)`` or ``(batch, len(times), n, n)``. Raises ``FloatingPointError`` on blow-up when
        ``check``."""
        from pinneapple_simulation.numerical_solvers.etdrk4 import ETDRK4

        w0 = np.asarray(w0, dtype=float)
        single = w0.ndim == 2
        wb = w0[None] if single else w0
        mask = self.mask

        def nonlinear(wh, t):
            w = np.fft.ifft2(wh).real
            u, v = self.velocity(w)
            nl = -mask * (1j * self.KX * np.fft.fft2(u * w) + 1j * self.KY * np.fft.fft2(v * w)) + self.fh
            if self.closure is not None:
                nl = nl + np.fft.fft2(self.closure(w, self))
            return nl

        times = [float(t) for t in times]
        out, t, wh = [], 0.0, np.fft.fft2(wb)
        integ, h_cur = None, None
        for target in times:
            while target - t > 1e-12:
                h = min(dt, target - t)
                if integ is None or abs(h - h_cur) > 1e-15:
                    integ, h_cur = ETDRK4(-self.nu * self.K2 - self.drag, nonlinear, h), h
                wh = integ.step(wh, t)
                t += h
                if check and not np.all(np.isfinite(wh)):
                    raise FloatingPointError(f"VorticityNS2D blew up at t = {t:.4g}")
            out.append(np.fft.ifft2(wh).real)
        res = np.stack(out, axis=1)
        return res[0] if single else res

    # ----------------------------------------------------------------------------------------- diagnostics
    def energy(self, w: np.ndarray) -> np.ndarray:
        u, v = self.velocity(w)
        return 0.5 * (u ** 2 + v ** 2).mean(axis=(-2, -1))

    def enstrophy(self, w: np.ndarray) -> np.ndarray:
        return 0.5 * (np.asarray(w) ** 2).mean(axis=(-2, -1))

    def spectrum(self, w: np.ndarray, kmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return energy_spectrum(w, self.length, kmax)


def exact_advection(w: np.ndarray, length: float = 2 * np.pi) -> np.ndarray:
    """``div(u w)`` of the trigonometric interpolant of ``w``, without aliasing: the products are formed on a grid
    3/2 times finer and truncated back (exact for the modes of the original grid)."""
    from .filters import prolong, restrict

    n = w.shape[-1]
    m = 3 * n // 2 + (3 * n // 2) % 2
    fine = VorticityNS2D(m, 0.0, length, dealias=False)
    wf = prolong(w, m)
    u, v = fine.velocity(wf)
    return restrict(fine.ddx(u * wf) + fine.ddy(v * wf), n)


def vorticity_residual(w_prev: np.ndarray, w_next: np.ndarray, dt: float, nu: float, length: float = 2 * np.pi,
                       forcing: np.ndarray | None = None, drag: float = 0.0, closure_tendency: np.ndarray | None = None
                       ) -> tuple[np.ndarray, float]:
    """Residual of the vorticity equation between two snapshots ``dt`` apart (trapezoidal in time, second order):
    ``R = (w_next - w_prev) / dt + mean(div(u w)) - nu lap(mean w) + mu mean(w) - f - C``, the means taken over the
    two snapshots, where ``C`` is the subfilter tendency given in ``closure_tendency`` (0 for the resolved
    equations). Returns ``(R, scale)``, where ``scale`` is the rms of the largest term, so ``rms(R) / scale`` is a
    relative residual."""
    w_prev, w_next = np.asarray(w_prev, float), np.asarray(w_next, float)
    n = w_prev.shape[-1]
    solver = VorticityNS2D(n, nu, length, dealias=False)
    wm = 0.5 * (w_prev + w_next)
    dwdt = (w_next - w_prev) / dt
    adv = 0.5 * (exact_advection(w_prev, length) + exact_advection(w_next, length))
    visc = nu * solver.laplacian(wm)
    R = dwdt + adv - visc + drag * wm
    if forcing is not None:
        R = R - forcing
    if closure_tendency is not None:
        R = R - closure_tendency
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))  # noqa: E731
    return R, max(rms(dwdt), rms(adv), rms(visc), 1e-300)
