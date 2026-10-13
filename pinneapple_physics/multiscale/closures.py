"""Subfilter closures for 2-D large-eddy simulation in vorticity form, and the tests that tell whether a closure can
be trusted.

Filtering the momentum equations leaves the subfilter stress ``tau_ij = bar(u_i u_j) - bar(u_i) bar(u_j)``; in
vorticity form the filtered equation gets the tendency ``-curl(div tau) = -div(sigma)`` with the subfilter vorticity
flux ``sigma_j = bar(u_j omega) - bar(u_j) bar(omega)`` (the two are equal because the filter commutes with
derivatives). A closure models ``tau`` from the resolved field. In 2-D a deviatoric symmetric tensor has two
components, ``a = tau_xx = -tau_yy`` and ``b = tau_xy``, and the tendency is ``-(b_xx - b_yy - 2 a_xy)``.

Closures (each a callable ``closure(omega, solver) -> tendency`` for :class:`~.ns2d.VorticityNS2D`, with
``stress(omega, solver) -> (a, b)``):

* :class:`Smagorinsky` (Smagorinsky 1963; Lilly 1967): ``tau = -2 (Cs Delta)^2 |S| S``, ``|S| = sqrt(2 S_ij S_ij)``;
* :class:`DynamicSmagorinsky` (Germano, Piomelli, Moin & Cabot 1991; least squares of Lilly 1992): the coefficient
  from the Germano identity ``L_ij = C M_ij`` between the grid filter and a test filter of width ``alpha Delta``,
  averaged over the (homogeneous) domain and clipped at 0;
* :class:`GradientModel` (Clark, Ferziger & Reynolds 1979): ``tau = Delta^2 / 12 d_k u_i d_k u_j``, the leading term
  of the Taylor expansion for Gaussian or box filters. It correlates well with the true stress a priori and is
  known to be unstable a posteriori without extra dissipation;
* :class:`NoModel`, and learned closures through :mod:`.learned`.

Tests:

* :func:`apriori_test` (from a filtered fine field, no simulation): correlation of the modelled and true stress
  components and tendencies, mean subfilter energy transfer ``Pi = -tau_ij S_ij`` and the backscatter fraction
  (``Pi < 0``), which an eddy-viscosity model cannot represent;
* :func:`aposteriori_test` (run the LES and compare with the filtered reference): energy-spectrum error,
  resolved-energy dissipation rate, long-run stability;
* :func:`closure_health`: label-free checks on an LES prediction (no reference): resolved enstrophy must not grow in
  unforced 2-D flow, the spectrum near the cut-off must not pile up or collapse against the power law of the
  resolved range, the field must stay finite. Used by the ensemble as a prediction check.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .filters import Filter, apply_filter, restrict
from .ns2d import VorticityNS2D, energy_spectrum, velocity_from_vorticity

__all__ = ["strain", "stress_tendency", "true_subfilter", "Smagorinsky", "DynamicSmagorinsky", "GradientModel",
           "NoModel", "AprioriReport", "apriori_test", "AposterioriReport", "aposteriori_test", "ClosureHealth",
           "closure_health", "spectrum_error", "germano_error"]


def _d(f: np.ndarray, solver: VorticityNS2D, axis: str) -> np.ndarray:
    return solver.ddx(f) if axis == "x" else solver.ddy(f)


def strain(w: np.ndarray, solver: VorticityNS2D) -> dict[str, np.ndarray]:
    """Velocity gradients, strain ``S11 = u_x``, ``S12 = (u_y + v_x) / 2`` and ``|S| = sqrt(2 S_ij S_ij)``."""
    u, v = solver.velocity(w)
    ux, uy, vx, vy = solver.ddx(u), solver.ddy(u), solver.ddx(v), solver.ddy(v)
    s11, s12 = ux, 0.5 * (uy + vx)
    return {"u": u, "v": v, "ux": ux, "uy": uy, "vx": vx, "vy": vy, "S11": s11, "S12": s12,
            "S": 2.0 * np.sqrt(s11 ** 2 + s12 ** 2)}


def stress_tendency(a: np.ndarray, b: np.ndarray, solver: VorticityNS2D) -> np.ndarray:
    """Vorticity tendency ``-curl(div tau)`` of a deviatoric stress ``(tau_xx, tau_xy) = (a, b)``."""
    KX, KY = solver.KX, solver.KY
    ah, bh = np.fft.fft2(a), np.fft.fft2(b)
    return -np.fft.ifft2(-(KX ** 2) * bh + KY ** 2 * bh + 2 * KX * KY * ah).real


def _dev(txx, txy, tyy):
    return 0.5 * (txx - tyy), txy


def true_subfilter(w_fine: np.ndarray, filt: Filter, n_coarse: int | None = None, length: float = 2 * np.pi
                   ) -> dict[str, np.ndarray]:
    """Exact subfilter quantities of a resolved vorticity field for the filter ``filt``: the filtered field
    ``w_bar``, the deviatoric stress ``(a, b)``, the vorticity flux ``sigma`` and the tendency ``-div(sigma)``,
    restricted to ``n_coarse`` points per axis (exact when the filter removes the modes the coarse grid lacks)."""
    u, v = velocity_from_vorticity(w_fine, length)
    F = lambda f: apply_filter(f, filt, length)  # noqa: E731
    ub, vb, wb = F(u), F(v), F(w_fine)
    a, b = _dev(F(u * u) - ub * ub, F(u * v) - ub * vb, F(v * v) - vb * vb)
    sx, sy = F(u * w_fine) - ub * wb, F(v * w_fine) - vb * wb
    fine = VorticityNS2D(w_fine.shape[-1], 0.0, length, dealias=False)
    tend = -(fine.ddx(sx) + fine.ddy(sy))
    out = {"w_bar": wb, "a": a, "b": b, "sigma_x": sx, "sigma_y": sy, "tendency": tend}
    if n_coarse is not None and n_coarse < w_fine.shape[-1]:
        out = {k: restrict(v_, n_coarse) for k, v_ in out.items()}
    return out


class _EddyViscosity:
    delta: float

    def nu_t(self, w, solver, st) -> np.ndarray | float:
        raise NotImplementedError

    def stress(self, w: np.ndarray, solver: VorticityNS2D) -> tuple[np.ndarray, np.ndarray]:
        st = strain(w, solver)
        nt = self.nu_t(w, solver, st)
        return -2 * nt * st["S11"], -2 * nt * st["S12"]

    def __call__(self, w: np.ndarray, solver: VorticityNS2D) -> np.ndarray:
        a, b = self.stress(w, solver)
        return stress_tendency(a, b, solver) * solver.mask


@dataclass
class Smagorinsky(_EddyViscosity):
    """``nu_t = (cs delta)^2 |S|``. ``delta`` defaults to the solver's implicit filter width."""
    cs: float = 0.17
    delta: float | None = None
    name: str = "Smagorinsky"

    def nu_t(self, w, solver, st):
        d = self.delta if self.delta is not None else solver.implicit_filter_width
        return (self.cs * d) ** 2 * st["S"]


@dataclass
class DynamicSmagorinsky(_EddyViscosity):
    """Dynamic Smagorinsky: ``nu_t = C delta^2 |S|`` with ``C = <L_ij M_ij> / <M_ij M_ij>`` (domain average, clipped
    at 0) from the test filter of width ``test_ratio * delta`` and kind ``test_kind``. The last coefficient is kept
    in ``self.last_c``."""
    delta: float | None = None
    test_ratio: float = 2.0
    test_kind: str = "sharp"
    name: str = "dynamic Smagorinsky"
    last_c: float = field(default=float("nan"), init=False)

    def coefficient(self, w: np.ndarray, solver: VorticityNS2D, st: dict | None = None) -> np.ndarray:
        st = st or strain(w, solver)
        d = self.delta if self.delta is not None else solver.implicit_filter_width
        T = Filter(self.test_kind, self.test_ratio * d)
        H = lambda f: apply_filter(f, T, solver.length)  # noqa: E731
        u, v = st["u"], st["v"]
        Lxx, Lxy = _dev(H(u * u) - H(u) ** 2, H(u * v) - H(u) * H(v), H(v * v) - H(v) ** 2)
        sh = strain(H(w), solver)
        Mxx = 2 * d ** 2 * (H(st["S"] * st["S11"]) - self.test_ratio ** 2 * sh["S"] * sh["S11"])
        Mxy = 2 * d ** 2 * (H(st["S"] * st["S12"]) - self.test_ratio ** 2 * sh["S"] * sh["S12"])
        ax = (-2, -1)
        num = (2 * Lxx * Mxx + 2 * Lxy * Mxy).mean(axis=ax)
        den = (2 * Mxx ** 2 + 2 * Mxy ** 2).mean(axis=ax)
        c = np.maximum(num / np.maximum(den, 1e-300), 0.0)
        self.last_c = float(np.mean(c))
        return c

    def nu_t(self, w, solver, st):
        d = self.delta if self.delta is not None else solver.implicit_filter_width
        c = self.coefficient(w, solver, st)
        c = np.asarray(c)[..., None, None] if np.ndim(c) else c
        return c * d ** 2 * st["S"]


@dataclass
class GradientModel:
    """Clark gradient (nonlinear) model ``tau_ij = coef * delta^2 / 12 * d_k u_i d_k u_j`` (deviatoric part)."""
    delta: float | None = None
    coef: float = 1.0
    name: str = "gradient (Clark)"

    def stress(self, w: np.ndarray, solver: VorticityNS2D) -> tuple[np.ndarray, np.ndarray]:
        st = strain(w, solver)
        d = self.delta if self.delta is not None else solver.implicit_filter_width
        c = self.coef * d ** 2 / 12
        return _dev(c * (st["ux"] ** 2 + st["uy"] ** 2), c * (st["ux"] * st["vx"] + st["uy"] * st["vy"]),
                    c * (st["vx"] ** 2 + st["vy"] ** 2))

    def __call__(self, w: np.ndarray, solver: VorticityNS2D) -> np.ndarray:
        a, b = self.stress(w, solver)
        return stress_tendency(a, b, solver) * solver.mask


@dataclass
class NoModel:
    """No subfilter model (an under-resolved DNS)."""
    name: str = "no model"

    def stress(self, w, solver):
        z = np.zeros_like(w)
        return z, z

    def __call__(self, w, solver):
        return np.zeros_like(w)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.ravel(x) - np.mean(x), np.ravel(y) - np.mean(y)
    d = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / d) if d > 0 else float("nan")


@dataclass
class AprioriReport:
    closure: str
    filter: str
    stress_correlation: float          # mean of the correlations of the two deviatoric components
    tendency_correlation: float        # correlation of -div(sigma), the term the vorticity equation sees
    transfer_true: float               # mean subfilter energy transfer <Pi>, Pi = -tau_ij S_ij (> 0: to the subfilter)
    transfer_model: float
    backscatter_true: float            # fraction of points with Pi < 0 in the filtered reference
    backscatter_model: float           # same for the model (0 for any eddy viscosity)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def apriori_test(w_fine: np.ndarray, filt: Filter, closure: Any, n_coarse: int, length: float = 2 * np.pi
                 ) -> AprioriReport:
    """A-priori test of ``closure`` on a resolved field: filter it with ``filt``, restrict to the LES grid of
    ``n_coarse`` points, and compare the closure's stress and tendency (computed from the filtered field) with the
    exact subfilter terms."""
    tr = true_subfilter(w_fine, filt, n_coarse, length)
    solver = VorticityNS2D(n_coarse, 0.0, length)
    wb = tr["w_bar"]
    a_m, b_m = closure.stress(wb, solver)
    t_m = closure(wb, solver)
    st = strain(wb, solver)
    pi_t = -2 * (tr["a"] * st["S11"] + tr["b"] * st["S12"])            # -tau_ij S_ij for a deviatoric pair
    pi_m = -2 * (a_m * st["S11"] + b_m * st["S12"])
    return AprioriReport(getattr(closure, "name", type(closure).__name__), filt.label(),
                         0.5 * (_corr(tr["a"], a_m) + _corr(tr["b"], b_m)), _corr(tr["tendency"], t_m),
                         float(pi_t.mean()), float(pi_m.mean()), float(np.mean(pi_t < 0)), float(np.mean(pi_m < 0)))


def spectrum_error(w: np.ndarray, w_ref: np.ndarray, kmax: float, length: float = 2 * np.pi, kmin: int = 1) -> float:
    """Mean absolute difference of ``log10 E(k)`` over the shells ``kmin <= k <= kmax`` (0: same spectrum; 0.3: a
    factor of 2 on average)."""
    k, E = energy_spectrum(w, length)
    k2, Er = energy_spectrum(w_ref, length)
    m = min(len(E), len(Er))
    sel = (k[:m] >= kmin) & (k[:m] <= kmax) & (Er[:m] > 0)
    return float(np.mean(np.abs(np.log10(np.maximum(E[:m][sel], 1e-300)) - np.log10(Er[:m][sel]))))


@dataclass
class AposterioriReport:
    closure: str
    times: list[float]
    spectrum_error: list[float]        # log10 spectrum error against the filtered reference, per time
    field_error: list[float]           # relative L2 against the filtered reference, per time
    dissipation_les: float             # mean rate of change of resolved energy, -dE/dt
    dissipation_ref: float
    stable: bool                       # finite and bounded until ``t_long``
    t_long: float
    enstrophy_growth: float            # max over the long run of Z(t) / Z(0)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def aposteriori_test(les: VorticityNS2D, w0_bar: np.ndarray, ref_bar: np.ndarray, times: Sequence[float],
                     dt: float = 5e-3, t_long: float | None = None) -> AposterioriReport:
    """Run the LES ``les`` (its closure included) from the filtered initial field ``w0_bar`` (on its grid) and compare
    with the filtered reference ``ref_bar`` (``(len(times), n, n)``, same grid) at ``times``; then keep running to
    ``t_long`` (default 4 x the last time) for the long-run stability."""
    name = getattr(les.closure, "name", "no model") if les.closure is not None else "no model"
    times = [float(t) for t in times]
    t_long = float(t_long if t_long is not None else 4 * times[-1])
    try:
        traj = les.simulate(w0_bar, times + ([t_long] if t_long > times[-1] else []), dt)
        stable = bool(np.all(np.isfinite(traj)))
    except FloatingPointError:
        traj, stable = None, False
    kmax = les.kmax * les.length / (2 * np.pi)
    if traj is None:
        nan = [float("nan")] * len(times)
        return AposterioriReport(name, times, nan, nan, float("nan"), float("nan"), False, t_long, float("inf"))
    W = traj[:len(times)]
    se = [spectrum_error(W[i], ref_bar[i], kmax, les.length) for i in range(len(times))]
    fe = [float(np.linalg.norm(W[i] - ref_bar[i]) / np.linalg.norm(ref_bar[i])) for i in range(len(times))]
    E0, El, Er = float(les.energy(w0_bar)), float(les.energy(W[-1])), float(les.energy(ref_bar[-1]))
    z0 = float(les.enstrophy(w0_bar))
    zmax = max(float(les.enstrophy(traj[i])) for i in range(traj.shape[0]))
    growth = zmax / max(z0, 1e-300)
    stable = stable and growth < 10.0
    return AposterioriReport(name, times, se, fe, (E0 - El) / times[-1], (E0 - Er) / times[-1], stable, t_long, growth)


@dataclass
class ClosureHealth:
    score: float                       # 1 healthy .. 0 failed
    ok: bool
    reasons: list[str]
    enstrophy_ratio: float
    tail_ratio: float
    germano: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def closure_health(w: np.ndarray, w0: np.ndarray, kc: float, length: float = 2 * np.pi, *, forced: bool = False,
                   max_enstrophy_growth: float = 1.05, tail_band: tuple[float, float] = (0.05, 5.0),
                   closure: Any = None, solver: VorticityNS2D | None = None, germano_limit: float = 1.05
                   ) -> ClosureHealth:
    """Label-free health of an LES (or any filtered) prediction ``w`` at cut-off ``kc`` (shells), started from
    ``w0`` (same filter):

    * finite values;
    * resolved enstrophy ``Z(t) / Z(0) <= max_enstrophy_growth`` in unforced flow (2-D advection conserves
      enstrophy, viscosity and the forward enstrophy cascade only remove it: growth is a closure that injects
      enstrophy, e.g. an unstable gradient model);
    * spectral tail: the energy in ``kc / 2 < k <= kc`` divided by the power law fitted on ``kc / 8 .. kc / 2`` and
      extrapolated, inside ``tail_band`` (above: pile-up at the cut-off, a closure too weak; below: a closure too
      dissipative).

    * with ``closure`` and ``solver`` (the LES grid): the Germano-identity error :func:`germano_error` at most
      ``germano_limit`` (a closure that explains the resolved transfer between the grid and test filters worse than
      no model at all is inconsistent across scales).

    ``score`` is the product of soft scores (1 inside the limits, ``exp(-(log excess)^2)`` outside). Arrays may
    carry leading axes (snapshots); every snapshot is checked and the worst counts."""
    reasons: list[str] = []
    w = np.asarray(w, float)
    if not np.all(np.isfinite(w)):
        return ClosureHealth(0.0, False, ["non-finite field"], float("inf"), float("nan"), float("nan"))
    if w.ndim > 2:                                                  # several snapshots: the worst one counts
        reps = [closure_health(x, w0, kc, length, forced=forced, max_enstrophy_growth=max_enstrophy_growth,
                               tail_band=tail_band, closure=closure, solver=solver, germano_limit=germano_limit)
                for x in w.reshape(-1, *w.shape[-2:])]
        worst = min(reps, key=lambda r: r.score)
        return ClosureHealth(worst.score, worst.ok, sorted({r_ for r in reps for r_ in r.reasons}),
                             max(r.enstrophy_ratio for r in reps), worst.tail_ratio, worst.germano)
    z0, z = 0.5 * float(np.mean(np.asarray(w0) ** 2)), 0.5 * float(np.mean(w ** 2))
    zr = z / max(z0, 1e-300)
    score = 1.0
    if not forced and zr > max_enstrophy_growth:
        score *= float(np.exp(-(np.log(zr / max_enstrophy_growth) / 0.05) ** 2))
        reasons.append(f"resolved enstrophy grew by {100 * (zr - 1):.1f} % in unforced flow")
    k, E = energy_spectrum(w, length)
    lo = max(2, int(np.floor(kc / 8)))
    mid, hi = int(np.floor(kc / 2)), int(np.floor(kc))
    tail = float("nan")
    if hi < len(E) and mid - lo >= 2 and np.all(E[lo:mid + 1] > 0):
        p = np.polyfit(np.log(k[lo:mid + 1]), np.log(E[lo:mid + 1]), 1)
        pred = np.exp(np.polyval(p, np.log(k[mid + 1:hi + 1])))
        tail = float(E[mid + 1:hi + 1].sum() / max(pred.sum(), 1e-300))
        if tail > tail_band[1]:
            score *= float(np.exp(-np.log(tail / tail_band[1]) ** 2))
            reasons.append(f"energy piles up at the cut-off ({tail:.2f} x the resolved power law)")
        elif tail < tail_band[0]:
            score *= float(np.exp(-np.log(tail_band[0] / tail) ** 2))
            reasons.append(f"spectrum collapses at the cut-off ({tail:.2f} x the resolved power law)")
    g = float("nan")
    if closure is not None and solver is not None and hasattr(closure, "stress"):
        g = germano_error(w, closure, solver)
        if g > germano_limit:
            score *= float(np.exp(-((g - germano_limit) / 0.1) ** 2))
            reasons.append(f"Germano identity error {g:.2f} > {germano_limit} (worse than no closure across scales)")
    return ClosureHealth(score, score > 0.5, reasons, zr, tail, g)


def germano_error(w: np.ndarray, closure: Any, solver: VorticityNS2D, test_ratio: float = 2.0,
                  test_kind: str = "sharp", delta: float | None = None) -> float:
    """Label-free cross-scale check of a closure through the Germano identity: the resolved stress between the grid
    filter and a test filter of width ``test_ratio * delta``, ``L_ij = hat(u_i u_j) - hat(u_i) hat(u_j)``, is known
    exactly from the LES field; a consistent closure must reproduce it as ``tau(test width, hat field) -
    hat(tau(delta, field))``. Compared through the vorticity tendency each stress produces (``-curl div``, the
    term the resolved equation sees; in 2-D the stress components themselves are nearly orthogonal to any
    eddy-viscosity stress, see :func:`apriori_test`). Returns ``||t(L) - t(L_model)|| / ||t(L)||``: 1 for no model,
    below 1 when the closure explains part of the resolved transfer, above 1 when it makes things worse (e.g. a
    fixed coefficient far too large for the flow). ``closure`` must accept a ``delta`` attribute (the classical
    closures here do); it is restored afterwards."""
    d = delta if delta is not None else solver.implicit_filter_width
    T = Filter(test_kind, test_ratio * d)
    H = lambda f: apply_filter(f, T, solver.length)  # noqa: E731
    u, v = solver.velocity(w)
    Lxx, Lxy = _dev(H(u * u) - H(u) ** 2, H(u * v) - H(u) * H(v), H(v * v) - H(v) ** 2)
    old = getattr(closure, "delta", None)
    try:
        closure.delta = d
        a1, b1 = closure.stress(w, solver)
        closure.delta = test_ratio * d
        a2, b2 = closure.stress(H(w), solver)
    finally:
        closure.delta = old
    tL = stress_tendency(Lxx, Lxy, solver)
    tM = stress_tendency(a2 - H(a1), b2 - H(b1), solver)
    den = float(np.sum(tL ** 2))
    return float(np.sqrt(np.sum((tL - tM) ** 2) / den)) if den > 0 else float("nan")
