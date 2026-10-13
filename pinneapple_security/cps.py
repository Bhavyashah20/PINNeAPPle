"""Physics-residual detection of false data injection in cyber-physical systems (water, process, HVAC, grids).

An attacker who rewrites a sensor stream keeps every value plausible on its own and close to its own history, which
is what a purely data-driven detector learns. What the attacker cannot keep without a model of the plant is the
physics that ties several sensors together: tank levels must integrate the measured in- and outflows, a valve or
orifice passes ``q = sqrt(a + b * dh)`` for the head difference across it, a heat exchanger obeys its energy balance.
This module turns those laws into residual channels, identifies their constants from trusted data (grey-box twin, no
hand-tuned coefficients) and runs a two-sided CUSUM (Page 1954) per channel:

1. components (:class:`MassBalance`, :class:`FlowLaw`, or any object with ``fit(D)`` and ``residual(D)``) map a
   dict of sensor columns to one residual series each;
2. :class:`PhysicsResidualDetector` standardises each channel with robust statistics of trusted data, accumulates
   the deviations with a CUSUM and calibrates the alarm threshold to a target false-alarm rate on attack-free data;
3. :meth:`PhysicsResidualDetector.detect` reports alarms and the channel that raised them (which balance broke →
   where the manipulated sensor sits).

What it cannot see: an attacker who knows the plant model and injects physically consistent data (a coordinated
replay of every sensor, a stealthy attack in the sense of Liu, Ning & Reiter 2009). Pair it with
:func:`pinneapple_security.anomaly.replay_score` and with integrity of the data path (``signing``).

References: E. S. Page, Biometrika 41 (1954) 100-115; Y. Liu, P. Ning, M. K. Reiter, CCS 2009 (false data injection
against state estimation); A. A. Cárdenas et al., ASIACCS 2011 (physics-based attack detection in process control);
D. I. Urbina et al., CCS 2016 (limiting the impact of stealthy attacks, CUSUM on physical residuals).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

__all__ = ["ResidualComponent", "MassBalance", "FlowLaw", "CusumAlarm", "PhysicsResidualDetector", "cusum", "roc_auc",
           "episode_metrics", "window_auc"]

Data = Mapping[str, np.ndarray]


class ResidualComponent(Protocol):
    name: str

    def fit(self, D: Data) -> Any: ...

    def residual(self, D: Data) -> np.ndarray: ...


def _col(D: Data, k: str) -> np.ndarray:
    if k not in D:
        raise KeyError(f"sensor column {k!r} missing (have {sorted(D)})")
    return np.asarray(D[k], dtype=float)


@dataclass
class MassBalance:
    """Storage balance of one volume over a window of ``window`` samples:
    ``area * (level[t] - level[t-w]) - dt * sum(inflows - outflows)[t-w:t]``, in volume units (m^3 for water).

    ``area`` (m^2, or heat capacity for an energy balance) is identified by least squares on trusted data when not
    given. The window trades detection delay for noise: level noise is divided by ``window``, flow noise grows as
    ``sqrt(window)``. Flows are taken at the start of each step (forward Euler of the plant's storage equation)."""
    level: str
    inflows: Sequence[str] = ()
    outflows: Sequence[str] = ()
    dt: float = 1.0
    window: int = 10
    area: float | None = None
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"balance[{self.level}]"

    def _parts(self, D: Data) -> tuple[np.ndarray, np.ndarray]:
        h = _col(D, self.level)
        q = np.zeros_like(h)
        for k in self.inflows:
            q = q + _col(D, k)
        for k in self.outflows:
            q = q - _col(D, k)
        w = int(self.window)
        S = np.concatenate([[0.0], np.cumsum(q * self.dt)])           # S[t] = dt * sum(q[:t])
        dh = np.full_like(h, np.nan)
        dv = np.full_like(h, np.nan)
        dh[w:] = h[w:] - h[:-w]
        dv[w:] = S[w:-1] - S[:-w - 1]                                  # sum of q[t-w .. t-1]
        return dh, dv

    def fit(self, D: Data) -> MassBalance:
        if self.area is None:
            dh, dv = self._parts(D)
            ok = np.isfinite(dh) & np.isfinite(dv)
            self.area = float(dh[ok] @ dv[ok] / max(dh[ok] @ dh[ok], 1e-300))
        return self

    def residual(self, D: Data) -> np.ndarray:
        if self.area is None:
            raise RuntimeError("fit() first or give the area")
        dh, dv = self._parts(D)
        return self.area * dh - dv


@dataclass
class FlowLaw:
    """Flow through a valve, orifice, pipe or weir driven by a head difference: ``q^2 = a + b * (up - down)``
    (turbulent square-root law, exponent 1/2; ``exponent`` 1 gives a linear laminar or pump-curve law).
    ``a`` absorbs the elevation offset between the two levels. Residual ``q - q_law`` in flow units.
    ``midpoint``: the flow meter reports the average over the sample interval and the levels are read at its start,
    so the law uses the mean head of samples t and t+1 (one sample of latency, no lag bias)."""
    flow: str
    up: str
    down: str | None = None
    exponent: float = 0.5
    midpoint: bool = False
    coef: tuple[float, float] | None = None
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"law[{self.flow}]"

    def _head(self, D: Data) -> np.ndarray:
        h = _col(D, self.up)
        h = h - _col(D, self.down) if self.down else h
        if self.midpoint:
            h = np.append(0.5 * (h[:-1] + h[1:]), np.nan)
        return h

    def fit(self, D: Data) -> FlowLaw:
        if self.coef is None:
            q, dh = _col(D, self.flow), self._head(D)
            ok = np.isfinite(q) & np.isfinite(dh)
            y = np.abs(q[ok]) ** (1.0 / self.exponent)
            A = np.c_[np.ones(ok.sum()), dh[ok]]
            self.coef = tuple(float(c) for c in np.linalg.lstsq(A, y, rcond=None)[0])
        return self

    def residual(self, D: Data) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit() first or give coef")
        a, b = self.coef
        q_law = np.maximum(a + b * self._head(D), 0.0) ** self.exponent
        return _col(D, self.flow) - q_law


def cusum(z: np.ndarray, k: float = 0.5, h: float | None = None, upper_only: bool = False
          ) -> tuple[np.ndarray, list[int]]:
    """Two-sided CUSUM of standardised scores ``z`` (T,) or (T, m). Returns the statistic (T,), the maximum over
    channels and both sides, and the alarm times when ``h`` is given (the sums restart after each alarm, so the
    alarm count is a false-alarm count on clean data). NaNs (warm-up) leave the sums unchanged. ``upper_only`` for
    anomaly scores where only large values are suspicious."""
    z = np.asarray(z, dtype=float)
    z2 = z[:, None] if z.ndim == 1 else z
    hi = np.zeros(z2.shape[1])
    lo = np.zeros(z2.shape[1])
    stat = np.zeros(len(z2))
    alarms: list[int] = []
    for t, zt in enumerate(z2):
        zt = np.where(np.isfinite(zt), zt, 0.0)
        hi = np.maximum(0.0, hi + zt - k)
        if not upper_only:
            lo = np.maximum(0.0, lo - zt - k)
        stat[t] = max(hi.max(), lo.max())
        if h is not None and stat[t] > h:
            alarms.append(t)
            hi[:] = 0.0
            lo[:] = 0.0
    return stat, alarms


class CusumAlarm:
    """Alarm logic shared by every detector here: standardised scores ``self.z(D)`` (T, m) → CUSUM → threshold
    calibrated to a false-alarm budget. Subclasses define ``z`` and ``channels``; ``upper_only`` for scores where
    only large values are suspicious. Physics and data-driven detectors use the same logic, so a comparison between
    them measures the residuals, not the alarm rule."""
    k: float
    h: float
    upper_only: bool = False

    @property
    def channels(self) -> list[str]:
        raise NotImplementedError

    def z(self, D: Data) -> np.ndarray:
        raise NotImplementedError

    def score(self, D: Data) -> np.ndarray:
        """CUSUM statistic per sample without restarts (for ROC curves)."""
        return cusum(self.z(D), self.k, upper_only=self.upper_only)[0]

    def calibrate(self, D: Data, false_alarms_per_day: float = 1.0, samples_per_day: int = 1440) -> float:
        """Smallest threshold whose restarted CUSUM gives at most the budget of alarms on clean data ``D`` (data the
        detector was not fitted on)."""
        Z = self.z(D)
        days = len(Z) / samples_per_day
        lo, hi = 0.0, float(cusum(Z, self.k, upper_only=self.upper_only)[0].max()) + 1.0
        for _ in range(40):                                             # alarms decrease with h: bisect
            mid = 0.5 * (lo + hi)
            if len(cusum(Z, self.k, mid, self.upper_only)[1]) / days <= false_alarms_per_day:
                hi = mid
            else:
                lo = mid
        self.h = hi
        return hi

    def detect(self, D: Data) -> dict[str, Any]:
        Z = self.z(D)
        stat, alarms = cusum(Z, self.k, self.h, self.upper_only)
        names = self.channels
        blamed = []
        for t in alarms:                                                # channel with the largest recent deviation
            seg = np.nan_to_num(Z[max(0, t - 30):t + 1])
            blamed.append(names[int(np.argmax(np.abs(seg).mean(axis=0)))])
        return {"n": len(Z), "threshold": self.h, "alarms": alarms, "channels": blamed,
                "first_alarm": alarms[0] if alarms else None, "statistic": stat}


@dataclass
class PhysicsResidualDetector(CusumAlarm):
    """Grey-box physics residuals + per-channel two-sided CUSUM, threshold calibrated to a false-alarm budget.

    ``fit(D_trusted)`` identifies the components' constants and each channel's robust centre and spread;
    ``calibrate(D_clean, false_alarms_per_day)`` picks the threshold on attack-free data not used for fitting;
    ``score`` gives the CUSUM statistic per sample (ROC curves); ``detect`` the alarms and the channel blamed for
    each (the balance that broke points at the manipulated sensor)."""
    components: Sequence[ResidualComponent] = ()
    k: float = 0.5
    h: float = 8.0
    mu_: np.ndarray = field(default=None, repr=False)
    sigma_: np.ndarray = field(default=None, repr=False)

    @property
    def channels(self) -> list[str]:
        return [c.name for c in self.components]

    def residuals(self, D: Data) -> np.ndarray:
        return np.stack([c.residual(D) for c in self.components], axis=1)

    def fit(self, D: Data) -> PhysicsResidualDetector:
        if not self.components:
            raise ValueError("give at least one residual component")
        for c in self.components:
            c.fit(D)
        R = self.residuals(D)
        self.mu_ = np.nanmedian(R, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(R - self.mu_), axis=0)
        self.sigma_ = np.where(mad > 0, mad, np.nanstd(R, axis=0) + 1e-300)
        return self

    def z(self, D: Data) -> np.ndarray:
        if self.mu_ is None:
            raise RuntimeError("fit() on trusted data first")
        return (self.residuals(D) - self.mu_) / self.sigma_


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties). NaN scores are dropped."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):                                                  # average rank over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def window_auc(stat_attacked: np.ndarray, episodes: Sequence[tuple[int, int]], stat_clean: np.ndarray) -> float:
    """Episode-level ROC AUC: the largest statistic inside each attack episode against the largest statistic in
    every non-overlapping window of the same length of a clean trace. Separability of attacked from clean windows,
    independent of any threshold."""
    length = int(np.median([e - s for s, e in episodes]))
    pos = [np.nanmax(stat_attacked[s:e]) for s, e in episodes]
    c = np.asarray(stat_clean, dtype=float)
    neg = [np.nanmax(c[i:i + length]) for i in range(0, len(c) - length + 1, length)]
    return roc_auc(np.r_[pos, neg], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)])


def episode_metrics(alarms: Sequence[int], episodes: Sequence[tuple[int, int]], n: int,
                    samples_per_day: int = 1440, grace: int = 60) -> dict[str, Any]:
    """Detection rate and delay per attack episode ``(start, end)`` (end exclusive) and false alarms per day
    outside the episodes. An alarm in ``[start, end)`` detects the episode; the delay is in samples. Alarms in the
    ``grace`` samples after an episode (the attack being withdrawn is itself a discontinuity) count as neither."""
    alarms = np.asarray(sorted(alarms), dtype=int)
    inside = np.zeros(n, dtype=bool)
    delays = []
    for s, e in episodes:
        inside[s:min(e + grace, n)] = True
        hit = alarms[(alarms >= s) & (alarms < e)]
        delays.append(int(hit[0] - s) if len(hit) else None)
    false = int(sum(1 for a in alarms if not inside[a]))
    clean_days = max((~inside).sum() / samples_per_day, 1e-12)
    found = [d for d in delays if d is not None]
    return {"detected": len(found), "episodes": len(episodes),
            "detection_rate": len(found) / max(len(episodes), 1),
            "median_delay": float(np.median(found)) if found else None, "delays": delays,
            "false_alarms": false, "false_alarms_per_day": false / clean_days}
