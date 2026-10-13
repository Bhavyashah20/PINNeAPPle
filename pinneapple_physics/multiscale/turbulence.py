"""Decaying 2-D turbulence as a multiscale benchmark: resolved reference data, filtered datasets at several widths,
query streams that shift the Reynolds number and the requested scale, and the closure benchmark.

:class:`DecayingTurbulence2D` holds the set-up (domain, output grid, output times, Reynolds bands, requested widths
and filter kinds, LES grid). Every case starts from a random vorticity field (energy spectrum peaked at ``k0``),
evolved for ``spinup`` time units at its own viscosity so the cascade is developed; the question is the vorticity
``t_out`` later, filtered at a requested width, on the ``n_out`` grid. Training data come from a pseudo-spectral DNS
at ``n_train``; the independent reference from a finer DNS at ``n_ref``. Data generation runs in parallel processes
(at low priority) and is cached in ``.npz`` files.

* :meth:`~DecayingTurbulence2D.generate_training_data`, :meth:`~DecayingTurbulence2D.generate_reference`;
* :meth:`~DecayingTurbulence2D.query_stream`: queries ``{"id", "seed", "nu", "re", "band", "width", "kind",
  "filter"}``; :meth:`~DecayingTurbulence2D.reference_output`: the filtered reference of a query;
* :meth:`~DecayingTurbulence2D.filtered_pairs`: (input, target) pairs at one filter for supervised surrogates;
* :func:`benchmark_closures`: a-priori, a-posteriori and label-free checks of several closures on held-out cases;
* :func:`scale_conditioning_study`: errors of a scale-conditioned model against fixed-width and scale-blind models,
  at trained and held-out widths, with its cross-scale self-consistency.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .filters import Filter, apply_filter, refilter, restrict
from .ns2d import VorticityNS2D, random_vorticity

__all__ = ["DecayingTurbulence2D", "benchmark_closures", "scale_conditioning_study"]


def _case(args) -> np.ndarray:
    """Worker: (seed, nu, n, dt, spinup, times, k0, energy, nice) -> (1 + len(times), n, n), t = 0 first."""
    seed, nu, n, dt, spinup, times, k0, energy, nice = args
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass
    w0 = random_vorticity(np.random.default_rng(seed), n, k0=k0, energy=energy)
    s = VorticityNS2D(n, nu)
    w = s.simulate(w0, [spinup], dt=dt)[0]
    traj = s.simulate(w, list(times), dt=dt)
    return np.concatenate([w[None], traj]).astype(np.float32)


@dataclass
class DecayingTurbulence2D:
    """Set-up of the benchmark (defaults: the experiment of ``docs/core_concepts/multiscale_ensembles.md``)."""
    length: float = 2 * math.pi
    n_out: int = 64
    t_out: tuple[float, ...] = (0.28, 0.30)
    spinup: float = 0.5
    k0: float = 6.0
    energy: float = 0.5
    re_bands: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "low": (150.0, 300.0), "mid": (700.0, 1400.0), "high": (2800.0, 5000.0)})
    widths: tuple[int, ...] = (1, 2, 4, 6, 8)            # requested width / h_out
    width_probabilities: tuple[float, ...] = (0.15, 0.2, 0.25, 0.15, 0.25)
    kinds: tuple[str, ...] = ("gaussian", "sharp", "box")
    n_train: int = 128
    dt_train: float = 5e-3
    train_times: tuple[float, ...] = (0.1, 0.2, 0.28, 0.30)
    n_ref: int = 256
    dt_ref: float = 2.5e-3
    les_n: int = 48
    nice: int = 5

    @property
    def h(self) -> float:
        """Spacing of the output grid."""
        return self.length / self.n_out

    @property
    def dt_pair(self) -> float:
        return self.t_out[-1] - self.t_out[-2]

    @property
    def les_filter(self) -> Filter:
        """Implicit sharp filter of the dealiased LES grid (``k <= les_n / 3``)."""
        return Filter("sharp", math.pi / (self.les_n // 3) * self.length / (2 * math.pi))

    def out_index(self) -> list[int]:
        """Indices of ``t_out`` in a training trajectory (t = 0 first)."""
        return [1 + self.train_times.index(t) for t in self.t_out]

    # -------------------------------------------------------------------------------------------- data
    def initial_state(self, seed: int, nu: float, n: int, dt: float) -> np.ndarray:
        return _case((seed, nu, n, dt, self.spinup, [], self.k0, self.energy, 0))[0].astype(float)

    def generate_training_data(self, path: str | Path, n_per_band: int = 200, workers: int = 4, seed: int = 12345
                               ) -> dict[str, np.ndarray]:
        """``n_per_band`` cases per Re band at ``n_train`` (cached in ``path``): ``w`` (cases, 1 + len(train_times),
        n, n) float32 with t = 0 first, ``nu``, ``band``, ``seed``."""
        path = Path(path)
        if path.exists():
            return dict(np.load(path))
        rng = np.random.default_rng(seed)
        jobs, band = [], []
        for b, (lo, hi) in self.re_bands.items():
            for _ in range(n_per_band):
                re = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                jobs.append((100000 + len(jobs), 1.0 / re))
                band.append(b)
        t0 = time.time()
        res = self._map([(s, nu, self.n_train, self.dt_train, self.spinup, self.train_times, self.k0, self.energy,
                          self.nice) for s, nu in jobs], workers)
        print(f"training DNS {self.n_train}^2: {len(jobs)} cases in {time.time() - t0:.0f} s", flush=True)
        data = {"w": np.stack(res), "nu": np.array([j[1] for j in jobs]), "band": np.array(band),
                "seed": np.array([j[0] for j in jobs])}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        return data

    @staticmethod
    def _map(jobs, workers):
        if workers <= 1:
            return [_case(j) for j in jobs]
        from multiprocessing import Pool
        with Pool(workers) as p:
            return p.map(_case, jobs, chunksize=1)

    def query_stream(self, seed: int, n: int = 60, order: Sequence[str] = ("low", "high", "mid")) -> list[dict]:
        """``n`` queries whose Re regime goes through ``order`` (equal blocks) and whose requested width and filter
        kind are drawn at random; new initial states (never in training)."""
        rng = np.random.default_rng(1000 + seed)
        qs = []
        for k in range(n):
            band = order[min(k * len(order) // n, len(order) - 1)]
            lo, hi = self.re_bands[band]
            re = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            width = int(rng.choice(self.widths, p=np.asarray(self.width_probabilities) / sum(self.width_probabilities)))
            kind = str(rng.choice(self.kinds))
            qs.append({"id": k, "seed": 500000 + 1000 * seed + k, "nu": 1.0 / re, "re": re, "band": band,
                       "width": width, "kind": kind, "filter": Filter(kind, width * self.h)})
        return qs

    def generate_reference(self, path: str | Path, queries: Sequence[dict], workers: int = 4) -> dict[str, np.ndarray]:
        """Reference DNS at ``n_ref`` for every query (cached in ``path``): ``w0`` (n, n_ref, n_ref), the state the
        query gives, and ``traj`` (n, len(t_out), n_ref, n_ref)."""
        path = Path(path)
        if path.exists():
            return dict(np.load(path))
        t0 = time.time()
        res = self._map([(q["seed"], q["nu"], self.n_ref, self.dt_ref, self.spinup, self.t_out, self.k0, self.energy,
                          self.nice) for q in queries], workers)
        print(f"reference DNS {self.n_ref}^2: {len(queries)} cases in {time.time() - t0:.0f} s", flush=True)
        out = {"w0": np.stack([r[0] for r in res]), "traj": np.stack([r[1:] for r in res])}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
        return out

    def reference_output(self, traj: np.ndarray, filt: Filter) -> np.ndarray:
        """Filtered reference on the output grid."""
        return restrict(apply_filter(np.asarray(traj, float), filt, self.length), self.n_out)

    def filtered_input(self, w0: np.ndarray, filt: Filter, n: int | None = None) -> np.ndarray:
        """A state filtered with ``filt`` on an ``n``-point grid (default: the output grid)."""
        return restrict(apply_filter(np.asarray(w0, float), filt, self.length), n or self.n_out)

    def split(self, data: Mapping[str, np.ndarray], bands: Sequence[str] | str, test_frac: float = 0.2
              ) -> tuple[np.ndarray, np.ndarray]:
        """(train, held-out) case indices of the given bands: the first ``test_frac`` of each selection is held out."""
        bands = (bands,) if isinstance(bands, str) else tuple(bands)
        idx = np.flatnonzero(np.isin(data["band"], bands))
        n_test = int(round(test_frac * len(idx)))
        return idx[n_test:], idx[:n_test]

    def filtered_pairs(self, data: Mapping[str, np.ndarray], idx: np.ndarray, filt: Filter, n: int | None = None,
                       scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Inputs ``(B, n, n)`` (initial state filtered with ``filt``) and targets ``(B, len(t_out), n, n)`` (the state
        at ``t_out``, same filter), divided by ``scale``."""
        w = np.asarray(data["w"][idx], float)
        x = self.filtered_input(w[:, 0], filt, n) / scale
        y = self.filtered_input(w[:, self.out_index()], filt, n) / scale
        return x, y


def benchmark_closures(bench: DecayingTurbulence2D, data: Mapping[str, np.ndarray],
                       closures: Mapping[str, Any], cases_per_band: int = 8, times: Sequence[float] = (0.1, 0.2, 0.3),
                       t_long: float = 2.0, dt: float = 5e-3) -> list[dict[str, Any]]:
    """Closures on held-out cases of every band, on the LES grid of ``bench`` (sharp filter of the dealiased grid).

    A priori (filtered training DNS): stress and tendency correlations, backscatter, Germano-identity error. A
    posteriori (LES from the filtered initial state): spectrum and field error at the last of ``times``, resolved
    dissipation against the reference, stability and enstrophy growth to ``t_long``, and whether the label-free
    :func:`~.closures.closure_health` flags the run. ``closures``: ``{name: closure}``, None for no model; closures
    without a ``stress`` method (learned ones) get the tendency-level tests only. ``times`` must be in
    ``bench.train_times``."""
    from .closures import (
        aposteriori_test,
        apriori_test,
        closure_health,
        germano_error,
        strain,
        true_subfilter,
    )

    filt = bench.les_filter
    n = bench.les_n
    s_les = VorticityNS2D(n, 0.0, bench.length)
    tidx = [1 + bench.train_times.index(t) for t in times]
    rows = []
    for band in bench.re_bands:
        _, te = bench.split(data, band)
        te = te[:cases_per_band]
        for name, cl in closures.items():
            acc: dict[str, list[float]] = {k: [] for k in (
                "stress_corr", "tend_corr", "backscatter_true", "backscatter_model", "germano", "spec_err",
                "field_err", "diss_ratio", "stable", "z_growth", "flagged")}
            for i in te:
                w = np.asarray(data["w"][i], float)
                nu = float(data["nu"][i])
                w0b = bench.filtered_input(w[0], filt, n)
                tr = true_subfilter(w[0], filt, n, bench.length)
                st = strain(tr["w_bar"], s_les)
                acc["backscatter_true"].append(float(np.mean(-2 * (tr["a"] * st["S11"] + tr["b"] * st["S12"]) < 0)))
                if cl is not None:
                    if hasattr(cl, "stress"):
                        a = apriori_test(w[0], filt, cl, n, bench.length)
                        acc["stress_corr"].append(a.stress_correlation)
                        acc["backscatter_model"].append(a.backscatter_model)
                        acc["germano"].append(germano_error(w0b, cl, s_les))
                    t_m = cl(tr["w_bar"], s_les)
                    if np.std(t_m) > 0:
                        acc["tend_corr"].append(float(np.corrcoef(tr["tendency"].ravel(), t_m.ravel())[0, 1]))
                ref = bench.filtered_input(w[tidx], filt, n)
                les = VorticityNS2D(n, nu, bench.length, closure=cl)
                ap = aposteriori_test(les, w0b, ref, list(times), dt=dt, t_long=t_long)
                acc["spec_err"].append(ap.spectrum_error[-1])
                acc["field_err"].append(ap.field_error[-1])
                acc["diss_ratio"].append(ap.dissipation_les / ap.dissipation_ref if ap.dissipation_ref else np.nan)
                acc["stable"].append(float(ap.stable))
                acc["z_growth"].append(ap.enstrophy_growth)
                if ap.stable:
                    wl = les.simulate(w0b, list(bench.t_out), dt=dt)
                    h = closure_health(wl, w0b, n // 3, bench.length,
                                       closure=cl if hasattr(cl, "stress") else None, solver=s_les)
                    acc["flagged"].append(float(not h.ok))
                else:
                    acc["flagged"].append(1.0)
            rows.append({"band": band, "closure": name, "n": len(te),
                         **{k: (float(np.nanmean(v)) if len(v) else float("nan")) for k, v in acc.items()}})
            print(rows[-1], flush=True)
    return rows


def scale_conditioning_study(bench: DecayingTurbulence2D, data: Mapping[str, np.ndarray], idx: np.ndarray,
                             conditioned: Callable[[np.ndarray, float, float], np.ndarray],
                             fixed: Mapping[float, Callable[[np.ndarray, float], np.ndarray]],
                             blind: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
                             widths: Sequence[float] = (1, 2, 4, 6, 8), kind: str = "gaussian") -> list[dict]:
    """Errors at each width (``widths`` in units of ``bench.h``) on the cases ``idx``:

    * ``conditioned(w0, nu, delta)``: the scale-conditioned model asked at ``delta``;
    * ``fixed``: ``{width: predict(w0, nu)}``, models trained at one width; evaluated at ``delta`` by post-filtering
      the narrowest one at or below ``delta`` (exact for Gaussians), the best of them reported;
    * ``blind(w0, nu, delta)``: the same network without the width input (its input is still filtered at ``delta``);

    plus the spectrum error of the conditioned model and its self-consistency (prediction at ``delta``, filtered to
    ``2 delta``, against the prediction at ``2 delta``)."""
    from .closures import spectrum_error

    rows = []
    h = bench.h
    for d in widths:
        tgt = Filter(kind, d * h)
        acc: dict[str, list[float]] = {k: [] for k in ("conditioned", "blind", "best fixed", "spectrum", "consistency")}
        for i in idx:
            w = np.asarray(data["w"][i], float)
            nu = float(data["nu"][i])
            ref = bench.filtered_input(w[bench.out_index()], tgt)
            rel = lambda p: float(np.linalg.norm(p - ref) / np.linalg.norm(ref))  # noqa: B023, E731
            p = conditioned(w[0], nu, d * h)
            acc["conditioned"].append(rel(p))
            acc["spectrum"].append(spectrum_error(p[-1], ref[-1], min(tgt.cutoff, bench.n_out / 2 - 1), bench.length))
            if blind is not None:
                acc["blind"].append(rel(blind(w[0], nu, d * h)))
            ok = [e for e in fixed if e <= d * h * (1 + 1e-9)]
            if ok:
                acc["best fixed"].append(min(rel(refilter(fixed[e](w[0], nu), Filter(kind, e), tgt, bench.length))
                                             for e in ok))
            if 2 * d <= max(widths):
                p2 = conditioned(w[0], nu, 2 * d * h)
                acc["consistency"].append(float(np.linalg.norm(refilter(p, tgt, Filter(kind, 2 * d * h), bench.length)
                                                               - p2) / np.linalg.norm(p2)))
        rows.append({"width": d, "n": len(idx), **{k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}})
        print(rows[-1], flush=True)
    return rows
