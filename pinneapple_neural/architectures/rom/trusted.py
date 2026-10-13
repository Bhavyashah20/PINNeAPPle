"""Latent surrogate that says when not to trust it: error bounds, envelope check and physics residual per query.

A parametric reduced-order model (:class:`~.parametric.ParametricPOD`: encode the fields in a POD latent space,
regress the latent coordinates on the design parameters, decode) answers in microseconds, and answers just as fast
where it is wrong. :class:`TrustedLatentSurrogate` wraps it with three independent checks per query, and turns them
into the library's :class:`~pinneapple_analysis.trust.TrustReport`:

1. **Error bound** — the latent regressor's predictive spread (a Gaussian process's own standard deviation, or a
   bootstrap ensemble of RBF latent regressors) gives a spread ``s(p)`` of the decoded field;
   normalised split-conformal prediction (Lei et al. 2018; Angelopoulos & Bates 2022) on a held-out calibration set
   scales it to a bound ``q * s(p)`` that covers the true relative error with probability ``1 - alpha`` for queries
   drawn like the calibration set.
2. **Envelope** — the mean distance to the ``k`` nearest training designs (scaled parameters) converted to a
   conformal p-value against the calibration designs. Catches holes inside the parameter box and extrapolation,
   where the coverage of 1. no longer holds (exchangeability is lost) and the spread may be overconfident.
3. **Physics residual** — optional ``residual(p, x) -> float``: the discrete governing equations evaluated on the
   decoded field (a sparse mat-vec, no solve). Independent of the data: it sees errors the ensemble agrees on.

Each check is calibrated on data the latent model did not see. The decision rule is the library's: an envelope or
physics failure is critical (REJECT), an error bound above the tolerance asks for REVIEW.

References: J. Lei, M. G'Sell, A. Rinaldo, R. Tibshirani, L. Wasserman, JASA 113 (2018) 1094-1111 (normalised
conformal regression); A. N. Angelopoulos, S. Bates, arXiv:2107.07511; V. Vovk et al., Algorithmic Learning in a
Random World (2005) (conformal p-values); B. Peherstorfer, K. Willcox, CMAME 306 (2016) (non-intrusive ROMs);
S. Hijazi et al. / A. Quarteroni et al. on a-posteriori residual estimators for reduced bases.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .parametric import ParametricPOD

__all__ = ["TrustedLatentSurrogate"]


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th order statistic, k = ceil((n + 1)(1 - alpha)); +inf when the calibration set is too small."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1.0 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def _pvalue(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Conformal p-value of nonconformity ``x`` against calibration scores ``cal`` (larger = stranger)."""
    cal = np.sort(np.asarray(cal, dtype=float))
    ge = len(cal) - np.searchsorted(cal, np.asarray(x, dtype=float), side="left")
    return (ge + 1.0) / (len(cal) + 1.0)


@dataclass
class TrustedLatentSurrogate:
    """POD latent surrogate + per-query error bound, envelope p-value and physics residual.

    ``rom``: the latent model (any :class:`ParametricPOD` setting; RBF regression by default). ``spread``: where
    the spread comes from — ``"gp"`` (the GP's predictive std; needs ``regressor="gpr"``), ``"ensemble"``
    (``members`` RBF refits of the latent regression on ``subsample`` of the training designs, same basis) or
    ``"auto"`` (gp for a GP regressor, ensemble otherwise). ``alpha``: miscoverage of the error bound. ``alpha_envelope``,
    ``alpha_residual``: false-rejection rate of the envelope and residual checks on in-distribution queries.
    ``residual``: ``f(p, x) -> float`` physics residual of the field ``x`` at parameters ``p`` (relative)."""
    rom: ParametricPOD = field(default_factory=lambda: ParametricPOD(regressor="rbf"))
    spread: str = "auto"
    members: int = 16
    subsample: float = 0.8
    neighbours: int = 5
    alpha: float = 0.1
    alpha_envelope: float = 0.02
    alpha_residual: float = 0.02
    residual: Callable[[np.ndarray, np.ndarray], float] | None = None
    seed: int = 0
    calibration_: dict[str, Any] = field(default=None, repr=False)

    # -- fitting -----------------------------------------------------------------------------------------------

    def fit(self, P_train: np.ndarray, X_train: np.ndarray, P_cal: np.ndarray, X_cal: np.ndarray
            ) -> TrustedLatentSurrogate:
        """Fit the latent model and the ensemble on the training designs, calibrate the three checks on the
        calibration designs (fields the model never saw, drawn from the same envelope)."""
        from scipy.interpolate import RBFInterpolator
        P_train = np.atleast_2d(np.asarray(P_train, float))
        self.rom.fit(P_train, X_train)
        Ps, A = self.rom._P, self.rom._A
        self._mode = self.spread if self.spread != "auto" else ("gp" if self.rom.regressor == "gpr" else "ensemble")
        if self._mode == "gp" and self.rom.regressor != "gpr":
            raise ValueError('spread="gp" needs ParametricPOD(regressor="gpr")')
        if self._mode not in ("gp", "ensemble"):
            raise ValueError(f"unknown spread {self.spread!r}")
        self._members = []
        if self._mode == "ensemble":
            n = len(Ps)
            m = max(int(round(self.subsample * n)), Ps.shape[1] + 2)
            rng = np.random.default_rng(self.seed)
            for _ in range(self.members):
                idx = rng.choice(n, size=min(m, n), replace=False)
                self._members.append(RBFInterpolator(Ps[idx], A[idx], kernel=self.rom.kernel,
                                                     smoothing=self.rom.smoothing))
        self._Ptrain = Ps
        P_cal = np.atleast_2d(np.asarray(P_cal, float))
        X_cal = np.asarray(X_cal, float).reshape(len(P_cal), -1)
        err = self.rom.error(P_cal, X_cal)
        Xh, s = self._predict_spread(P_cal)
        self._s0 = float(np.median(s))                               # floor: the spread vanishes at training designs
        ratio = err / (s + self._s0)
        dist = self._knn(P_cal)
        cal: dict[str, Any] = {"n": len(P_cal), "q": _conformal_quantile(ratio, self.alpha),
                               "spread_floor": self._s0, "distance": dist, "error": err}
        if self.residual is not None:
            res = np.array([self.residual(p, x) for p, x in zip(P_cal, Xh, strict=True)])
            cal["residual"] = res
            cal["residual_limit"] = _conformal_quantile(res, self.alpha_residual)
        self.calibration_ = cal
        return self

    # -- prediction --------------------------------------------------------------------------------------------

    def _knn(self, P: np.ndarray) -> np.ndarray:
        Ps = self.rom._scale(np.atleast_2d(P))
        d = np.sqrt(((Ps[:, None, :] - self._Ptrain[None]) ** 2).sum(-1))
        k = min(self.neighbours, d.shape[1])
        return np.sort(d, axis=1)[:, :k].mean(1)

    def _predict_spread(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.atleast_2d(np.asarray(P, float))
        A, std = self.rom.coefficients(P, return_std=True)
        if self._mode == "ensemble":
            Ps = self.rom._scale(P)
            std = np.stack([mem(Ps) for mem in self._members]).std(0)     # (n, r)
        shape = A @ self.rom.basis_.T + self.rom.mean_                       # before the amplitude, if normalised
        rel = np.linalg.norm(std, axis=1) / np.maximum(np.linalg.norm(shape, axis=1), 1e-300)  # orthonormal basis
        amp = np.exp(self.rom._amp(self.rom._scale(P)))[:, None] if self.rom.normalize else 1.0
        return shape * amp, rel

    def predict(self, P: np.ndarray) -> np.ndarray:
        return self.rom.predict(np.atleast_2d(np.asarray(P, float)))

    def assess(self, P: np.ndarray, tol: float, *, with_residual: bool = True) -> dict[str, np.ndarray]:
        """Prediction and the three checks for each row of ``P``. ``tol``: largest acceptable relative L2 error.
        Returns arrays: ``X``, ``error_bound``, ``envelope_p``, ``residual`` (NaN if not run), and ``decision``
        (APPROVED / REVIEW / REJECT with the rule of :meth:`trust_report`)."""
        if self.calibration_ is None:
            raise RuntimeError("fit() first")
        P = np.atleast_2d(np.asarray(P, float))
        X, s = self._predict_spread(P)
        bound = self.calibration_["q"] * (s + self._s0)
        pval = _pvalue(self.calibration_["distance"], self._knn(P))
        res = np.full(len(P), np.nan)
        if with_residual and self.residual is not None:
            res = np.array([self.residual(p, x) for p, x in zip(P, X, strict=True)])
        decision = np.array([self._decide(b, pv, r, tol) for b, pv, r in zip(bound, pval, res, strict=True)])
        return {"X": X, "error_bound": bound, "envelope_p": pval, "residual": res, "decision": decision}

    def _decide(self, bound: float, pval: float, res: float, tol: float) -> str:
        if pval < self.alpha_envelope:
            return "REJECT"
        if np.isfinite(res) and res > self.calibration_["residual_limit"]:
            return "REJECT"
        return "APPROVED" if bound <= tol else "REVIEW"

    def trust_report(self, p: np.ndarray, tol: float, *, model: str = "latent surrogate"):
        """:class:`~pinneapple_analysis.trust.TrustReport` for one query: error bound (non-critical), envelope and
        physics residual (critical). The report's decision equals :meth:`assess`'s."""
        from pinneapple_analysis.trust.trust_report import (
            CONTRADICTS,
            NOT_RUN,
            SUPPORTS,
            Check,
            TrustReport,
        )
        a = self.assess(np.atleast_2d(p), tol)
        b, pv, r = float(a["error_bound"][0]), float(a["envelope_p"][0]), float(a["residual"][0])
        src = "pinneapple_neural.architectures.rom.TrustedLatentSurrogate"
        checks = [
            Check("conformal error bound", SUPPORTS if b <= tol else CONTRADICTS, value=b, limit=tol, source=src,
                  reason=f"{1 - self.alpha:.0%} split-conformal bound on the relative L2 error"),
            Check("design envelope", SUPPORTS if pv >= self.alpha_envelope else CONTRADICTS, critical=True,
                  value=pv, limit=self.alpha_envelope, source=src,
                  reason="conformal p-value of the distance to the training designs"),
        ]
        if np.isfinite(r):
            lim = self.calibration_["residual_limit"]
            checks.append(Check("physics residual", SUPPORTS if r <= lim else CONTRADICTS, critical=True, value=r,
                                limit=lim, source=src, reason="governing equations on the decoded field"))
        else:
            checks.append(Check("physics residual", NOT_RUN, critical=True, source=src,
                                reason="no residual function given"))
        return TrustReport(checks=checks, min_coverage=0.6, min_score=1.0, model=model,
                           details={"tol": tol, "alpha": self.alpha, "rank": self.rom.rank})
