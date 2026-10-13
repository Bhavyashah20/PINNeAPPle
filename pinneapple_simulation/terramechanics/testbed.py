"""Single-wheel testbed: drawbar pull, torque and sinkage versus slip at a fixed vertical load, and comparison with
published measurements.

    from pinneapple_simulation.terramechanics import testbed
    data = testbed.load_dataset("ding2011", "grousered")         # bundled points, see data/single_wheel_tests.csv
    rep = testbed.validate(data)                                   # model with the cited soil, errors per quantity
    cal = testbed.calibrate_holdout(data)                          # slip-sinkage fit on a training subset only
    cal["test_before"], cal["test_after"]                          # errors on the held-out points
    testbed.plot_validation([rep], "validation.png", calibrated=[cal])

Slip-sinkage calibration (:func:`calibrate_holdout`): the sinkage exponent of the Bekker pressure-sinkage law grows
with slip, ``n = n0 + n1 |s|`` (Ding et al. 2010, 2011: the slip-sinkage model they identified from single-wheel tests
on lunar simulant). ``n0`` and ``n1`` (and the shear modulus ``K``) are fitted by least squares to the sinkage and
drawbar pull of a *training* subset of the published points (every other point by slip, see :func:`holdout_split`);
the remaining points are held out and the errors on them are reported with the cited soil (before) and the fitted
soil (after). The published points are never changed.

The model side is :meth:`WongReeceModel.steady_state` (sinkage from the load balance, then forces). A testbed runs on
Earth, so the bundled cases use g = 9.81 m/s^2 (it only enters the bulldozing term; the load is prescribed).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .soils import Soil, get_soil
from .wheel import Grousers, Wheel
from .wong_reece import WongReeceModel

DATA = os.path.join(os.path.dirname(__file__), "data", "single_wheel_tests.csv")
SOURCES = {
    "ding2011": "Ding et al. (2011), J. Terramechanics 48(1):27-45, Figs 8/9 (80 N)",
    "hurrell2025": "Hurrell et al. (2025), Space Sci. Rev. 221, art. 37, Figs 5/6 (24.5 N)",
    "wanghan2016": "Wang & Han (2016), J. Korean Geotech. Soc. 32(11):97-108, Fig. 14 (58.86 N)",
}


@dataclass
class TestbedData:
    """Measured single-wheel points of one wheel on one soil at one load."""
    __test__ = False
    dataset: str
    wheel_kind: str
    wheel: Wheel
    soil: Soil
    load: float
    slip: np.ndarray
    drawbar_pull: np.ndarray
    sinkage: np.ndarray
    torque: np.ndarray
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.dataset} {self.wheel_kind}"


def list_datasets() -> list[tuple[str, str]]:
    rows = _rows()
    return list(dict.fromkeys((r["dataset"], r["wheel"]) for r in rows))


def _rows():
    with open(DATA) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def load_dataset(dataset: str, wheel: str, lambda_grouser: float = 1.0) -> TestbedData:
    """Bundled points of ``dataset`` ("ding2011", "hurrell2025", "wanghan2016") for ``wheel`` ("smooth" or
    "grousered"), sorted by slip; the soil comes from the database entry named in the file."""
    rows = sorted((r for r in _rows() if r["dataset"] == dataset and r["wheel"] == wheel), key=lambda r: float(r["slip"]))
    if not rows:
        raise KeyError(f"no data for {dataset!r} / {wheel!r}; available: {list_datasets()}")
    r0 = rows[0]
    g = Grousers(count=int(r0["grouser_count"]), height=float(r0["grouser_height_m"]),
                 shear_radius_factor=lambda_grouser)
    whl = Wheel(radius=float(r0["radius_m"]), width=float(r0["width_m"]), grousers=g)
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])  # noqa: E731
    return TestbedData(dataset, wheel, whl, get_soil(r0["soil"]), float(r0["load_N"]), col("slip"),
                       col("drawbar_pull_N"), col("sinkage_m"), col("torque_Nm"), SOURCES.get(dataset, ""))


@dataclass
class ValidationReport:
    data: TestbedData
    model: dict                         # model curves at the measured slips
    curve: dict                         # model curves on a fine slip grid
    errors: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"dataset": self.data.label, "source": self.data.source, "soil": self.data.soil.name,
                "load_N": self.data.load, **self.errors}


def _err(model, meas, scale):
    ok = np.isfinite(meas)
    if not ok.any():
        return {}
    e = model[ok] - meas[ok]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs": float(np.max(np.abs(e))),
            "rmse_rel_scale": float(np.sqrt(np.mean(e ** 2)) / scale), "bias": float(np.mean(e))}


def run_testbed(model: WongReeceModel, load: float, slips) -> dict:
    """Model drawbar pull [N], torque [N m], sinkage [m] and DP/W versus slip at a fixed load [N]."""
    return model.steady_state(np.full(len(np.atleast_1d(slips)), float(load)), np.atleast_1d(slips))


def validate(data: TestbedData, soil: Soil | None = None, g: float = 9.81, **model_kw) -> ValidationReport:
    """Run the testbed at the measured slips with ``soil`` (default: the dataset's cited soil) and report the errors:
    drawbar pull and torque RMSE [N, N m] and RMSE relative to the load / load x radius; sinkage RMSE [m] and
    relative to the largest measured sinkage."""
    m = WongReeceModel(soil or data.soil, data.wheel, g=g, **model_kw)
    at = run_testbed(m, data.load, data.slip)
    fine = run_testbed(m, data.load, np.linspace(0.0, max(0.7, float(data.slip.max())), 41))
    errors = {"drawbar_pull_N": _err(at["drawbar_pull"], data.drawbar_pull, data.load),
              "sinkage_m": _err(at["sinkage"], data.sinkage, float(np.nanmax(data.sinkage))),
              "torque_Nm": _err(at["torque"], data.torque, data.load * data.wheel.radius)}
    return ValidationReport(data, at, fine, {k: v for k, v in errors.items() if v})


CALIBRATION_BOUNDS = {"a0": (0.05, 0.8), "a1": (0.0, 0.8), "n0": (0.3, 2.0), "n1": (0.0, 2.0), "K": (0.002, 0.05),
                      "c3": (0.0, 0.6)}


def _errors_at(data: TestbedData, model_at: dict, mask) -> dict:
    zs = float(np.nanmax(data.sinkage))
    out = {"drawbar_pull_N": _err(model_at["drawbar_pull"][mask], data.drawbar_pull[mask], data.load),
           "sinkage_m": _err(model_at["sinkage"][mask], data.sinkage[mask], zs),
           "torque_Nm": _err(model_at["torque"][mask], data.torque[mask], data.load * data.wheel.radius)}
    return {k: v for k, v in out.items() if v}


def holdout_split(data: TestbedData, every: int = 2, offset: int = 0) -> np.ndarray:
    """Boolean training mask over the points of ``data`` (sorted by slip): every ``every``-th point starting at
    ``offset`` is used for fitting, the others are held out. The default (0, 2, 4 ...) keeps the lowest slip in the
    training set, so the static sinkage is constrained, and holds out points across the whole slip range."""
    idx = np.arange(len(data.slip))
    return (idx % every) == (offset % every)


def calibrate(data: TestbedData, params=("n0", "n1", "K"), bounds=None, g: float = 9.81, train=None) -> dict:
    """Least-squares fit of soil ``params`` to the drawbar pull and sinkage of ``data`` (scaled by the load and the
    largest measured sinkage). ``train`` (boolean mask) restricts the fit to those points; by default all points are
    used, which is a fit, not a validation. Returns the fitted soil, its parameters and the report on all points."""
    from scipy.optimize import least_squares
    base = data.soil
    bounds = {**CALIBRATION_BOUNDS, "k_phi": (0.2 * base.k_phi, 5 * base.k_phi), **(bounds or {})}
    train = np.ones(len(data.slip), bool) if train is None else np.asarray(train, bool)
    x0 = np.array([getattr(base, p) for p in params], float)
    lo = np.array([bounds[p][0] for p in params])
    hi = np.array([bounds[p][1] for p in params])
    x0 = np.clip(x0, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
    ok_z = np.isfinite(data.sinkage[train])
    zs = float(np.nanmax(data.sinkage[train]))          # scale from the training points only (no held-out leak)
    slips = data.slip[train]

    def res(x):
        soil = base.with_(**dict(zip(params, x, strict=True)))
        r = run_testbed(WongReeceModel(soil, data.wheel, g=g), data.load, slips)
        return np.r_[(r["drawbar_pull"] - data.drawbar_pull[train]) / data.load,
                     (r["sinkage"][ok_z] - data.sinkage[train][ok_z]) / zs]

    sol = least_squares(res, x0, bounds=(lo, hi), x_scale=np.maximum(np.abs(x0), 1e-3))
    soil = base.with_(**dict(zip(params, sol.x, strict=True)))
    rep = validate(data, soil=soil, g=g)
    at_bound = [p for p, v, a, b in zip(params, sol.x, lo, hi, strict=True) if min(v - a, b - v) < 1e-3 * (b - a)]
    return {"soil": soil, "params": dict(zip(params, sol.x.tolist(), strict=True)), "report": rep,
            "at_bound": at_bound}


def calibrate_holdout(data: TestbedData, params=("n0", "n1", "K"), every: int = 2, offset: int = 0,
                      g: float = 9.81) -> dict:
    """Slip-sinkage calibration with held-out points: fit ``params`` (default: the slip-dependent sinkage exponent
    ``n = n0 + n1 |s|`` of Ding et al. 2010/2011 and the shear modulus K) on :func:`holdout_split` training points
    only, then report the errors on the training and held-out points with the cited soil (``*_before``) and the
    fitted soil (``*_after``)."""
    train = holdout_split(data, every, offset)
    test = ~train
    before = validate(data, g=g)
    fit = calibrate(data, params, g=g, train=train)
    return {"dataset": data.label, "params": fit["params"], "at_bound": fit["at_bound"], "soil": fit["soil"],
            "train": train, "report_before": before, "report": fit["report"],
            "train_before": _errors_at(data, before.model, train), "train_after": _errors_at(data, fit["report"].model,
                                                                                           train),
            "test_before": _errors_at(data, before.model, test), "test_after": _errors_at(data, fit["report"].model,
                                                                                        test)}


def plot_validation(reports, path: str, title: str = "", calibrated=None) -> str:
    """Drawbar pull / load and sinkage versus slip: measured points and model curves, one colour per report.

    With ``calibrated`` (results of :func:`calibrate_holdout`, one per report, same order) the solid lines are the
    calibrated model, the dashed lines the cited soil; filled points were used for fitting, open points are held
    out, and the legend gives the held-out RMSE before and after."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for k, rep in enumerate(reports):
        c = pal[k % len(pal)]
        d, W = rep.data, rep.data.load
        cal = calibrated[k] if calibrated else None
        if cal is None:
            lab_dp = d.label + (f" ({rep.errors['drawbar_pull_N']['rmse_rel_scale']:.3f} W rmse)"
                                if "drawbar_pull_N" in rep.errors else "")
            lab_z = d.label
            ax[0].plot(rep.curve["slip"], rep.curve["drawbar_pull"] / W, color=c, lw=1.8)
            ax[1].plot(rep.curve["slip"], 1e3 * rep.curve["sinkage"], color=c, lw=1.8)
            ax[0].plot(d.slip, d.drawbar_pull / W, "o", color=c, ms=5, label=lab_dp)
            ax[1].plot(d.slip, 1e3 * d.sinkage, "o", color=c, ms=5, label=lab_z)
            continue
        tb, ta = cal["test_before"], cal["test_after"]
        lab_dp = (f"{d.label}: held-out {tb['drawbar_pull_N']['rmse_rel_scale']:.3f} -> "
                  f"{ta['drawbar_pull_N']['rmse_rel_scale']:.3f} W")
        lab_z = f"{d.label}: held-out {1e3 * tb['sinkage_m']['rmse']:.1f} -> {1e3 * ta['sinkage_m']['rmse']:.1f} mm"
        cur = cal["report"].curve
        for a, key, sc, lab in ((ax[0], "drawbar_pull", 1.0 / W, lab_dp), (ax[1], "sinkage", 1e3, lab_z)):
            meas = (d.drawbar_pull / W) if key == "drawbar_pull" else 1e3 * d.sinkage
            a.plot(rep.curve["slip"], sc * rep.curve[key], color=c, lw=1.0, ls=(0, (4, 3)), alpha=0.75)
            a.plot(cur["slip"], sc * cur[key], color=c, lw=2.0)
            tr = cal["train"]
            a.plot(d.slip[tr], meas[tr], "o", color=c, ms=5.5, label=lab)
            a.plot(d.slip[~tr], meas[~tr], "o", mfc="white", mec=c, mew=1.6, ms=6)
    ax[0].set_ylabel("drawbar pull / load DP/W [-]")
    ax[1].set_ylabel("sinkage [mm]")
    for a in ax:
        a.set_xlabel("slip [-]")
        a.grid(True, color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.legend(fontsize=7.2, frameon=False, loc="upper left")
    if calibrated:
        ax[0].set_title("filled: fitted points   open: held out   solid: calibrated   dashed: cited soil",
                        fontsize=8.5, loc="left")
        ax[1].set_title("slip-sinkage n = n0 + n1 s (Ding et al. 2010, 2011)", fontsize=8.5, loc="left")
    else:
        ax[0].set_title("points: published tests   lines: Wong-Reece model", fontsize=9, loc="left")
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


__all__ = ["TestbedData", "ValidationReport", "load_dataset", "list_datasets", "run_testbed", "validate",
           "calibrate", "calibrate_holdout", "holdout_split", "plot_validation", "SOURCES", "CALIBRATION_BOUNDS"]
