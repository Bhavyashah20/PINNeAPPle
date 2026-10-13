"""Single-wheel testbed: drawbar pull, torque and sinkage versus slip at a fixed vertical load, and comparison with
published measurements.

    from pinneapple_simulation.terramechanics import testbed
    data = testbed.load_dataset("ding2011", "grousered")         # bundled points, see data/single_wheel_tests.csv
    rep = testbed.validate(data)                                   # model with the cited soil, errors per quantity
    testbed.plot_validation([rep], "validation.png")

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


def calibrate(data: TestbedData, params=("a0", "a1", "n1", "K"), bounds=None, g: float = 9.81) -> dict:
    """Least-squares fit of soil ``params`` to the drawbar pull and sinkage of ``data`` (a fit, not a validation:
    report it separately). Returns the fitted soil and the errors after the fit."""
    from scipy.optimize import least_squares
    base = data.soil
    bounds = bounds or {"a0": (0.05, 0.8), "a1": (0.0, 0.8), "n1": (0.0, 1.0), "K": (0.002, 0.05),
                        "c3": (0.0, 0.6), "k_phi": (0.2 * base.k_phi, 5 * base.k_phi)}
    x0 = np.array([getattr(base, p) for p in params], float)
    lo = np.array([bounds[p][0] for p in params])
    hi = np.array([bounds[p][1] for p in params])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    ok_z = np.isfinite(data.sinkage)
    zs = float(np.nanmax(data.sinkage))

    def res(x):
        soil = base.with_(**dict(zip(params, x, strict=True)))
        r = run_testbed(WongReeceModel(soil, data.wheel, g=g), data.load, data.slip)
        return np.r_[(r["drawbar_pull"] - data.drawbar_pull) / data.load, (r["sinkage"][ok_z] - data.sinkage[ok_z]) / zs]

    sol = least_squares(res, x0, bounds=(lo, hi), x_scale=np.maximum(np.abs(x0), 1e-3))
    soil = base.with_(**dict(zip(params, sol.x, strict=True)))
    rep = validate(data, soil=soil, g=g)
    return {"soil": soil, "params": dict(zip(params, sol.x.tolist(), strict=True)), "report": rep}


def plot_validation(reports, path: str, title: str = "") -> str:
    """Drawbar pull / load and sinkage versus slip: measured points and model curves, one colour per report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for k, rep in enumerate(reports):
        c = pal[k % len(pal)]
        d, W = rep.data, rep.data.load
        lab = f"{d.label}" + (f" ({rep.errors['drawbar_pull_N']['rmse_rel_scale']:.3f} W rmse)"
                              if "drawbar_pull_N" in rep.errors else "")
        ax[0].plot(rep.curve["slip"], rep.curve["drawbar_pull"] / W, color=c, lw=1.8)
        ax[0].plot(d.slip, d.drawbar_pull / W, "o", color=c, ms=5, label=lab)
        ax[1].plot(rep.curve["slip"], 1e3 * rep.curve["sinkage"], color=c, lw=1.8)
        ax[1].plot(d.slip, 1e3 * d.sinkage, "o", color=c, ms=5, label=d.label)
    ax[0].set_ylabel("drawbar pull / load DP/W [-]")
    ax[1].set_ylabel("sinkage [mm]")
    for a in ax:
        a.set_xlabel("slip [-]")
        a.grid(True, color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].legend(fontsize=7.5, frameon=False)
    ax[0].set_title("points: published tests   lines: Wong-Reece model", fontsize=9, loc="left")
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


__all__ = ["TestbedData", "ValidationReport", "load_dataset", "list_datasets", "run_testbed", "validate",
           "calibrate", "plot_validation", "SOURCES"]
