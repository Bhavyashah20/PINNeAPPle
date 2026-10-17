"""Engineering Sensitivity Lab: web API + single-page UI. Which parameters really control the result?

Three ways in:
- a model (pp.zoo model, built-in benchmark with exact indices, or an uploaded ONNX graph): Morris screening, Sobol
  first/total/second-order indices with bootstrap intervals, effect curves (PDP + ICE), local derivatives by autodiff
  when the model is differentiable, and the "parameters you can freeze" recommendation with its direct check;
- runs done elsewhere (OpenFOAM, CalculiX, a campaign runner): download the Saltelli design, run it, upload the results;
- a dataset (experiments, past simulations): first-order indices from the data, with the input correlation reported.

Every number comes from pinneapple_analysis.sensitivity.

Run:  uvicorn sensitivity_lab.api:app --port 8099   (from apps/sensitivity_lab)

Environment (all optional):
  SEN_USER / SEN_PASSWORD   HTTP Basic login on everything except /health
  SEN_MAX_HEAVY             concurrent studies per worker (default 2)
  SEN_MAX_MB                largest upload in MB (default 50)
  SEN_MAX_EVALS             model evaluations per study (default 400000)
  SEN_MAX_PARAMS            varied parameters per study (default 30)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.sensitivity import (
    first_order_given_data,
    freeze_recommendation,
    saltelli_design,
    sensitivity_study,
    sobol_from_runs,
    study_cost,
)
from pinneapple_analysis.study_models import StudyModel, builtin, columns
from pinneapple_analysis.uncertainty.distributions import as_space
from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.data_health import load_table

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
MAX_MB = float(os.environ.get("SEN_MAX_MB") or "50")
MAX_EVALS = int(os.environ.get("SEN_MAX_EVALS") or "400000")
MAX_PARAMS = int(os.environ.get("SEN_MAX_PARAMS") or "30")

ZOO = {"heatsink_forced_mlp": ["r_base_k_w", "pressure_drop_pa"], "airfoil_cst_rans_mlp": ["Cl", "Cd", "Cm"]}
BUILTIN = ("ishigami", "sobol_g")

app = FastAPI(title="Engineering Sensitivity Lab", version=VERSION,
              description="Which parameters really control the result: Sobol indices with confidence intervals, Morris "
                          "screening, interactions, effect curves, local derivatives and the parameters you can "
                          "freeze, for a model (POST /api/study), runs done elsewhere (/api/design + /api/runs) or a "
                          "dataset (/api/given-data).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="SEN")
_HEAVY = BusyLimiter("SEN_MAX_HEAVY")


def _u(name, low, high, **kw):
    return {"name": name, "dist": "uniform", "low": low, "high": high, **kw}


HS_SPACE = [_u("base_width_mm", 90, 110), _u("base_depth_mm", 90, 110), _u("base_thickness_mm", 4, 8),
            _u("n_fins", 15, 21), _u("fin_thickness_mm", 1.0, 1.4), _u("fin_height_mm", 30, 40),
            _u("k_w_mk", 180, 210), _u("t_ambient_c", 20, 45),
            {"name": "air_velocity_m_s", "dist": "normal", "mean": 2.5, "std": 0.3}, _u("power_w", 80, 160)]

EXAMPLES = {
    "ishigami": {
        "title": "Ishigami function (exact answer known)",
        "description": "The standard benchmark: 3 inputs, a strong interaction between x1 and x3, x3 with no main "
                       "effect. The estimated indices are compared with the exact ones.",
        "config": {"model": {"source": "builtin", "name": "ishigami"}, "parameters": builtin("ishigami").metadata["space"],
                   "n_sobol": 2048, "second_order": True, "r_morris": 30, "n_grid": 15, "n_background": 64,
                   "threshold": 0.01, "nominal": {"x1": 1.0, "x2": 1.0, "x3": 1.0}}},
    "heatsink": {
        "title": "Heat-sink design: 10 parameters, 2 responses",
        "description": "Zoo model heatsink_forced_mlp around a 100 x 100 mm design: which of geometry, material, "
                       "airflow, ambient and power control the thermal resistance and the pressure drop, and which can "
                       "be frozen. Derivatives by automatic differentiation.",
        "config": {"model": {"source": "zoo", "name": "heatsink_forced_mlp"}, "parameters": HS_SPACE,
                   "fixed": {"source_width_mm": 40.0, "source_depth_mm": 40.0}, "outputs": ["r_base_k_w",
                                                                                          "pressure_drop_pa"],
                   "n_sobol": 1024, "second_order": True, "r_morris": 20, "n_grid": 15, "n_background": 48,
                   "threshold": 0.01}},
    "dataset": {
        "title": "From data only: 3,900 heat-sink simulations",
        "description": "No model: first-order indices straight from the heatsink_forced_convection training data "
                       "(12 inputs). Shows what given-data indices can and cannot say when inputs are correlated.",
        "config": {"mode": "given_data", "dataset": "heatsink_forced_convection", "split": "train",
                   "outputs": ["r_base_k_w", "pressure_drop_pa"]}},
}

VALIDATION = [
    "Sobol first-order, total and second-order estimates match the exact Ishigami indices (Sobol & Levitan 1999) "
    "within 0.02, with the exact values inside the bootstrap intervals (tests/test_sensitivity.py)",
    "Sobol' G function (a = 0, 1, 4.5, 9, 99, 99): first-order and total indices within 0.02 of the exact values; "
    "x5 and x6 recommended for freezing and the re-run check confirms it",
    "Morris on a linear model returns exactly |coefficients| with zero spread; ranks the G-function inputs correctly",
    "Local derivatives: automatic differentiation and central differences agree with the analytic Ishigami gradient "
    "(1e-5) and with each other on the heat-sink zoo model",
    "Given-data first-order indices of a random Ishigami sample within 0.02 of the exact values",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the indices mean",
            "band": "Indices describe the model over the ranges you give, not the physics everywhere.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can mislead", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Ranges and distributions", "effect": "check",
                 "detail": "Sobol indices are shares of output variance for the input distributions given; widen a "
                           "range and that input matters more.",
                 "today": "Use the real manufacturing / operating spread of each parameter.", "planned": ""},
                {"topic": "Independent inputs", "effect": "optimistic",
                 "detail": "The designed studies sample inputs independently. Given-data indices of correlated inputs "
                           "can add up to more than 1 and credit one input with another's effect.",
                 "today": "Read the input-correlation warning; prefer a designed study.",
                 "planned": "Shapley effects for correlated inputs."},
                {"topic": "Freeze recommendation", "effect": "conservative",
                 "detail": "Only when the upper bound of the total index is below the threshold for every output, then "
                           "checked by re-running with the parameter fixed.",
                 "today": "Inconclusive means more samples, not freeze.", "planned": ""},
                {"topic": "Sample size", "effect": "optimistic",
                 "detail": "At N = 256-512 the bootstrap intervals can be narrower than the real error (Ishigami at "
                           "N = 512: total index of x1 0.45, interval 0.37-0.55, exact 0.56).",
                 "today": "Use N >= 1024 for decisions; the cost is shown before running.", "planned": ""},
                {"topic": "Surrogate models", "effect": "check",
                 "detail": "The indices are those of the model; a surrogate can carry spurious small dependencies "
                           "(e.g. on power for a resistance) or miss real ones outside its data.",
                 "today": "Check the model's own validation (Uncertainty Lab) first.", "planned": ""},
            ]}


# ---------------------------------------------------------------------------------------------- helpers
def _model(spec: dict, files: dict) -> StudyModel:
    src = spec.get("source", "builtin")
    if src == "builtin":
        if spec.get("name") not in BUILTIN:
            raise HTTPException(422, f"built-in model must be one of {', '.join(BUILTIN)}")
        return builtin(spec["name"])
    if src == "zoo":
        if spec.get("name") not in ZOO:
            raise HTTPException(422, f"zoo model must be one of {', '.join(ZOO)}")
        return StudyModel.from_zoo(spec["name"], outputs=ZOO[spec["name"]])
    if src == "onnx":
        if "model" not in files:
            raise HTTPException(422, "Upload the ONNX file (field 'model').")
        return StudyModel.from_onnx(files["model"][1], spec.get("input_names") or None,
                                    spec.get("output_names") or None, name=files["model"][0])
    raise HTTPException(422, "model.source must be 'builtin', 'zoo' or 'onnx'.")


def _check_size(files: dict):
    if sum(len(b) for _, b in files.values()) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Uploads larger than {MAX_MB:g} MB.")


def _cost(cfg: dict, d: int) -> dict:
    return study_cost(d, int(cfg.get("n_sobol", 1024)), bool(cfg.get("second_order", False)),
                      int(cfg.get("r_morris", 20)), int(cfg.get("n_grid", 15)), int(cfg.get("n_background", 64)),
                      morris_on=cfg.get("morris", True), effects_on=cfg.get("effects", True))


def _space(cfg):
    params = cfg.get("parameters") or []
    if not params:
        raise HTTPException(422, "Give at least one varied parameter (name, dist and its parameters).")
    if len(params) > MAX_PARAMS:
        raise HTTPException(413, f"{len(params)} parameters; the limit is {MAX_PARAMS}.")
    try:
        return as_space(params)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def run_model_study(cfg: dict, files: dict) -> dict:
    t0 = time.time()
    _check_size(files)
    space = _space(cfg)
    cost = _cost(cfg, len(space))
    if cost["total"] > MAX_EVALS:
        raise HTTPException(413, f"{cost['total']:,} model evaluations; the limit is {MAX_EVALS:,} per study "
                                 "(lower the Sobol sample size, the effect grid or the number of parameters).")
    try:
        with _HEAVY:
            model = _model(cfg.get("model") or {}, files)
            exact = model.metadata.get("exact") if model.kind == "function" else None
            out = sensitivity_study(model, space, cfg.get("outputs") or None, n_sobol=int(cfg.get("n_sobol", 1024)),
                                    second_order=bool(cfg.get("second_order", False)),
                                    r_morris=int(cfg.get("r_morris", 20)), n_grid=int(cfg.get("n_grid", 15)),
                                    n_background=int(cfg.get("n_background", 64)),
                                    threshold=float(cfg.get("threshold", 0.01)), nominal=cfg.get("nominal") or None,
                                    fixed=cfg.get("fixed") or None, morris_on=cfg.get("morris", True),
                                    effects_on=cfg.get("effects", True), exact=exact, seed=int(cfg.get("seed", 0)))
    except HTTPException:
        raise
    except (ValueError, KeyError, ImportError) as e:
        raise HTTPException(422, str(e)) from e
    out["mode"] = "model"
    out["cost"] = cost
    out["usage"] = {"parameters": len(space), "model_evaluations": out["n_evals"]["total"],
                    "seconds": round(time.time() - t0, 2)}
    out["config"] = cfg
    out["scope"] = scope()
    out["study_result"] = _result("sensitivity_study", model.kind, model.name, out)
    return to_jsonable(out)


def _result(kind, source, name, out) -> dict:
    s = out.get("sobol") or {}
    return {"kind": kind, "app": "sensitivity", "app_version": VERSION, "model": {"source": source, "name": name},
            "outputs": out.get("outputs"), "metrics": {"ranking": s.get("ranking"),
                                                       "freeze": [r["name"] for r in out.get("freeze", {}).get("rows", [])
                                                                  if r["decision"] == "freeze"]}}


def _design_spec(cfg: dict) -> dict:
    space = _space(cfg)
    n = int(cfg.get("n_sobol", 256))
    so = bool(cfg.get("second_order", False))
    seed = int(cfg.get("seed", 0))
    return {"parameters": [p.to_dict() for p in space], "n_sobol": n, "second_order": so, "seed": seed,
            "outputs": cfg.get("outputs") or [], "threshold": float(cfg.get("threshold", 0.01))}


def _table(item):
    df, _ = load_table(item[1], item[0])
    return df


# ---------------------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "examples": [{"name": k, "title": v["title"], "description": v["description"]}
                                             for k, v in EXAMPLES.items()],
            "zoo": [{"name": k, "outputs": v} for k, v in ZOO.items()], "builtin": list(BUILTIN),
            "limits": {"max_mb": MAX_MB, "max_evals": MAX_EVALS, "max_params": MAX_PARAMS}, "scope": scope()}


@app.get("/api/model/{source}/{name}")
def model_info(source: str, name: str):
    """Inputs, outputs, units, validity range and a default parameter space (uniform on the validity range or a
    +-10 % band around the data median) for a built-in or zoo model."""
    m = _model({"source": source, "name": name}, {})
    info = m.info()
    if m.kind == "function":
        info["space"] = m.metadata["space"]
        info["fixed"] = {}
    else:
        X, _ = m.reference("test")
        med = np.median(X, axis=0)
        space, fixed = [], {}
        for i, n in enumerate(m.input_names):
            if n in m.bounds:
                lo, hi = m.bounds[n]
                c, w = med[i], 0.1 * abs(med[i]) if med[i] else 0.1 * (hi - lo)
                space.append(_u(n, round(max(lo, c - w), 6), round(min(hi, c + w), 6)))
            else:
                fixed[n] = float(np.round(med[i], 6))
        info["space"], info["fixed"] = space, fixed
    return to_jsonable(info)


@app.post("/api/cost")
def api_cost(config: dict):
    """Model evaluations a study would take, before running it."""
    return _cost(config, len(_space(config)))


# FastAPI keeps per-parameter state on these objects: one object per parameter, never shared
_CONFIG = Form(..., description="study configuration (JSON, see /api/example/heatsink/config)")
_GD_CONFIG = Form(..., description="{inputs, outputs} or {dataset, split, outputs} (JSON)")
_SPEC = Form(..., description="the JSON on the first line of the design CSV")
_MODEL_FILE, _DATA_FILE, _RUNS_FILE = File(None), File(None), File(None)


@app.post("/api/study")
async def api_study(config: str = _CONFIG, model: UploadFile | None = _MODEL_FILE):
    """Sensitivity study of a model. ``config``: {model: {source: builtin|zoo|onnx, name, input_names, output_names},
    parameters: [{name, dist, ...}], fixed: {input: value}, outputs, n_sobol, second_order, r_morris, n_grid,
    n_background, threshold, nominal, morris, effects}."""
    cfg = _json(config)
    files = {"model": (model.filename, await model.read())} if model is not None and model.filename else {}
    return run_model_study(cfg, files)


@app.post("/api/design")
def api_design(config: dict):
    """Saltelli design to run elsewhere (OpenFOAM, CalculiX, a campaign runner): CSV with run_id and one column per
    parameter. Keep the returned spec (header lines starting with #) to upload the results to /api/runs."""
    spec = _design_spec(config)
    X, layout = saltelli_design(spec["parameters"], spec["n_sobol"], spec["second_order"], spec["seed"])
    names = layout["names"]
    buf = io.StringIO()
    buf.write("# pinneapple.sensitivity.design/1 " + json.dumps(spec) + "\n")
    buf.write("run_id," + ",".join(names) + "\n")
    for i, row in enumerate(X):
        buf.write(f"{i}," + ",".join(f"{v:.10g}" for v in row) + "\n")
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="sobol_design.csv"',
                             "X-Runs": str(X.shape[0])})


@app.post("/api/runs")
async def api_runs(spec: str = _SPEC, runs: UploadFile = _RUNS_FILE):
    """Sobol indices from the runs of a design: ``spec`` is the JSON on the design's first line, ``runs`` a table with
    ``run_id`` and one column per output (extra columns are ignored)."""
    if runs is None or not runs.filename:
        raise HTTPException(422, "Upload the results table (field 'runs').")
    t0 = time.time()
    cfg = _json(spec)
    files = {"runs": (runs.filename, await runs.read())}
    _check_size(files)
    df = _table(files["runs"])
    if "run_id" not in df.columns:
        raise HTTPException(422, "The results need a run_id column (the row number of the design).")
    params = [p["name"] for p in cfg.get("parameters", [])]
    outs = cfg.get("outputs") or [c for c in df.columns if c != "run_id" and c not in params]
    if not outs:
        raise HTTPException(422, "No output columns in the results.")
    try:
        with _HEAVY:
            s = sobol_from_runs(cfg["parameters"], int(cfg["n_sobol"]), df["run_id"].to_numpy(),
                                columns(df, outs), second_order=bool(cfg.get("second_order", False)),
                                seed=int(cfg.get("seed", 0)), outputs=outs)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    out = {"mode": "runs", "outputs": outs, "parameters": cfg["parameters"], "sobol": s.to_dict(),
           "threshold": float(cfg.get("threshold", 0.01))}
    out["sobol"]["tables"] = {o: s.table(k) for k, o in enumerate(outs)}
    out["sobol"]["ranking"] = {o: s.ranking(k) for k, o in enumerate(outs)}
    out["freeze"] = {"threshold": out["threshold"], "rows": freeze_recommendation(s, out["threshold"]),
                     "note": "No direct re-run check: the model is not available here. Re-run your solver with the "
                             "recommended parameters frozen to confirm."}
    out["n_evals"] = {"sobol": s.n_evals, "total": s.n_evals}
    out["usage"] = {"parameters": len(params), "model_evaluations": s.n_evals, "seconds": round(time.time() - t0, 2)}
    out["scope"] = scope()
    out["study_result"] = _result("sensitivity_study", "external runs", runs.filename, out)
    return to_jsonable(out)


@app.post("/api/given-data")
async def api_given_data(config: str = _GD_CONFIG, data: UploadFile | None = _DATA_FILE):
    """First-order indices from a dataset: ``config`` {inputs: [...], outputs: [...]} (or {dataset, split} for a
    pp.datasets table), ``data`` a CSV / Parquet / JSON table."""
    cfg = _json(config)
    files = {"data": (data.filename, await data.read())} if data is not None and data.filename else {}
    return run_given_data(cfg, files)


def run_given_data(cfg: dict, files: dict) -> dict:
    t0 = time.time()
    _check_size(files)
    if "data" in files:
        df = _table(files["data"])
        src = files["data"][0]
    elif cfg.get("dataset"):
        from pinneapple_hub.zoo import datasets
        try:
            df = datasets.load(cfg["dataset"], split=cfg.get("split", "train")).to_pandas()
        except Exception as e:  # noqa: BLE001 - unknown dataset or split: report it
            raise HTTPException(422, str(e)) from e
        src = f"pp.datasets {cfg['dataset']} ({cfg.get('split', 'train')})"
    else:
        raise HTTPException(422, "Upload a data table (field 'data') or name a pp.datasets dataset.")
    outs = cfg.get("outputs") or []
    if not outs:
        raise HTTPException(422, "Name the output columns (outputs).")
    ins = cfg.get("inputs") or [c for c in df.columns if c not in outs and np.issubdtype(df[c].dtype, np.number)
                                and df[c].nunique() > 1 and c not in ("run_id",)]
    if cfg.get("dataset") == "heatsink_forced_convection" and not cfg.get("inputs"):
        ins = ["base_width_mm", "base_depth_mm", "base_thickness_mm", "n_fins", "fin_thickness_mm", "fin_height_mm",
               "k_w_mk", "source_width_mm", "source_depth_mm", "t_ambient_c", "air_velocity_m_s", "power_w"]
    try:
        X = columns(df, ins)
        res = {}
        with _HEAVY:
            for o in outs:
                y = columns(df, [o])[:, 0]
                ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                r = first_order_given_data(X[ok], y[ok], names=ins, n_boot=int(cfg.get("n_boot", 100)),
                                           seed=int(cfg.get("seed", 0)))
                r["sum_S1"] = float(np.sum(r["S1"]))
                r["dropped_rows"] = int((~ok).sum())
                res[o] = r
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    first = res[outs[0]]
    out = {"mode": "given_data", "source": src, "inputs": ins, "outputs": outs, "given_data": res,
           "n_rows": int(len(df)), "max_input_correlation": first["max_input_correlation"],
           "usage": {"parameters": len(ins), "rows": int(len(df)), "model_evaluations": 0,
                     "seconds": round(time.time() - t0, 2)},
           "config": cfg, "scope": scope()}
    out["study_result"] = {"kind": "sensitivity_given_data", "app": "sensitivity", "app_version": VERSION,
                           "source": src, "outputs": outs,
                           "metrics": {o: dict(zip(ins, res[o]["S1"], strict=True)) for o in outs}}
    return to_jsonable(out)


def _json(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError as e:
        raise HTTPException(422, f"config is not valid JSON: {e}") from e


@app.get("/api/example/{name}")
def api_example(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    cfg = json.loads(json.dumps(EXAMPLES[name]["config"]))
    out = run_given_data(cfg, {}) if cfg.get("mode") == "given_data" else run_model_study(cfg, {})
    out["example"] = {"name": name, "title": EXAMPLES[name]["title"], "description": EXAMPLES[name]["description"]}
    return out


@app.get("/api/example/{name}/config")
def api_example_config(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "No such example.")
    return EXAMPLES[name]["config"]


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
