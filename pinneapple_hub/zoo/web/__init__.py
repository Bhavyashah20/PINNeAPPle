"""Zoo models for the browser: ONNX graphs with their query checks, model cards and the JS runtime.

    python -m pinneapple_hub.zoo.web data/web/zoo        # what the Playground (apps/playground) serves

:func:`export_web` writes, per model, ``<name>.onnx`` (predictions for ``x`` and the query checks for ``query``,
:func:`pinneapple_hub.zoo.export.export_onnx` with ``checks=True``) and ``<name>.json`` (the model card, the
declarative check spec of :mod:`pinneapple_hub.zoo.checks`, the graph's layout and reference data to plot next to
the predictions). :func:`export_bundle` adds the CST airfoil geometry graphs
(:mod:`pinneapple_design.aero.geometry_onnx`) and ``fixtures.json`` (Python answers for fixed inputs, used by the
JS tests and the Playground's browser test).

The JS runtime that reads these files ships next to this module: ``onnx.js`` (a dependency-free ONNX interpreter)
and ``zoo.js`` (``loadModel``, ``loadAirfoilGeometry``, ``evaluate``: the JS twin of
:func:`pinneapple_hub.zoo.checks.evaluate`). :func:`runtime_dir` returns its folder.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper")
GEOMETRY_PRESETS = {"NACA 2412": "2412", "NACA 4412": "4412", "NACA 0012": "0012", "NACA 4415": "4415",
                    "NACA 0006": "0006", "NACA 0025": "0025", "NACA 8412": "8412"}
CARD_KEYS = ("name", "description", "task", "version", "inputs", "outputs", "validity_range", "known_limitations",
             "architecture", "equations", "training_config", "data_lineage", "validation_metrics", "reference_source",
             "citation", "license", "trust_report")


def runtime_dir() -> Path:
    """Folder of the JS runtime (``onnx.js``, ``zoo.js``)."""
    return Path(__file__).resolve().parent


def _round(o, nd=8):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}") if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, np.generic):
        return _round(o.item(), nd)
    return o


def reference_data(zm) -> dict[str, Any]:
    """Reference points to plot next to the predictions: for the airfoil, the held-out OpenFOAM runs and the XFOIL
    polar of each NACA preset; for the rover, the held-out Wong-Reece states."""
    from .. import datasets
    if zm.adapter == "airfoil_mlp":
        from pinneapple_design.aero.geometry import REFERENCE
        refs = []
        for split, label in (("naca_reference", "OpenFOAM RANS, not trained on"),
                             ("xfoil_naca2412", "XFOIL 6.99, independent solver")):
            d = datasets.load(zm.card.data_lineage["dataset"], split=split, offline=True)
            S = np.stack([np.asarray(d[k], float) for k in zm.input_names[:6]], 1)
            for preset, shape in REFERENCE.items():
                hit = np.all(np.abs(S - shape) < 2e-4, 1)
                if hit.any():
                    refs.append({"preset": preset, "source": label, "split": split,
                                 **{k: np.asarray(d[c], float)[hit].tolist()
                                    for k, c in (("alpha", "alpha_deg"), ("Cl", "Cl"), ("Cd", "Cd"), ("Cm", "Cm"))}})
        return {"polars": refs}
    if zm.adapter == "rover_contact":
        X, Y = zm.reference("test", offline=True)
        return {"heldout": {"slip": X[:, 0].round(4).tolist(), "sinkage_m": X[:, 1].round(5).tolist(),
                            "Fx_N": Y[:, 0].round(2).tolist(), "Fz_N": Y[:, 1].round(2).tolist(),
                            "My_Nm": Y[:, 2].round(2).tolist()}, "physics": zm.config["physics"]}
    return {}


def export_web(name: str, directory: str | Path, *, offline: bool | None = None) -> dict[str, Any]:
    """Write ``<name>.onnx`` and ``<name>.json`` for the browser into ``directory``; returns the JSON content."""
    from .. import load
    from ..checks import check_spec
    from ..export import export_onnx
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zm = load(name, offline=offline)
        info = export_onnx(zm, d / f"{name}.onnx", checks=True)
    card = zm.card.to_dict()
    meta = {"card": {k: card.get(k) for k in CARD_KEYS},
            "onnx": {"file": f"{name}.onnx", "bytes": info["bytes"], "sha256": info["sha256"], "opset": info["opset"],
                     "inputs": info["inputs"], "input_columns": info["input_columns"], "outputs": info["outputs"],
                     "parity": info["parity"], "checks_parity": info["checks_parity"]},
            "checks": check_spec(zm), "reference": reference_data(zm),
            "hub": f"https://huggingface.co/PINNeAPPle-Labs/{name}",
            "code": {"adapter": "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/pinneapple_hub/zoo/_models.py",
                     "checks": "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/pinneapple_hub/zoo/checks.py",
                     "export": "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/pinneapple_hub/zoo/export.py"}}
    meta = _round(meta)
    (d / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return meta


def fixtures(*, offline: bool | None = None) -> dict[str, Any]:
    """Python answers for fixed inputs: predictions and query-check evaluations of each model (in range and out of
    range), and the geometry of a few CST shapes. The JS runtime tests and the Playground's browser test compare
    against these."""
    from pinneapple_design.aero import geometry as g
    from pinneapple_design.aero import geometry_onnx as go

    from .. import load
    from ..checks import check_spec, check_values, evaluate
    cases = {
        "airfoil_cst_rans_mlp": [[*g.naca4("2412"), 4.0], [*g.naca4("0012"), 0.0], [*g.naca4("4415"), 8.0],
                                 [*g.naca4("2412"), 16.0], [0.30, 0.15, 0.15, 0.02, 0.05, 0.03, 4.0]],
        "heatsink_forced_mlp": [[80, 100, 6, 14, 1.5, 35, 200, 40, 40, 25, 2.5, 60],
                                [80, 100, 6, 14, 4.5, 35, 200, 40, 40, 25, 2.5, 60],
                                [150, 120, 8, 30, 1.0, 50, 390, 60, 60, 35, 5.5, 300]],
        "rover_contact_grc1_viper": [[0.2, 0.02], [-0.1, 0.035], [0.6, 0.05], [0.2, 0.075]],
    }
    out: dict[str, Any] = {"models": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, rows in cases.items():
            zm = load(name, offline=offline)
            X = np.asarray(rows, float)
            pred = zm._forward(X)
            v, a = check_values(zm, X)
            spec = check_spec(zm)
            out["models"][name] = {
                "inputs": X.tolist(), "in_range": zm.in_range(X).tolist(),
                "predict": {k: np.asarray(p, float).tolist() for k, p in pred.items()},
                "check_values": v.tolist(), "check_aux": a.tolist(),
                "evaluation": [{"statuses": [c["status"] for c in e["checks"]],
                                "details": [c["detail_text"] for c in e["checks"]], "verdict": e["verdict"]["status"]}
                               for e in (evaluate(spec, v[i], a[i]) for i in range(len(X)))]}
    rng = np.random.default_rng(11)
    S = g.BOUNDS[:, 0] + rng.random((6, 6)) * (g.BOUNDS[:, 1] - g.BOUNDS[:, 0])
    S = np.r_[S, [[0.3, 0.02, 0.3, 0, 0, 0], [0.12, 0.3, -0.3, 0, 0, 0]]]
    ref = go.reference(S)
    out["geometry"] = {"shapes": S.tolist(), **{k: v.tolist() for k, v in ref.items()},
                       "valid": [list(g.valid(s)) for s in S]}
    return _round(out, 12)


def export_bundle(directory: str | Path, models=MODELS, *, offline: bool | None = None) -> dict[str, Any]:
    """Everything the Playground loads: each model (:func:`export_web`), the CST geometry graphs and fixtures."""
    from pinneapple_design.aero import geometry_onnx
    d = Path(directory)
    metas = {n: export_web(n, d, offline=offline) for n in models}
    geo = geometry_onnx.export_onnx(d, presets=GEOMETRY_PRESETS)
    (d / "fixtures.json").write_text(json.dumps(fixtures(offline=offline), indent=1) + "\n")
    index = {"models": list(models), "geometry": "cst_geometry.json", "fixtures": "fixtures.json",
             "runtime": ["onnx.js", "zoo.js"], "generator": "python -m pinneapple_hub.zoo.web"}
    (d / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return {"models": metas, "geometry": geo, "index": index}


__all__ = ["export_web", "export_bundle", "fixtures", "reference_data", "runtime_dir", "MODELS"]
