"""Simulation metadata extraction (PINNeAPPle): solver, mesh, time step, boundary conditions, parameters
and convergence from the files a simulation leaves behind.

>>> from pinneapple_data.simulation_metadata import extract, read_upload
>>> record = extract(read_upload([("case.zip", open("case.zip", "rb").read())]))

Supported inputs
----------------
* **OpenFOAM** case folders (zip, or loose files): ``system/controlDict``, ``fvSchemes``, ``fvSolution``,
  ``blockMeshDict``, ``constant/*Properties``, ``constant/polyMesh/{owner,boundary,points}``, ``0/*`` fields,
  solver / blockMesh / checkMesh logs, ``postProcessing/**.dat`` monitors.
* **CalculiX / Abaqus** input decks (``.inp``, with ``*INCLUDE`` files) and CalculiX run files
  (``.sta``, ``.cvg``, ``.dat``, solver output).
* **Residual histories** from any solver as a table (Fluent / STAR-CCM+ / SU2 ``history.csv`` exports).

The record (schema ``pinneapple.simmeta/1``) is JSON-ready. Every value says where it came from, and nothing is
inferred that the files do not contain, except the clearly labelled ``derived`` quantities and the convergence
verdict, whose reasons are listed.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Sequence
from typing import Any

import numpy as np

from pinneapple_core.jsonable import to_jsonable

from . import calculix, convergence, openfoam

SCHEMA = "pinneapple.simmeta/1"
MAX_FILES = 5000
MAX_UNZIPPED = 400 * 1024 * 1024
SKIP_EXT = (".vtk", ".vtu", ".vtp", ".frd", ".rmed", ".odb", ".h5", ".hdf5", ".png", ".jpg", ".gz", ".foam", ".stl",
            ".obj", ".msh", ".unv", ".cgns", ".pdf", ".pyc", ".so", ".o")
FOAM_SYSTEM = {"controlDict", "fvSchemes", "fvSolution", "blockMeshDict", "snappyHexMeshDict", "decomposeParDict"}
FOAM_CONSTANT = {"transportProperties", "physicalProperties", "thermophysicalProperties", "turbulenceProperties",
                 "momentumTransport", "g", "thermophysicalTransport", "radiationProperties"}


def read_upload(files: Sequence[tuple[str, bytes]]) -> dict[str, bytes]:
    """Turn uploaded files (zips are expanded, with size limits) into {relative path: bytes}."""
    fs: dict[str, bytes] = {}
    total = 0
    for name, data in files:
        if name.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = [i for i in z.infolist() if not i.is_dir()]
                if len(infos) > MAX_FILES:
                    raise ValueError(f"The archive has {len(infos)} files (limit {MAX_FILES}).")
                for i in infos:
                    p = i.filename.replace("\\", "/")
                    if "__MACOSX" in p or p.split("/")[-1].startswith("._") or p.lower().endswith(SKIP_EXT):
                        continue
                    if re.search(r"(^|/)(processor\d+|VTK|\.git)(/|$)", p):
                        continue
                    if re.search(r"(^|/)\d+(\.\d+)?/[^/]+$", p) and not re.search(r"(^|/)0(\.orig)?/[^/]+$", p):
                        fs[p] = b""                      # result time directories: note presence, skip content
                        continue
                    total += i.file_size
                    if total > MAX_UNZIPPED:
                        raise ValueError("The archive expands to more than 400 MB of text; upload the case without "
                                         "result time directories.")
                    fs[p] = z.read(i)
        else:
            fs[name.replace("\\", "/").lstrip("/")] = data
    # strip a common leading folder
    keys = [k for k in fs]
    if keys and all("/" in k for k in keys):
        first = {k.split("/", 1)[0] for k in keys}
        if len(first) == 1:
            pre = next(iter(first)) + "/"
            fs = {k[len(pre):]: v for k, v in fs.items()}
    return fs


def _virtual_foam(fs: dict[str, bytes]) -> dict[str, bytes]:
    """Loose OpenFOAM files (controlDict, log.simpleFoam, ...) placed into a case layout."""
    out = dict(fs)
    for k, v in fs.items():
        b = k.rsplit("/", 1)[-1]
        if b in FOAM_SYSTEM and f"system/{b}" not in out:
            out[f"system/{b}"] = v
        elif b in FOAM_CONSTANT and f"constant/{b}" not in out:
            out[f"constant/{b}"] = v
    if "system/controlDict" not in out and any(_is_foam_log(v) for v in fs.values()):
        out["system/controlDict"] = b"FoamFile { class dictionary; object controlDict; }\n"
    return out


def _is_foam_log(b: bytes) -> bool:
    head = b[:6000].decode("utf-8", errors="ignore")
    return ("Exec   :" in head or "Exec :" in head) or "Solving for" in b[:200000].decode("utf-8", errors="ignore")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# ── generic residual tables ─────────────────────────────────────────────────
_ITER = re.compile(r"^(iter(ation)?s?|inner_iter|outer_iter|time_iter|time|step|it)$", re.I)


def parse_residual_table(text: str) -> dict[str, Any] | None:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hdr_i = None
    for i, ln in enumerate(lines[:60]):
        toks = [t.strip().strip('"').strip() for t in re.split(r"[,\t]|\s{2,}|\s(?=\")", ln.lstrip("#").strip()) if t.strip()]
        if len(toks) >= 2 and any(_ITER.match(t.replace(" ", "_")) for t in toks) and \
                not all(re.fullmatch(r"[-+.\deE]+", t) for t in toks):
            hdr_i, header = i, toks
            break
    if hdr_i is None:
        return None
    rows = []
    for ln in lines[hdr_i + 1:]:
        p = [t.strip().strip('"') for t in re.split(r"[,\s]+", ln.strip()) if t.strip()]
        try:
            vals = [float(x) for x in p]
        except ValueError:
            continue
        if len(vals) == len(header):
            rows.append(vals)
    if len(rows) < 3:
        return None
    a = np.array(rows)
    it_col = next(i for i, h in enumerate(header) if _ITER.match(h.replace(" ", "_")))
    for pref in ("inner_iter", "iteration", "iter"):
        for i, h in enumerate(header):
            if h.lower().replace(" ", "_") == pref:
                it_col = i
                break
    skip = {i for i, h in enumerate(header) if _ITER.match(h.replace(" ", "_")) or re.search(r"time|cfl|wall|cpu|dt", h, re.I)}
    res = {h: a[:, i].tolist() for i, h in enumerate(header) if i not in skip}
    monitors = {h: v for h, v in res.items() if not re.search(r"res|rms|continuity|velocity|energy|^[kpU]$|epsilon|omega|"
                                                            r"nut|\[|x-|y-|z-", h, re.I)}
    residuals = {h: v for h, v in res.items() if h not in monitors}
    if not residuals:
        return None
    vals = np.concatenate([np.array(v) for v in residuals.values()])
    is_log10 = bool(np.nanmax(vals) <= 2 and np.nanmin(vals) < -0.5)
    return {"iterations": a[:, it_col].tolist(), "residuals": residuals, "monitors": monitors, "log10": is_log10,
            "header": header}


# ── main entry ──────────────────────────────────────────────────────────────
def extract(fs: dict[str, bytes]) -> dict[str, Any]:
    if not fs:
        raise ValueError("No files.")
    files = [{"path": k, "bytes": len(v), "sha256": _sha(v) if v else None} for k, v in sorted(fs.items())]
    root = openfoam.find_root(fs)
    loose_foam = root is None and (any(k.rsplit("/", 1)[-1] in FOAM_SYSTEM | FOAM_CONSTANT for k in fs)
                                   or any(_is_foam_log(v) for k, v in fs.items() if v and not k.lower().endswith(".inp")))
    inps = [k for k in fs if k.lower().endswith(".inp") and fs[k]]
    if root is not None or loose_foam:
        if root is None:
            fs = _virtual_foam(fs)
            root = ""
        rec = _openfoam_record(fs, root)
    elif inps:
        rec = _calculix_record(fs, inps)
    else:
        rec = _table_record(fs)
    used = set(rec.pop("_used", []))
    for f in files:
        f["used"] = f["path"] in used or (root is not None and root + f["path"] in used)
    rec["files"] = files
    rec["schema"] = SCHEMA
    return _clean(rec)


def _openfoam_record(fs: dict[str, bytes], root: str) -> dict[str, Any]:
    case = openfoam.extract_case(fs, root)
    s = openfoam.summarise(case)
    log = s.pop("_log")
    warnings: list[str] = []
    if log is None:
        conv = {"status": "unknown", "reasons": ["No solver log in the upload: add log.<solver> to assess convergence."],
                "recommendations": ["Run the solver with `| tee log.simpleFoam` (or foamJob) and include the log."]}
    elif s.pop("_steady"):
        conv = convergence.assess_steady(log, s["_residual_control"])
    else:
        conv = convergence.assess_transient(log, s["_linear_tolerances"], s["time"].get("max_courant"),
                                            s["analysis"].get("algorithm"), s["time"].get("reached_end"))
    s.pop("_steady", None)
    s.pop("_residual_control", None)
    s.pop("_linear_tolerances", None)
    if log:
        x = log["time_steps"] if not s["analysis"]["type"] == "steady" else list(range(1, len(log["time_steps"]) + 1))
        conv["history"] = convergence.downsample(x, log["residuals"])
        conv["history"]["x_label"] = "iteration" if s["analysis"]["type"] == "steady" else "time [s]"
        if log.get("courant_max"):
            conv["courant_history"] = convergence.downsample(x[: len(log["courant_max"])],
                                                             {"max": log["courant_max"], "mean": log["courant_mean"]}, 800)
        conv["linear_solvers"] = log["linear_solvers"]
    mon = _monitors(fs, root)
    if mon:
        conv["monitors"] = mon
    if case["time_dirs"]:
        s["time"]["result_times_present"] = len(case["time_dirs"])
    if not s["mesh"].get("cells"):
        warnings.append("Mesh size unknown: include constant/polyMesh (owner, boundary) or the blockMesh/checkMesh log.")
    if not case["boundary_conditions"]:
        warnings.append("No initial/boundary condition files (0/ or 0.orig/) found.")
    if root == "" and "system/fvSolution" not in fs:
        warnings.append("Only some case files were uploaded; upload the whole case folder as a zip for full metadata.")
    return {"detected": {"format": "OpenFOAM case", "root": root or "(upload root)"}, **s,
            "convergence": {**conv, "label": convergence.LABEL[conv["status"]], "level": convergence.LEVEL[conv["status"]]},
            "warnings": warnings, "_used": case["files_used"]}


def _monitors(fs: dict[str, bytes], root: str) -> list[dict[str, Any]]:
    out = []
    for k, v in fs.items():
        if not k.startswith(f"{root}postProcessing/") or not v or not k.endswith(".dat"):
            continue
        cols, a = openfoam.parse_dat(v.decode("utf-8", errors="replace"))
        if not cols or cols[0] != "Time" or len(a) < 5:
            continue
        for j, c in enumerate(cols[1:], start=1):
            y = a[:, j]
            tail = y[-max(5, len(y) // 10):]
            ref = np.abs(np.mean(tail)) or 1.0
            out.append({"file": k[len(root):], "name": c, "final": float(y[-1]),
                        "tail_variation_pct": float(np.ptp(tail) / ref * 100),
                        "history": convergence.downsample(a[:, 0].tolist(), {c: y.tolist()}, 600)})
    return out[:20]


def _calculix_record(fs: dict[str, bytes], inps: list[str]) -> dict[str, Any]:
    texts = {k.rsplit("/", 1)[-1]: v.decode("utf-8", errors="replace") for k, v in fs.items() if v}
    included = set()
    for k in inps:
        for m in re.finditer(r"^\*INCLUDE\s*,\s*INPUT\s*=\s*\"?([^\"\n,]+)", texts[k.rsplit("/", 1)[-1]], re.I | re.M):
            included.add(m.group(1).strip().split("/")[-1])
    main = next((k for k in inps if k.rsplit("/", 1)[-1] not in included), inps[0])
    deck = calculix.parse_inp(texts[main.rsplit("/", 1)[-1]], include=texts)
    stem = main.rsplit("/", 1)[-1][:-4]
    used = [main] + [k for k in fs if k.rsplit("/", 1)[-1] in included]
    find = lambda ext: next((k for k in fs if k.lower().endswith(ext) and fs[k] and k.rsplit("/", 1)[-1][: -len(ext)] == stem), None) \
        or next((k for k in fs if k.lower().endswith(ext) and fs[k]), None)   # noqa: E731
    sta_p, cvg_p, dat_p = find(".sta"), find(".cvg"), find(".dat")
    log_p = next((k for k in fs if fs[k] and ("CalculiX" in fs[k][:4000].decode("utf-8", "ignore") or
                                             k.rsplit("/", 1)[-1].startswith("log"))
                  and not k.lower().endswith((".inp", ".sta", ".cvg", ".dat"))), None)
    sta = calculix.parse_sta(texts[sta_p.rsplit("/", 1)[-1]]) if sta_p else []
    cvg = calculix.parse_cvg(texts[cvg_p.rsplit("/", 1)[-1]]) if cvg_p else []
    dat = calculix.parse_dat(texts[dat_p.rsplit("/", 1)[-1]]) if dat_p else {}
    log = calculix.parse_ccx_log(texts[log_p.rsplit("/", 1)[-1]]) if log_p else None
    used += [p for p in (sta_p, cvg_p, dat_p, log_p) if p]
    abaqus = bool(re.search(r"\*(PART|INSTANCE|ASSEMBLY)\b", texts[main.rsplit("/", 1)[-1]], re.I)) or \
        "Abaqus" in (deck.get("heading") or "")
    conv = calculix.assess(sta, cvg, deck, log)
    if cvg:
        hist_x = list(range(1, len(cvg) + 1))
        floor = 1e-6                                   # CalculiX prints exactly 0 when the residual vanishes
        conv["history"] = convergence.downsample(hist_x, {"force residual %": [max(r["force_residual_pct"], floor) for r in cvg],
                                                          "displacement correction %": [max(r["displacement_correction_pct"], floor) for r in cvg]})
        conv["history"]["x_label"] = "Newton iteration (all increments)"
        conv["history"]["y_label"] = "% (0 shown as 1e-6)"
    steps = deck["steps"]
    analysis_type = ", ".join(sorted({s["procedure"] for s in steps if s.get("procedure")})) or "unknown"
    if any(s.get("nlgeom") for s in steps):
        analysis_type += ", geometrically nonlinear"
    if any(m.get("plastic_points") for m in deck["materials"]):
        analysis_type += ", plasticity"
    bcs = [{**b, "step": "model"} for b in deck["model_boundary_conditions"]] + \
          [{**b, "step": s["index"]} for s in steps for b in s["boundary_conditions"]]
    loads = [{**ld, "step": s["index"]} for s in steps for ld in s["loads"]]
    params = []
    for m in deck["materials"]:
        for k, unit in (("young_modulus", "(model units, e.g. MPa)"), ("poisson_ratio", "-"), ("density", "(model units)"),
                        ("conductivity", ""), ("specific_heat", ""), ("expansion", "1/K"), ("yield_stress", "")):
            if m.get(k) is not None:
                params.append({"name": f"{m['name']}: {k.replace('_', ' ')}", "value": m[k], "unit": unit, "source": main})
    for ld in loads:
        params.append({"name": f"step {ld['step']} {ld['type']} {ld['component']} on {ld['target']}",
                       "value": ld.get("total", ld.get("value")), "unit": "(model units)", "source": main})
    results = [{"name": f"max |u| on set {v['set']}", "value": v["max_magnitude"], "time": v["time"]} if "max_magnitude" in v
               else {"name": f"{k} range", "value": [v["min"], v["max"]], "time": v["time"]} for k, v in dat.items()]
    total_inc = len(sta)
    rec = {
        "detected": {"format": "Abaqus input deck" if abaqus else "CalculiX input deck", "root": main},
        "solver": {"name": "CalculiX" if (log or sta or not abaqus) else "Abaqus", "family": "FEA · finite element",
                   "application": "ccx" if (log or sta) else None, "version": (log or {}).get("version"),
                   "executed": {"wall_time_s": (log or {}).get("wall_time_s"), "n_procs": (log or {}).get("threads")}},
        "analysis": {"type": analysis_type, "physics": sorted({s["procedure"] for s in steps if s.get("procedure")}),
                     "steps": [{k: s.get(k) for k in ("index", "name", "procedure", "nlgeom", "time", "modes", "max_increments")}
                               for s in steps], "interactions": deck["interactions"]},
        "mesh": {"nodes": deck["nodes"], "elements": deck["elements"], "element_types": deck["element_types"],
                 "bounding_box": deck["bounding_box"], "node_sets": deck["node_sets"], "element_sets": deck["element_sets"],
                 "source": main},
        "time": {"steps": len(steps), "increments_run": total_inc,
                 "periods": [s["time"].get("period") for s in steps if s.get("time")],
                 "initial_increment": steps[0]["time"].get("initial_increment") if steps and steps[0].get("time") else None,
                 "unit": "step time"},
        "boundary_conditions": bcs, "loads": loads, "materials": deck["materials"], "sections": deck["sections"],
        "parameters": params, "results": results, "derived": [], "numerics": {"keywords": deck["other_keywords"]},
        "convergence": {**conv, "label": convergence.LABEL[conv["status"]], "level": convergence.LEVEL[conv["status"]]},
        "warnings": ([f"*INCLUDE file(s) not uploaded: {', '.join(i for i in deck['includes'] if i not in texts)}"]
                     if any(i not in texts for i in deck["includes"]) else [])
                    + ([] if sta or log else ["No .sta or solver output: only the input deck was analysed."]),
        "_used": used,
    }
    return rec


def _table_record(fs: dict[str, bytes]) -> dict[str, Any]:
    for k, v in fs.items():
        if not v or not k.lower().endswith((".csv", ".dat", ".out", ".txt", ".trn", ".log", ".tsv")):
            continue
        tab = parse_residual_table(v.decode("utf-8", errors="replace"))
        if tab:
            conv = convergence.assess_table(tab["iterations"], tab["residuals"], tab["log10"])
            series = {f: (10.0 ** np.array(r)).tolist() if tab["log10"] else r for f, r in tab["residuals"].items()}
            conv["history"] = convergence.downsample(tab["iterations"], series)
            conv["history"]["x_label"] = "iteration"
            if tab["monitors"]:
                conv["monitors"] = [{"file": k, "name": n, "final": vv[-1],
                                     "tail_variation_pct": float(np.ptp(vv[-max(5, len(vv) // 10):]) /
                                                                 (abs(np.mean(vv[-max(5, len(vv) // 10):])) or 1) * 100),
                                     "history": convergence.downsample(tab["iterations"], {n: vv}, 600)}
                                    for n, vv in tab["monitors"].items()]
            su2 = any("[" in h for h in tab["header"])
            return {"detected": {"format": "SU2 history" if su2 else "residual history table", "root": k},
                    "solver": {"name": "SU2" if su2 else "unknown (residual table)", "family": "CFD", "executed": {}},
                    "analysis": {"type": "iterative (from residual history)"}, "mesh": {}, "time": {"iterations": len(tab["iterations"])},
                    "boundary_conditions": [], "parameters": [], "derived": [], "numerics": {"columns": tab["header"],
                                                                                           "residuals_log10": tab["log10"]},
                    "convergence": {**conv, "label": convergence.LABEL[conv["status"]], "level": convergence.LEVEL[conv["status"]]},
                    "warnings": ["Only a residual history was found: solver, mesh and boundary conditions need the case "
                                 "files (OpenFOAM case, .inp deck)."], "_used": [k]}
    raise ValueError("No recognised simulation files. Upload an OpenFOAM case (zip of the case folder, or controlDict / "
                     "fvSolution / logs), a CalculiX/Abaqus .inp deck (+ .sta/.cvg/.dat), or a residual-history table.")


def _clean(o: Any) -> Any:
    return to_jsonable(o)


__all__ = ["extract", "read_upload", "parse_residual_table", "SCHEMA"]
