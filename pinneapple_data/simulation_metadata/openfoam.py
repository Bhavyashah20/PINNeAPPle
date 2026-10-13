"""OpenFOAM case + log extraction for the simulation metadata record."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .foam_dict import boundary, header, parse, scalar, vector

STEADY_SOLVERS = {"simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam", "potentialFoam", "porousSimpleFoam",
                  "SRFSimpleFoam", "laplacianFoam", "solidDisplacementFoam", "chtMultiRegionSimpleFoam",
                  "adjointShapeOptimizationFoam", "boundaryFoam", "scalarTransportFoam"}
SOLVER_PHYSICS = {
    "icoFoam": ["incompressible", "laminar", "transient"], "simpleFoam": ["incompressible", "steady", "turbulent/laminar"],
    "pisoFoam": ["incompressible", "transient"], "pimpleFoam": ["incompressible", "transient"],
    "rhoSimpleFoam": ["compressible", "steady"], "rhoPimpleFoam": ["compressible", "transient"],
    "buoyantSimpleFoam": ["compressible", "buoyant", "heat transfer", "steady"],
    "buoyantPimpleFoam": ["compressible", "buoyant", "heat transfer", "transient"],
    "interFoam": ["two-phase VOF", "transient"], "chtMultiRegionFoam": ["conjugate heat transfer", "transient"],
    "chtMultiRegionSimpleFoam": ["conjugate heat transfer", "steady"], "laplacianFoam": ["diffusion"],
    "potentialFoam": ["potential flow"], "sonicFoam": ["compressible", "transonic", "transient"],
    "rhoCentralFoam": ["compressible", "high-speed", "transient"], "reactingFoam": ["reacting", "transient"],
    "solidDisplacementFoam": ["solid mechanics"], "scalarTransportFoam": ["passive scalar transport"],
}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE_SOLVE = re.compile(rf"(\w+):\s+Solving for (\w+), Initial residual = ({_NUM}), Final residual = ({_NUM}), No Iterations (\d+)")
_RE_TIME = re.compile(rf"^Time = ({_NUM})\s*$")
_RE_CO = re.compile(rf"Courant Number mean: ({_NUM}) max: ({_NUM})")
_RE_DT = re.compile(rf"^deltaT = ({_NUM})")
_RE_CONT = re.compile(rf"time step continuity errors : sum local = ({_NUM}), global = ({_NUM}), cumulative = ({_NUM})")
_RE_EXEC = re.compile(rf"ExecutionTime = ({_NUM}) s\s+ClockTime = ({_NUM}) s")


def parse_log(text: str) -> dict[str, Any]:
    """Parse an OpenFOAM solver log (or blockMesh / checkMesh log)."""
    lines = text.splitlines()
    head: dict[str, str] = {}
    for ln in lines[:40]:
        m = re.match(r"^(Build|Exec|Date|Time|Host|PID|Case|nProcs|Arch|I/O)\s*:\s*(.*)$", ln)
        if m:
            head[m.group(1)] = m.group(2).strip()
    m = re.search(r"Version:\s*(\S+)", text[:3000])
    out: dict[str, Any] = {"header": head, "exec": head.get("Exec", "").split()[0] if head.get("Exec") else None,
                           "version": m.group(1) if m and m.group(1) != "|" else None}
    bm = re.search(r"OPENFOAM=(\d+)", head.get("Build", ""))
    if bm and not out["version"]:
        out["version"] = f"v{bm.group(1)}"
    elif head.get("Build") and not out["version"]:
        out["version"] = head["Build"].split()[0]

    steps: list[float] = []
    res_init: dict[str, list[float]] = {}
    res_final: dict[str, list[float]] = {}
    iters: dict[str, list[int]] = {}
    lin_solver: dict[str, str] = {}
    co_mean, co_max, dts, cont, exec_t = [], [], [], [], []
    seen_this_step: set = set()
    bounding = 0
    for ln in lines:
        mt = _RE_TIME.match(ln)
        if mt:
            steps.append(float(mt.group(1)))
            seen_this_step = set()
            for d in (res_init, res_final, iters):
                for k in d:
                    d[k].append(np.nan) if d is not iters else d[k].append(0)
            continue
        if not steps:
            continue
        ms = _RE_SOLVE.search(ln)
        if ms:
            solver, field, r0, r1, n = ms.group(1), ms.group(2), float(ms.group(3)), float(ms.group(4)), int(ms.group(5))
            lin_solver.setdefault(field, solver)
            if field not in res_init:
                res_init[field] = [np.nan] * len(steps)
                res_final[field] = [np.nan] * len(steps)
                iters[field] = [0] * len(steps)
            if field not in seen_this_step:                       # first solve of the step: OpenFOAM convention
                res_init[field][-1] = r0
                seen_this_step.add(field)
            res_final[field][-1] = r1
            iters[field][-1] += n
            continue
        mc = _RE_CO.search(ln)
        if mc:
            co_mean.append(float(mc.group(1)))
            co_max.append(float(mc.group(2)))
            continue
        md = _RE_DT.match(ln)
        if md:
            dts.append(float(md.group(1)))
            continue
        mo = _RE_CONT.search(ln)
        if mo:
            cont.append((float(mo.group(1)), float(mo.group(2)), float(mo.group(3))))
            continue
        me = _RE_EXEC.search(ln)
        if me:
            exec_t.append((float(me.group(1)), float(me.group(2))))
            continue
        if ln.startswith("bounding "):
            bounding += 1
    conv = re.search(r"(SIMPLE|PIMPLE|PISO)?\s*solution converged in (\d+) iterations", text)
    fatal = re.search(r"--> FOAM FATAL (IO )?ERROR[^\n]*\n(.*?)(?:\n\n|\n\s+From)", text, re.S)
    fpe = re.search(r"^#\d+\s+Foam::(?:error::printStack|sigFpe)|Floating point exception(?! trapping)", text, re.M)
    ended = bool(re.search(r"^End\s*$", text, re.M))
    out.update({
        "time_steps": steps, "residuals": res_init, "final_residuals": res_final, "linear_iterations": iters,
        "linear_solvers": lin_solver, "courant_mean": co_mean, "courant_max": co_max, "delta_t": dts,
        "continuity": cont, "execution_time_s": exec_t[-1][0] if exec_t else None,
        "clock_time_s": exec_t[-1][1] if exec_t else None, "converged_message": conv.group(0).strip() if conv else None,
        "converged_iterations": int(conv.group(2)) if conv else None,
        "fatal": (fatal.group(2).strip()[:300] if fatal else None), "floating_point_exception": bool(fpe),
        "ended": ended, "bounding_warnings": bounding,
    })
    # mesh utilities
    bmc = re.search(r"nCells:\s*(\d+)", text)
    if out["exec"] == "blockMesh" and bmc:
        out["mesh_cells"] = int(bmc.group(1))
    if out["exec"] == "checkMesh" or "Checking geometry" in text:
        q: dict[str, Any] = {}
        for k, pat in (("cells", r"cells:\s+(\d+)"), ("points", r"points:\s+(\d+)"), ("faces", r"faces:\s+(\d+)"),
                       ("max_non_orthogonality", rf"Mesh non-orthogonality Max: ({_NUM})"),
                       ("avg_non_orthogonality", rf"average: ({_NUM})"), ("max_skewness", rf"Max skewness = ({_NUM})"),
                       ("max_aspect_ratio", rf"Max aspect ratio = ({_NUM})")):
            mm = re.search(pat, text)
            if mm:
                q[k] = float(mm.group(1))
        failed = re.search(r"Failed (\d+) mesh checks", text)
        q["status"] = "Mesh OK" if "Mesh OK" in text else (f"Failed {failed.group(1)} mesh checks" if failed else None)
        out["mesh_quality"] = q
    return out


def _read(fs: dict[str, bytes], path: str) -> str | None:
    b = fs.get(path)
    return b.decode("utf-8", errors="replace") if b is not None else None


def _first(fs: dict[str, bytes], root: str, *rels: str) -> tuple[str | None, str | None]:
    for r in rels:
        p = f"{root}{r}"
        if p in fs:
            return p, _read(fs, p)
    return None, None


def find_root(fs: dict[str, bytes]) -> str | None:
    """Case root: the directory that holds system/controlDict (shortest such path)."""
    roots = [p[: -len("system/controlDict")] for p in fs if p.endswith("system/controlDict")]
    return min(roots, key=len) if roots else None


def _points_bbox(text: str) -> dict[str, list[float]] | None:
    h = header(text)
    if h.get("format", "ascii") != "ascii":
        return None
    body = text[text.find("(", text.find("}", text.find("FoamFile")) + 1):]
    arr = np.array(re.findall(rf"\(({_NUM})\s+({_NUM})\s+({_NUM})\)", body), dtype=float)
    if not len(arr):
        return None
    mn, mx = arr.min(0), arr.max(0)
    return {"min": mn.tolist(), "max": mx.tolist(), "size": (mx - mn).tolist()}


def _block_mesh(text: str) -> dict[str, Any]:
    d = parse(text)
    scale = scalar(d.get("convertToMeters", d.get("scale", "1"))) or 1.0
    verts = re.findall(rf"\(\s*({_NUM})\s+({_NUM})\s+({_NUM})\s*\)", str(d.get("vertices", "")))
    cells = 0
    for m in re.finditer(r"hex\s*\(([^)]*)\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)):
        cells += int(m.group(2)) * int(m.group(3)) * int(m.group(4))
    out: dict[str, Any] = {"cells_from_blocks": cells or None}
    if verts:
        a = np.array(verts, dtype=float) * scale
        out["bounding_box"] = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "size": (a.max(0) - a.min(0)).tolist()}
    return out


def _field_bcs(text: str, field: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = parse(text)
    rows = []
    for patch, spec in (d.get("boundaryField") or {}).items():
        if not isinstance(spec, dict):
            continue
        val = spec.get("value", spec.get("inletValue", spec.get("uniformValue", spec.get("freestreamValue"))))
        extra = {k: v for k, v in spec.items() if k not in ("type", "value") and isinstance(v, str) and len(v) < 60}
        rows.append({"field": field, "patch": patch, "type": spec.get("type"),
                     "value": (str(val)[:80] if val is not None else None), "settings": extra})
    info = {"dimensions": d.get("dimensions"), "internal": str(d.get("internalField", ""))[:80]}
    return rows, info


def _match_key(d: dict[str, Any], field: str) -> Any | None:
    if field in d:
        return d[field]
    for k, v in d.items():
        pat = k.strip('"')
        try:
            if re.fullmatch(pat, field):
                return v
        except re.error:
            continue
    return None


# ------------------------------------------------------------------ function-object tables (postProcessing/**.dat)
_RE_HEADER_NAME = re.compile(r"[^\s(]+(?:\([^)]*\))?|\([^)]*\)")


def _nested(line: str) -> list[Any]:
    """Whitespace tokens of a data row, parenthesised groups as (nested) lists: ``1 (2 3 4)`` -> [1, [2, 3, 4]]."""
    stack: list[list[Any]] = [[]]
    for tok in line.replace("(", " ( ").replace(")", " ) ").split():
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced parentheses")
            grp = stack.pop()
            stack[-1].append(grp)
        else:
            stack[-1].append(float(tok))
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses")
    return stack[0]


def _expand(name: str, item: Any) -> list[str]:
    """Column names for one row item: a scalar keeps ``name``; a 3-vector gives ``name_x/_y/_z``; other groups get
    ``name_0, name_1...`` or, for a header such as ``forces(pressure viscous)``, ``forces_pressure_x...``."""
    if not isinstance(item, list):
        return [name]
    m = re.fullmatch(r"([^\s(]+)\((.+\s.+)\)", name)
    subs = m.group(2).split() if m else []
    base = m.group(1) if m else name
    if len(subs) == len(item) and all(isinstance(x, list) for x in item):
        return [c for sub, x in zip(subs, item, strict=True) for c in _expand(f"{base}_{sub}", x)]
    if len(item) == 3 and not any(isinstance(x, list) for x in item):
        return [f"{base}_x", f"{base}_y", f"{base}_z"]
    return [c for j, x in enumerate(item) for c in _expand(f"{base}_{j}", x)]


def _flatten(item: Any) -> list[float]:
    return [v for x in item for v in _flatten(x)] if isinstance(item, list) else [item]


def parse_dat(text: str) -> tuple[list[str], np.ndarray]:
    """Parse an OpenFOAM function-object table (forceCoeffs.dat, forces.dat, probes, residuals.dat, ...).

    Header lines start with ``#``; the last non-empty one names the columns (``# Time Cd Cl Cd(f) ...``). Data
    values in parentheses are vectors (or nested groups) and are expanded to one column each: a 3-vector under
    the header name ``U`` becomes ``U_x, U_y, U_z``. Rows that do not parse or have another layout than the first
    data row are skipped. Returns ``(columns, array)`` with ``array.shape == (n_rows, len(columns))``.
    """
    names: list[str] = []
    rows: list[list[float]] = []
    columns: list[str] = []
    width = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            h = s.lstrip("#").strip()
            if h and not rows:
                names = _RE_HEADER_NAME.findall(h)
            continue
        try:
            items = _nested(s)
        except ValueError:
            continue
        flat = _flatten(items)
        if width is None:
            if len(names) == len(items):
                columns = [c for n, it in zip(names, items, strict=True) for c in _expand(n, it)]
            else:
                columns = [f"col_{j}" for j in range(len(flat))]
            width = len(flat)
            if len(columns) != width:
                columns = [f"col_{j}" for j in range(width)]
        if len(flat) == width:
            rows.append(flat)
    if not rows:
        return names, np.zeros((0, len(names)))
    return columns, np.asarray(rows, dtype=np.float64)


def read_dat(path: str) -> tuple[list[str], np.ndarray]:
    """:func:`parse_dat` of the file at ``path`` (an OpenFOAM ``postProcessing/<name>/<time>/<table>.dat``)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_dat(fh.read())


def extract_case(fs: dict[str, bytes], root: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"files_used": []}

    def use(path: str | None) -> None:
        if path:
            rec["files_used"].append(path)

    p, t = _first(fs, root, "system/controlDict")
    cd = parse(t) if t else {}
    use(p)
    p, t = _first(fs, root, "system/fvSchemes")
    schemes = parse(t) if t else {}
    use(p)
    p, t = _first(fs, root, "system/fvSolution")
    sol = parse(t) if t else {}
    use(p)
    for k in (cd, schemes, sol):
        k.pop("FoamFile", None)
    rec["controlDict"], rec["fvSchemes"], rec["fvSolution"] = cd, schemes, sol

    # physical properties / models
    props: dict[str, Any] = {}
    for rel in ("constant/transportProperties", "constant/physicalProperties", "constant/thermophysicalProperties",
                "constant/turbulenceProperties", "constant/momentumTransport", "constant/g",
                "constant/thermophysicalTransport", "constant/radiationProperties"):
        p, t = _first(fs, root, rel)
        if t:
            d = parse(t)
            d.pop("FoamFile", None)
            props[rel.split("/")[-1]] = d
            use(p)
    rec["properties"] = props

    # mesh
    mesh: dict[str, Any] = {}
    p, t = _first(fs, root, "constant/polyMesh/owner")
    if t:
        note = header(t).get("note", "")
        for k, v in re.findall(r"(nPoints|nCells|nFaces|nInternalFaces):\s*(\d+)", note):
            mesh[{"nPoints": "points", "nCells": "cells", "nFaces": "faces", "nInternalFaces": "internal_faces"}[k]] = int(v)
        mesh["source"] = "constant/polyMesh"
        use(p)
    p, t = _first(fs, root, "constant/polyMesh/boundary")
    if t:
        mesh["patches"] = boundary(t)
        use(p)
    p, t = _first(fs, root, "constant/polyMesh/points")
    if t:
        bb = _points_bbox(t)
        if bb:
            mesh["bounding_box"] = bb
        use(p)
    p, t = _first(fs, root, "system/blockMeshDict", "constant/polyMesh/blockMeshDict")
    if t:
        bmd = _block_mesh(t)
        use(p)
        mesh.setdefault("bounding_box", bmd.get("bounding_box"))
        if "cells" not in mesh and bmd.get("cells_from_blocks"):
            mesh["cells"] = bmd["cells_from_blocks"]
            mesh["source"] = "system/blockMeshDict (blocks; snappyHexMesh not counted)"
    if fs.get(f"{root}system/snappyHexMeshDict") is not None:
        mesh["snappyHexMesh"] = True
        use(f"{root}system/snappyHexMeshDict")
    rec["mesh"] = mesh

    # initial / boundary conditions (0 or 0.orig)
    bcs: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}
    zero = next((d for d in ("0/", "0.orig/") if any(k.startswith(f"{root}{d}") for k in fs)), None)
    if zero:
        for path in sorted(k for k in fs if k.startswith(f"{root}{zero}") and k.count("/") == (root + zero).count("/")):
            name = path.rsplit("/", 1)[-1]
            if name.startswith(".") or name.endswith((".orig", ".gz")):
                continue
            t = _read(fs, path)
            if not t or "boundaryField" not in t:
                continue
            rows, info = _field_bcs(t, name)
            bcs += rows
            fields[name] = info
            use(path)
    rec["boundary_conditions"], rec["fields"] = bcs, fields

    # logs
    logs = []
    for path in sorted(fs):
        if not path.startswith(root):
            continue
        base = path.rsplit("/", 1)[-1]
        if base.startswith("log") or base.endswith(".log"):
            t = _read(fs, path) or ""
            if "Exec" in t[:4000] or "Solving for" in t or "Time =" in t:
                lg = parse_log(t)
                lg["path"] = path
                logs.append(lg)
                use(path)
    rec["logs"] = logs
    rec["time_dirs"] = sorted({float(m.group(1)) for k in fs if (m := re.match(re.escape(root) + rf"({_NUM})/", k))
                               and m.group(1) not in ("0",)})
    return rec


def summarise(case: dict[str, Any]) -> dict[str, Any]:
    """Turn the raw case extraction into the record sections."""
    cd, schemes, sol, props = case["controlDict"], case["fvSchemes"], case["fvSolution"], case["properties"]
    solver_log = next((ln for ln in case["logs"] if ln.get("exec") and ln["exec"] not in
                       ("blockMesh", "checkMesh", "snappyHexMesh", "decomposePar", "reconstructPar", "setFields",
                        "surfaceFeatureExtract", "surfaceFeatures", "mapFields", "foamToVTK", "postProcess")), None)
    if solver_log is None:
        solver_log = next((ln for ln in case["logs"] if ln.get("time_steps")), None)
    app = cd.get("application") or (solver_log or {}).get("exec")
    ddt = (schemes.get("ddtSchemes") or {}).get("default", "")
    steady = "steadyState" in ddt or app in STEADY_SOLVERS
    turb = props.get("turbulenceProperties") or props.get("momentumTransport") or {}
    sim_type = turb.get("simulationType")
    model = None
    if isinstance(turb.get("RAS"), dict):
        model = turb["RAS"].get("RASModel") or turb["RAS"].get("model")
    if isinstance(turb.get("LES"), dict):
        model = turb["LES"].get("LESModel") or turb["LES"].get("model")
    algo_name = next((k for k in ("SIMPLE", "PIMPLE", "PISO", "SIMPLEC") if k in sol), None)
    algo = dict(sol.get(algo_name, {})) if algo_name else {}
    if algo_name == "SIMPLE" and str(algo.get("consistent", "")).lower() in ("yes", "true", "on"):
        algo_name = "SIMPLEC"
    phys = list(SOLVER_PHYSICS.get(app or "", []))
    if sim_type:
        phys = [p for p in phys if p != "turbulent/laminar"] + [{"laminar": "laminar", "RAS": "turbulent (RANS)",
                                                                  "LES": "turbulent (LES)"}.get(sim_type, sim_type)]
    tp = props.get("transportProperties") or props.get("physicalProperties") or {}
    nu = scalar(tp.get("nu")) if tp.get("nu") is not None else None
    thermo = props.get("thermophysicalProperties") or {}
    params: list[dict[str, Any]] = []
    if nu is not None:
        params.append({"name": "kinematic viscosity ν", "value": nu, "unit": "m²/s", "source": "transportProperties"})
    if tp.get("rho") is not None and scalar(tp.get("rho")) is not None:
        params.append({"name": "density ρ", "value": scalar(tp.get("rho")), "unit": "kg/m³", "source": "transportProperties"})
    if isinstance(thermo.get("thermoType"), dict):
        params.append({"name": "thermophysical model", "value": " / ".join(str(v) for v in thermo["thermoType"].values()),
                       "unit": "", "source": "thermophysicalProperties"})
    g = props.get("g", {})
    if g.get("value"):
        params.append({"name": "gravity", "value": g["value"], "unit": "m/s²", "source": "constant/g"})
    # inlet velocity from the U boundary conditions
    u_in = None
    for bc in case["boundary_conditions"]:
        if bc["field"] == "U" and bc["type"] in ("fixedValue", "flowRateInletVelocity", "freestream",
                                                 "freestreamVelocity", "inletOutlet", "uniformFixedValue"):
            v = vector(bc.get("value"))
            if v and any(abs(x) > 0 for x in v):
                u_in = (bc["patch"], v)
                params.append({"name": f"velocity at '{bc['patch']}'", "value": v, "unit": "m/s", "source": "0/U"})
                break
    for bc in case["boundary_conditions"]:
        if bc["field"] in ("T", "p") and bc["type"] == "fixedValue" and scalar(bc.get("value")) not in (None,):
            if bc["field"] == "T" or abs(scalar(bc.get("value")) or 0) > 0:
                params.append({"name": f"{bc['field']} at '{bc['patch']}'", "value": scalar(bc.get("value")),
                               "unit": "K" if bc["field"] == "T" else ("m²/s² (kinematic)" if nu is not None else "Pa"),
                               "source": f"0/{bc['field']}"})
    derived = []
    bb = case["mesh"].get("bounding_box")
    if nu and u_in and bb:
        umag = math.sqrt(sum(x * x for x in u_in[1]))
        sizes = [s for s in bb["size"] if s > 0]
        dims2 = sorted(sizes)
        L = dims2[1] if len(dims2) == 3 and case["mesh"].get("patches") and any(
            p["type"] == "empty" for p in case["mesh"]["patches"]) else dims2[0]
        if L > 0:
            derived.append({"name": "Reynolds number (indicative)", "value": umag * L / nu,
                            "definition": f"|U| at '{u_in[0]}' ({umag:g} m/s) × L / ν, with L = {L:.4g} m, the smallest "
                                          "in-plane extent of the domain. Use the case's own reference length for a "
                                          "definitive value."})
    lin = {}
    for k, v in (sol.get("solvers") or {}).items():
        if isinstance(v, dict):
            lin[k.strip('"')] = {kk: v.get(kk) for kk in ("solver", "preconditioner", "smoother", "tolerance", "relTol",
                                                           "maxIter", "nCellsInCoarsestLevel") if v.get(kk) is not None}
    numerics = {
        "time_scheme": ddt or None, "convection": {k: v for k, v in (schemes.get("divSchemes") or {}).items()},
        "gradient": (schemes.get("gradSchemes") or {}).get("default"),
        "laplacian": (schemes.get("laplacianSchemes") or {}).get("default"),
        "linear_solvers": lin,
        "pressure_velocity": {"algorithm": algo_name,
                              **{k: v for k, v in algo.items() if not isinstance(v, dict)}},
        "residual_control": algo.get("residualControl") if isinstance(algo.get("residualControl"), dict) else None,
        "relaxation": sol.get("relaxationFactors"),
    }
    executed = {}
    if solver_log:
        h = solver_log["header"]
        executed = {"date": " ".join(x for x in (h.get("Date"), h.get("Time")) if x) or None, "host": h.get("Host"),
                    "n_procs": int(h["nProcs"]) if h.get("nProcs", "").isdigit() else None,
                    "wall_time_s": solver_log.get("clock_time_s"), "cpu_time_s": solver_log.get("execution_time_s"),
                    "log": solver_log["path"]}
    mesh = dict(case["mesh"])
    q = next((ln.get("mesh_quality") for ln in case["logs"] if ln.get("mesh_quality")), None)
    if q:
        mesh["quality"] = q
        for k in ("cells", "points", "faces"):
            if k not in mesh and q.get(k):
                mesh[k] = int(q[k])
    if "cells" not in mesh:
        bm = next((ln.get("mesh_cells") for ln in case["logs"] if ln.get("mesh_cells")), None)
        if bm:
            mesh["cells"], mesh["source"] = bm, "blockMesh log"
    end = scalar(cd.get("endTime"))
    time_rec = {"start": scalar(cd.get("startTime")), "end": end, "delta_t": scalar(cd.get("deltaT")),
                "adjustable": str(cd.get("adjustTimeStep", "no")).lower() in ("yes", "on", "true"),
                "max_courant": scalar(cd.get("maxCo")), "write_control": cd.get("writeControl"),
                "write_interval": scalar(cd.get("writeInterval")),
                "unit": "iterations" if steady else "s"}
    if solver_log and solver_log["time_steps"]:
        last = solver_log["time_steps"][-1]
        time_rec.update({"steps_run": len(solver_log["time_steps"]), "last_time": last,
                         "reached_end": bool(end is not None and last >= end - 1e-9 * max(1.0, abs(end)))})
    return {
        "solver": {"name": "OpenFOAM", "family": "CFD · finite volume", "application": app,
                   "version": (solver_log or {}).get("version"), "executed": executed},
        "analysis": {"type": "steady" if steady else "transient", "physics": phys,
                     "turbulence": {"type": sim_type, "model": model} if sim_type else None,
                     "algorithm": algo_name},
        "mesh": mesh, "time": time_rec, "boundary_conditions": case["boundary_conditions"],
        "fields": case["fields"], "parameters": params, "derived": derived, "numerics": numerics,
        "_log": solver_log, "_steady": steady, "_residual_control": numerics["residual_control"],
        "_linear_tolerances": {k: scalar(v.get("tolerance")) for k, v in lin.items()},
    }
