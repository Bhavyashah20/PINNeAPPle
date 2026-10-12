"""Generate the planned meshes and measure their quality.

- solids: Gmsh from the STEP file, quadratic tetrahedra, sizes from the plan (curvature-based refinement puts the
  planned number of elements around holes and fillets), written as ``.msh`` and Abaqus/CalculiX ``.inp``;
- 3-D flow: OpenFOAM blockMesh + snappyHexMesh with prism layers sized by the plan (external with far field and,
  for vehicles, a ground; internal with inlet/outlet patches taken from the CAD faces), checked with checkMesh;
- 2-D sections: Gmsh with a boundary-layer field and quadrilaterals, extruded one cell (OpenFOAM-ready ``.msh``).

Every mesh then goes through ``pinneapple_data.cae`` (non-orthogonality, skewness, aspect ratio, scaled Jacobian,
...) and the report keeps the verdict, the statistics and where the bad cells are.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cad import CadVariant
from .catalog import Template
from .mesh_plan import MeshPlan

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")


@dataclass
class MeshResult:
    variant: int
    level: float                     # size multiplier (1 = the planned, finest mesh)
    tool: str
    path: str = ""
    files: list[str] = field(default_factory=list)
    cells: int = 0
    seconds: float = 0.0
    quality: dict[str, Any] = field(default_factory=dict)      # summary, checks and verdict from pinneapple_data.cae
    layers: dict[str, Any] = field(default_factory=dict)       # snappyHexMesh: achieved layers and thickness
    error: str = ""

    @property
    def verdict(self) -> str:
        if self.error:
            return "fail"
        return self.quality.get("verdict", "unknown")


def generate(tpl: Template, var: CadVariant, plan: MeshPlan, out_dir: str, level: float = 1.0, procs: int = 1,
             max_cells: float = 3e6, log: Callable[[str], None] = print) -> MeshResult:
    """Mesh one CAD variant at ``level`` x the planned sizes (1.0 = the plan; 1.5, 2.25 = the coarser family)."""
    res = MeshResult(variant=var.index, level=level, tool=plan.tool)
    est = plan.estimated_cells / level ** 3
    if est > max_cells:
        res.error = f"estimated {est:,.0f} cells is above max_cells = {max_cells:,.0f}; raise it to generate"
        return res
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:
        if plan.kind == "solid":
            _gmsh_solid(var, plan, out_dir, level, res)
        elif plan.domain.get("two_d") and tpl.section is not None:
            _gmsh_2d(tpl, var, plan, out_dir, level, res)
        else:
            _snappy(tpl, var, plan, out_dir, level, procs, res, log)
    except Exception as exc:                                       # report, never hide
        res.error = f"{type(exc).__name__}: {exc}"[:400]
    res.seconds = round(time.time() - t0, 1)
    if not res.error and res.path:
        res.quality = quality(res.path, "fea" if plan.kind == "solid" else "cfd")
        res.cells = int(res.quality.get("summary", {}).get("cells", res.cells))
        for name, st in (res.layers.get("patches") or {}).items():
            pct = st["coverage_pct"]
            ok = pct >= 80.0
            res.quality["checks"].append({
                "key": f"prism_layers_{name}", "status": "pass" if ok else "warn", "value": pct, "limit": 80.0,
                "message": f"{st['layers']:.1f} of {res.layers['requested']['count']} layers on average, "
                           f"{pct:.0f} % of the planned thickness",
                "action": "" if ok else "layers collapsed near sharp edges or thin gaps (a known snappyHexMesh "
                          "limit): check y+ in the solution, relax featureAngle / minThickness, or use a "
                          "prism-layer mesher for this geometry"})
            if not ok and res.quality["verdict"] == "pass":
                res.quality["verdict"] = "warn"
    return res


# ---------------------------------------------------------------- quality through pinneapple_data.cae
def quality(path: str, target: str) -> dict[str, Any]:
    from pinneapple_data.cae import io as cae_io
    from pinneapple_data.cae import report as cae_report

    fs: dict[str, bytes] = {}
    if os.path.isdir(path):                                         # OpenFOAM case: the polyMesh files
        poly = os.path.join(path, "constant", "polyMesh")
        for n in os.listdir(poly):
            fp = os.path.join(poly, n)
            if os.path.isfile(fp):
                fs[f"case/constant/polyMesh/{n}"] = open(fp, "rb").read()
    else:
        fs[os.path.basename(path)] = open(path, "rb").read()
    mesh = cae_io.read_any(fs, fields=False)
    rep = cae_report.build(mesh, target=target)
    st = [c.get("status") for c in rep.get("checks", [])]
    verdict = "fail" if "fail" in st else "warn" if "warn" in st else "pass"
    return {"summary": rep["summary"], "checks": rep["checks"], "regions": rep.get("regions", [])[:6],
            "notes": rep.get("notes", []), "verdict": verdict}


# ---------------------------------------------------------------- Gmsh: solids
def _gmsh_solid(var: CadVariant, plan: MeshPlan, out: str, level: float, res: MeshResult) -> None:
    import gmsh

    s = plan.sizes
    h_max = s["global"] * level
    h_min = min(s.values()) * level
    # elements per 2*pi of curvature: the planned count around holes, else 20 (the Gmsh tutorial default)
    if "at_hole" in s:
        around = max(8, int(round(math.pi * var.params["hole_diameter"] / (s["at_hole"] * level))))
    else:
        around = max(8, int(round(20 / level)))
    stem = os.path.join(out, f"v{var.index:03d}_L{level:g}")
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(var.step)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", around)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        vols = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], name="solid")
        gmsh.model.mesh.generate(3)
        gmsh.write(stem + ".msh")
        gmsh.write(stem + ".inp")
        _, tags, _ = gmsh.model.mesh.getElements(3)
        res.cells = int(sum(len(t) for t in tags))
    finally:
        gmsh.finalize()
    res.path, res.files = stem + ".msh", [stem + ".msh", stem + ".inp"]


# ---------------------------------------------------------------- Gmsh: 2-D sections with a boundary layer
def _gmsh_2d(tpl: Template, var: CadVariant, plan: MeshPlan, out: str, level: float, res: MeshResult) -> None:
    import gmsh

    role, pts = tpl.section(var.params)
    hs = plan.sizes["surface"] * level
    h0 = plan.sizes["background"] * level
    lay = plan.layers
    L = plan.sizes["reference_length"]
    stem = os.path.join(out, f"v{var.index:03d}_L{level:g}")
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        geo = gmsh.model.geo
        def loop(points, size):
            tags = [geo.addPoint(float(x), float(z), 0.0, size) for x, z in points]
            lines = [geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
            return geo.addCurveLoop(lines), lines
        if role == "body":
            d = plan.domain
            lo = pts.min(0)
            hi = pts.max(0)
            box = [(lo[0] - d["upstream"], lo[1] - d["lateral"]), (hi[0] + d["downstream"], lo[1] - d["lateral"]),
                   (hi[0] + d["downstream"], hi[1] + d["lateral"]), (lo[0] - d["upstream"], hi[1] + d["lateral"])]
            outer, _ = loop(box, h0)
            inner, wall = loop(pts, hs)
            surf = geo.addPlaneSurface([outer, inner])
        else:
            outer, wall = loop(pts, hs)
            surf = geo.addPlaneSurface([outer])
        geo.synchronize()
        if lay.get("first_height"):
            f = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(f, "CurvesList", wall)
            gmsh.model.mesh.field.setNumber(f, "Size", lay["first_height"])
            gmsh.model.mesh.field.setNumber(f, "Ratio", lay["growth"])
            gmsh.model.mesh.field.setNumber(f, "Thickness", lay["total_thickness"])
            gmsh.model.mesh.field.setNumber(f, "Quads", 1)
            gmsh.model.mesh.field.setAsBoundaryLayer(f)
        if role == "body":
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "CurvesList", wall)
            thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr, "InField", dist)
            gmsh.model.mesh.field.setNumber(thr, "SizeMin", hs)
            gmsh.model.mesh.field.setNumber(thr, "SizeMax", h0)
            gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.05 * L)
            gmsh.model.mesh.field.setNumber(thr, "DistMax", 2.0 * L)
            gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)                  # frontal-Delaunay for quads
        ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, hs, numElements=[1], recombine=True)
        geo.synchronize()
        vol = [e[1] for e in ext if e[0] == 3]
        gmsh.model.addPhysicalGroup(3, vol, name="fluid")
        gmsh.model.addPhysicalGroup(2, [surf] + [e[1] for e in ext if e[0] == 2 and e is ext[0]], name="frontAndBack")
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)           # gmshToFoam reads 2.2
        gmsh.write(stem + ".msh")
        _, tags, _ = gmsh.model.mesh.getElements(3)
        res.cells = int(sum(len(t) for t in tags))
    finally:
        gmsh.finalize()
    res.path, res.files, res.tool = stem + ".msh", [stem + ".msh"], "gmsh (2-D, extruded one cell)"


# ---------------------------------------------------------------- snappyHexMesh: 3-D flows
def _foam(case: str, cmd: str, name: str) -> int:
    full = f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} > log.{name} 2>&1'
    return subprocess.run(["bash", "-c", full]).returncode


def _hdr(cls: str, obj: str) -> str:
    return f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n"


def _write(case: str, rel: str, body: str, cls: str = "dictionary") -> None:
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(_hdr(cls, os.path.basename(rel)) + body)


def _stl_regions(tpl: Template, var: CadVariant, path: str, tol: float) -> list[str]:
    """ASCII STL with one solid per patch: the ports from the CAD faces, the rest as 'wall'."""
    import cadquery as cq

    wp = tpl.build(var.params)
    used, names, chunks = set(), [], []
    groups = [(n, wp.faces(sel).vals()) for n, sel in tpl.ports.items()]
    for _, faces in groups:
        used |= {f.hashCode() for f in faces}
    groups.append(("wall", [f for f in wp.faces().vals() if f.hashCode() not in used]))
    tmp = path + ".part.stl"
    for name, faces in groups:
        if not faces:
            continue
        cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(faces)), tmp, tolerance=tol,
                            angularTolerance=0.1, opt={"ascii": True})
        txt = open(tmp).read()
        txt = re.sub(r"^solid.*$", f"solid {name}", txt, count=1, flags=re.M)
        txt = re.sub(r"^endsolid.*$", f"endsolid {name}", txt, flags=re.M)
        chunks.append(txt.strip() + "\n")
        names.append(name)
    os.remove(tmp)
    with open(path, "w") as f:
        f.write("".join(chunks))
    return names


def _snappy(tpl, var, plan, out, level, procs, res, log) -> None:
    import shutil as _sh
    case = os.path.join(out, f"v{var.index:03d}_L{level:g}")
    if os.path.isdir(case):
        _sh.rmtree(case)
    s, lay, dom = plan.sizes, plan.layers, plan.domain
    h0, levels = s["background"] * level, s["levels"]
    bb_lo, bb_hi = _bounds(var.stl)
    internal = plan.kind == "internal_flow"
    tri = os.path.join(case, "constant", "triSurface", "body.stl")
    os.makedirs(os.path.dirname(tri))
    if internal:
        regions = _stl_regions(tpl, var, tri, 0.001 * s["smallest_feature"])
        pad = 2 * h0
        lo, hi = bb_lo - pad, bb_hi + pad
        loc = tpl.inside(var.params)
    else:
        shutil.copy(var.stl, tri)
        regions = []
        lo = bb_lo - np.array([dom["upstream"], dom["lateral"], dom["top"]])
        hi = bb_hi + np.array([dom["downstream"], dom["lateral"], dom["top"]])
        if dom.get("ground"):
            lo[2] = 0.0
        loc = (lo[0] + 1.37 * h0, hi[1] - 1.61 * h0, hi[2] - 1.53 * h0)
    n = [max(4, int(math.ceil((b - a) / h0))) for a, b in zip(lo, hi, strict=True)]
    hi = lo + np.array(n) * h0
    loc = tuple(float(x) for x in loc)
    ground = "\n    ground { type wall; faces ( (0 3 2 1) ); }" if dom.get("ground") else ""
    far_faces = "(0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (4 5 6 7)" + ("" if dom.get("ground") else " (0 3 2 1)")
    lo, hi = [float(x) for x in lo], [float(x) for x in hi]
    v = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
         (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    _write(case, "system/blockMeshDict", "convertToMeters 1;\nvertices (" + " ".join(
        f"({a!r} {b!r} {c!r})" for a, b, c in v) + f");\nblocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) "
           f"simpleGrading (1 1 1) );\nedges ();\nboundary\n(\n    farfield {{ type patch; faces ( {far_faces} ); }}"
           f"{ground}\n);\nmergePatchPairs ();\n")
    _write(case, "system/controlDict", "application snappyHexMesh;\nstartFrom startTime;\nstartTime 0;\n"
           "stopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
    _write(case, "system/fvSchemes", "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
           "divSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\n"
           "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
    _write(case, "system/fvSolution", "")
    _write(case, "system/decomposeParDict", f"numberOfSubdomains {procs};\nmethod scotch;\n")
    _write(case, "system/surfaceFeatureExtractDict", "body.stl { extractionMethod extractFromSurface; "
           "extractFromSurfaceCoeffs { includedAngle 150; } subsetFeatures { nonManifoldEdges no; openEdges yes; } "
           "writeObj no; }\n")
    if internal:
        reg = " ".join(f"{r} {{ name {r}; }}" for r in regions)
        geom = f"    body.stl {{ type triSurfaceMesh; name body; regions {{ {reg} }} }}"
        rtypes = {"inlet": "patch", "outlet": "patch", "sides": "symmetry", "wall": "wall"}
        surf_regions = " ".join(f"{r} {{ level ({levels} {levels}); patchInfo {{ type {rtypes.get(r, 'wall')}; }} }}"
                                for r in regions)
        surfaces = f"        body {{ level ({levels} {levels}); regions {{ {surf_regions} }} }}"
        layer_patch = '"wall"'
    else:
        geom = "    body.stl { type triSurfaceMesh; name body; }"
        surfaces = f"        body {{ level ({max(levels - 1, 0)} {levels}); patchInfo {{ type wall; }} }}"
        layer_patch = '"body.*"'
    y1 = lay.get("first_height", 0.0) * level
    nl = int(lay.get("count", 0))
    add_layers = "true" if nl and y1 > 0 else "false"
    _write(case, "system/snappyHexMeshDict", f"""castellatedMesh true;
snap true;
addLayers {add_layers};
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 20000000; minRefinementCells 5; maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ( {{ file "body.eMesh"; level {levels}; }} );
    refinementSurfaces
    {{
{surfaces}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{ }}
    locationInMesh ({loc[0]!r} {loc[1]!r} {loc[2]!r});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes false;
    layers {{ {layer_patch} {{ nSurfaceLayers {nl}; }} }}
    firstLayerThickness {max(y1, 1e-12)!r};
    expansionRatio {lay.get('growth', 1.2)!r};
    minThickness {max(0.1 * y1, 1e-13)!r};
    nGrow 0; featureAngle 130; slipFeatureAngle 30; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-30;
    minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
writeFlags ();
mergeTolerance 1e-6;
""")
    for cmd, name in (("blockMesh", "blockMesh"), ("surfaceFeatureExtract", "surfaceFeatureExtract")):
        if _foam(case, cmd, name):
            raise RuntimeError(f"{name} failed: see {case}/log.{name}")
    mpi = f"mpirun --allow-run-as-root --oversubscribe -np {procs}"
    if procs > 1:
        _foam(case, "decomposePar -force", "decomposePar")
        rc = _foam(case, f"{mpi} snappyHexMesh -overwrite -parallel", "snappyHexMesh")
        _foam(case, "reconstructParMesh -constant", "reconstructParMesh")
    else:
        rc = _foam(case, "snappyHexMesh -overwrite", "snappyHexMesh")
    if rc:
        raise RuntimeError(f"snappyHexMesh failed: see {case}/log.snappyHexMesh")
    _foam(case, "checkMesh", "checkMesh")
    res.layers = _layer_stats(os.path.join(case, "log.snappyHexMesh"))
    res.layers["requested"] = {"count": nl, "first_height": y1}
    chk = open(os.path.join(case, "log.checkMesh")).read()
    m = re.search(r"cells:\s+(\d+)", chk)
    res.cells = int(m.group(1)) if m else 0
    res.layers["checkMesh_ok"] = "Mesh OK." in chk
    for d in os.listdir(case):
        if d.startswith("processor"):
            shutil.rmtree(os.path.join(case, d))
    res.path, res.files = case, [case]


def _bounds(stl: str):
    import trimesh
    m = trimesh.load(stl, force="mesh")
    return np.asarray(m.bounds[0], float), np.asarray(m.bounds[1], float)


def _layer_stats(log_path: str) -> dict[str, Any]:
    """Achieved layers from the snappyHexMesh layer summary table (patch, faces, layers, thickness, %)."""
    txt = open(log_path, errors="ignore").read()
    rows = re.findall(r"^\s*(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.eE+-]+)\s+([\d.]+)\s*$", txt, flags=re.M)
    out = {}
    for name, faces, nlay, thick, pct in rows:
        out[name] = {"faces": int(faces), "layers": float(nlay), "thickness": float(thick), "coverage_pct": float(pct)}
    return {"patches": out}
