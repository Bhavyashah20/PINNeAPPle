"""Recommend a mesh that is fit for the purpose: fine enough for the quantities that matter, no finer.

The recommendation is a set of explicit rules, each written into the plan's rationale with its source, so the
choice can be reviewed and overridden. Three purposes set the resolution:

- ``screening`` (trends and ranking, ~10 %): wall functions, coarse surface, one mesh;
- ``design`` (a few %): resolved boundary layers where separation or wall quantities matter, one mesh plus a
  coarser one to see the trend;
- ``verification`` (a number to sign off): three systematically refined meshes (ratio 1.5) for Richardson
  extrapolation and the grid-convergence index (Roache; Celik et al. 2008).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import Template
from .context import MATERIALS, ProblemContext, wall_unit_height
from .layers import layer_total

# per purpose: surface cells per reference length, cells across the smallest feature, background cells per
# reference length, prism growth, wake length in body lengths, CPU budget hint
CFD_RULES = {
    "screening": dict(per_length=40, per_feature=4, background=8, growth=1.3, wake=10.0, y_plus=50.0),
    "design": dict(per_length=80, per_feature=8, background=10, growth=1.2, wake=15.0, y_plus=1.0),
    "verification": dict(per_length=120, per_feature=12, background=12, growth=1.15, wake=20.0, y_plus=1.0),
}
INTERNAL_ACROSS = {"screening": 12, "design": 24, "verification": 36}        # cells across the diameter / step
FE_RULES = {   # elements around a hole, through the thinnest wall, per characteristic length
    "screening": dict(around_hole=16, through=2, per_length=6),
    "design": dict(around_hole=32, through=3, per_length=10),
    "verification": dict(around_hole=48, through=4, per_length=16),
}
CELL_COST_S = 1e-6           # CPU seconds per cell and iteration for a steady RANS solver (order of magnitude)
TWO_D = ("naca4_wing", "cylinder_crossflow", "backward_facing_step")
SEPARATION_QUANTITIES = {"drag coefficient", "reattachment length", "wall shear", "Strouhal number",
                         "pressure drop", "erosion rate", "Nusselt number", "thermal resistance"}


@dataclass
class MeshPlan:
    kind: str                          # external_flow / internal_flow / solid
    tool: str                          # snappyHexMesh / gmsh
    elements: str
    purpose: str
    sizes: dict[str, float] = field(default_factory=dict)
    layers: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)
    estimated_cells: int = 0
    estimated_cost: str = ""
    family: list[float] = field(default_factory=list)    # size multipliers of the meshes to build (1 = finest)
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mesh_kind(tpl: Template, ctx: ProblemContext) -> str:
    phys = ctx.physics
    if "structural" in tpl.physics and ("structural" in phys or not phys):
        return "solid"
    if tpl.key == "heat_sink" and not any(p in phys for p in ("external_flow", "internal_flow")):
        return "solid"
    if "internal_flow" in tpl.physics and ("internal_flow" in phys or "external_flow" not in tpl.physics):
        return "internal_flow"
    return "external_flow" if "external_flow" in tpl.physics else "solid"


def plan_mesh(tpl: Template, params: dict[str, float], ctx: ProblemContext, budget_cells: float = 5e6) -> MeshPlan:
    purpose = ctx.accuracy if ctx.accuracy in CFD_RULES else "design"
    kind = mesh_kind(tpl, ctx)
    if kind == "solid":
        plan = _plan_fe(tpl, params, ctx, purpose)
    else:
        plan = _plan_cfd(tpl, params, ctx, purpose, kind)
    plan.family = [1.0, 1.5, 2.25] if purpose == "verification" else [1.0, 1.5] if purpose == "design" else [1.0]
    if purpose == "verification":
        plan.rationale.append("Three meshes with a constant refinement ratio r = 1.5 (> 1.3) for Richardson "
                              "extrapolation and the grid-convergence index (Roache 1994; Celik et al., J. Fluids "
                              "Eng. 130:078001, 2008).")
    elif purpose == "design":
        plan.rationale.append("A second mesh 1.5x coarser shows whether the quantities still move with the mesh; "
                              "a sign-off number needs the three-mesh study (purpose 'verification').")
    if plan.estimated_cells > budget_cells:
        plan.warnings.append(f"about {plan.estimated_cells:,} cells, above the budget of {int(budget_cells):,}: "
                             "use a symmetry plane, a 2-D section, or the 'screening' purpose first.")
    return plan


# ---------------------------------------------------------------- CFD
def _plan_cfd(tpl, v, ctx, purpose, kind) -> MeshPlan:
    r = CFD_RULES[purpose]
    L = tpl.ref_length(v)
    feat = tpl.min_feature(v)
    rho, mu = ctx.fluid_props()
    nu = mu / rho
    re = ctx.reynolds(L)
    U = ctx.get("speed") or (re * nu / L if re else None)
    plan = MeshPlan(kind=kind, tool="snappyHexMesh", elements="hexahedra with prism layers at the walls",
                    purpose=purpose)
    rat = plan.rationale
    if kind == "internal_flow":
        n_across = INTERNAL_ACROSS[purpose]
        hs = feat / n_across
        h0 = 4 * hs
        rat.append(f"{n_across} cells across the {('diameter' if 'diameter' in v else 'step height')} "
                   f"({feat:.4g} m): the core flow and the secondary flow in bends are resolved "
                   f"({'screening' if purpose == 'screening' else 'design practice'} level).")
    else:
        hs = min(L / r["per_length"], feat / r["per_feature"])
        h0 = L / r["background"]
        rat.append(f"Surface cells: min(L/{r['per_length']}, smallest feature/{r['per_feature']}) = {hs:.4g} m "
                   f"(L = {L:.4g} m, smallest feature {feat:.4g} m).")
    levels = max(0, math.ceil(math.log2(h0 / hs)))
    hs = h0 / 2 ** levels
    plan.sizes = {"background": h0, "surface": hs, "levels": levels, "reference_length": L, "smallest_feature": feat}
    rat.append(f"Background cells {h0:.4g} m refined {levels} times to {hs:.4g} m at the walls (snappyHexMesh "
               "halves the cell size per level).")

    # wall treatment and prism layers
    qty = set(ctx.get("quantities", [])) or set(tpl.quantities[:1])
    resolve = purpose != "screening" and bool(qty & SEPARATION_QUANTITIES | {"lift coefficient"} & qty)
    y_plus = 1.0 if resolve else 50.0
    if re is None or U is None:
        plan.warnings.append("no speed or Reynolds number: prism layers sized for y+ cannot be computed; "
                             "answer the speed question.")
        plan.layers = {"wall_treatment": "unknown"}
    else:
        wu = wall_unit_height(y_plus, U, L, nu, re, internal=(kind == "internal_flow"))
        y1 = wu["y1"]
        g = r["growth"]
        n = 1 + int(math.ceil(math.log(max(0.5 * hs / y1, 1.0)) / math.log(g)))
        n = min(max(n, 3), 30)
        total = layer_total(y1, g, n)
        plan.layers = {"wall_treatment": "resolved (low-Re, y+ ~ 1)" if resolve else "wall functions (y+ 30-100)",
                       "y_plus": y_plus, "first_height": y1, "growth": g, "count": n, "total_thickness": total,
                       "boundary_layer_thickness": wu["delta"], "reynolds": re, "turbulent": wu["turbulent"],
                       "skin_friction": wu["cf"]}
        rat.append(f"Re = {re:.3g} ({'turbulent' if wu['turbulent'] else 'laminar'}, "
                   f"{'pipe' if kind == 'internal_flow' else 'flat-plate'} estimate). "
                   + ("Separation-sensitive quantities (" + ", ".join(sorted(qty)) + ") call for a resolved "
                      "viscous sublayer: y+ ~ 1 with a low-Re model such as k-omega SST (Menter 1994)."
                      if resolve else "Wall functions are enough for screening: target y+ ~ 50 (log layer, "
                                      "30 < y+ < 300)."))
        rat.append(f"First cell {y1:.3g} m from Cf = {wu['cf']:.3g} "
                   f"({_cf_source(kind, wu['turbulent'])}); {n} layers growing "
                   f"{g} ({'1.1-1.3 is the usual range' }) reach {total:.3g} m, so the last layer matches about half "
                   f"the surface cell; the boundary layer at L is about {wu['delta']:.3g} m.")
        if resolve and wu["turbulent"] and kind != "internal_flow" and total < 0.5 * wu["delta"]:
            plan.warnings.append("the prism stack covers less than half the boundary-layer thickness; refine the "
                                 "surface or allow more layers if the outer boundary layer matters.")

    # domain and cell estimate
    bb = _bbox(tpl, v)
    if kind == "external_flow":
        lx, ly, lz = bb
        up, down, side = 5 * L, r["wake"] * L, 5 * max(L, ly, lz)
        ground = tpl.key == "ahmed_body"
        plan.domain = {"upstream": up, "downstream": down, "lateral": side, "top": side,
                       "ground": "moving wall at the body's floor clearance" if ground else None}
        area = ly * lz
        dom_area = (ly + 2 * side) * ((lz + side) if ground else (lz + 2 * side))
        plan.domain["blockage"] = area / dom_area
        rat.append(f"Domain: {up / L:.0f} L upstream, {down / L:.0f} L downstream, {side / max(L, ly, lz):.0f} body "
                   f"sizes to the sides; blockage {area / dom_area:.2%} (keep below about 3 %).")
        vol = (lx + up + down) * (ly + 2 * side) * ((lz + side) if ground else (lz + 2 * side))
        n_bg = vol / h0 ** 3
        surf = _area(tpl, v)
        n_ref = sum(surf / (h0 / 2 ** k) ** 2 * 3 for k in range(1, levels + 1))
        n_wake = (2 * L) * ly * lz * 4 / (h0 / 2 ** max(levels - 2, 0)) ** 3
    else:
        vol = _fluid_volume(tpl, v)
        n_bg, n_ref, n_wake = vol / hs ** 3, 0.0, 0.0
        surf = _area(tpl, v)
        plan.domain = {"inlet": "planar inlet face", "outlet": "planar outlet face",
                       "development": "inlet run >= 10 D (bend) / 4 h (step), set by the template"}
    n_layers = surf / hs ** 2 * plan.layers.get("count", 0)
    total = n_bg + n_ref + n_wake + n_layers
    if "span" in v and tpl.key in TWO_D and not ctx.get("three_d"):
        n_span = max(v["span"] / hs, 1.0)
        total /= n_span
        plan.domain["two_d"] = True
        rat.append("Extruded section: meshed as 2-D (one cell across the span, 'empty' front and back) unless a "
                   "3-D wake (LES, spanwise instabilities) is needed; answer three_d = true for that.")
    plan.estimated_cells = int(total)
    iters = {"screening": 1000, "design": 2000, "verification": 3000}[purpose]
    core_h = plan.estimated_cells * iters * CELL_COST_S / 3600
    plan.estimated_cost = f"~{core_h:.2g} core-hours for {iters} steady iterations (order of magnitude)"
    return plan


def _cf_source(kind: str, turbulent: bool) -> str:
    if kind == "internal_flow":
        return "pipe friction factor 0.316 Re^-1/4, Blasius" if turbulent else "laminar pipe friction 64/Re"
    return "0.026 Re^-1/7, White" if turbulent else "0.664 Re^-1/2, Blasius"


# ---------------------------------------------------------------- FE
def _plan_fe(tpl, v, ctx, purpose) -> MeshPlan:
    r = FE_RULES[purpose]
    plan = MeshPlan(kind="solid", tool="gmsh", elements="quadratic tetrahedra (10 nodes)", purpose=purpose)
    rat = plan.rationale
    L = max(_bbox(tpl, v))
    thin = tpl.min_feature(v)
    h_global = min(L / r["per_length"], thin / r["through"] * 1.0)
    sizes = {"global": h_global, "through_thinnest": thin / r["through"]}
    rat.append("Quadratic tetrahedra: linear tetrahedra are too stiff in bending and under-predict peak stress; "
               "10-node elements converge with far fewer elements.")
    rat.append(f"At least {r['through']} quadratic elements through the thinnest wall ({thin:.4g} m) and "
               f"{r['per_length']} along the largest dimension: global size {h_global:.4g} m.")
    if "hole_diameter" in v:
        h_hole = math.pi * v["hole_diameter"] / r["around_hole"]
        sizes["at_hole"] = h_hole
        rat.append(f"{r['around_hole']} elements around the hole ({h_hole:.4g} m): the stress concentration is a "
                   "local peak and needs the finest elements; far from it the global size is enough.")
        d = v["hole_diameter"] / v["width"]
        kt_n = 2 + (1 - d) ** 3
        rat.append(f"Closed-form check for the result: net-section Kt = 2 + (1 - d/W)^3 = {kt_n:.3f} "
                   "(Heywood form; Pilkey & Pilkey, Peterson's).")
    if tpl.key == "cantilever_beam":
        rat.append("Closed-form checks for the result: tip deflection P L^3 / (3 E I), root stress 6 P L / (b h^2).")
    plan.sizes = sizes
    vol = tpl.volume(v) if tpl.volume else math.prod(_bbox(tpl, v))
    h_eff = min(sizes.values())
    n_glob = vol / (0.118 * h_global ** 3)
    n_loc = 0.0
    if "at_hole" in sizes:
        rv = math.pi * ((v["hole_diameter"] / 2 + 4 * sizes["at_hole"]) ** 2 - (v["hole_diameter"] / 2) ** 2)
        n_loc = rv * v["thickness"] / (0.118 * sizes["at_hole"] ** 3)
    plan.estimated_cells = int(n_glob + n_loc)
    plan.estimated_cost = (f"~{3 * 1.6 * plan.estimated_cells:,.0f} degrees of freedom; seconds to minutes for a "
                           "linear static solve with a sparse direct solver")
    mat = ctx.get("material")
    if mat in MATERIALS:
        E, nu, rho, k = MATERIALS[mat]
        plan.domain = {"material": mat, "E": E, "poisson": nu, "density": rho, "conductivity": k}
    else:
        plan.warnings.append("no material given: the mesh does not depend on it, the analysis will.")
    _ = h_eff
    return plan


# ---------------------------------------------------------------- geometry measures (from the template formulas)
def _bbox(tpl, v):
    try:
        bb = tpl.build(v).val().BoundingBox()
        return (bb.xlen, bb.ylen, bb.zlen)
    except Exception:
        L = tpl.ref_length(v)
        return (L, L, L)


def _area(tpl, v) -> float:
    try:
        return float(tpl.build(v).val().Area())
    except Exception:
        return 6 * tpl.ref_length(v) ** 2


def _fluid_volume(tpl, v) -> float:
    return float(tpl.volume(v)) if tpl.volume else math.prod(_bbox(tpl, v))
