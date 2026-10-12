"""Parametric geometry templates with the references that make them checkable.

Each template is a reference geometry from the literature (or a textbook case with a closed-form answer), with:

- the physics it is meant for and the words that point to it in a problem description;
- its parameters, each with the value of the reference object, a typical study range and hard validity limits;
- the published reference objects it can be compared with (dimensions and what was measured on them);
- a CadQuery builder, the exact volume where it has one, the smallest feature (which drives the mesh) and the
  reference length (which defines the Reynolds number or the element size).

All lengths are in metres and angles in degrees.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    value: float                       # value of the reference object
    unit: str
    typical: tuple[float, float]       # range normally studied
    limits: tuple[float, float]        # outside this the template (or its references) stops being valid
    role: str                          # what it changes physically, in one line
    sensitivity: str = "medium"        # expected influence on the quantities of interest: high / medium / low


@dataclass(frozen=True)
class Reference:
    citation: str
    what: str                          # what was measured or derived, and how it is used here
    params: dict[str, float] = field(default_factory=dict)   # the reference object's parameter values


@dataclass
class Template:
    key: str
    title: str
    physics: tuple[str, ...]           # external_flow, internal_flow, structural, thermal
    keywords: tuple[str, ...]
    description: str
    params: tuple[Param, ...]
    references: tuple[Reference, ...]
    quantities: tuple[str, ...]        # quantities of interest usually extracted
    build: Callable[[dict[str, float]], object]
    volume: Callable[[dict[str, float]], float] | None
    min_feature: Callable[[dict[str, float]], float]
    ref_length: Callable[[dict[str, float]], float]
    notes: str = ""
    derive: Callable[[dict[str, float]], dict[str, float]] | None = None   # keeps dependent sizes consistent
    ports: dict[str, str] = field(default_factory=dict)       # internal flows: patch name -> CadQuery face selector
    inside: Callable[[dict[str, float]], tuple[float, float, float]] | None = None   # a point in the fluid
    section: Callable[[dict[str, float]], tuple[str, np.ndarray]] | None = None      # 2-D profile (x, z)

    def complete(self, values: dict[str, float]) -> dict[str, float]:
        """Defaults, then ``values``, then the dependent sizes (e.g. development lengths in diameters)."""
        v = {**self.defaults(), **values}
        return self.derive(v) if self.derive else v

    def defaults(self) -> dict[str, float]:
        return {p.name: p.value for p in self.params}

    def param(self, name: str) -> Param:
        return next(p for p in self.params if p.name == name)

    def check_params(self, values: dict[str, float]) -> list[str]:
        out = []
        for p in self.params:
            v = values.get(p.name, p.value)
            if not (p.limits[0] <= v <= p.limits[1]):
                out.append(f"{p.name} = {v:g} {p.unit} is outside the valid range {p.limits[0]:g}-{p.limits[1]:g}")
        out += _EXTRA_CHECKS.get(self.key, lambda v: [])({**self.defaults(), **values})
        return out


# ---------------------------------------------------------------- geometry helpers

def naca4_points(m: float, p: float, t: float, chord: float, n: int = 80) -> np.ndarray:
    """Closed NACA 4-digit section (x, z), trailing edge closed (a4 = -0.1036), cosine spacing."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    pts = np.concatenate([np.c_[xu[::-1], zu[::-1]], np.c_[xl[1:-1], zl[1:-1]]]) * chord
    return pts


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cq():
    import cadquery as cq
    return cq


def _naca_build(v):
    cq = _cq()
    pts = naca4_points(v["camber"] / 100, v["camber_pos"] / 10, v["thickness"] / 100, v["chord"])
    return cq.Workplane("XZ").polyline([tuple(p) for p in pts]).close().extrude(-v["span"])


def _naca_volume(v):
    pts = naca4_points(v["camber"] / 100, v["camber_pos"] / 10, v["thickness"] / 100, v["chord"])
    return polygon_area(pts) * v["span"]


def _ahmed_profile(v):
    L, H, ls, phi = v["length"], v["height"], v["slant_length"], math.radians(v["slant_angle"])
    return [(0.0, 0.0), (L, 0.0), (L, H - ls * math.sin(phi)), (L - ls * math.cos(phi), H), (0.0, H)]


def _ahmed_build(v):
    cq = _cq()
    body = (cq.Workplane("XZ").polyline(_ahmed_profile(v)).close().extrude(v["width"] / 2, both=True)
            .translate((0, 0, v["clearance"])))
    r = v["nose_radius"]
    if r > 0:
        body = body.edges("<X").fillet(r)
    return body


def _ahmed_volume_upper(v):
    return polygon_area(np.array(_ahmed_profile(v))) * v["width"]


def _cylinder_build(v):
    return _cq().Workplane("XZ").circle(v["diameter"] / 2).extrude(v["span"] / 2, both=True)


def _sphere_build(v):
    return _cq().Workplane("XY").sphere(v["diameter"] / 2)


def _bend_build(v):
    cq = _cq()
    D, R, L1, L2 = v["diameter"], v["bend_ratio"] * v["diameter"], v["inlet_length"], v["outlet_length"]
    path = cq.Workplane("XY").moveTo(-L1, 0).lineTo(0, 0).radiusArc((R, R), -R).lineTo(R, R + L2)
    return cq.Workplane("YZ").workplane(offset=-L1).circle(D / 2).sweep(path, transition="round")


def _bend_volume(v):
    D, R = v["diameter"], v["bend_ratio"] * v["diameter"]
    return math.pi * D ** 2 / 4 * (v["inlet_length"] + v["outlet_length"] + math.pi * R / 2)   # Pappus


def _bfs_profile(v):
    h, er, li, lo = v["step_height"], v["expansion_ratio"], v["inlet_length"], v["outlet_length"]
    H_out = h * er / (er - 1)            # ER = H_out / H_in and H_out = H_in + h
    return [(-li, h), (0.0, h), (0.0, 0.0), (lo, 0.0), (lo, H_out), (-li, H_out)]


def _bfs_build(v):
    return _cq().Workplane("XZ").polyline(_bfs_profile(v)).close().extrude(v["span"] / 2, both=True)


def _bfs_volume(v):
    return polygon_area(np.array(_bfs_profile(v))) * v["span"]


def _plate_build(v):
    cq = _cq()
    plate = cq.Workplane("XY").box(v["length"], v["width"], v["thickness"])
    return plate.faces(">Z").workplane().hole(v["hole_diameter"])


def _plate_volume(v):
    return (v["length"] * v["width"] - math.pi * v["hole_diameter"] ** 2 / 4) * v["thickness"]


def _beam_build(v):
    return _cq().Workplane("XY").box(v["length"], v["width"], v["height"]).translate((v["length"] / 2, 0, 0))


def _heatsink_build(v):
    cq = _cq()
    W, Lb, tb = v["base_width"], v["base_length"], v["base_thickness"]
    n, tf, hf = int(round(v["fin_count"])), v["fin_thickness"], v["fin_height"]
    sink = cq.Workplane("XY").box(Lb, W, tb, centered=(True, True, False))
    pitch = (W - tf) / (n - 1)
    for i in range(n):
        y = -W / 2 + tf / 2 + i * pitch
        sink = sink.union(cq.Workplane("XY").box(Lb, tf, hf, centered=(True, True, False)).translate((0, y, tb)))
    return sink


def _heatsink_volume(v):
    n = int(round(v["fin_count"]))
    return v["base_length"] * (v["base_width"] * v["base_thickness"] + n * v["fin_thickness"] * v["fin_height"])


def _heatsink_gap(v):
    n = int(round(v["fin_count"]))
    return (v["base_width"] - n * v["fin_thickness"]) / (n - 1)


_EXTRA_CHECKS = {
    "heat_sink": lambda v: ([] if _heatsink_gap(v) > 0 else ["fins overlap: fin_count x fin_thickness >= base_width"]),
    "plate_with_hole": lambda v: ([] if v["hole_diameter"] < 0.9 * v["width"] else
                                  ["the hole leaves less than 10 % of the width as ligament"]),
    "ahmed_body": lambda v: ([] if v["slant_length"] * math.sin(math.radians(v["slant_angle"])) < v["height"] else
                             ["the rear slant cuts through the whole height"]),
}


# ---------------------------------------------------------------- the catalogue

TEMPLATES: dict[str, Template] = {}


def _add(t: Template) -> None:
    TEMPLATES[t.key] = t


_add(Template(
    key="naca4_wing", title="NACA 4-digit wing section (extruded)",
    physics=("external_flow",),
    keywords=("airfoil", "aerofoil", "wing", "blade", "lift", "naca", "hydrofoil", "fan", "propeller", "aerofólio",
              "asa", "pá", "sustentação"),
    description="Straight extruded wing of a NACA 4-digit section; 2-D section results when meshed with one cell span.",
    params=(
        Param("chord", 1.0, "m", (0.1, 2.0), (0.01, 50.0), "sets the Reynolds number with the speed", "high"),
        Param("thickness", 12.0, "% chord", (6.0, 18.0), (1.0, 40.0), "drag and stall behaviour", "high"),
        Param("camber", 0.0, "% chord", (0.0, 4.0), (0.0, 9.5), "lift at zero angle", "high"),
        Param("camber_pos", 4.0, "tenths of chord", (2.0, 6.0), (1.0, 9.0), "where the camber peaks", "low"),
        Param("span", 0.1, "m", (0.05, 5.0), (0.001, 100.0), "3-D span (one cell for a 2-D run)", "low"),
    ),
    references=(
        Reference("Abbott & von Doenhoff, Theory of Wing Sections, Dover (1959)",
                  "section definition and measured lift, drag and moment of NACA 4-digit sections",
                  {"thickness": 12.0, "camber": 0.0}),
        Reference("Ladson, NASA TM-4074 (1988)",
                  "NACA 0012 lift and drag at Re = 3-6 million, a standard CFD validation set",
                  {"thickness": 12.0, "camber": 0.0, "chord": 0.6096}),
    ),
    quantities=("lift coefficient", "drag coefficient", "moment coefficient", "pressure distribution"),
    build=_naca_build, volume=_naca_volume,
    section=lambda v: ("body", naca4_points(v["camber"] / 100, v["camber_pos"] / 10, v["thickness"] / 100,
                                             v["chord"], n=160)),
    min_feature=lambda v: 1.1019 * (v["thickness"] / 100) ** 2 * v["chord"],          # leading-edge radius
    ref_length=lambda v: v["chord"],
))

_add(Template(
    key="ahmed_body", title="Ahmed body (simplified car)",
    physics=("external_flow",),
    keywords=("car", "vehicle", "automotive", "bluff", "truck", "bus", "wake", "carro", "veículo", "automotivo",
              "caminhão", "ônibus", "esteira", "ahmed"),
    description="Bluff body with rounded front and a rear slant; the reference case for vehicle aerodynamics. "
                "The four support legs of the wind-tunnel model are left out.",
    params=(
        Param("length", 1.044, "m", (0.5, 5.0), (0.1, 30.0), "Reynolds number (based on length)", "medium"),
        Param("width", 0.389, "m", (0.2, 2.0), (0.05, 5.0), "frontal area", "medium"),
        Param("height", 0.288, "m", (0.15, 2.0), (0.05, 5.0), "frontal area", "medium"),
        Param("slant_angle", 25.0, "deg", (0.0, 40.0), (0.0, 50.0), "rear separation regime: drag jumps near 30 deg",
              "high"),
        Param("slant_length", 0.222, "m", (0.1, 0.5), (0.0, 2.0), "extent of the rear slant", "medium"),
        Param("nose_radius", 0.1, "m", (0.05, 0.15), (0.0, 1.0), "front separation", "medium"),
        Param("clearance", 0.05, "m", (0.02, 0.2), (0.0, 1.0), "underbody flow and ground effect", "medium"),
    ),
    references=(
        Reference("Ahmed, Ramm & Faltin, SAE 840300 (1984)",
                  "drag and wake topology versus rear slant angle 0-40 deg at Re = 4.29 million (length based)",
                  {"length": 1.044, "width": 0.389, "height": 0.288, "slant_length": 0.222, "nose_radius": 0.1,
                   "clearance": 0.05}),
        Reference("Lienhart & Becker, SAE 2003-01-0656 (2003)",
                  "LDA velocity profiles behind the 25 and 35 deg Ahmed bodies, used for wake validation",
                  {"slant_angle": 25.0}),
    ),
    quantities=("drag coefficient", "lift coefficient", "wake length", "surface pressure"),
    build=_ahmed_build, volume=None,
    min_feature=lambda v: min(v["nose_radius"] if v["nose_radius"] > 0 else v["height"], v["clearance"]),
    ref_length=lambda v: v["length"],
    notes="Volume check: the rounded nose makes the exact volume formula long; the CAD volume is compared with the "
          "sharp-nose volume as an upper bound.",
))

_add(Template(
    key="cylinder_crossflow", title="Circular cylinder in cross-flow",
    physics=("external_flow", "thermal"),
    keywords=("cylinder", "pipe", "tube", "riser", "cable", "vortex", "shedding", "strouhal", "chimney", "pile",
              "cilindro", "tubo", "vórtice", "chaminé", "estaca"),
    description="Cylinder normal to the flow; vortex shedding, drag and heat transfer reference.",
    params=(
        Param("diameter", 0.1, "m", (0.01, 2.0), (0.0005, 20.0), "Reynolds number", "high"),
        Param("span", 0.4, "m", (0.1, 2.0), (0.001, 100.0), "3-D wake effects (spanwise length >= 4 D for LES)",
              "low"),
    ),
    references=(
        Reference("Williamson, Annu. Rev. Fluid Mech. 28:477-539 (1996)",
                  "wake regimes and Strouhal number versus Reynolds number (St close to 0.2 for 300 < Re < 2e5)", {}),
        Reference("Zdravkovich, Flow Around Circular Cylinders, vol. 1, Oxford (1997)",
                  "drag coefficient versus Reynolds number, including the drag crisis near Re = 2-5e5", {}),
    ),
    quantities=("drag coefficient", "Strouhal number", "lift fluctuation", "Nusselt number"),
    build=_cylinder_build, volume=lambda v: math.pi * v["diameter"] ** 2 / 4 * v["span"],
    section=lambda v: ("body", v["diameter"] / 2 * np.c_[np.cos(np.linspace(0, 2 * np.pi, 160, endpoint=False)),
                                                          np.sin(np.linspace(0, 2 * np.pi, 160, endpoint=False))]),
    min_feature=lambda v: v["diameter"], ref_length=lambda v: v["diameter"],
))

_add(Template(
    key="sphere", title="Sphere",
    physics=("external_flow", "thermal"),
    keywords=("sphere", "ball", "particle", "droplet", "bubble", "esfera", "bola", "partícula", "gota"),
    description="Sphere in a uniform stream; drag and heat or mass transfer reference.",
    params=(Param("diameter", 0.1, "m", (0.001, 1.0), (1e-6, 10.0), "Reynolds number", "high"),),
    references=(
        Reference("Schlichting & Gersten, Boundary-Layer Theory, 9th ed., Springer (2017)",
                  "sphere drag coefficient versus Reynolds number (Cd close to 0.47 for 1e3 < Re < 2e5)", {}),
        Reference("Stokes (1851)", "Cd = 24/Re for Re << 1, the creeping-flow limit", {}),
    ),
    quantities=("drag coefficient", "Nusselt number"),
    build=_sphere_build, volume=lambda v: math.pi * v["diameter"] ** 3 / 6,
    min_feature=lambda v: v["diameter"], ref_length=lambda v: v["diameter"],
))

_add(Template(
    key="pipe_bend_90", title="90-degree pipe bend (flow domain)",
    physics=("internal_flow",),
    keywords=("pipe", "bend", "elbow", "duct", "erosion", "pressure drop", "pneumatic", "slurry", "piping", "curva",
              "cotovelo", "duto", "tubulação", "erosão", "perda de carga"),
    description="Fluid volume of a circular pipe with a 90-degree bend and straight inlet and outlet runs.",
    params=(
        Param("diameter", 0.0762, "m", (0.025, 0.5), (0.002, 5.0), "Reynolds number and particle Stokes number", "high"),
        Param("bend_ratio", 1.5, "R/D", (1.0, 5.0), (0.6, 20.0), "secondary flow and impact angle on the wall", "high"),
        Param("inlet_length", 0.762, "m", (0.3, 3.0), (0.0, 100.0), "flow development before the bend (>= 10 D)",
              "medium"),
        Param("outlet_length", 0.381, "m", (0.2, 2.0), (0.0, 100.0), "recovery after the bend (>= 5 D)", "low"),
    ),
    references=(
        Reference("Vieira et al., Powder Technology 288:339-353 (2016)",
                  "erosion depth by ultrasonic thickness on a 76.2 mm, R/D = 1.5 elbow with air and sand",
                  {"diameter": 0.0762, "bend_ratio": 1.5}),
        Reference("Idelchik, Handbook of Hydraulic Resistance, 4th ed. (2007)",
                  "pressure-loss coefficient of smooth bends versus R/D and Reynolds number", {}),
    ),
    quantities=("pressure drop", "loss coefficient", "wall shear", "erosion rate"),
    build=_bend_build, volume=_bend_volume,
    derive=lambda v: {**v, "inlet_length": max(v["inlet_length"], 10 * v["diameter"]),
                      "outlet_length": max(v["outlet_length"], 5 * v["diameter"])},
    ports={"inlet": "<X", "outlet": ">Y"},
    inside=lambda v: (-v["inlet_length"] / 2, 0.0, 0.0),
    min_feature=lambda v: v["diameter"], ref_length=lambda v: v["diameter"],
))

_add(Template(
    key="backward_facing_step", title="Backward-facing step (channel)",
    physics=("internal_flow", "thermal"),
    keywords=("step", "expansion", "separation", "reattachment", "diffuser", "recirculation", "degrau", "expansão",
              "separação", "recolamento", "recirculação"),
    description="Channel with a sudden expansion; separation and reattachment reference.",
    params=(
        Param("step_height", 0.0127, "m", (0.005, 0.1), (0.0005, 2.0), "Reynolds number (step based)", "high"),
        Param("expansion_ratio", 1.125, "-", (1.1, 2.0), (1.01, 5.0), "strength of the adverse pressure gradient",
              "high"),
        Param("inlet_length", 0.0508, "m", (0.02, 0.5), (0.0, 20.0), "inflow development (>= 4 h)", "low"),
        Param("outlet_length", 0.381, "m", (0.2, 1.0), (0.01, 50.0), "must exceed the reattachment length (>= 30 h)",
              "medium"),
        Param("span", 0.0127, "m", (0.005, 0.2), (0.0001, 10.0), "one cell for 2-D runs", "low"),
    ),
    references=(
        Reference("Driver & Seegmiller, AIAA J. 23(2):163-171 (1985)",
                  "turbulent reattachment length close to 6.26 h at ER = 1.125, Re_h = 36,000",
                  {"step_height": 0.0127, "expansion_ratio": 1.125}),
        Reference("Armaly et al., J. Fluid Mech. 127:473-496 (1983)",
                  "laminar to turbulent reattachment lengths at ER = 1.94", {"expansion_ratio": 1.94}),
    ),
    quantities=("reattachment length", "skin friction", "pressure recovery"),
    build=_bfs_build, volume=_bfs_volume,
    derive=lambda v: {**v, "inlet_length": max(v["inlet_length"], 4 * v["step_height"]),
                      "outlet_length": max(v["outlet_length"], 30 * v["step_height"])},
    ports={"inlet": "<X", "outlet": ">X", "sides": "|Y"},
    inside=lambda v: (v["outlet_length"] / 2, 0.0, v["step_height"] / 2),
    section=lambda v: ("domain", np.array(_bfs_profile(v))),
    min_feature=lambda v: v["step_height"], ref_length=lambda v: v["step_height"],
))

_add(Template(
    key="plate_with_hole", title="Plate with a central hole (tension)",
    physics=("structural",),
    keywords=("hole", "plate", "stress concentration", "notch", "bracket", "bolt hole", "fatigue", "furo", "placa",
              "concentração de tensão", "entalhe", "fadiga", "suporte"),
    description="Flat plate with a central circular hole under uniaxial tension; stress-concentration reference.",
    params=(
        Param("length", 0.2, "m", (0.05, 1.0), (0.005, 10.0), "distance to the loaded ends (>= 2 W)", "low"),
        Param("width", 0.05, "m", (0.02, 0.5), (0.002, 5.0), "net section with the hole", "high"),
        Param("hole_diameter", 0.01, "m", (0.002, 0.03), (0.0002, 4.5), "d/W sets the stress concentration", "high"),
        Param("thickness", 0.005, "m", (0.001, 0.02), (0.0001, 1.0), "plane stress when thin", "low"),
    ),
    references=(
        Reference("Kirsch (1898); Timoshenko & Goodier, Theory of Elasticity, 3rd ed. (1970)",
                  "Kt = 3 for a hole in an infinite plate in tension", {}),
        Reference("Pilkey & Pilkey, Peterson's Stress Concentration Factors, 3rd ed., Wiley (2008)",
                  "Kt versus d/W for a finite-width plate (net-section; Heywood form Kt_n = 2 + (1 - d/W)^3)", {}),
    ),
    quantities=("peak stress", "stress concentration factor", "displacement"),
    build=_plate_build, volume=_plate_volume,
    min_feature=lambda v: min(v["hole_diameter"], v["thickness"]), ref_length=lambda v: v["hole_diameter"],
))

_add(Template(
    key="cantilever_beam", title="Cantilever beam (rectangular section)",
    physics=("structural",),
    keywords=("beam", "cantilever", "deflection", "bending", "bar", "shaft", "arm", "viga", "balanço", "flexão",
              "deflexão", "barra", "eixo", "braço"),
    description="Rectangular beam clamped at one end; deflection, stress and natural-frequency reference.",
    params=(
        Param("length", 1.0, "m", (0.1, 5.0), (0.01, 100.0), "deflection grows with L^3", "high"),
        Param("width", 0.05, "m", (0.01, 0.3), (0.001, 10.0), "second moment of area", "medium"),
        Param("height", 0.1, "m", (0.01, 0.5), (0.001, 10.0), "second moment of area (h^3)", "high"),
    ),
    references=(
        Reference("Timoshenko & Gere, Mechanics of Materials (1972); Euler-Bernoulli theory",
                  "tip deflection P L^3 / (3 E I), root stress 6 P L / (b h^2), first frequency "
                  "(1.875^2 / L^2) sqrt(E I / (rho A))", {}),
    ),
    quantities=("tip deflection", "maximum stress", "natural frequencies"),
    build=_beam_build, volume=lambda v: v["length"] * v["width"] * v["height"],
    min_feature=lambda v: min(v["width"], v["height"]), ref_length=lambda v: v["height"],
))

_add(Template(
    key="heat_sink", title="Plate-fin heat sink",
    physics=("thermal", "internal_flow"),
    keywords=("heat sink", "fin", "cooling", "electronics", "thermal", "junction", "dissipador", "aleta",
              "resfriamento", "eletrônica", "térmico", "junção", "convection", "convecção"),
    description="Base plate with straight rectangular fins; conduction plus forced or natural convection.",
    params=(
        Param("base_length", 0.05, "m", (0.02, 0.2), (0.005, 1.0), "flow length along the fins", "medium"),
        Param("base_width", 0.05, "m", (0.02, 0.2), (0.005, 1.0), "number of channels that fit", "medium"),
        Param("base_thickness", 0.004, "m", (0.002, 0.01), (0.0005, 0.05), "heat spreading", "medium"),
        Param("fin_count", 10, "-", (5, 25), (2, 200), "area versus channel blockage", "high"),
        Param("fin_thickness", 0.0012, "m", (0.0005, 0.003), (0.0001, 0.02), "fin efficiency", "medium"),
        Param("fin_height", 0.02, "m", (0.01, 0.05), (0.001, 0.3), "area and fin efficiency", "high"),
    ),
    references=(
        Reference("Teertstra, Yovanovich & Culham, IEEE Trans. Comp. Packag. Technol. 23(4):666-676 (2000)",
                  "analytical forced-convection model for plate-fin heat sinks, validated against measurements", {}),
        Reference("Incropera et al., Fundamentals of Heat and Mass Transfer, 7th ed. (2011)",
                  "fin efficiency tanh(mL)/(mL), the closed-form check of a single fin", {}),
    ),
    quantities=("thermal resistance", "base temperature", "pressure drop"),
    build=_heatsink_build, volume=_heatsink_volume,
    min_feature=lambda v: min(v["fin_thickness"], _heatsink_gap(v)), ref_length=lambda v: _heatsink_gap(v),
))
