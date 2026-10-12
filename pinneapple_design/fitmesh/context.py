"""Read a problem description into the facts the geometry and mesh choices depend on, and ask for what is missing.

The parser is deliberately simple (keywords and numbers with units) so that it is predictable and testable; an
optional ``llm`` callable can fill the same fields from free text, and every answer the user gives later overrides
both. Nothing is guessed silently: each field records where it came from (text, answer, default).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

PHYSICS_WORDS = {
    "external_flow": ("drag", "lift", "aerodynamic", "aerodinâmic", "wind", "vento", "around", "em torno", "wake",
                      "esteira", "arrasto", "sustentação", "airfoil", "aerofólio", "vehicle", "veículo", "car",
                      "carro", "wing", "asa", "cylinder", "cilindro", "sphere", "esfera"),
    "internal_flow": ("pipe", "tubo", "tubulação", "duct", "duto", "channel", "canal", "bend", "curva", "elbow",
                      "cotovelo", "pressure drop", "perda de carga", "valve", "válvula", "step", "degrau", "erosion",
                      "erosão", "pneumatic", "pneumático"),
    "structural": ("stress", "tensão", "strain", "deformação", "deflection", "deflexão", "load", "carga", "fatigue",
                   "fadiga", "beam", "viga", "plate", "placa", "bracket", "suporte", "stiffness", "rigidez", "fea",
                   "elementos finitos", "von mises"),
    "thermal": ("heat", "calor", "temperature", "temperatura", "thermal", "térmic", "cooling", "resfriamento",
                "heat sink", "dissipador", "junction", "junção", "convection", "convecção", "conduction", "condução"),
}
ACCURACY_WORDS = {
    "screening": ("screening", "triagem", "quick", "rápid", "rough", "estimate", "estimativa", "preliminar",
                  "preliminary", "ordem de grandeza", "order of magnitude"),
    "verification": ("verification", "verificação", "certif", "independence", "independência", "gci", "validation",
                     "validação", "sign-off", "aprovação", "high fidelity", "alta fidelidade"),
}
QUANTITY_WORDS = {
    "drag coefficient": ("drag", "arrasto", "cd"), "lift coefficient": ("lift", "sustentação", "cl"),
    "pressure drop": ("pressure drop", "perda de carga", "pressure loss"),
    "wall shear": ("shear", "cisalhamento", "erosion", "erosão", "wear", "desgaste"),
    "reattachment length": ("reattachment", "recolamento", "recirculation", "recirculação"),
    "peak stress": ("stress", "tensão", "von mises", "fatigue", "fadiga"),
    "tip deflection": ("deflection", "deflexão", "displacement", "deslocamento"),
    "natural frequencies": ("frequency", "frequência", "modal", "vibration", "vibração"),
    "thermal resistance": ("thermal resistance", "resistência térmica", "junction", "junção", "temperature",
                           "temperatura"),
    "Strouhal number": ("strouhal", "shedding", "desprendimento"),
}
FLUIDS = {   # density [kg/m^3], dynamic viscosity [Pa s] at about 20 C
    "air": (1.204, 1.81e-5), "water": (998.0, 1.0e-3), "oil": (870.0, 0.1), "steam": (0.6, 1.2e-5),
}
FLUID_WORDS = {"air": ("air", "ar", "gas", "gás", "wind", "vento"), "water": ("water", "água", "agua", "liquid"),
               "oil": ("oil", "óleo", "oleo"), "steam": ("steam", "vapor")}
MATERIALS = {   # E [Pa], nu, rho [kg/m^3], k [W/m K]
    "steel": (210e9, 0.3, 7850.0, 45.0), "aluminium": (70e9, 0.33, 2700.0, 200.0), "copper": (117e9, 0.34, 8960.0, 390.0),
    "titanium": (114e9, 0.34, 4430.0, 7.0),
}
MATERIAL_WORDS = {"steel": ("steel", "aço", "aco"), "aluminium": ("aluminium", "aluminum", "alumínio", "aluminio"),
                  "copper": ("copper", "cobre"), "titanium": ("titanium", "titânio")}
_LEN = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3, "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "polegadas": 0.0254, "pol": 0.0254, "\"": 0.0254}
_SPEED = {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "knots": 0.514444, "nós": 0.514444}


@dataclass
class Fact:
    value: Any
    source: str              # "text", "answer", "default", "llm"


@dataclass
class ProblemContext:
    text: str
    facts: dict[str, Fact] = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def get(self, key: str, default=None):
        f = self.facts.get(key)
        return f.value if f is not None else default

    def set(self, key: str, value, source: str = "answer") -> None:
        self.facts[key] = Fact(value, source)

    @property
    def physics(self) -> list[str]:
        return list(self.get("physics", []))

    @property
    def accuracy(self) -> str:
        return self.get("accuracy", "design")

    def fluid_props(self):
        return FLUIDS.get(self.get("fluid", "air"), FLUIDS["air"])

    def reynolds(self, length: float) -> float | None:
        if self.get("reynolds") is not None:
            return float(self.get("reynolds"))
        U = self.get("speed")
        if U is None or length is None:
            return None
        rho, mu = self.fluid_props()
        return rho * float(U) * float(length) / mu

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "facts": {k: asdict(v) for k, v in self.facts.items()}}

    # ------------------------------------------------------------------ what is still missing
    def questions(self) -> list[dict[str, Any]]:
        """Questions for the facts the geometry and mesh choices need and the text did not give."""
        q = []

        def ask(key, text, options=None, unit=None, why=""):
            if key not in self.facts or self.facts[key].source == "default":
                q.append({"key": key, "question": text, "options": options, "unit": unit, "why": why,
                          "current": self.get(key)})
        if not self.physics:
            ask("physics", "Which physics is the study about?", list(PHYSICS_WORDS), why="chooses the templates")
        flow = any(p in self.physics for p in ("external_flow", "internal_flow"))
        if flow:
            ask("fluid", "Which fluid?", list(FLUIDS), why="density and viscosity for the Reynolds number")
            if self.get("reynolds") is None:
                ask("speed", "Flow speed?", unit="m/s", why="Reynolds number, first-cell height (y+)")
        if "structural" in self.physics or "thermal" in self.physics:
            ask("material", "Material?", list(MATERIALS), why="stiffness and conductivity of the solid")
        if "structural" in self.physics:
            ask("load", "Load (force in N, or pressure in Pa)?", unit="N", why="checks against the closed form")
        ask("length", "Characteristic size of the part (chord, length, diameter)?", unit="m",
            why="scales the template and the mesh")
        ask("accuracy", "How will the result be used?", ["screening", "design", "verification"],
            why="screening: ~10 % is enough; design: a few %; verification: three meshes and a grid-convergence "
                "index")
        ask("quantities", "Which quantities matter most?", sorted(QUANTITY_WORDS),
            why="where the mesh must be finest")
        return q


def _has(text: str, words) -> int:
    """Words found in ``text``: short words (<= 3 letters) whole, longer ones as a stem at a word start."""
    n = 0
    for w in words:
        w = w.strip()
        tail = r"(?![a-zà-ú])" if len(w) <= 3 else ""
        n += bool(re.search(r"(?<![a-zà-ú])" + re.escape(w) + tail, text))
    return n


def _number(s: str) -> float:
    return float(s.replace(",", "."))


def parse(text: str, llm: Callable[[str], dict[str, Any]] | None = None) -> ProblemContext:
    """Facts from a free-text problem description (English or Portuguese)."""
    t = " " + text.lower() + " "
    ctx = ProblemContext(text=text)
    phys = [p for p, words in PHYSICS_WORDS.items() if _has(t, words)]
    if phys:
        ctx.set("physics", phys, "text")
    for fl, words in FLUID_WORDS.items():
        if _has(t, words):
            ctx.set("fluid", fl, "text")
            break
    for mat, words in MATERIAL_WORDS.items():
        if _has(t, words):
            ctx.set("material", mat, "text")
            break
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(m/s|km/h|mph|knots|nós)", t)
    if m:
        ctx.set("speed", _number(m.group(1)) * _SPEED[m.group(2)], "text")
    m = re.search(r"\bre(?:ynolds)?\s*(?:number)?\s*[=≈~:]?\s*(\d+(?:[.,]\d+)?(?:\s*[e×x]\s*\^?\s*[+-]?\d+)?)", t)
    if m:
        raw = re.sub(r"\s", "", m.group(1)).replace("×10^", "e").replace("x10^", "e").replace("×", "e")
        try:
            ctx.set("reynolds", float(raw.replace(",", ".")), "text")
        except ValueError:
            pass
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|km|inches|inch|polegadas|pol|in|m|\")(?![a-zà-ú/])", t)
    if m:
        ctx.set("length", _number(m.group(1)) * _LEN[m.group(2)], "text")
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(kn|n)\b", t)
    if m:
        ctx.set("load", _number(m.group(1)) * (1e3 if m.group(2) == "kn" else 1.0), "text")
    acc = next((a for a, words in ACCURACY_WORDS.items() if _has(t, words)), None)
    ctx.set("accuracy", acc or "design", "text" if acc else "default")
    qs = [q for q, words in QUANTITY_WORDS.items() if _has(t, words)]
    if qs:
        ctx.set("quantities", qs, "text")
    if llm is not None:
        for k, v in (llm(text) or {}).items():
            if k not in ctx.facts or ctx.facts[k].source == "default":
                ctx.set(k, v, "llm")
    return ctx


def wall_unit_height(y_plus: float, speed: float, length: float, nu: float, re: float, internal: bool = False
                     ) -> dict[str, float]:
    """First-cell height for a target y+ from flat-plate skin friction at the reference length.

    Turbulent (Re > 5e5): Cf = 0.026 Re^-1/7 (White); laminar: Cf = 0.664 Re^-1/2 (Blasius, local); internal flows
    use the pipe friction factor on the diameter (Re > 2300 turbulent). Returns the
    height, the friction velocity, Cf and the boundary-layer thickness at the reference length (0.37 L Re^-1/5
    turbulent, 5 L Re^-1/2 laminar)."""
    if internal:      # fully developed pipe flow on the diameter: Blasius f = 0.316 Re^-1/4 (turbulent), 64/Re laminar
        turbulent = re > 2300
        cf = (0.316 * re ** -0.25 if turbulent else 64 / re) / 4
        u_tau = speed * math.sqrt(cf / 2)
        return {"y1": y_plus * nu / u_tau, "u_tau": u_tau, "cf": cf, "delta": length / 2, "turbulent": turbulent}
    turbulent = re > 5e5
    cf = 0.026 * re ** (-1 / 7) if turbulent else 0.664 * re ** -0.5
    u_tau = speed * math.sqrt(cf / 2)
    delta = 0.37 * length * re ** -0.2 if turbulent else 5 * length * re ** -0.5
    return {"y1": y_plus * nu / u_tau, "u_tau": u_tau, "cf": cf, "delta": delta, "turbulent": turbulent}
