"""Checks shared by results and models: finite values, bounds, physical admissibility, units, references."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .. import _reuse
from ..config import DEFAULT_TOLERANCES, Config
from ..result import Location, Outcome
from ..rules import RULES

__all__ = ["Ctx", "check_values", "check_units", "rel_l2", "fmt"]


def fmt(x: float) -> str:
    if x is None:
        return "n/a"
    if not np.isfinite(x):
        return str(x)
    a = abs(x)
    return f"{x:.3g}" if 1e-3 <= a < 1e4 or a == 0 else f"{x:.2e}"


class Ctx:
    """Where outcomes are collected for one target."""

    def __init__(self, cfg: Config, target, enabled: set[str], outcomes: list[Outcome]):
        self.cfg, self.target, self.enabled, self.outcomes = cfg, target, enabled, outcomes
        self.ran: set[str] = set()
        self.notes: list[str] = []

    def note(self, msg: str) -> None:
        """Something that could not be checked (reported, never a violation)."""
        self.notes.append(f"{self.target.name if self.target is not None else 'integrity'}: {msg}")

    def on(self, rule: str) -> bool:
        return rule in self.enabled

    def default_location(self) -> Location:
        t = self.target
        if t is not None and t.declared and self.cfg.file is not None:
            return Location(self.cfg.rel(self.cfg.file), t.line)
        p = Path(str(t.path).rsplit(":", 1)[0]) if t is not None else Path(".")
        if p.is_dir():
            for c in ("system/controlDict", "constant/polyMesh/boundary"):
                if (p / c).is_file():
                    return Location(self.cfg.rel(p / c), 1)
        return Location(self.cfg.rel(p), 1)

    def file_location(self, path: Path | None, needle: str | None = None, after: str | None = None,
                      then: str | None = None) -> Location:
        """A line in a text file: the first ``needle`` (a patch name, ``internalField``) after ``after``, moved to the
        ``then`` entry of the block that follows when there is one (the ``value`` of a patch). The target
        declaration for binary files."""
        if path is None or not Path(path).is_file():
            return self.default_location()
        if self.target.declared and Path(path).suffix.lower() in (".npz", ".h5", ".hdf5", ".vtu", ".pt", ".pth"):
            return self.default_location()
        line = 1
        if needle:
            try:
                text = Path(path).read_text(encoding="latin-1")
            except OSError:
                text = ""
            start = max(text.find(after) if after else 0, 0)
            m = re.compile(r'(?:^|\s)"?(' + re.escape(needle) + r')"?(?=\s|$|\{)', re.M).search(text, start)
            if m:
                pos = m.start(1)
                if then:
                    end = text.find("}", pos)
                    t = re.compile(r"\b" + re.escape(then) + r"\b").search(text, pos, end if end > 0 else len(text))
                    pos = t.start() if t else pos
                line = text.count("\n", 0, pos) + 1
        return Location(self.cfg.rel(path), line)

    def emit(self, rule: str, subject: str, passed: bool, message: str, location: Location | None = None,
             value: Any = None, threshold: Any = None, hint: str = "") -> None:
        if rule not in self.enabled:
            return
        self.ran.add(rule)
        sev = self.cfg.severity.get(rule, RULES[rule].severity)
        self.outcomes.append(Outcome(rule, self.target.name if self.target is not None else "integrity", subject,
                                     bool(passed), message, location or self.default_location(), value, threshold,
                                     sev, hint=hint))

    def tol(self, key: str) -> float:
        return self.target.tol(key) if self.target is not None else DEFAULT_TOLERANCES[key]


# ---------------------------------------------------------------------------------------------- values
_FRACTION = re.compile(r"^(alpha(\.\w+|\d)?|Y_?\w*|vof|vf|phase_?fraction|volume_?fraction|mass_?fraction)$", re.I)
_NONNEG = re.compile(r"^(k|epsilon|omega|nut|nuTilda|alphat|mut|tke|tdr|kinetic_energy)$")
_DENSITY = re.compile(r"^(rho|density)$", re.I)
_TEMPERATURE = re.compile(r"^(T|temp|temperature)$", re.I)


def admissible(name: str, unit: str | None) -> tuple[float | None, float | None, str] | None:
    """Hard physical limits for a field name (and unit): (low, high, what)."""
    if _FRACTION.match(name) and not name.lower().startswith("alphat"):
        return 0.0, 1.0, "a fraction"
    if _NONNEG.match(name):
        return 0.0, None, "non-negative by definition"
    if _DENSITY.match(name):
        return 0.0, None, "a density"
    if _TEMPERATURE.match(name) and unit:
        U = _reuse.units()
        u = U.try_parse_unit(unit)
        if u is not None and u.dim == U.parse_unit("K").dim:
            zero = float(U.parse_unit("K").to_si(0.0))
            low = float(u.from_si(zero))                                # absolute zero in the field's unit
            return low, None, "an absolute temperature" + ("" if abs(low) < 1e-12 else f" (0 K = {fmt(low)} {unit})")
    return None


def check_values(ctx: Ctx, name: str, v: np.ndarray, unit: str | None, loc: Location, where=None) -> None:
    """PC403 non-finite, PC401 declared bounds, PC402 admissibility, for one field."""
    v = np.asarray(v, float)
    finite = np.isfinite(v)
    nbad = int((~finite).sum())
    ctx.emit("PC403", name, nbad == 0,
             f"{name}: {nbad} of {v.size} values are NaN or infinite" if nbad else f"{name}: all values finite",
             loc, value=nbad, threshold=0)
    if not finite.any():
        return
    vf = v[finite]
    b = ctx.target.bounds.get(name) if ctx.target is not None else None
    if b is not None:
        lo = None if b[0] is None or b[0] is False or b[0] in ("-inf", "") else float(b[0])
        hi = None if b[1] is None or b[1] is False or b[1] in ("inf", "+inf", "") else float(b[1])
        _bound(ctx, "PC401", name, v, vf, lo, hi, "declared bounds", 0.0, loc, where)
    adm = admissible(name, unit)
    if adm is not None and b is None:
        slack = ctx.tol("fraction-slack") if adm[2] == "a fraction" else 0.0
        _bound(ctx, "PC402", name, v, vf, adm[0], adm[1], adm[2], slack, loc, where)


def _bound(ctx, rule, name, v, vf, lo, hi, what, slack, loc, where):
    below = (v < lo - slack) if lo is not None else np.zeros(v.shape, bool)
    above = (v > hi + slack) if hi is not None else np.zeros(v.shape, bool)
    out = (below | above) & np.isfinite(v)
    rng = f"[{fmt(lo) if lo is not None else '-inf'}, {fmt(hi) if hi is not None else 'inf'}]"
    n = int(out.sum())
    if n:
        excess = np.full(v.shape, -np.inf)
        if lo is not None:
            excess = np.maximum(excess, lo - v)
        if hi is not None:
            excess = np.maximum(excess, v - hi)
        excess[~np.isfinite(v)] = -np.inf
        i = int(np.argmax(excess.ravel()))
        per = v.size // max(v.shape[0], 1) if v.ndim > 1 else 1
        at = f" (worst {fmt(float(v.flat[i]))}" + (f" at {where(i // per)}" if where else "") + ")"
        msg = f"{name}: {n} of {v.size} values outside {rng} ({what}); min {fmt(float(vf.min()))}, " \
              f"max {fmt(float(vf.max()))}{at}"
    else:
        msg = f"{name} within {rng} ({what})"
    ctx.emit(rule, name, n == 0, msg, loc, value=[float(vf.min()), float(vf.max())], threshold=[lo, hi])


# ---------------------------------------------------------------------------------------------- units
_PLAUSIBLE = {"T": "temperature", "temp": "temperature", "temperature": "temperature", "p": "pressure",
              "rho": "density", "density": "density", "U": "velocity", "u": "velocity", "nu": "kinematic_viscosity"}


def check_units(ctx: Ctx, name: str, file_unit: str | None, declared: str | None, values: np.ndarray | None,
                loc: Location, used: bool) -> None:
    U = _reuse.units()
    if declared and file_unit:
        a, b = U.try_parse_unit(file_unit), U.try_parse_unit(declared)
        if a is not None and b is not None:
            same = a.dim == b.dim and abs(a.factor / b.factor - 1) < 1e-9 and abs(a.offset - b.offset) < 1e-9
            if a.dim != b.dim:
                msg = (f"{name}: the file says {file_unit} ({U.dim_str(a.dim)}), the configuration says {declared} "
                       f"({U.dim_str(b.dim)})")
            elif not same:
                msg = f"{name}: the file is in {file_unit}, the configuration declares {declared} (same dimension, " \
                      f"different scale)"
            else:
                msg = f"{name}: {file_unit} as declared"
            ctx.emit("PC501", name, same, msg, loc, value=file_unit, threshold=declared)
    unit = declared or file_unit
    if used and not unit:
        ctx.emit("PC502", name, False, f"{name} is used by a physics check but has no unit in the file or the "
                                       f"configuration", loc,
                 hint=f'Declare it under the target: units = {{ {name} = "<unit>" }}.')
    elif used:
        ctx.emit("PC502", name, True, f"{name} in {unit}", loc)
    q = _PLAUSIBLE.get(name)
    if q and unit and values is not None:
        u = U.try_parse_unit(unit)
        Q = U.QUANTITIES.get(q)
        if u is None or Q is None or u.dim != Q.dim or Q.typical == (None, None):
            return
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if v.ndim > 1 or v.size == 0:
            v = np.linalg.norm(np.atleast_2d(v), axis=-1) if v.ndim > 1 else v
        if v.size == 0:
            return
        si = np.asarray(u.to_si(v), float)
        lo, hi = Q.typical
        inside = np.ones(si.shape, bool)
        if lo is not None:
            inside &= si >= lo
        if hi is not None:
            inside &= si <= hi
        frac = 1 - float(inside.mean())
        typ = f"{fmt(lo) if lo is not None else '-inf'} .. {fmt(hi) if hi is not None else 'inf'} {Q.si_unit}"
        ctx.emit("PC503", name, frac < 0.5,
                 f"{name}: {frac:.0%} of the values are outside the typical range of a {q.replace('_', ' ')} "
                 f"({typ}) when read as {unit}; median {fmt(float(np.median(v)))} {unit}" if frac >= 0.5 else
                 f"{name}: magnitudes plausible for {unit}", loc, value=frac, threshold=0.5)


# ---------------------------------------------------------------------------------------------- regression
def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    nb = float(np.linalg.norm(b[m]))
    d = float(np.linalg.norm(a[m] - b[m]))
    if nb <= 1e-300:
        return 0.0 if d <= 1e-300 else float("inf")
    return d / nb
