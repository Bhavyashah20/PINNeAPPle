"""Geometry library: realistic parametric 3-D geometries for Physics-AI, with the code that regenerates them and
ready-made files.

    import pinneapple as pp
    pp.geometries.list()                                   # every entry
    pp.geometries.info("wind_turbine_nrel5mw")             # dimensions, parameters with ranges, sources, uses
    g = pp.geometries.build("wind_turbine_nrel5mw", pitch_deg=5.0)   # -> Geometry (closed bodies, materials)
    g.export("turbine.stl"); g.export("turbine.glb"); g.export("turbine.step")
    g.render("turbine.png")                                # thumbnail (software renderer)
    pp.geometries.load("airliner_a320", format="glb")      # path to the pre-generated file (repo, cache or Hub)

Every entry is in metres in a documented frame, made of closed, outward-oriented bodies, at two levels of detail
(``lod="light"``: committed, ~20k triangles; ``"fine"``: on the Hugging Face Hub). ``build`` with the default
parameters reproduces the stored files (tested). ``Geometry.solid()`` fuses the bodies into one watertight surface
for meshing (``manifold3d``). See ``docs/guides/geometry_library.md``.
"""
from __future__ import annotations

import builtins
import builtins as _builtins
from typing import Any

from .core import (
    FRAME_VEHICLE,
    LODS,
    Body,
    Geometry,
    GeometrySpec,
    Param,
    Source,
    build,
    get_spec,
    info,
    names,
    register,
    summary_table,
)
from .store import HUB_REPO, load


def list(category: str | None = None) -> builtins.list[str]:  # noqa: A001 - the documented API name
    """Names of the library geometries (optionally of one ``category``: aircraft, ground_vehicle, rotorcraft,
    energy, marine, space, thermal, turbomachinery, civil, biomedical)."""
    out = names()
    if category is not None:
        out = [n for n in out if get_spec(n).category == category]
    return out


def categories() -> dict[str, builtins.list[str]]:
    cats: dict[str, builtins.list[str]] = {}
    for n in names():
        cats.setdefault(get_spec(n).category, []).append(n)
    return cats


def export(name: str, path: str, fmt: str | None = None, lod: str = "light", **params) -> str:
    """Build ``name`` with ``params`` and write it (format from ``fmt`` or the extension)."""
    return build(name, lod=lod, **params).export(path, fmt)


def render(name: str, path: str | None = None, lod: str = "light", size=(960, 640), **kw: Any):
    """Thumbnail of ``name`` with its default parameters (``kw``: azimuth, elevation, ... and physical
    parameters)."""
    spec = get_spec(name)
    pnames = {p.name for p in spec.params}
    params = {k: v for k, v in kw.items() if k in pnames}
    view = {k: v for k, v in kw.items() if k not in pnames}
    return build(name, lod=lod, **params).render(path, size=size, **view)


def thumbnail(name: str):
    """The committed thumbnail PNG (path), rendering it if absent."""
    from .store import thumbnail_path
    p = thumbnail_path(name)
    if p.exists():
        return p
    import tempfile
    from pathlib import Path
    out = Path(tempfile.mkdtemp()) / f"{name}.png"
    render(name, str(out))
    return out


__all__ = ["list", "categories", "info", "build", "load", "export", "render", "thumbnail", "register", "Geometry",
           "GeometrySpec", "Body", "Param", "Source", "LODS", "FRAME_VEHICLE", "HUB_REPO", "summary_table",
           "get_spec"]
del _builtins
