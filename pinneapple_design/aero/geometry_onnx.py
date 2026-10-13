"""The CST airfoil geometry of :mod:`pinneapple_design.aero.geometry` as ONNX graphs.

Two graphs, both float64 and both reproducing the NumPy functions to round-off:

* ``cst_geometry.onnx``: shape ``(n, 6)`` -> ``upper`` / ``lower`` (y on a cosine-spaced x), ``props`` (the
  :func:`~pinneapple_design.aero.geometry.properties` values in :data:`PROPS` order), ``validity`` (smallest interior
  thickness and largest |d2y/dx2| of the upper surface, the two quantities :func:`~...geometry.valid` tests) and
  ``handles`` (upper and lower y at :data:`~...geometry.HANDLE_X`).
* ``cst_from_handles.onnx``: handle heights ``(n, 6)`` (3 upper, 3 lower) -> shape ``(n, 6)``.

A browser (or any ONNX runtime) can then draw, measure and edit CST airfoils without a re-implementation of the
geometry. :func:`torch_geometry` is also used inside the zoo airfoil model's query checks.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry as g

PROPS = ("thickness", "thickness_at", "camber", "camber_at", "le_radius", "area", "min_thickness_aft", "spar_depth")
VALIDITY = ("min_interior_thickness", "max_abs_d2_upper")
VALID_LIMITS = {"min_interior_thickness": [">", 0.0, "surfaces cross"], "max_abs_d2_upper": ["<=", 40.0, "wavy surface"]}
N_OUTLINE = 121
OPSET = 17


def _x_grid() -> np.ndarray:
    return np.linspace(0, 1, 401)


def torch_geometry(n_outline: int = N_OUTLINE):
    """A ``torch.nn.Module``: shape (n, 6) -> dict of the geometry outputs (float64)."""
    import torch
    from torch import nn

    x = _x_grid()
    xo = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_outline)))

    def basis(xs):
        xs = np.clip(xs, 0.0, 1.0)
        return (np.sqrt(xs) * (1 - xs))[:, None] * g._bern(g.N_CST, xs)       # (m, 3)

    aft = np.nonzero(x >= 0.9)[0]
    spar = np.nonzero((x >= 0.2) & (x <= 0.35))[0]

    class Geometry(nn.Module):
        def __init__(self):
            super().__init__()
            T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
            self.register_buffer("A", T(basis(x).T))                            # (3, 401)
            self.register_buffer("Ao", T(basis(xo).T))
            self.register_buffer("Ah", T(basis(g.HANDLE_X).T))
            self.register_buffer("x", T(x))
            self.register_buffer("tex", T(x * g.TE_THICKNESS / 2))
            self.register_buffer("texo", T(np.clip(xo, 0, 1) * g.TE_THICKNESS / 2))
            self.register_buffer("texh", T(g.HANDLE_X * g.TE_THICKNESS / 2))
            self.dx = float(x[1] - x[0])
            self.aft = (int(aft[0]), int(aft[-1]) + 1)
            self.spar = (int(spar[0]), int(spar[-1]) + 1)

        def surfaces(self, s, A, te):
            T, M = s[:, 0:3], s[:, 3:6]
            return (M + T) @ A + te, (M - T) @ A - te

        def forward(self, s):
            yu, yl = self.surfaces(s, self.A, self.tex)
            t, c = yu - yl, (yu + yl) / 2
            i = torch.argmax(t, dim=1)
            k = torch.argmax(torch.abs(c), dim=1)
            thick = torch.gather(t, 1, i[:, None])[:, 0]
            camber = torch.gather(c, 1, k[:, None])[:, 0]
            area = ((t[:, 1:] + t[:, :-1]) * 0.5).sum(1) * self.dx
            aft_min = torch.amin(t[:, self.aft[0]:self.aft[1]], dim=1)
            spar_min = torch.amin(t[:, self.spar[0]:self.spar[1]], dim=1)
            props = torch.stack([thick, self.x[i], camber, self.x[k], s[:, 0] ** 2 / 2, area, aft_min, spar_min], 1)

            def grad(y):                                                       # numpy.gradient, uniform grid
                return torch.cat([(y[:, 1:2] - y[:, 0:1]) / self.dx, (y[:, 2:] - y[:, :-2]) / (2 * self.dx),
                                  (y[:, -1:] - y[:, -2:-1]) / self.dx], 1)
            d2 = grad(grad(yu))
            validity = torch.stack([torch.amin(t[:, 1:-1], dim=1), torch.amax(torch.abs(d2[:, 20:-5]), dim=1)], 1)
            up, lo = self.surfaces(s, self.Ao, self.texo)
            hu, hl = self.surfaces(s, self.Ah, self.texh)
            return {"props": props, "validity": validity, "upper": up, "lower": lo,
                    "handles": torch.cat([hu, hl], 1)}

    return Geometry()


def torch_from_handles():
    """A ``torch.nn.Module``: handle heights (n, 6) -> shape (n, 6) (float64), as :func:`shape_from_handles`."""
    import torch
    from torch import nn

    Ainv = np.linalg.inv(g._handle_matrix())
    te = g.HANDLE_X * g.TE_THICKNESS / 2

    class FromHandles(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("AinvT", torch.tensor(Ainv.T, dtype=torch.float64))
            self.register_buffer("te", torch.tensor(te, dtype=torch.float64))

        def forward(self, h):
            wu = (h[:, 0:3] - self.te) @ self.AinvT
            wl = (h[:, 3:6] + self.te) @ self.AinvT
            return torch.cat([(wu - wl) / 2, (wu + wl) / 2], 1)

    return FromHandles()


def _export(module, n_in: int, outputs: list[str]) -> bytes:
    import onnx
    import torch
    x = torch.zeros(3, n_in, dtype=torch.float64)
    buf = io.BytesIO()
    kw = dict(input_names=["x"], output_names=outputs, opset_version=OPSET,
              dynamic_axes={"x": {0: "n"}, **{o: {0: "n"} for o in outputs}}, do_constant_folding=True)

    class Ordered(torch.nn.Module):                                   # dict outputs -> fixed tuple order
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, s):
            out = self.m(s)
            return tuple(out[o] for o in outputs) if isinstance(out, dict) else out
    try:
        torch.onnx.export(Ordered(module).eval(), (x,), buf, dynamo=False, **kw)
    except TypeError:                                                  # torch < 2.5 has no dynamo flag
        torch.onnx.export(Ordered(module).eval(), (x,), buf, **kw)
    m = onnx.load_from_string(buf.getvalue())
    m.producer_name, m.producer_version = "pinneapple", "1"
    onnx.checker.check_model(m)
    return m.SerializeToString()


GEOMETRY_OUTPUTS = ["props", "validity", "upper", "lower", "handles"]


def export_onnx(directory: str | Path, *, presets: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``cst_geometry.onnx``, ``cst_from_handles.onnx`` and ``cst_geometry.json`` (grids, bounds, output
    layout, validity limits and the NACA presets as CST weights) into ``directory``. Returns the JSON content."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    geo = _export(torch_geometry(), 6, GEOMETRY_OUTPUTS)
    fh = _export(torch_from_handles(), 6, ["shape"])
    (d / "cst_geometry.onnx").write_bytes(geo)
    (d / "cst_from_handles.onnx").write_bytes(fh)
    codes = presets if presets is not None else {k: k.split()[-1] for k in g.REFERENCE}
    meta = {
        "params": list(g.PARAMS), "bounds": g.BOUNDS.tolist(), "te_thickness": g.TE_THICKNESS,
        "outline_x": (0.5 * (1 - np.cos(np.linspace(0, np.pi, N_OUTLINE)))).tolist(),
        "handle_x": g.HANDLE_X.tolist(), "props": list(PROPS), "validity": list(VALIDITY),
        "validity_limits": VALID_LIMITS,
        "presets": {name: {"shape": (w := g.naca4(code)).tolist(), "out_of_bounds": g.out_of_bounds(w)}
                    for name, code in codes.items()},
        "files": {"geometry": "cst_geometry.onnx", "from_handles": "cst_from_handles.onnx"},
        "source": "pinneapple_design.aero.geometry_onnx",
    }
    (d / "cst_geometry.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def reference(shapes: np.ndarray) -> dict[str, np.ndarray]:
    """The NumPy values the graphs must reproduce (used by the tests)."""
    S = np.atleast_2d(np.asarray(shapes, float))
    xo = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N_OUTLINE)))
    x = _x_grid()
    out: dict[str, list] = {k: [] for k in GEOMETRY_OUTPUTS}
    for s in S:
        p = g.properties(s)
        yu, yl = g.surfaces(s, x)
        d2 = np.gradient(np.gradient(yu, x), x)
        out["props"].append([p[k] for k in PROPS])
        out["validity"].append([float((yu - yl)[1:-1].min()), float(np.abs(d2[20:-5]).max())])
        u, lo = g.surfaces(s, xo)
        out["upper"].append(u)
        out["lower"].append(lo)
        hu, hl = g.handles(s)
        out["handles"].append(np.r_[hu, hl])
    return {k: np.asarray(v, float) for k, v in out.items()}
