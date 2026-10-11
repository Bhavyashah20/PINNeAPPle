"""Self-contained interactive 3-D HTML with Plotly: one file that opens in any browser (rotate, zoom, hover values),
to share a result without a server or the PINNeAPPle viewer.

* :func:`scene_html`: a ``studio.Scene`` (surfaces coloured by a field, with a colour bar; streamlines);
* :func:`molecule_html`: a molecule and its metal slab from :mod:`..molecular` (hover: residue, atom);
* :func:`field_slices_html`: a 3-D scalar field on a grid as orthogonal slices and an isosurface.

Plotly's JavaScript is embedded (``include_plotlyjs="cdn"`` makes the file small but needs the internet).
Needs ``pip install plotly``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["scene_html", "molecule_html", "field_slices_html"]


def _layout(title: str):
    return dict(title=dict(text=title, x=0.02), paper_bgcolor="#0B0E14", font=dict(color="#E8E8E8"),
                scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                           zaxis=dict(visible=False), bgcolor="#0B0E14"),
                margin=dict(l=0, r=0, t=40, b=0))


def scene_html(scene, path: str, field: str | None = None, cmap: str = "Turbo", title: str = "",
               include_plotlyjs: bool | str = True) -> str:
    import plotly.graph_objects as go

    traces = []
    shown_bar = False
    for s in scene.surfaces:
        V, F = s.vertices, s.faces
        kw = dict(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], name=s.name,
                  flatshading=False, lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.5))
        vals = s.fields.get(field) if field else None
        if vals is not None:
            vals = np.linalg.norm(vals, axis=-1) if np.ndim(vals) > 1 else np.asarray(vals)
            kw.update(intensity=vals, colorscale=cmap, showscale=not shown_bar,
                      colorbar=dict(title=scene.labels.get(field, field)))
            shown_bar = True
        else:
            kw.update(color="#C7CBD1")
        traces.append(go.Mesh3d(**kw))
    for k, ln in enumerate(scene.lines):
        for p in ln.points:
            p = np.asarray(p)
            traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", showlegend=False,
                                       line=dict(color="#9CC9FF", width=2), name=f"line {k}"))
    fig = go.Figure(traces, layout=_layout(title or scene.title))
    fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return path


def molecule_html(mol, path: str, xyz=None, slab=None, metal: str = "Au", title: str = "",
                  include_plotlyjs: bool | str = True) -> str:
    import plotly.graph_objects as go

    from ..molecular import COLORS, METAL_COLORS

    X = mol.xyz if xyz is None else np.asarray(xyz)
    classes = mol.classes()
    traces = []
    for c in dict.fromkeys(classes):
        idx = [i for i, k in enumerate(classes) if k == c]
        hover = [f"{mol.resnames[i]} {mol.resids[i]} {mol.names[i]}" for i in idx] if mol.resnames else None
        traces.append(go.Scatter3d(x=X[idx, 0], y=X[idx, 1], z=X[idx, 2], mode="markers", name=c, text=hover,
                                   hoverinfo="text+name",
                                   marker=dict(size=4.5, color=COLORS[c], line=dict(width=0), opacity=1.0)))
    if slab is not None:
        traces.append(go.Scatter3d(x=slab[:, 0], y=slab[:, 1], z=slab[:, 2], mode="markers", name=metal,
                                   hoverinfo="name", marker=dict(size=5, color=METAL_COLORS.get(metal, "#D4AF37"))))
    fig = go.Figure(traces, layout=_layout(title))
    fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return path


def field_slices_html(x, y, z, values, path: str, iso: float | None = None, cmap: str = "Viridis", title: str = "",
                      include_plotlyjs: bool | str = True) -> str:
    """``values[i, j, k]`` on the grid ``x[i], y[j], z[k]``: three mid-plane slices and, with ``iso``, an isosurface."""
    import plotly.graph_objects as go

    V = np.asarray(values, float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    lo, hi = float(np.nanmin(V)), float(np.nanmax(V))
    traces = [go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=V.ravel(), isomin=lo, isomax=hi,
                        opacity=0.08, surface_count=12, colorscale=cmap, caps=dict(x_show=False, y_show=False,
                                                                                    z_show=False))]
    if iso is not None:
        traces.append(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=V.ravel(), isomin=iso, isomax=iso,
                                    surface_count=1, colorscale=cmap, showscale=False, opacity=0.9))
    fig = go.Figure(traces, layout=_layout(title))
    fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return path
