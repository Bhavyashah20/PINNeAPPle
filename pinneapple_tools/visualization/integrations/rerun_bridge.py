"""Rerun (rerun.io) bridge: stream PINNeAPPle results into the Rerun viewer, live or into an ``.rrd`` file.

Rerun is a time-aware multimodal viewer: 3-D scenes, point clouds, meshes, time series, text and graphs on shared
timelines, scrubbable after the fact. That fits digital twins and simulations, where a field, a sensor reading, the
active model and a diagnosis all change with time. This bridge logs:

* :meth:`RerunLogger.scene`: a ``studio.Scene`` (surfaces coloured by a field, streamlines);
* :meth:`RerunLogger.points` / :meth:`RerunLogger.mesh`: raw point clouds and triangle meshes;
* :meth:`RerunLogger.molecule`: a molecule (and its metal slab) from :mod:`..molecular`, frame by frame;
* :meth:`RerunLogger.scalars`: time series (sensor values, errors, weights of an ensemble);
* :meth:`RerunLogger.ensemble_run`: a :class:`~pinneapple_physics.ensemble.EnsembleRun` (weights, errors, active
  model per case);
* :meth:`RerunLogger.graph`: a knowledge graph or a proof graph as a graph view, nodes coloured by status;
* :meth:`RerunLogger.text`: a Markdown report (physics audit, proof report).

    from pinneapple_tools.visualization.integrations.rerun_bridge import RerunLogger
    log = RerunLogger("pinneapple", path="run.rrd")          # or spawn=True for a live viewer
    log.molecule(mol, frames, times, slab=slab)
    log.close()                                              # then: rerun run.rrd

Needs ``pip install rerun-sdk``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["RerunLogger"]


def _rgba(colors) -> np.ndarray:
    c = np.asarray(colors, dtype=float)
    if c.ndim == 1:
        c = c[None]
    if c.max() <= 1.0:
        c = c * 255.0
    if c.shape[1] == 3:
        c = np.concatenate([c, np.full((len(c), 1), 255.0)], axis=1)
    return c.astype(np.uint8)


def _hex(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [255]


class RerunLogger:
    """``path``: write an ``.rrd`` file; ``spawn``: open a live viewer; ``connect``: a gRPC URL of a running
    viewer. With none of them the data stay in memory (useful in tests)."""

    def __init__(self, app_id: str = "pinneapple", path: str | None = None, spawn: bool = False,
                 connect: str | None = None):
        import rerun as rr

        self.rr = rr
        self.rec = rr.RecordingStream(app_id)
        self.path = path
        if path:
            self.rec.save(path)
        elif spawn:
            self.rec.spawn()
        elif connect:
            self.rec.connect_grpc(connect)

    # ------------------------------------------------------------------------------------------ time
    def at(self, *, seconds: float | None = None, step: int | None = None, timeline: str | None = None) -> RerunLogger:
        """Set the current time: ``seconds`` on the "time" timeline, ``step`` on the "step" timeline."""
        if seconds is not None:
            self.rec.set_time(timeline or "time", duration=float(seconds))
        if step is not None:
            self.rec.set_time(timeline or "step", sequence=int(step))
        return self

    # ------------------------------------------------------------------------------------------ geometry
    def points(self, entity: str, xyz, colors=None, radii=None, labels: Sequence[str] | None = None) -> None:
        kw: dict[str, Any] = {}
        if colors is not None:
            kw["colors"] = _rgba(colors)
        if radii is not None:
            kw["radii"] = np.asarray(radii, dtype=np.float32)
        if labels is not None:
            kw["labels"] = list(labels)
        self.rec.log(entity, self.rr.Points3D(np.asarray(xyz, dtype=np.float32), **kw))

    def mesh(self, entity: str, vertices, faces, colors=None, values=None, cmap: str = "turbo",
             value_range: tuple[float, float] | None = None) -> None:
        """A triangle mesh; ``values`` (one per vertex) are mapped to ``cmap`` colours."""
        kw: dict[str, Any] = {}
        if values is not None:
            from matplotlib import colormaps

            v = np.asarray(values, float)
            lo, hi = value_range or (np.nanpercentile(v, 1), np.nanpercentile(v, 99))
            kw["vertex_colors"] = _rgba(colormaps[cmap](np.clip((v - lo) / max(hi - lo, 1e-30), 0, 1))[:, :3])
        elif colors is not None:
            kw["vertex_colors"] = _rgba(colors)
        self.rec.log(entity, self.rr.Mesh3D(vertex_positions=np.asarray(vertices, np.float32),
                                            triangle_indices=np.asarray(faces, np.uint32), **kw))

    def scene(self, scene, entity: str = "scene", field: str | None = None, cmap: str = "turbo") -> None:
        """Every surface of a ``studio.Scene`` (coloured by ``field`` if given) and its streamlines."""
        for s in scene.surfaces:
            vals = s.fields.get(field) if field else None
            if vals is not None and np.ndim(vals) > 1:
                vals = np.linalg.norm(vals, axis=-1)
            self.mesh(f"{entity}/{s.name}", s.vertices, s.faces, values=vals, cmap=cmap,
                      colors=None if vals is not None else [0.75, 0.77, 0.8])
        for k, ln in enumerate(scene.lines):
            self.rec.log(f"{entity}/lines_{k}", self.rr.LineStrips3D([np.asarray(p, np.float32) for p in ln.points],
                                                                     radii=0.002))

    def molecule(self, mol, frames: Sequence[np.ndarray] | None = None, times: Sequence[float] | None = None,
                 slab: np.ndarray | None = None, entity: str = "molecule", color_by: str = "residue") -> None:
        """A molecule from :mod:`pinneapple_tools.visualization.molecular`; with ``frames`` one point cloud per
        time (``times`` in ps on the "time" timeline, else frame index on "step")."""
        from matplotlib import colors as mcolors

        from ..molecular import COLORS, ELEMENT_COLORS, METAL_COLORS

        if color_by == "element":
            rgb = np.array([mcolors.to_rgb(ELEMENT_COLORS.get(e, "#C0C0C0")) for e in mol.elements])
        else:
            rgb = np.array([mcolors.to_rgb(COLORS[c]) for c in mol.classes()])
        if slab is not None:
            self.rec.log(f"{entity}/slab", self.rr.Points3D(np.asarray(slab, np.float32), radii=1.44,
                                                            colors=_hex(METAL_COLORS["Au"])), static=True)
        frames = list(frames) if frames is not None else [mol.xyz]
        for k, X in enumerate(frames):
            if times is not None:
                self.at(seconds=float(times[k]))
            else:
                self.at(step=k)
            self.rec.log(f"{entity}/atoms", self.rr.Points3D(np.asarray(X, np.float32), radii=0.8 * mol.radii,
                                                             colors=_rgba(rgb)))

    # ------------------------------------------------------------------------------------------ series
    def scalars(self, entity: str, values: Sequence[float], times: Sequence[float] | None = None) -> None:
        for k, v in enumerate(values):
            if times is not None:
                self.at(seconds=float(times[k]))
            else:
                self.at(step=k)
            self.rec.log(entity, self.rr.Scalars(float(v)))

    def ensemble_run(self, run, entity: str = "ensemble") -> None:
        """Weights and errors of each expert per case, the ensemble error and the active model."""
        for k in range(len(run.active)):
            self.at(step=k)
            for i, n in enumerate(run.names):
                self.rec.log(f"{entity}/weights/{n}", self.rr.Scalars(float(run.weights[k, i])))
                if np.isfinite(run.errors[k, i]):
                    self.rec.log(f"{entity}/errors/{n}", self.rr.Scalars(float(run.errors[k, i])))
            if np.isfinite(run.ensemble_errors[k]):
                self.rec.log(f"{entity}/errors/ensemble", self.rr.Scalars(float(run.ensemble_errors[k])))
            self.rec.log(f"{entity}/active", self.rr.TextLog(run.active[k]))

    # ------------------------------------------------------------------------------------------ graphs and text
    def graph(self, nodes: Mapping[str, str] | Sequence[str], edges: Sequence[tuple[str, str]],
              entity: str = "graph", colors: Mapping[str, str] | None = None, directed: bool = True) -> None:
        """A graph view. ``nodes``: ids, or {id: label}; ``colors``: {id: hex colour}."""
        ids = list(nodes)
        labels = [nodes[i] for i in ids] if isinstance(nodes, Mapping) else ids
        kw: dict[str, Any] = {"labels": labels, "show_labels": True}
        if colors:
            kw["colors"] = [_hex(colors.get(i, "#9AA5B1")) for i in ids]
        self.rec.log(entity, self.rr.GraphNodes(ids, **kw), static=True)
        self.rec.log(entity, self.rr.GraphEdges([tuple(e) for e in edges],
                                                graph_type="directed" if directed else "undirected"), static=True)

    def proof_graph(self, g, entity: str = "proof") -> None:
        """A :class:`pinneapple_analysis.proof.ProofGraph` (after ``run()``): nodes coloured by status, its report."""
        from pinneapple_analysis.proof.graph import COLORS as STATUS_COLORS

        nodes = {i: f"{i} [{n.status}]" for i, n in g.nodes.items()}
        edges = [(d, i) for i, n in g.nodes.items() for d in n.depends_on + n.supported_by]
        self.graph(nodes, edges, entity=f"{entity}/graph",
                   colors={i: STATUS_COLORS[n.status] for i, n in g.nodes.items()})
        self.text(f"{entity}/report", g.report())

    def knowledge_graph(self, kg, entity: str = "knowledge") -> None:
        """A :class:`pinneapple_physics.physics_audit.PhysicsKnowledgeGraph`, nodes coloured by kind."""
        palette = {"phenomenon": "#D9534F", "equation": "#4C78D8", "assumption": "#F2B233", "parameter": "#7FB8A4",
                   "mechanism": "#B38BD9", "model": "#2E9E5B"}
        nodes = {n.id: n.label for n in kg.nodes.values()}
        self.graph(nodes, [(e.src, e.dst) for e in kg.edges], entity=entity,
                   colors={n.id: palette.get(n.kind, "#9AA5B1") for n in kg.nodes.values()})

    def text(self, entity: str, markdown: str) -> None:
        self.rec.log(entity, self.rr.TextDocument(markdown, media_type=self.rr.MediaType.MARKDOWN), static=True)

    def close(self) -> str | None:
        self.rec.flush()
        self.rec.disconnect()
        return self.path
