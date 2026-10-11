"""Longitudinal section of a melt runner (trough) as a 2-D OpenFOAM mesh: a sloped floor, a weir (skimmer) at the
downstream end, a jet entering through the top, and open air above.

    from pinneapple_design.geometry.trough import TroughSection
    sec = TroughSection(length=8.0, slope_deg=2.0)         # + slope: the floor descends toward the weir
    sec.write_mesh("cases/trough/mesh")                     # blockMesh; patches floor, upstream, weir,
                                                            # outlet, inlet, atmosphere, frontAndBack (empty)

The mesh is in the trough's own frame (x along the floor, z normal to it) and stays rectangular: the slope enters
through gravity, tilted by ``slope_deg`` (``gravity()``), with the bath surface normal to it (``level()``) and the
jet velocity rotated into the same frame (``to_mesh``). The floor pivots about the weir end, so a slope keeps the
weir crest and the bath level and changes the depth under the jet: +2° (floor descending toward the weir) gives a
shallow pool at impact, -2° a deep one. A sloped mesh with vertical gravity was unstable in the shallow corner in
interFoam (v1912); the tilted-gravity frame is the standard equivalent. Six blocks: three columns (upstream, jet,
downstream) times two rows (below and above the weir crest).
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass

__all__ = ["TroughSection"]


@dataclass
class TroughSection:
    length: float = 8.0
    slope_deg: float = 0.0
    weir_height: float = 0.5          # crest above the floor at the weir end, m
    top: float = 1.6                   # top of the air region above z = 0, m
    inlet_x: float = 1.0               # jet start along the floor, m
    inlet_width: float = 0.12          # jet width in the section, m
    cell: float = 0.04                 # cell size, m
    thickness: float = 0.05            # depth of the 2-D slab, m

    @property
    def theta(self) -> float:
        return math.radians(self.slope_deg)

    def floor(self, x: float) -> float:
        """Floor height above the weir-end floor in the world frame (for plots)."""
        return (self.length - x) * math.tan(self.theta)

    def depth_at(self, x: float) -> float:
        """Initial bath depth, measured vertically (the bath fills to the weir crest)."""
        return self.weir_height - self.floor(x)

    def gravity(self, g: float = 9.81) -> tuple:
        """Gravity in the mesh frame."""
        return (g * math.sin(self.theta), 0.0, -g * math.cos(self.theta))

    def to_mesh(self, v) -> tuple:
        """A world-frame vector (x downstream, z up) in the mesh frame."""
        c, s_ = math.cos(self.theta), math.sin(self.theta)
        return (v[0] * c - v[2] * s_, v[1], v[0] * s_ + v[2] * c)

    def level(self) -> float:
        """The bath surface through the weir crest, as a height along -gravity (FreeSurfaceFlow ``level``)."""
        return self.weir_height * math.cos(self.theta) - self.length * math.sin(self.theta)

    def blockmesh(self) -> str:
        L, w, H, t = self.length, self.weir_height, self.top, self.thickness
        if self.depth_at(0.0) < 2 * self.cell:
            raise ValueError("the slope leaves less than two cells of bath at the upstream end")
        xs = [0.0, self.inlet_x, self.inlet_x + self.inlet_width, L]
        pts = []
        for y in (0.0, t):
            for x in xs:
                pts += [(x, y, 0.0), (x, y, w), (x, y, H)]
        idx = lambda iy, ix, iz: (iy * 4 + ix) * 3 + iz                 # noqa: E731
        blocks, faces = [], {k: [] for k in ("floor", "upstream", "weir", "outlet", "inlet", "atmosphere",
                                              "frontAndBack")}
        nz_row = (max(2, round(w / self.cell)), max(2, round((H - w) / self.cell)))
        for ix in range(3):
            nx = max(1, round((xs[ix + 1] - xs[ix]) / self.cell))
            for iz in range(2):
                v = [idx(0, ix, iz), idx(0, ix + 1, iz), idx(1, ix + 1, iz), idx(1, ix, iz),
                     idx(0, ix, iz + 1), idx(0, ix + 1, iz + 1), idx(1, ix + 1, iz + 1), idx(1, ix, iz + 1)]
                nz = nz_row[iz]
                blocks.append(f"    hex ({' '.join(map(str, v))}) ({nx} 1 {nz}) simpleGrading (1 1 1)")
                faces["frontAndBack"] += [f"({v[0]} {v[1]} {v[5]} {v[4]})", f"({v[3]} {v[7]} {v[6]} {v[2]})"]
                if iz == 0:
                    faces["floor"].append(f"({v[0]} {v[3]} {v[2]} {v[1]})")
                if iz == 1:
                    faces["inlet" if ix == 1 else "atmosphere"].append(f"({v[4]} {v[5]} {v[6]} {v[7]})")
                if ix == 0:
                    faces["upstream"].append(f"({v[0]} {v[4]} {v[7]} {v[3]})")
                if ix == 2:
                    faces["weir" if iz == 0 else "outlet"].append(f"({v[1]} {v[2]} {v[6]} {v[5]})")
        types = {"floor": "wall", "upstream": "wall", "weir": "wall", "outlet": "patch", "inlet": "patch",
                 "atmosphere": "patch", "frontAndBack": "empty"}
        bnd = "\n".join(f"    {k} {{ type {types[k]}; faces ( {' '.join(f)} ); }}" for k, f in faces.items())
        vs = "\n".join(f"    ({x!r} {y!r} {z!r})" for x, y, z in pts)
        return (f"FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}\n"
                f"convertToMeters 1;\nvertices\n(\n{vs}\n);\nblocks\n(\n" + "\n".join(blocks) +
                f"\n);\nedges ();\nboundary\n(\n{bnd}\n);\nmergePatchPairs ();\n")

    def write_mesh(self, case: str, bashrc: str | None = None) -> str:
        """Write system/blockMeshDict (with a minimal controlDict) and run blockMesh; returns ``case``."""
        bashrc = bashrc or os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")
        os.makedirs(os.path.join(case, "system"), exist_ok=True)
        os.makedirs(os.path.join(case, "constant"), exist_ok=True)
        open(os.path.join(case, "system", "blockMeshDict"), "w").write(self.blockmesh())
        open(os.path.join(case, "system", "controlDict"), "w").write(
            "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
            "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\n"
            "writeControl timeStep; writeInterval 1;\n")
        r = subprocess.run(["bash", "-c", f'source "{bashrc}" >/dev/null 2>&1; cd "{case}" && blockMesh > '
                                          f'log.blockMesh 2>&1'])
        if r.returncode:
            raise RuntimeError(f"blockMesh failed, see {case}/log.blockMesh")
        return case
