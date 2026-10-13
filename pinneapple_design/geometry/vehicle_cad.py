"""Parametric vehicle CAD: closed-mesh primitives for rover-class vehicles, parts with materials and masses, mass
properties (centre of mass and inertia from the meshes) and assembly into bodies for multibody simulation.

    from pinneapple_design.geometry.vehicle_cad import VehicleCAD, beveled_box, rover_wheel, solar_panel
    cad = VehicleCAD("my rover")
    cad.add("chassis", "body", beveled_box((1.2, 0.9, 0.5), (0, 0, 0.8), 0.03), "white_paint", mass=200.0)
    cad.add("wheel_FL", "wheel_FL", *rover_wheel(0.25, 0.2, grousers=24, grouser_height=0.025))  # several parts
    props = cad.mass_properties()          # per body: mass, com, inertia about the com (body frame = world here)
    cad.scene().save("rover.glb")          # glTF with PBR materials, .stl, .usda

A *part* is one closed triangle mesh with a material and a mass (uniform density inside the mesh); a *body* is a
group of parts that move together (chassis, a suspension arm, a wheel). Parts that are thin plates are still closed
boxes so volume, centroid and inertia are well defined (divergence theorem, Eberly's polyhedral mass properties).

Coordinates: metres, z up, x forward, y left. Every function returns ``(vertices (V, 3), faces (F, 3))`` or, for the
assemblies (wheel, panel, camera head ...), a list of ``(suffix, (V, F), material, mass_share)``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------- basic closed primitives
def _faces_grid(nr, nc, closed=True):
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc if closed else nc - 1), indexing="ij")
    a = i * nc + j
    b = i * nc + (j + 1) % nc
    return np.concatenate([np.stack([a, b, b + nc], -1).reshape(-1, 3), np.stack([a, b + nc, a + nc], -1)
                           .reshape(-1, 3)])


def orient_outward(mesh: Mesh) -> Mesh:
    """Flip the winding of a closed mesh whose signed volume is negative."""
    V, F = mesh
    if signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def signed_volume(V, F) -> float:
    t = V[F]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def box(size, center=(0, 0, 0), R=None) -> Mesh:
    sx, sy, sz = np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, d in q for t in ([a, b, c], [a, c, d])], np.int64)
    if R is not None:
        v = v @ np.asarray(R).T
    return orient_outward((v + np.asarray(center, float), f))


def beveled_box(size, center=(0, 0, 0), bevel: float = 0.02, n: int = 3, R=None) -> Mesh:
    """Box with rounded edges of radius ``bevel`` (convex hull of eight corner spheres)."""
    from scipy.spatial import ConvexHull
    half = np.asarray(size, float) / 2 - bevel
    if np.any(half <= 0):
        return box(size, center, R)
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 4 * n, endpoint=False), np.linspace(0, np.pi, 2 * n + 1))
    s = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1).reshape(-1, 3)
    pts = np.concatenate([np.asarray(c) * half + bevel * s for c in
                          [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]])
    pts = np.unique(np.round(pts, 9), axis=0)
    h = ConvexHull(pts)
    V = pts
    t = pts[h.simplices]
    flip = np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), h.equations[:, :3]) < 0
    h.simplices[flip] = h.simplices[flip][:, ::-1]               # qhull facets are not consistently wound
    used = np.unique(h.simplices)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[h.simplices]
    if R is not None:
        V = V @ np.asarray(R).T
    return orient_outward((V + np.asarray(center, float), F))


def cylinder(p0, p1, radius, n: int = 32) -> Mesh:
    """Closed cylinder (or frustum, ``radius=(r0, r1)``) between two points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r0, r1 = (radius, radius) if np.isscalar(radius) else radius
    ax = p1 - p0
    L = np.linalg.norm(ax)
    ax = ax / L
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ax, seed)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    V = np.vstack([ring(p0, r0), ring(p1, r1), p0, p1])
    F = _faces_grid(2, n)
    c0, c1 = 2 * n, 2 * n + 1
    fan = [[c0, (j + 1) % n, j] for j in range(n)] + [[c1, n + j, n + (j + 1) % n] for j in range(n)]
    return orient_outward((V, np.vstack([F, fan])))


def tube_shell(axis_p0, axis_p1, r_in, r_out, n: int = 48) -> Mesh:
    """Closed annular tube (a wheel rim): inner and outer cylinders with annular end faces."""
    p0, p1 = np.asarray(axis_p0, float), np.asarray(axis_p1, float)
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(ax, seed)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    V = np.vstack([ring(p0, r_out), ring(p1, r_out), ring(p1, r_in), ring(p0, r_in)])
    F = _ring_faces(4, n)
    return orient_outward((V, F))


def _ring_faces(n_rings, n):
    """Faces joining ``n_rings`` rings cyclically (ring k to k+1, last to first)."""
    out = []
    for k in range(n_rings):
        a0, b0 = k * n, ((k + 1) % n_rings) * n
        for j in range(n):
            a, b = a0 + j, a0 + (j + 1) % n
            c, d = b0 + j, b0 + (j + 1) % n
            out += [[a, b, d], [a, d, c]]
    return np.array(out, np.int64)


def revolve_closed(profile, axis_origin=(0, 0, 0), axis=(0, 0, 1), n: int = 48) -> Mesh:
    """Closed solid of revolution of a closed polygon ``profile`` of (r, h) points about ``axis``."""
    p = np.asarray(profile, float)
    o, a = np.asarray(axis_origin, float), np.asarray(axis, float) / np.linalg.norm(axis)
    seed = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, seed)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = (o + p[:, 1, None, None] * a + p[:, 0, None, None] * (np.cos(th)[None, :, None] * u
                                                              + np.sin(th)[None, :, None] * w)).reshape(-1, 3)
    return orient_outward((V, _ring_faces(len(p), n)))


def paraboloid_dish(diameter: float, depth: float, thickness: float, center, axis, n: int = 40, m: int = 8) -> Mesh:
    """Dish antenna reflector: a paraboloid shell of ``depth`` over ``diameter`` (its vertex at ``center``)."""
    r = np.linspace(0, diameter / 2, m)
    z = depth * (r / (diameter / 2)) ** 2
    prof = np.r_[np.c_[r, z], np.c_[r[::-1], z[::-1] - thickness]]
    prof[0, 0] = prof[-1, 0] = 1e-4
    return revolve_closed(prof, center, axis, n)


def rotate(mesh: Mesh, R, about=(0, 0, 0)) -> Mesh:
    V, F = mesh
    c = np.asarray(about, float)
    return (V - c) @ np.asarray(R).T + c, F


def translate(mesh: Mesh, t) -> Mesh:
    return mesh[0] + np.asarray(t, float), mesh[1]


def rot(axis, angle_rad):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * K @ K


def merge(meshes: Sequence[Mesh]) -> Mesh:
    vs, fs, off = [], [], 0
    for V, F in meshes:
        vs.append(np.asarray(V, float))
        fs.append(np.asarray(F) + off)
        off += len(V)
    return np.vstack(vs), np.vstack(fs)


# ---------------------------------------------------------------------- mass properties
def mesh_mass_properties(V, F, mass: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre of mass and inertia tensor about it [kg m^2] of a closed mesh of uniform density holding ``mass``,
    and its volume [m^3] (Eberly, "Polyhedral mass properties", 2002)."""
    V, F = np.asarray(V, float), np.asarray(F)
    t = V[F]
    x0, x1, x2 = t[:, 0], t[:, 1], t[:, 2]
    d = np.cross(x1 - x0, x2 - x0)
    f1 = x0 + x1 + x2
    f2 = x0 * x0 + x1 * (x0 + x1) + x2 * f1
    f3 = x0 ** 3 + x1 * (x0 * x0 + x0 * x1 + x1 * x1) + x2 * f2
    g0 = f2 + x0 * (f1 + x0)
    g1 = f2 + x1 * (f1 + x1)
    g2 = f2 + x2 * (f1 + x2)
    vol = (d[:, 0] * f1[:, 0]).sum() / 6.0
    first = (d * f2).sum(0) / 24.0
    second = (d * f3).sum(0) / 60.0
    yx = (d[:, 0] * (x0[:, 1] * g0[:, 0] + x1[:, 1] * g1[:, 0] + x2[:, 1] * g2[:, 0])).sum() / 120.0
    zy = (d[:, 1] * (x0[:, 2] * g0[:, 1] + x1[:, 2] * g1[:, 1] + x2[:, 2] * g2[:, 1])).sum() / 120.0
    xz = (d[:, 2] * (x0[:, 0] * g0[:, 2] + x1[:, 0] * g1[:, 2] + x2[:, 0] * g2[:, 2])).sum() / 120.0
    if abs(vol) < 1e-15:
        c = V.mean(0)
        return c, np.zeros((3, 3)), 0.0
    sgn = 1.0 if vol > 0 else -1.0
    vol, first, second, yx, zy, xz = (sgn * v for v in (vol, first, second, yx, zy, xz))
    c = first / vol
    rho = mass / vol
    Ixx = rho * (second[1] + second[2])
    Iyy = rho * (second[2] + second[0])
    Izz = rho * (second[0] + second[1])
    Ixy, Iyz, Ixz = -rho * yx, -rho * zy, -rho * xz
    Io = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    Ic = Io - mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))       # parallel axis back to the centroid
    return c, Ic, vol


def combine_mass(items) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass, centre of mass and inertia about it of (mass, com, inertia about own com) items."""
    m = sum(i[0] for i in items)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = sum(i[0] * np.asarray(i[1]) for i in items) / m
    inertia = np.zeros((3, 3))
    for mi, ci, Ii in items:
        r = np.asarray(ci) - c
        inertia += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return float(m), c, inertia


# ---------------------------------------------------------------------- assemblies
def rover_wheel(radius: float, width: float, *, rim_thickness: float = 0.004, grousers: int = 0,
                grouser_height: float = 0.0, grouser_thickness: float = 0.004, grouser_shape: str = "straight",
                chevron_angle_deg: float = 20.0, spokes: int = 8, spoke_width: float = 0.03, hub_radius: float = 0.07,
                hub_length: float | None = None, n: int = 72, inner_side: int = -1):
    """A rigid rover wheel in its own frame (axle along y, centre at the origin): rim shell, grousers (straight lugs
    or chevrons), a spoked web on the inner side, hub (drive actuator housing). Returns parts
    ``(suffix, mesh, material, mass_share)``; mass shares are by volume x material density ratio."""
    parts = []
    rim = tube_shell((0, -width / 2, 0), (0, width / 2, 0), radius - rim_thickness, radius, n)
    parts.append(("rim", rim, "wheel_metal", 0.0))
    for k in range(grousers):
        a = 2 * math.pi * k / grousers
        Rk = rot((0, 1, 0), a)
        if grouser_shape == "chevron":
            half = width / 2
            sweep = math.tan(math.radians(chevron_angle_deg)) * half
            L = math.hypot(half, sweep)
            for side in (-1, 1):
                ang = side * math.atan2(sweep, half)
                g = box((grouser_thickness, L, grouser_height), (0, 0, radius + grouser_height / 2 - 0.0005),
                        rot((0, 0, 1), ang))
                g = translate(g, (-sweep / 2, side * half / 2, 0))
                parts.append((f"grouser{k}_{side}", rotate(g, Rk), "wheel_metal", 0.0))
        else:
            g = box((grouser_thickness, width, grouser_height), (0, 0, radius + grouser_height / 2 - 0.0005))
            parts.append((f"grouser{k}", rotate(g, Rk), "wheel_metal", 0.0))
    y_web = inner_side * (width / 2 - 0.01)
    web = []
    for k in range(spokes):
        a = 2 * math.pi * (k + 0.5) / spokes
        r_mid = 0.5 * (hub_radius + radius - rim_thickness)
        sp = box((spoke_width, 0.006, radius - rim_thickness - hub_radius * 0.8), (0, y_web, r_mid))
        web.append(rotate(sp, rot((0, 1, 0), a)))
    parts.append(("spokes", merge(web), "wheel_metal", 0.0))
    ring = tube_shell((0, y_web - 0.004, 0), (0, y_web + 0.004, 0), radius - rim_thickness - 0.012,
                      radius - rim_thickness + 1e-4, n)
    parts.append(("web_ring", ring, "wheel_metal", 0.0))
    hl = hub_length if hub_length is not None else 0.75 * width
    parts.append(("hub", cylinder((0, inner_side * hl * 0.9, 0), (0, -inner_side * 0.1 * hl, 0), hub_radius, 32),
                  "anodized_black", 0.0))
    parts.append(("hub_cap", cylinder((0, -inner_side * 0.1 * hl, 0), (0, -inner_side * 0.14 * hl, 0),
                                      hub_radius * 0.6, 24), "aluminium", 0.0))
    return _shares(parts)


def solar_panel(width: float, height: float, thickness: float = 0.025, cells=(6, 4), gap: float = 0.012,
                frame: float = 0.03):
    """Vertical solar panel in the x-z plane (facing -y... rotate as needed), centred at the origin: aluminium frame
    and substrate plus a grid of cells standing proud by 1 mm."""
    parts = [("substrate", box((width, thickness, height), (0, 0, 0)), "anodized_black", 0.0)]
    nx, nz = cells
    cw = (width - 2 * frame - (nx - 1) * gap) / nx
    ch = (height - 2 * frame - (nz - 1) * gap) / nz
    cs = []
    for i in range(nx):
        for j in range(nz):
            x = -width / 2 + frame + cw / 2 + i * (cw + gap)
            z = -height / 2 + frame + ch / 2 + j * (ch + gap)
            cs.append(box((cw, 0.002, ch), (x, -thickness / 2 - 0.0008, z)))
    parts.append(("cells", merge(cs), "solar_cell", 0.0))
    parts.append(("frame_top", box((width, thickness * 1.2, frame), (0, 0, height / 2 - frame / 2)), "aluminium", 0.0))
    parts.append(("frame_bot", box((width, thickness * 1.2, frame), (0, 0, -height / 2 + frame / 2)), "aluminium",
                  0.0))
    return _shares(parts)


def radiator(width: float, depth: float, thickness: float = 0.02, fins: int = 12, fin_height: float = 0.03):
    """Horizontal radiator plate (x-y plane, top at z = 0) with ``fins`` parallel fins along x."""
    parts = [("plate", box((width, depth, thickness), (0, 0, -thickness / 2)), "radiator", 0.0)]
    fs = [box((width * 0.96, 0.004, fin_height), (0, -depth / 2 + (k + 0.5) * depth / fins, fin_height / 2))
          for k in range(fins)]
    parts.append(("fins", merge(fs), "radiator", 0.0))
    return _shares(parts)


def camera_head(width: float, depth: float, height: float, baseline: float, lens_radius: float = 0.022,
                lights: int = 2):
    """Mast camera head facing +x: housing, a stereo pair of lenses ``baseline`` apart, LED light bars."""
    parts = [("housing", beveled_box((depth, width, height), (0, 0, 0), 0.015), "white_paint", 0.0)]
    for s in (-1, 1):
        y = s * baseline / 2
        parts.append((f"lens_barrel_{'L' if s > 0 else 'R'}", cylinder((depth / 2 - 0.005, y, 0.01),
                                                                     (depth / 2 + 0.03, y, 0.01), lens_radius * 1.35,
                                                                     24), "anodized_black", 0.0))
        parts.append((f"lens_{'L' if s > 0 else 'R'}", cylinder((depth / 2 + 0.03, y, 0.01),
                                                              (depth / 2 + 0.034, y, 0.01), lens_radius, 24),
                      "lens", 0.0))
    for k in range(lights):
        y = (k - (lights - 1) / 2) * width * 0.55
        parts.append((f"light{k}", box((0.012, width * 0.22, 0.03), (depth / 2 + 0.006, y, -height / 2 + 0.03)),
                      "led", 0.0))
    return _shares(parts)


_DENSITY = {"wheel_metal": 2700.0, "aluminium": 2700.0, "anodized_black": 2700.0, "solar_cell": 2300.0,
            "radiator": 2700.0, "white_paint": 1500.0, "lens": 2500.0, "led": 1500.0}


def _shares(parts):
    vols = []
    for _name, (V, F), mat, _ in parts:
        vols.append(abs(signed_volume(V, F)) * _DENSITY.get(mat, 1500.0))
    tot = sum(vols) or 1.0
    return [(n, m, mat, v / tot) for (n, m, mat, _), v in zip(parts, vols, strict=True)]


# ---------------------------------------------------------------------- the assembly
@dataclass
class CADPart:
    name: str
    body: str
    vertices: np.ndarray
    faces: np.ndarray
    material: str
    mass: float = 0.0
    group: str = ""

    def mass_properties(self):
        return mesh_mass_properties(self.vertices, self.faces, self.mass)


@dataclass
class VehicleCAD:
    """Parts grouped into bodies; every coordinate is in one assembly frame (the vehicle at its reference pose)."""
    title: str = "vehicle"
    parts: list[CADPart] = field(default_factory=list)
    frames: dict = field(default_factory=dict)        # body -> (R, p) of its joint frame in the assembly frame
    notes: dict = field(default_factory=dict)

    def add(self, name: str, body: str, mesh: Mesh, material: str, mass: float = 0.0, group: str = "") -> CADPart:
        p = CADPart(name, body, np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64), material, float(mass),
                    group or body)
        self.parts.append(p)
        return p

    def add_assembly(self, prefix: str, body: str, parts, mass: float, R=None, t=(0, 0, 0), group: str = ""):
        """Add an assembly (list of (suffix, mesh, material, mass_share)) rotated by R, moved by t, with ``mass``
        split by the shares."""
        R = np.eye(3) if R is None else np.asarray(R)
        for suffix, (V, F), mat, share in parts:
            self.add(f"{prefix}_{suffix}", body, (V @ R.T + np.asarray(t, float), F), mat, mass * share, group)

    def bodies(self) -> list[str]:
        return list(dict.fromkeys(p.body for p in self.parts))

    def body_parts(self, body: str) -> list[CADPart]:
        return [p for p in self.parts if p.body == body]

    def mass_properties(self) -> dict:
        """Per body: ``mass`` [kg], ``com`` [m] and ``inertia`` about the com [kg m^2], in the assembly frame."""
        out = {}
        for b in self.bodies():
            items = []
            for p in self.body_parts(b):
                if p.mass > 0:
                    c, inertia, _ = p.mass_properties()
                    items.append((p.mass, c, inertia))
            m, c, inertia = combine_mass(items)
            out[b] = {"mass": m, "com": c, "inertia": inertia}
        return out

    def total(self) -> dict:
        mp = self.mass_properties()
        m, c, inertia = combine_mass([(v["mass"], v["com"], v["inertia"]) for v in mp.values() if v["mass"] > 0])
        return {"mass": m, "com": c, "inertia": inertia}

    def body_mesh(self, body: str) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
        return [(p.name, p.vertices, p.faces, p.material) for p in self.body_parts(body)]

    def posed(self, transforms: dict) -> list[CADPart]:
        """Parts moved rigidly: ``transforms[body] = (R, t)`` maps assembly coordinates of that body to the world."""
        out = []
        for p in self.parts:
            R, t = transforms.get(p.body, (np.eye(3), np.zeros(3)))
            out.append(CADPart(p.name, p.body, p.vertices @ np.asarray(R).T + t, p.faces, p.material, p.mass,
                               p.group))
        return out

    def scene(self, transforms: dict | None = None, title: str | None = None):
        from pinneapple_tools.visualization.studio.scene import Scene, Surface
        parts = self.posed(transforms) if transforms else self.parts
        return Scene([Surface(p.name, p.vertices, p.faces, p.material, group=p.group) for p in parts],
                     title=title or self.title)

    def export(self, path: str, transforms: dict | None = None) -> str:
        """glTF (.glb, PBR materials), STL or USD by extension."""
        return self.scene(transforms).save(path)

    def bounds(self):
        V = np.concatenate([p.vertices for p in self.parts])
        return V.min(0), V.max(0)


__all__ = ["VehicleCAD", "CADPart", "box", "beveled_box", "cylinder", "tube_shell", "revolve_closed",
           "paraboloid_dish", "rover_wheel", "solar_panel", "radiator", "camera_head", "mesh_mass_properties",
           "combine_mass", "rot", "rotate", "translate", "merge", "signed_volume", "orient_outward"]
