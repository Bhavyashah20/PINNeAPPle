"""Parametric six-wheel rocker-bogie rover: parameters, kinematics and meshes (glTF / STL export).

Every default is generic (not taken from a real rover). The wheel (R = 0.125 m, b = 0.06 m), the six driven wheels
and the 40 kg mass are the values of the terramechanics use case (``WheelParams``), so the PINN surrogate trained
there applies to this rover unchanged.

Body frame: origin on the rocker pivot axis midway between the two pivots, x forward, y left, z up.
Per side the rocker carries the front wheel and, at its rear end, the bogie pivot; the bogie carries the middle and
rear wheels. A differential makes the right rocker angle the negative of the left one (relative to the body).
With the default proportions (front arm = 2 x rear arm, bogie pivot midway between its wheels, centre of mass
above the rocker pivots) the six wheels carry equal loads on level ground and the wheels are equally spaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WHEEL_NAMES = ("L_front", "L_mid", "L_rear", "R_front", "R_mid", "R_rear")


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Body-to-world rotation. ``pitch`` > 0 is nose up, ``roll`` > 0 is left side up [rad]."""
    return rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


@dataclass
class RoverParams:
    """Generic small lunar rover (every value is a generic choice; see the module docstring)."""
    wheel_radius: float = 0.125          # [m]  (terramechanics use case WheelParams.R)
    wheel_width: float = 0.060           # [m]  (WheelParams.b)
    n_grousers: int = 0                  # drawn only; the Bekker-Wong contact ignores grousers
    grouser_height: float = 0.005        # [m]
    track: float = 0.60                  # lateral distance between wheel centres [m]
    wheelbase: float = 0.70              # front to rear wheel centre [m]
    pivot_height: float = 0.15           # rocker pivot above the wheel centres on level ground [m]
    bogie_drop: float = 0.05             # bogie pivot below the rocker pivot [m]
    body_mass: float = 40.0              # rover mass without payload [kg] (WheelParams.mass_rover)
    body_size: tuple = (0.50, 0.40, 0.15)  # body box (x, y, z) [m], for drawing and the default inertia
    body_center: tuple = (0.0, 0.0, 0.10)  # body box centre in the body frame [m]
    com: tuple = (0.0, 0.0, 0.10)        # centre of mass of the rover without payload, body frame [m]
    payload_mass: float = 0.0            # [kg]
    payload_com: tuple = (0.0, 0.0, 0.22)  # [m] body frame
    inertia: tuple | None = None         # body inertia about the CoM (Ixx, Iyy, Izz) [kg m^2]; None = uniform box

    # -- derived geometry (equal spacing, equal static loads) -------------
    @property
    def e(self) -> float:
        return self.wheelbase / 4.0

    @property
    def front_arm(self) -> float:        # rocker pivot -> front wheel, along x
        return 2.0 * self.e

    @property
    def rear_arm(self) -> float:         # rocker pivot -> bogie pivot, along x
        return self.e

    @property
    def mass(self) -> float:
        return self.body_mass + self.payload_mass

    @property
    def com_total(self) -> np.ndarray:
        return (self.body_mass * np.asarray(self.com) + self.payload_mass * np.asarray(self.payload_com)) / self.mass

    def body_inertia(self) -> np.ndarray:
        """Inertia tensor about the total centre of mass (uniform box body + point payload) [kg m^2]."""
        if self.inertia is not None:
            inertia = np.diag(self.inertia).astype(float)
        else:
            a, b, c = self.body_size
            m = self.body_mass
            inertia = np.diag([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12.0
        G = self.com_total
        for m, p in ((self.body_mass, np.asarray(self.com)), (self.payload_mass, np.asarray(self.payload_com))):
            d = p - G
            inertia = inertia + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return inertia

    def check(self) -> None:
        for k in ("wheel_radius", "wheel_width", "track", "wheelbase", "pivot_height", "body_mass"):
            if not getattr(self, k) > 0:
                raise ValueError(f"RoverParams.{k} must be > 0")
        if not 0 < self.bogie_drop < self.pivot_height:
            raise ValueError("bogie_drop must be in (0, pivot_height)")
        if self.payload_mass < 0:
            raise ValueError("payload_mass must be >= 0")


@dataclass
class Pose:
    """Rover configuration: body origin (rocker pivot axis centre), attitude and suspension angles [m, rad]."""
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    rocker: float = 0.0                  # left rocker angle relative to the body (right = -rocker)
    bogie_l: float = 0.0
    bogie_r: float = 0.0
    extra: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        return body_rotation(self.yaw, self.pitch, self.roll)


def link_frames(p: RoverParams, pose: Pose) -> dict:
    """World positions of the pivots and wheel centres plus link rotations.

    Returns a dict with ``R`` (body), ``origin``, ``rocker_pivot`` (2, 3), ``bogie_pivot`` (2, 3), ``wheel`` (6, 3)
    in :data:`WHEEL_NAMES` order, ``R_rocker`` (2, 3, 3), ``R_bogie`` (2, 3, 3) and ``com`` (3,).
    """
    R = pose.rotation()
    origin = np.array([pose.x, pose.y, pose.z], float)
    H, cz = p.pivot_height, p.bogie_drop
    ez = H - cz
    piv, bog, whl, Rr, Rb = [], [], [], [], []
    for side, sgn in ((0, 1.0), (1, -1.0)):
        rho = sgn * pose.rocker
        beta = pose.bogie_l if side == 0 else pose.bogie_r
        Pr = origin + R @ np.array([0.0, sgn * p.track / 2, 0.0])
        Rrock = R @ rot_y(rho)
        Rbog = Rrock @ rot_y(beta)
        Pb = Pr + Rrock @ np.array([-p.rear_arm, 0.0, -cz])
        whl += [Pr + Rrock @ np.array([p.front_arm, 0.0, -H]),
                Pb + Rbog @ np.array([p.e, 0.0, -ez]),
                Pb + Rbog @ np.array([-p.e, 0.0, -ez])]
        piv.append(Pr)
        bog.append(Pb)
        Rr.append(Rrock)
        Rb.append(Rbog)
    return {"R": R, "origin": origin, "rocker_pivot": np.array(piv), "bogie_pivot": np.array(bog),
            "wheel": np.array(whl), "R_rocker": np.array(Rr), "R_bogie": np.array(Rb),
            "com": origin + R @ p.com_total}


# ---------------------------------------------------------------------- meshes
def _wheel_mesh(p: RoverParams, n_theta: int = 40):
    from pinneapple_twin3d.primitives import box, cylinder, merge, rotation, transform
    R, b = p.wheel_radius, p.wheel_width
    parts = [cylinder((0, -b / 2, 0), (0, b / 2, 0), R, n_theta),
             cylinder((0, -b / 2 - 0.004, 0), (0, b / 2 + 0.004, 0), 0.25 * R, 16)]
    for k in range(p.n_grousers):
        a = 360.0 * k / p.n_grousers
        g = box((0.006, b, p.grouser_height), (0, 0, R + p.grouser_height / 2))
        parts.append(transform(g, R=rotation((0, 1, 0), a)))
    return merge(parts)


def rover_meshes(p: RoverParams) -> dict:
    """Meshes in their own link frames: ``body`` (body frame), ``rocker_L/R`` (rocker frame, origin at the pivot),
    ``bogie_L/R`` (bogie frame, origin at the bogie pivot) and ``wheel`` (wheel frame, axle along y)."""
    from pinneapple_twin3d.primitives import box, cylinder, merge
    H, cz = p.pivot_height, p.bogie_drop
    ez = H - cz
    rod = 0.012
    body = [box(p.body_size, p.body_center),
            cylinder((0, -p.track / 2, 0), (0, p.track / 2, 0), 0.015, 16)]
    if p.payload_mass > 0:
        body.append(box((0.2, 0.2, 0.08), p.payload_com))
    out = {"body": merge(body), "wheel": _wheel_mesh(p)}
    for s in ("L", "R"):
        out[f"rocker_{s}"] = merge([cylinder((0, 0, 0), (p.front_arm, 0, -H), rod, 12),
                                    cylinder((0, 0, 0), (-p.rear_arm, 0, -cz), rod, 12)])
        out[f"bogie_{s}"] = merge([cylinder((0, 0, 0), (p.e, 0, -ez), rod, 12),
                                   cylinder((0, 0, 0), (-p.e, 0, -ez), rod, 12)])
    return out


def assembled_mesh(p: RoverParams, pose: Pose, wheel_spin: np.ndarray | None = None):
    """All parts in world coordinates: list of (name, vertices, faces)."""
    fr = link_frames(p, pose)
    m = rover_meshes(p)
    spin = np.zeros(6) if wheel_spin is None else np.asarray(wheel_spin, float)
    out = [("body", m["body"][0] @ fr["R"].T + fr["origin"], m["body"][1])]
    for k, s in enumerate(("L", "R")):
        out.append((f"rocker_{s}", m[f"rocker_{s}"][0] @ fr["R_rocker"][k].T + fr["rocker_pivot"][k],
                    m[f"rocker_{s}"][1]))
        out.append((f"bogie_{s}", m[f"bogie_{s}"][0] @ fr["R_bogie"][k].T + fr["bogie_pivot"][k],
                    m[f"bogie_{s}"][1]))
    for i, name in enumerate(WHEEL_NAMES):
        Rw = fr["R_bogie"][i // 3] if i % 3 else fr["R_rocker"][i // 3]
        Rw = Rw @ rot_y(spin[i])
        out.append((name, m["wheel"][0] @ Rw.T + fr["wheel"][i], m["wheel"][1]))
    return out


def export_rover(p: RoverParams, path: str, pose: Pose | None = None) -> str:
    """Write the rover (level, on its wheels) as .glb or .stl with the studio writer."""
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    pose = pose or Pose(0.0, 0.0, p.wheel_radius + p.pivot_height)
    mats = {"body": "paint", "rocker_L": "aluminium", "rocker_R": "aluminium", "bogie_L": "aluminium",
            "bogie_R": "aluminium"}
    sc = Scene([Surface(n, V, F, mats.get(n, "plastic")) for n, V, F in assembled_mesh(p, pose)],
               title="Rocker-bogie rover")
    return sc.save(path)


__all__ = ["RoverParams", "Pose", "WHEEL_NAMES", "link_frames", "rover_meshes", "assembled_mesh", "export_rover",
           "body_rotation", "quat_from_matrix", "rot_x", "rot_y", "rot_z"]
