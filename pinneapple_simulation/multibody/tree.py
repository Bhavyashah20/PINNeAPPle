"""Floating-base multibody tree with revolute joints: kinematics, Jacobians, mass matrix, bias forces and a
linearly implicit time step.

Generalised coordinates ``q = (p_base (3), quat_base (4, w x y z), joint angles (nj))`` and velocities
``v = (v_base (3, world, of the base origin), omega_base (3, world), joint rates (nj))``. Equations of motion

    M(q) dv/dt + h(q, v) = Q_gravity + Q_external(q, v) + tau_joints

with ``M = sum_i (Jv_i^T m_i Jv_i + Jw_i^T I_i Jw_i)`` over the bodies' centres of mass and ``h`` the Coriolis and
centrifugal terms, obtained by a forward recursion of the velocity-product accelerations (accelerations of the bodies
with dv/dt = 0). :meth:`MultibodyTree.step` takes a linearly implicit Euler step: force providers return their
generalised force and its derivative with respect to v (contact damping, PD gains), so stiff terramechanics and
servo terms stay stable at millisecond steps.

Every body has a mass, a centre of mass and an inertia tensor about it in its own frame; the base is body 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def cross(a, b):
    """Cross product of (..., 3) arrays (much cheaper than ``np.cross`` for small arrays)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = skew(a)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_matrix(R):
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
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


@dataclass
class Body:
    """A rigid body: ``mass`` [kg], ``com`` [m] and ``inertia`` (3, 3) about the com [kg m^2], both in the body frame."""
    name: str
    mass: float
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-6)
    parent_joint: int = -1

    def __post_init__(self):
        self.com = np.asarray(self.com, float)
        self.inertia = np.asarray(self.inertia, float).reshape(3, 3)
        if not self.mass > 0:
            raise ValueError(f"body {self.name}: mass must be > 0")


@dataclass
class RevoluteJoint:
    """Child frame = parent frame * (translate ``origin``, rotate ``R0``) * rotation(``axis``, q). ``axis`` in the
    child frame (equal to the joint frame). ``limits`` (lo, hi) [rad] act through a stiff penalty spring."""
    name: str
    parent: int
    child: int
    origin: np.ndarray
    axis: np.ndarray
    R0: np.ndarray = field(default_factory=lambda: np.eye(3))
    limits: tuple | None = None
    damping: float = 0.0           # viscous joint damping [N m s/rad]

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.axis = np.asarray(self.axis, float) / np.linalg.norm(self.axis)
        self.R0 = np.asarray(self.R0, float)


@dataclass
class Kinematics:
    """World poses and velocities of every body for one state."""
    R: np.ndarray            # (nb, 3, 3) body-to-world
    p: np.ndarray            # (nb, 3) body frame origins
    c: np.ndarray            # (nb, 3) centres of mass
    axis: np.ndarray         # (nj, 3) joint axes, world
    o: np.ndarray            # (nj, 3) joint origins, world
    Jv: np.ndarray           # (nb, 3, nv) com linear-velocity Jacobians
    Jw: np.ndarray           # (nb, 3, nv) angular-velocity Jacobians
    omega: np.ndarray        # (nb, 3)
    vel: np.ndarray          # (nb, 3) com velocities


class MultibodyTree:
    """Build with :meth:`add_body` (the first body is the floating base) and :meth:`add_revolute`."""

    def __init__(self, gravity=(0.0, 0.0, -1.62)):
        self.bodies: list[Body] = []
        self.joints: list[RevoluteJoint] = []
        self.gravity = np.asarray(gravity, float)
        self._order = None

    # ------------------------------------------------------------------ building
    def add_body(self, body: Body) -> int:
        if self.bodies and body.parent_joint < 0 and len(self.bodies) > 0:
            pass
        self.bodies.append(body)
        self._order = None
        return len(self.bodies) - 1

    def add_revolute(self, name, parent: int | str, child: Body, origin, axis, R0=None, limits=None,
                     damping: float = 0.0) -> int:
        parent = self.index(parent) if isinstance(parent, str) else parent
        ci = self.add_body(child)
        j = RevoluteJoint(name, parent, ci, origin, axis, np.eye(3) if R0 is None else R0, limits, damping)
        self.joints.append(j)
        child.parent_joint = len(self.joints) - 1
        return len(self.joints) - 1

    def index(self, name: str) -> int:
        for i, b in enumerate(self.bodies):
            if b.name == name:
                return i
        raise KeyError(name)

    def joint_index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)

    @property
    def nj(self) -> int:
        return len(self.joints)

    @property
    def nv(self) -> int:
        return 6 + self.nj

    @property
    def nq(self) -> int:
        return 7 + self.nj

    def total_mass(self) -> float:
        return float(sum(b.mass for b in self.bodies))

    def ancestors(self, i: int) -> list[int]:
        """Joint indices from body ``i`` up to the base."""
        out = []
        while self.bodies[i].parent_joint >= 0:
            j = self.bodies[i].parent_joint
            out.append(j)
            i = self.joints[j].parent
        return out

    def initial_state(self, p=(0.0, 0.0, 0.0), R=None, angles=None):
        q = np.zeros(self.nq)
        q[:3] = p
        q[3:7] = quat_from_matrix(np.eye(3) if R is None else R)
        if angles is not None:
            q[7:] = angles
        return q, np.zeros(self.nv)

    # ------------------------------------------------------------------ kinematics
    def kinematics(self, q, v=None) -> Kinematics:
        nb, nj, nv = len(self.bodies), self.nj, self.nv
        R = np.zeros((nb, 3, 3))
        p = np.zeros((nb, 3))
        ax = np.zeros((nj, 3))
        o = np.zeros((nj, 3))
        R[0] = quat_to_matrix(q[3:7] / np.linalg.norm(q[3:7]))
        p[0] = q[:3]
        for j, jt in enumerate(self.joints):              # joints are added parent-first
            Rp, pp = R[jt.parent], p[jt.parent]
            Rj = Rp @ jt.R0
            o[j] = pp + Rp @ jt.origin
            ax[j] = Rj @ jt.axis
            R[jt.child] = Rj @ axis_angle(jt.axis, q[7 + j])
            p[jt.child] = o[j]
        c = p + np.einsum("bij,bj->bi", R, np.array([b.com for b in self.bodies]))
        Jv = np.zeros((nb, 3, nv))
        Jw = np.zeros((nb, 3, nv))
        for i in range(nb):
            Jv[i, :, 0:3] = np.eye(3)
            Jv[i, :, 3:6] = -skew(c[i] - p[0])
            Jw[i, :, 3:6] = np.eye(3)
            for j in self.ancestors(i):
                Jv[i, :, 6 + j] = cross(ax[j], c[i] - o[j])
                Jw[i, :, 6 + j] = ax[j]
        omega = vel = None
        if v is not None:
            omega = Jw @ v
            vel = Jv @ v
        return Kinematics(R, p, c, ax, o, Jv, Jw, omega, vel)

    def point_jacobian(self, kin: Kinematics, body: int, point) -> np.ndarray:
        """(3, nv) Jacobian of the world velocity of a point fixed to ``body`` (point given in world coordinates)."""
        point = np.asarray(point, float)
        J = np.zeros((3, self.nv))
        J[:, 0:3] = np.eye(3)
        J[:, 3:6] = -skew(point - kin.p[0])
        for j in self.ancestors(body):
            J[:, 6 + j] = cross(kin.axis[j], point - kin.o[j])
        return J

    # ------------------------------------------------------------------ dynamics
    def mass_matrix(self, kin: Kinematics) -> np.ndarray:
        M = np.zeros((self.nv, self.nv))
        for i, b in enumerate(self.bodies):
            Iw = kin.R[i] @ b.inertia @ kin.R[i].T
            M += b.mass * kin.Jv[i].T @ kin.Jv[i] + kin.Jw[i].T @ Iw @ kin.Jw[i]
        return M

    def bias(self, kin: Kinematics, v) -> np.ndarray:
        """Coriolis and centrifugal generalised forces h(q, v)."""
        nb = len(self.bodies)
        acc = np.zeros((nb, 3))      # com acceleration with dv/dt = 0
        alpha = np.zeros((nb, 3))
        w = kin.omega
        acc[0] = cross(w[0], cross(w[0], kin.c[0] - kin.p[0]))
        for j, jt in enumerate(self.joints):
            pa, ch = jt.parent, jt.child
            qd = v[6 + j]
            alpha[ch] = alpha[pa] + cross(w[pa], qd * kin.axis[j])
            r_po = kin.o[j] - kin.c[pa]
            a_o = acc[pa] + cross(alpha[pa], r_po) + cross(w[pa], cross(w[pa], r_po))
            r = kin.c[ch] - kin.o[j]
            acc[ch] = a_o + cross(alpha[ch], r) + cross(w[ch], cross(w[ch], r))
        h = np.zeros(self.nv)
        for i, b in enumerate(self.bodies):
            Iw = kin.R[i] @ b.inertia @ kin.R[i].T
            h += kin.Jv[i].T @ (b.mass * acc[i]) + kin.Jw[i].T @ (Iw @ alpha[i] + cross(w[i], Iw @ w[i]))
        return h

    def gravity_force(self, kin: Kinematics) -> np.ndarray:
        Q = np.zeros(self.nv)
        for i, b in enumerate(self.bodies):
            Q += kin.Jv[i].T @ (b.mass * self.gravity)
        return Q

    def joint_limit_force(self, q, v, k: float = 2e4, c: float = 400.0):
        Q = np.zeros(self.nv)
        D = np.zeros(self.nv)
        for j, jt in enumerate(self.joints):
            if jt.damping:
                Q[6 + j] -= jt.damping * v[6 + j]
                D[6 + j] -= jt.damping
            if jt.limits is None:
                continue
            lo, hi = jt.limits
            x = q[7 + j]
            if x < lo:
                Q[6 + j] += k * (lo - x) - c * min(v[6 + j], 0.0)
                D[6 + j] -= c
            elif x > hi:
                Q[6 + j] += k * (hi - x) - c * max(v[6 + j], 0.0)
                D[6 + j] -= c
        return Q, np.diag(D)

    def integrate(self, q, v, dt):
        """Positions after a step with velocity ``v`` (exponential map for the base orientation)."""
        q = q.copy()
        q[:3] += dt * v[:3]
        w = v[3:6]
        th = np.linalg.norm(w) * dt
        if th > 1e-15:
            a = w / np.linalg.norm(w)
            dq = np.r_[np.cos(th / 2), np.sin(th / 2) * a]
            q[3:7] = quat_mul(dq, q[3:7])
        q[3:7] /= np.linalg.norm(q[3:7])
        q[7:] += dt * v[6:]
        return q

    def step(self, q, v, dt, forces) -> tuple[np.ndarray, np.ndarray, Kinematics]:
        """One linearly implicit Euler step. ``forces(kin, q, v)`` returns (Q (nv,), D (nv, nv)) with ``D = dQ/dv``
        (and any position stiffness already folded in as ``dt K``). Returns the new (q, v) and the kinematics at the
        start of the step."""
        kin = self.kinematics(q, v)
        M = self.mass_matrix(kin)
        Q = self.gravity_force(kin) - self.bias(kin, v)
        Ql, Dl = self.joint_limit_force(q, v)
        Qe, De = forces(kin, q, v)
        A = M - dt * (Dl + De)
        dv = np.linalg.solve(A, dt * (Q + Ql + Qe))
        v_new = v + dv
        return self.integrate(q, v_new, dt), v_new, kin

    def energy(self, kin: Kinematics, v) -> tuple[float, float]:
        """Kinetic and gravitational potential energy [J]."""
        T = 0.0
        U = 0.0
        for i, b in enumerate(self.bodies):
            Iw = kin.R[i] @ b.inertia @ kin.R[i].T
            vi = kin.Jv[i] @ v
            wi = kin.Jw[i] @ v
            T += 0.5 * b.mass * vi @ vi + 0.5 * wi @ Iw @ wi
            U -= b.mass * self.gravity @ kin.c[i]
        return float(T), float(U)


__all__ = ["MultibodyTree", "Body", "RevoluteJoint", "Kinematics", "skew", "quat_to_matrix", "quat_from_matrix",
           "axis_angle", "quat_mul"]
