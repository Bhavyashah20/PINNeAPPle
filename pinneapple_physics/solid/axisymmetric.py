"""Axisymmetric linear elasticity in cylindrical coordinates ``(r, z)``, by autograd.

A displacement field ``u(x)`` with ``x = [r, z]`` (``(N, 2)``, ``requires_grad``) and ``u = [u_r, u_z]`` or
``[u_r, u_z, u_theta]`` (``(N, 2)`` or ``(N, 3)``, computed from ``x``) is independent of ``theta``. With the
optional torsional component ``u_theta`` (torsion of a solid of revolution) the small strains are

    e_rr = du_r/dr,   e_tt = u_r / r,   e_zz = du_z/dz,   e_rz = (du_r/dz + du_z/dr) / 2,
    e_rt = (du_t/dr - u_t / r) / 2,     e_tz = (du_t/dz) / 2          (tensor shear strains),

Hooke's law ``sigma = lambda tr(e) I + 2 mu e`` with ``lambda = E nu / ((1 + nu)(1 - 2 nu))``,
``mu = E / (2 (1 + nu))``, and the equilibrium equations without body force are

    R_r = d s_rr/dr + d s_rz/dz + (s_rr - s_tt) / r
    R_z = d s_rz/dr + d s_zz/dz + s_rz / r
    R_t = d s_rt/dr + d s_tz/dz + 2 s_rt / r.

The meridional (``u_r, u_z``) and torsional (``u_theta``) problems decouple. Reference: S. P. Timoshenko and
J. N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill (1970): equilibrium and strain-displacement relations in
cylindrical coordinates, the solid of revolution under axisymmetric load and torsion of shafts of variable diameter;
the thick-walled cylinder (Lame) solution ``u_r = A r + B / r`` is the test case of ``tests/test_axisymmetric.py``.

Units are whatever ``x``, ``u`` and ``E`` use (``E`` in Pa with ``u`` and ``x`` in the same length unit gives Pa).
"""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["lame_parameters", "strains_rz", "stresses_rz", "equilibrium_residual_rz", "von_mises_rz",
           "von_mises_from_components"]


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """``(lambda, mu)`` from Young's modulus ``E`` and Poisson's ratio ``nu`` (isotropic)."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def _grad(f: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=create_graph, retain_graph=True)[0]


def _radius(x: torch.Tensor, r_min: float | None) -> torch.Tensor:
    r = x[:, 0:1]
    return r.clamp(min=r_min) if r_min is not None else r


def strains_rz(u: torch.Tensor, x: torch.Tensor, *, r_min: float | None = None,
               create_graph: bool = True) -> dict[str, torch.Tensor]:
    """Small strains ``e_rr, e_tt, e_zz, e_rz`` (and ``e_rt, e_tz`` when ``u`` has ``u_theta``), each ``(N, 1)``.

    Shear entries are tensor components (half the engineering shear strain). ``r_min`` clamps ``r`` in ``u_r / r``
    (points on the axis). ``create_graph=True`` keeps the result differentiable (needed for residuals and training).
    """
    if not x.requires_grad:
        raise ValueError("x must require grad and u must be computed from it")
    r = _radius(x, r_min)
    du_r = _grad(u[:, 0:1], x, create_graph)
    du_z = _grad(u[:, 1:2], x, create_graph)
    e = {"e_rr": du_r[:, 0:1], "e_tt": u[:, 0:1] / r, "e_zz": du_z[:, 1:2],
         "e_rz": 0.5 * (du_r[:, 1:2] + du_z[:, 0:1])}
    if u.shape[1] >= 3:
        du_t = _grad(u[:, 2:3], x, create_graph)
        e["e_rt"] = 0.5 * (du_t[:, 0:1] - u[:, 2:3] / r)
        e["e_tz"] = 0.5 * du_t[:, 1:2]
    return e


def stresses_rz(u: torch.Tensor, x: torch.Tensor, E: float, nu: float, *, r_min: float | None = None,
                create_graph: bool = True) -> dict[str, torch.Tensor]:
    """Stresses ``s_rr, s_tt, s_zz, s_rz`` (and ``s_rt, s_tz`` with torsion), each ``(N, 1)``, by Hooke's law."""
    lam, mu = lame_parameters(E, nu)
    e = strains_rz(u, x, r_min=r_min, create_graph=create_graph)
    tr = e["e_rr"] + e["e_tt"] + e["e_zz"]
    s = {"s_rr": lam * tr + 2 * mu * e["e_rr"], "s_tt": lam * tr + 2 * mu * e["e_tt"],
         "s_zz": lam * tr + 2 * mu * e["e_zz"], "s_rz": 2 * mu * e["e_rz"]}
    if "e_rt" in e:
        s["s_rt"] = 2 * mu * e["e_rt"]
        s["s_tz"] = 2 * mu * e["e_tz"]
    return s


def equilibrium_residual_rz(u: torch.Tensor, x: torch.Tensor, E: float, nu: float, *,
                            r_min: float | None = None) -> torch.Tensor:
    """Pointwise equilibrium residual ``[R_r, R_z]`` (``(N, 2)``), plus ``R_theta`` (``(N, 3)``) with torsion.

    Units of stress per length; zero for an exact elastic solution without body force. Differentiable in the model
    parameters (use ``(R ** 2).mean()`` as a physics loss; divide ``R_theta`` by ``mu`` for the torsion equation
    ``d2u/dr2 + (1/r) du/dr - u/r^2 + d2u/dz2 = 0`` in displacement form).
    """
    s = stresses_rz(u, x, E, nu, r_min=r_min, create_graph=True)
    r = _radius(x, r_min)
    d_rr = _grad(s["s_rr"], x, True)
    d_rz = _grad(s["s_rz"], x, True)
    d_zz = _grad(s["s_zz"], x, True)
    res = [d_rr[:, 0:1] + d_rz[:, 1:2] + (s["s_rr"] - s["s_tt"]) / r,
           d_rz[:, 0:1] + d_zz[:, 1:2] + s["s_rz"] / r]
    if "s_rt" in s:
        d_rt = _grad(s["s_rt"], x, True)
        d_tz = _grad(s["s_tz"], x, True)
        res.append(d_rt[:, 0:1] + d_tz[:, 1:2] + 2 * s["s_rt"] / r)
    return torch.cat(res, dim=1)


def von_mises_from_components(s_rr: Any, s_zz: Any, s_tt: Any, s_rz: Any = 0.0, s_rt: Any = 0.0,
                              s_tz: Any = 0.0) -> Any:
    """Von Mises stress ``sqrt(((s_rr-s_zz)^2 + (s_zz-s_tt)^2 + (s_tt-s_rr)^2 + 6 (s_rz^2 + s_rt^2 + s_tz^2)) / 2)``.

    Works on numpy arrays and torch tensors alike.
    """
    j = ((s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2 + (s_tt - s_rr) ** 2
         + 6 * (s_rz ** 2 + s_rt ** 2 + s_tz ** 2)) / 2
    return j ** 0.5


def von_mises_rz(u: torch.Tensor, x: torch.Tensor, E: float, nu: float, *, r_min: float | None = None,
                 create_graph: bool = False) -> torch.Tensor:
    """Von Mises stress ``(N, 1)`` of ``u`` (all six stress components when ``u`` has ``u_theta``)."""
    s = stresses_rz(u, x, E, nu, r_min=r_min, create_graph=create_graph)
    return von_mises_from_components(s["s_rr"], s["s_zz"], s["s_tt"], s["s_rz"],
                                     s.get("s_rt", 0.0), s.get("s_tz", 0.0))
