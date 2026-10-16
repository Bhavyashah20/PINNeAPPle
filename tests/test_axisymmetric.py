"""pinneapple_physics.solid.axisymmetric against closed-form elasticity solutions.

References: S. P. Timoshenko and J. N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill (1970): Lame's
thick-walled cylinder under internal pressure (plane strain), sigma_rr = C (1 - b^2/r^2), sigma_tt = C (1 + b^2/r^2),
C = p a^2 / (b^2 - a^2); Saint-Venant torsion of a circular shaft, u_theta = kappa r z, tau_tz = mu kappa r.
"""
import math

import pytest
import torch

from pinneapple_physics.solid.axisymmetric import (
    equilibrium_residual_rz,
    lame_parameters,
    stresses_rz,
    von_mises_from_components,
    von_mises_rz,
)

E, NU = 210e3, 0.3          # MPa (steel), lengths in mm -> stresses in MPa
A_IN, B_OUT, P = 50.0, 80.0, 20.0


def _points(n=200):
    g = torch.Generator().manual_seed(0)
    r = A_IN + (B_OUT - A_IN) * torch.rand(n, 1, generator=g, dtype=torch.float64)
    z = 100.0 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    return torch.cat([r, z], 1).requires_grad_(True)


def _lame_field(x):
    """Plane-strain Lame displacement u_r = A r + B / r with A, B from the pressure boundary conditions."""
    lam, mu = lame_parameters(E, NU)
    C = P * A_IN ** 2 / (B_OUT ** 2 - A_IN ** 2)
    A, B = C / (2 * (lam + mu)), C * B_OUT ** 2 / (2 * mu)
    r = x[:, 0:1]
    return torch.cat([A * r + B / r, torch.zeros_like(r)], 1), C


def test_lame_cylinder_stresses_and_zero_residual():
    x = _points()
    u, C = _lame_field(x)
    s = stresses_rz(u, x, E, NU)
    r = x[:, 0:1].detach()
    s_rr, s_tt = C * (1 - B_OUT ** 2 / r ** 2), C * (1 + B_OUT ** 2 / r ** 2)
    torch.testing.assert_close(s["s_rr"].detach(), s_rr, rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(s["s_tt"].detach(), s_tt, rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(s["s_zz"].detach(), NU * (s_rr + s_tt), rtol=1e-10, atol=1e-9)   # plane strain
    assert s["s_rz"].abs().max() == 0
    R = equilibrium_residual_rz(u, x, E, NU)
    assert R.shape == (200, 2) and R.abs().max().item() < 1e-9 * C

    vm = von_mises_rz(u, x, E, NU)
    s_zz = NU * (s_rr + s_tt)
    vm_ref = torch.sqrt(((s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2 + (s_tt - s_rr) ** 2) / 2)
    torch.testing.assert_close(vm.detach(), vm_ref, rtol=1e-10, atol=1e-9)


def test_lame_pressure_boundary_conditions():
    x = torch.tensor([[A_IN, 0.0], [B_OUT, 0.0]], dtype=torch.float64, requires_grad=True)
    u, _ = _lame_field(x)
    s_rr = stresses_rz(u, x, E, NU)["s_rr"].detach().ravel()
    torch.testing.assert_close(s_rr, torch.tensor([-P, 0.0], dtype=torch.float64), rtol=0, atol=1e-10)


def test_residual_matches_navier_displacement_form():
    """u_r = r^2, u_z = 0: R_r = (lambda + 2 mu)(u'' + u'/r - u/r^2) = 3 (lambda + 2 mu), R_z = 0."""
    x = _points(50)
    u = torch.cat([x[:, 0:1] ** 2, torch.zeros_like(x[:, 0:1])], 1)
    lam, mu = lame_parameters(E, NU)
    R = equilibrium_residual_rz(u, x, E, NU).detach()
    torch.testing.assert_close(R[:, 0], torch.full((50,), 3 * (lam + 2 * mu), dtype=torch.float64))
    assert R[:, 1].abs().max() == 0


def test_saint_venant_torsion():
    """u_theta = kappa r z: tau_tz = mu kappa r, tau_rt = 0, equilibrium holds, von Mises = sqrt(3) mu kappa r."""
    x = _points(80)
    kappa = 1e-4
    r, z = x[:, 0:1], x[:, 1:2]
    u = torch.cat([torch.zeros_like(r), torch.zeros_like(r), kappa * r * z], 1)
    _, mu = lame_parameters(E, NU)
    s = stresses_rz(u, x, E, NU)
    torch.testing.assert_close(s["s_tz"].detach(), mu * kappa * r.detach())
    assert s["s_rt"].abs().max().item() < 1e-12
    R = equilibrium_residual_rz(u, x, E, NU)
    assert R.shape == (80, 3) and R.abs().max().item() < 1e-9
    torch.testing.assert_close(von_mises_rz(u, x, E, NU).detach(), math.sqrt(3) * mu * kappa * r.detach())


def test_torsional_residual_is_mu_times_displacement_laplacian():
    """u_theta = r^3: R_theta / mu = u'' + u'/r - u/r^2 = 6r + 3r - r = 8 r."""
    x = _points(40)
    r = x[:, 0:1]
    u = torch.cat([torch.zeros_like(r), torch.zeros_like(r), r ** 3], 1)
    _, mu = lame_parameters(E, NU)
    R = equilibrium_residual_rz(u, x, E, NU).detach()
    torch.testing.assert_close(R[:, 2:3] / mu, 8 * r.detach())


def test_residual_is_differentiable_in_model_parameters():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)).double()
    x = _points(16)
    loss = (equilibrium_residual_rz(net(x), x, E, NU, r_min=1e-3) ** 2).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_von_mises_components_numpy_and_uniaxial():
    np = pytest.importorskip("numpy")
    assert von_mises_from_components(np.array([5.0]), np.array([0.0]), np.array([0.0]))[0] == pytest.approx(5.0)
    assert von_mises_from_components(0.0, 0.0, 0.0, 2.0) == pytest.approx(math.sqrt(3) * 2.0)   # pure shear


def test_x_must_require_grad():
    x = torch.ones(3, 2)
    with pytest.raises(ValueError):
        stresses_rz(x * 2, x, E, NU)
