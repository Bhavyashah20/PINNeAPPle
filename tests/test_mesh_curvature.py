"""Reference tests for pinneapple_design.geometry.ops.curvature: closed-form
curvatures of sphere / cylinder / plane and the discrete Gauss-Bonnet theorem
(exact identity for the angle-defect operator)."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_design.geometry.ops.curvature import (
    boundary_vertices, euler_characteristic, gaussian_curvature, laplace_beltrami,
    mean_curvature, mixed_areas, principal_curvatures)


def icosphere(subdiv: int, radius: float = 1.0):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        cache, F2 = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return radius * np.array(V), np.array(F)


def grid_mesh(n, fn):
    """Triangulated (u, v) grid mapped by fn(u, v) -> (x, y, z)."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    V = np.stack(fn(u.ravel(), v.ravel()), axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return V, F


def test_sphere_closed_form_curvatures_and_area():
    R = 2.0
    V, F = icosphere(4, R)
    K = gaussian_curvature(V, F)
    H = mean_curvature(V, F)
    assert np.abs(K * R ** 2 - 1).max() < 0.03          # K = 1/R^2
    assert np.abs(H * R - 1).max() < 0.03               # H = +1/R (outward winding)
    assert mixed_areas(V, F).sum() == pytest.approx(4 * np.pi * R ** 2, rel=2e-3)
    k1, k2 = principal_curvatures(V, F)
    assert np.nanmax(np.abs(k1 - k2)) < 0.4 / R         # umbilic: k1 = k2 = 1/R


@pytest.mark.parametrize("subdiv", [0, 1, 3])
def test_gauss_bonnet_is_exact_on_closed_surface(subdiv):
    V, F = icosphere(subdiv)
    total = (gaussian_curvature(V, F) * mixed_areas(V, F)).sum()
    assert euler_characteristic(V, F) == 2
    assert total == pytest.approx(2 * np.pi * 2, abs=1e-10)


def test_gauss_bonnet_with_boundary_on_a_bent_patch():
    """Disk-topology patch (chi = 1): interior defects + boundary geodesic
    turning sum to 2 pi chi exactly."""
    V, F = grid_mesh(15, lambda u, v: (u, v, 0.3 * np.sin(2 * u) * np.cos(3 * v)))
    total = (gaussian_curvature(V, F, boundary="geodesic") * mixed_areas(V, F)).sum()
    assert euler_characteristic(V, F) == 1
    assert total == pytest.approx(2 * np.pi, abs=1e-10)


def test_cylinder_closed_form():
    """Cylinder radius R: K = 0, |H| = 1/(2R) away from the boundary."""
    R = 1.5
    V, F = grid_mesh(41, lambda u, v: (R * np.cos(2 * np.pi * u * 0.5), R * np.sin(2 * np.pi * u * 0.5), 2 * v))
    inner = ~boundary_vertices(V, F)
    K = gaussian_curvature(V, F, boundary="nan")
    H = np.abs(mean_curvature(V, F))
    assert np.abs(K[inner]).max() < 1e-2
    assert np.abs(H[inner] * 2 * R - 1).max() < 0.02


def test_flat_plane_and_linear_fields():
    V, F = grid_mesh(12, lambda u, v: (u, v, 0 * u))
    inner = ~boundary_vertices(V, F)
    assert np.abs(gaussian_curvature(V, F)[inner]).max() < 1e-12
    assert np.abs(mean_curvature(V, F)[inner]).max() < 1e-12
    f = 3 * V[:, 0] - 2 * V[:, 1] + 1                     # harmonic
    assert np.abs(laplace_beltrami(f, V, F)[inner]).max() < 1e-10
    g = V[:, 0] ** 2 + V[:, 1] ** 2                       # Laplacian = 4
    assert np.allclose(laplace_beltrami(g, V, F)[inner], 4.0, atol=1e-9)
