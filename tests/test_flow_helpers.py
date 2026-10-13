"""Signal frequency, prism-layer sizing, finite-volume Courant number / wall distance and function-object tables."""
import numpy as np
import pytest

from pinneapple_data.cae import (
    Mesh,
    courant_number,
    mesh_courant_number,
    mesh_wall_distance,
    wall_distance,
)
from pinneapple_data.cae.model import CellBlock
from pinneapple_data.simulation_metadata.openfoam import parse_dat, read_dat
from pinneapple_design.fitmesh import size_layers
from pinneapple_physics.physics_audit.regime import dominant_frequency, strouhal


# ----------------------------------------------------------------------------------------------- frequency
def test_dominant_frequency_refines_a_non_bin_centred_sinusoid():
    dt, n = 0.01, 4000
    t = np.arange(n) * dt
    f0 = 3.217                                     # 2000 kept samples -> bin width 0.05 Hz, f0 between bins
    x = 2.0 + np.sin(2 * np.pi * f0 * t + 0.3)
    coarse = dominant_frequency(x, dt, refine=False)
    fine = dominant_frequency(x, dt)
    assert abs(coarse - f0) <= 0.5 / (0.5 * n * dt) + 1e-12      # within half a bin
    assert fine == pytest.approx(f0, rel=5e-3)
    assert abs(fine - f0) < abs(coarse - f0)
    noisy = x + 0.5 * np.random.default_rng(0).standard_normal(n)
    assert dominant_frequency(noisy, dt) == pytest.approx(f0, rel=5e-3)
    assert strouhal(fine, 0.1, 2.0) == pytest.approx(f0 * 0.05, rel=5e-3)


def test_dominant_frequency_options():
    dt = 0.5
    t = np.arange(512) * dt
    x = np.sin(2 * np.pi * 0.11 * t)
    assert dominant_frequency(x, dt, discard=0.0, window=True, refine=False, pad=16) == pytest.approx(0.11, rel=2e-3)
    assert dominant_frequency(np.ones(3), 1.0, discard=0.0) == 0.0


# ----------------------------------------------------------------------------------------------- layers
def test_size_layers_round_trips():
    full = size_layers(1e-5, expansion=1.2, n_layers=12)
    assert full["total_thickness"] == pytest.approx(1e-5 * (1.2 ** 12 - 1) / 0.2, rel=1e-14)
    assert full["last_height"] == pytest.approx(1e-5 * 1.2 ** 11)
    r = size_layers(1e-5, n_layers=12, total_thickness=full["total_thickness"])
    assert r["expansion"] == pytest.approx(1.2, rel=1e-10)
    n = size_layers(1e-5, expansion=1.2, total_thickness=full["total_thickness"])
    assert n["n_layers"] == 12 and n["total_thickness"] == pytest.approx(full["total_thickness"])
    up = size_layers(1e-5, expansion=1.2, total_thickness=1.01 * full["total_thickness"])
    assert up["n_layers"] == 13 and up["total_thickness"] >= 1.01 * full["total_thickness"]
    assert size_layers(2.0, n_layers=5, total_thickness=10.0)["expansion"] == 1.0
    assert size_layers(1.0, expansion=0.8, n_layers=4)["total_thickness"] == pytest.approx(1 + 0.8 + 0.64 + 0.512)
    shrink = size_layers(1.0, n_layers=4, total_thickness=2.952)
    assert shrink["expansion"] == pytest.approx(0.8, rel=1e-9)
    with pytest.raises(ValueError):
        size_layers(1e-5, expansion=1.2)
    with pytest.raises(ValueError):
        size_layers(1.0, expansion=0.5, total_thickness=5.0)        # converges to 2, never reaches 5


# ----------------------------------------------------------------------------------------------- FV metrics
def _hex_grid(nx, ny, nz, lx, ly, lz):
    xs, ys, zs = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1), np.linspace(0, lz, nz + 1)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange(len(P)).reshape(nx + 1, ny + 1, nz + 1)
    i, j, k = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    conn = np.stack([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                     idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]], 1)
    bottom = np.stack([idx[:-1, 0, :-1].ravel(), idx[1:, 0, :-1].ravel(), idx[1:, 0, 1:].ravel(),
                       idx[:-1, 0, 1:].ravel()], 1)
    top = np.stack([idx[:-1, -1, :-1].ravel(), idx[1:, -1, :-1].ravel(), idx[1:, -1, 1:].ravel(),
                    idx[:-1, -1, 1:].ravel()], 1)
    return Mesh(points=P, blocks=[CellBlock("hexahedron", conn)],
                boundary_blocks=[("lower", CellBlock("quad", bottom)), ("upper", CellBlock("quad", top))])


def test_courant_number_uniform_flow_on_hex_grid():
    m = _hex_grid(6, 5, 4, 1.2, 0.5, 0.8)            # dx = 0.2, dy = 0.1, dz = 0.2
    dt = 0.01
    co = mesh_courant_number(m, np.tile([3.0, 0.0, 0.0], (m.n_cells, 1)), dt)
    assert np.allclose(co, 3.0 * dt / 0.2, rtol=1e-12)
    co = mesh_courant_number(m, np.tile([0.0, -2.0, 0.0], (m.n_cells, 1)), dt)
    assert np.allclose(co, 2.0 * dt / 0.1, rtol=1e-12)
    co = mesh_courant_number(m, np.tile([3.0, -2.0, 0.0], (m.n_cells, 1)), dt)      # sum over directions
    assert np.allclose(co, dt * (3.0 / 0.2 + 2.0 / 0.1), rtol=1e-12)


def test_courant_number_from_face_velocities():
    # two unit cubes side by side: faces 0 internal (x = 1), then boundary faces
    sf = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]], float)
    own = np.array([0, 0, 1, 0, 0, 1, 1])
    nei = np.array([1])
    uf = np.tile([2.0, 0.0, 0.0], (7, 1))
    co = courant_number(sf, own, nei, uf, np.ones(2), 0.1)
    assert np.allclose(co, 2.0 * 0.1)
    with pytest.raises(ValueError):
        courant_number(sf, own, nei, np.ones((3, 3)), np.ones(2), 0.1)


def test_wall_distance_on_a_channel():
    H, ny = 1.0, 10
    m = _hex_grid(8, ny, 2, 2.0, H, 0.5)
    from pinneapple_data.cae.compare import cell_centres_volumes
    cc, _ = cell_centres_volumes(m)
    exact = np.minimum(cc[:, 1], H - cc[:, 1])
    d = mesh_wall_distance(m, ["lower", "upper"])
    assert np.allclose(d, exact, atol=1e-12)
    d_raw = mesh_wall_distance(m, ["lower", "upper"], project=False)
    assert np.all(d_raw >= exact - 1e-12) and np.all(d_raw - exact <= 0.5 * H / ny)
    assert np.allclose(wall_distance(cc, np.array([[0.0, 0.0, 0.0]])), np.linalg.norm(cc, axis=1))
    with pytest.raises(ValueError):
        mesh_wall_distance(m)                        # FE patches carry no "wall" type: name them


# ----------------------------------------------------------------------------------------------- .dat tables
FORCE_COEFFS = """# Force coefficients
# dragDir     : (1.000000e+00 0.000000e+00 0.000000e+00)
# liftDir     : (0.000000e+00 1.000000e+00 0.000000e+00)
#
# Time          Cd              Cs              Cl              CmRoll          CmPitch         CmYaw           Cd(f)           Cd(r)
1               1.2             0.0             0.31            0.0             0.05            0.0             0.6             0.6
2               1.1             0.0             0.30            0.0             0.04            0.0             0.55            0.55
3               1.05            0.0             0.29            0.0             0.04            0.0             0.525           0.525
"""

PROBES = """# Probe 0 (0.1 0 0)
# Probe 1 (0.2 0 0)
#       Probe             0             1
#        Time
0.1	(1 2 3)	(4 5 6)
0.2	(1.5 2.5 3.5)	(4.5 5.5 6.5)
"""

FORCES_OLD = """# Time forces(pressure viscous) moment(pressure viscous)
1 ((1 2 3) (4 5 6)) ((7 8 9) (10 11 12))
"""


def test_read_force_coeffs_dat(tmp_path):
    f = tmp_path / "forceCoeffs.dat"
    f.write_text(FORCE_COEFFS)
    cols, a = read_dat(str(f))
    assert cols == ["Time", "Cd", "Cs", "Cl", "CmRoll", "CmPitch", "CmYaw", "Cd(f)", "Cd(r)"]
    assert a.shape == (3, 9)
    assert a[:, cols.index("Cl")].tolist() == [0.31, 0.30, 0.29]


def test_parse_dat_vectors():
    cols, a = parse_dat(PROBES.replace("#        Time\n", "").replace("#       Probe", "# Time"))
    assert cols == ["Time", "0_x", "0_y", "0_z", "1_x", "1_y", "1_z"]
    assert a[1].tolist() == [0.2, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    cols, a = parse_dat(FORCES_OLD)
    assert cols[:4] == ["Time", "forces_pressure_x", "forces_pressure_y", "forces_pressure_z"]
    assert cols[-1] == "moment_viscous_z" and a[0, -1] == 12.0
    cols, a = parse_dat("1 2\n3 4\n")
    assert cols == ["col_0", "col_1"] and a.shape == (2, 2)


def test_extract_monitors_uses_the_reader():
    from pinneapple_data.simulation_metadata import _monitors
    rows = "".join(f"{i} {1 / (i + 1)} (1 {i} 0)\n" for i in range(10))
    fs = {"case/postProcessing/probes/0/U.dat": ("# Time p U\n" + rows).encode()}
    mon = _monitors(fs, "case/")
    assert [m["name"] for m in mon] == ["p", "U_x", "U_y", "U_z"]
    assert mon[2]["final"] == 9.0
