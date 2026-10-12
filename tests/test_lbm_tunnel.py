import numpy as np
import pytest

from pinneapple_simulation.external_solvers import fluidx3d as fx
from pinneapple_simulation.numerical_solvers.lbm_tunnel import (
    Units,
    WindTunnelLBM,
    read_vtk,
    sphere,
    stokes_sphere_field,
    velocity_set,
    voxelize_stl,
    write_vtk,
)

PERIODIC_XY = {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic", "z-": "wall", "z+": "wall"}


@pytest.mark.parametrize("name", ["D3Q19", "D3Q27"])
def test_velocity_sets_are_isotropic(name):
    c, w, opp = velocity_set(name)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(np.einsum("q,qa,qb->ab", w, c, c), np.eye(3) / 3)
    assert (c[opp] == -c).all()


@pytest.mark.parametrize("collision", ["srt", "trt"])
def test_force_driven_poiseuille(collision):
    nz, nu, F = 18, 0.1, 1e-5
    lbm = WindTunnelLBM((2, 2, nz), nu=nu, force=(F, 0, 0), faces=PERIODIC_XY, collision=collision)
    lbm.run(3500)
    u = lbm.fields()["u"][0, 1, 1, 1:-1]
    H = nz - 2
    z = np.arange(1, nz - 1) - 0.5                       # halfway bounce-back: walls at z = 0.5 and nz - 1.5
    exact = F / (2 * nu) * z * (H - z)
    assert np.abs(u - exact).max() / exact.max() < 0.01


def test_fp16_storage_halves_memory_and_keeps_mass():
    a = WindTunnelLBM((12, 8, 8), nu=0.05, u_in=(0.05, 0, 0), faces={k: "periodic" for k in WindTunnelLBM.FACES})
    b = WindTunnelLBM((12, 8, 8), nu=0.05, u_in=(0.05, 0, 0), faces={k: "periodic" for k in WindTunnelLBM.FACES},
                      storage="fp16s")
    assert b.bytes_per_cell() * 2 == a.bytes_per_cell() == 76
    for lbm in (a, b):
        lbm.add_solid(sphere(lbm.shape, (6, 4, 4), 2))
        m0 = lbm.fields()["rho"][~lbm.solid.numpy()].sum()
        lbm.run(200)
        m1 = lbm.fields()["rho"][~lbm.solid.numpy()].sum()
        assert abs(m1 / m0 - 1) < 1e-3
    ua, ub = a.fields()["u"], b.fields()["u"]
    assert np.abs(ua - ub).max() < 2e-3                    # same flow within the 16-bit rounding


def test_stokes_drag_on_a_small_sphere():
    R, nu, u = 4.0, 1 / 6, 0.002
    L = int(8 * R)
    c = ((L - 1) / 2,) * 3
    lbm = WindTunnelLBM((L, L, L), nu=nu, u_in=(u, 0, 0), faces={k: "open" for k in WindTunnelLBM.FACES})
    lbm.add_solid(sphere(lbm.shape, c, R))
    lbm.initialize(stokes_sphere_field(lbm.shape, c, R, (u, 0, 0)))
    lbm.run(400)
    Fx = lbm.force_on_solid()[0]
    assert Fx == pytest.approx(6 * np.pi * nu * R * u, rel=0.08)


def test_units_round_trip():
    un = Units.from_si(lbm_length=40, lbm_velocity=0.05, lbm_rho=1, si_length=2.0, si_velocity=30.0, si_rho=1.2)
    assert un.si_x(un.x(1.3)) == pytest.approx(1.3)
    assert un.u(30.0) == pytest.approx(0.05)
    assert un.si_F(un.F(12.0)) == pytest.approx(12.0)
    assert un.nu(1.5e-5) == pytest.approx(1.5e-5 * un.s / un.m ** 2)


def test_vtk_round_trip(tmp_path):
    a = np.random.default_rng(0).normal(size=(3, 5, 4, 3)).astype(np.float32)
    write_vtk(str(tmp_path / "u.vtk"), a, origin=(1, 2, 3), spacing=0.5)
    r = read_vtk(str(tmp_path / "u.vtk"))
    assert np.allclose(r["data"], a) and r["dimensions"] == [5, 4, 3] and r["spacing"] == [0.5] * 3


def test_voxelize_stl_volume():
    trimesh = pytest.importorskip("trimesh")
    mask = voxelize_stl(trimesh.creation.icosphere(subdivisions=3), (40, 40, 40), (20, 20, 20), size=20.0)
    assert mask.sum().item() == pytest.approx(4 / 3 * np.pi * 10 ** 3, rel=0.08)


# ------------------------------------------------------------------------------- FluidX3D bridge (no GPU needed)
@pytest.fixture
def checkout(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "setup.cpp").write_text('#include "setup.hpp"\n\n/*void main_setup() { // benchmark\n}*/\n'
                                                "void main_setup() { // shipped default\n}\n")
    (tmp_path / "src" / "defines.hpp").write_text("#define D3Q19\n#define BENCHMARK\n//#define SUBGRID\n"
                                                  "//#define EQUILIBRIUM_BOUNDARIES\n//#define FORCE_FIELD\n")
    monkeypatch.setenv("FLUIDX3D_HOME", str(tmp_path))
    return tmp_path


def test_setup_and_defines_are_written_with_backups(checkout):
    s = fx.WindTunnelSetup(length_m=1.0, velocity_m_s=10.0, nu_m2_s=1.5e-5, stl="stl/car.stl", floor=True)
    cpp = s.to_cpp()
    for call in ("units.set_m_kg_s", "LBM lbm(lbm_N, lbm_nu)", "voxelize_stl", "object_force(TYPE_S|TYPE_X)",
                 "u.write_device_to_vtk()", fx.MARK):
        assert call in cpp
    fx.write_setup(s)
    fx.enable_features(s.features())
    setup = (checkout / "src" / "setup.cpp").read_text()
    assert setup.count("void main_setup()") == 1 and "PINNeAPPle wind tunnel" in setup
    defines = (checkout / "src" / "defines.hpp").read_text()
    assert "\n#define SUBGRID" in defines and "//#define BENCHMARK" in defines
    assert (checkout / "src" / "setup.cpp.pinneapple.bak").exists()


def test_results_are_read_from_log_and_vtk(checkout):
    (checkout / "log.pinneapple").write_text(f"| {fx.MARK} 0.1 1.5 -0.2 0.0\n| {fx.MARK} 0.2 1.4 -0.1 0.0\n")
    exp = checkout / "bin" / "export"
    exp.mkdir(parents=True)
    u = np.ones((3, 4, 4, 4), np.float32)
    write_vtk(str(exp / "u-000000100.vtk"), u)
    write_vtk(str(exp / "u-000000200.vtk"), 2 * u)
    res = fx.read_results()
    assert res["forces"].shape == (2, 4) and res["forces"][-1, 1] == pytest.approx(1.4)
    assert res["fields"]["u"]["step"] == 200 and np.allclose(res["fields"]["u"]["data"], 2.0)
    samples = fx.results_to_samples(res)
    assert len(samples) == 1 and tuple(samples[0].state["u"].shape) == (3, 4, 4, 4)
