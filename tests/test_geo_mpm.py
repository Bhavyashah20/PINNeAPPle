"""Geomechanics MPM (granular Drucker-Prager, weakly compressible fluid, elastic) and the GeoTaichi bridge."""
import numpy as np
import pytest
import torch

from pinneapple_simulation.particle_dynamics.geo_mpm import (
    Elastic,
    Fluid,
    GeoMPM,
    Sand,
    box,
    read_particle_frames,
    runout,
    write_particle_frame,
)

torch.set_num_threads(1)


def test_free_fall_and_conservation():
    sim = GeoMPM(domain=(0.6, 1.0), dx=0.02)
    sim.add_body(box((0.2, 0.6), (0.4, 0.8)), Elastic(E=1e5, rho=1000.0))
    m0 = float(sim.m.sum())
    y0 = float((sim.m * sim.x[:, 1]).sum()) / m0
    fr = sim.run(t_end=0.2, save_every=0.2)
    y1 = float((sim.m * sim.x[:, 1]).sum()) / m0
    assert y0 - y1 == pytest.approx(0.5 * 9.81 * fr[-1]["time"] ** 2, rel=0.01)   # COM falls at g before impact
    assert float(sim.m.sum()) == pytest.approx(m0)
    assert np.allclose(fr[-1]["velocity"][:, 1], -9.81 * fr[-1]["time"], rtol=0.02)  # rigid fall, no spurious strain


def test_fluid_column_is_hydrostatic():
    sim = GeoMPM(domain=(0.4, 0.5), dx=0.01, floor="slip", walls="slip")
    sim.add_body(box((0.0, 0.0), (0.4, 0.3)), Fluid(K=2e5, rho=1000.0))
    v0 = float(sim.vol0.sum())
    frames = sim.run(t_end=0.4, save_every=0.01)[10:]       # the column breathes (weak compressibility, no damping):
    Y = np.concatenate([f["position"][:, 1] for f in frames])  # fit the time-pooled pressure profile over 0.1-0.4 s
    P = np.concatenate([-0.5 * np.trace(f["stress"], axis1=1, axis2=2) for f in frames])
    m = (Y > 0.03) & (Y < 0.25)
    slope = -np.polyfit(Y[m], P[m], 1)[0]
    assert slope == pytest.approx(1000 * 9.81, rel=0.05)      # dp/dz = rho g
    assert np.sqrt((frames[-1]["velocity"] ** 2).sum(1).mean()) < 0.3    # rms: breathing, no sloshing
    assert frames[-1]["volume"].sum() == pytest.approx(v0, rel=0.02)     # weakly compressible: J stays near 1


def test_sand_column_spreads_and_stops():
    L0, a = 0.2, 0.5
    sim = GeoMPM(domain=(0.8, 0.25), dx=0.02, floor="sticky", walls="slip")
    sim.add_body(box((0, 0), (L0, a * L0)), Sand(friction_deg=30.0))
    fr = sim.run(t_end=0.8, save_every=0.4)
    dl = (runout(fr[-1]["position"]) - L0) / L0
    assert 0.2 < dl < 1.0                                      # Lube et al. (2005): 1.2 a = 0.6
    assert np.linalg.norm(fr[-1]["velocity"], axis=1).max() < 0.05   # the deposit has come to rest
    sand_heap = np.quantile(fr[-1]["position"][:, 1], 0.995)
    assert sand_heap < a * L0 + 0.02                           # no particle climbs above the column


def test_frames_on_disk_and_geotaichi_layout(tmp_path):
    fr = {"time": 0.1, "position": np.random.rand(5, 3), "velocity": np.zeros((5, 3)), "volume": np.ones(5)}
    write_particle_frame(str(tmp_path), 0, fr)
    back = read_particle_frames(str(tmp_path))
    assert len(back) == 1 and np.allclose(back[0]["position"], fr["position"])
    # a GeoTaichi-style frame with velocity as an (vx, vy, vz) tuple and a pickled dict entry
    d = tmp_path / "gt" / "particles"
    d.mkdir(parents=True)
    vel = np.empty(3, dtype=object)
    vel[:] = [np.ones(4), 2 * np.ones(4), np.zeros(4)]
    np.savez(d / "MPMParticle000000.npz", position=np.zeros((4, 3)), velocity=np.c_[np.ones(4), 2 * np.ones(4),
                                                                                    np.zeros(4)],
             state_vars=np.array({"epdev": np.zeros(4)}, dtype=object), vel_tuple=vel)
    from pinneapple_simulation.external_solvers.geotaichi import frames_to_samples, read_geotaichi

    g = read_geotaichi(str(tmp_path / "gt"))                   # pickled entries skipped by default
    assert "state_vars" not in g[0]
    assert g[0]["velocity"].shape == (4, 3) and np.allclose(g[0]["velocity"][:, 1], 2.0)
    t = read_geotaichi(str(tmp_path / "gt"), allow_pickle=True)    # trusted file: tuples stacked into (n, d)
    assert t[0]["vel_tuple"].shape == (4, 3) and "state_vars" in t[0]
    s = frames_to_samples(g)
    assert tuple(s[0].state["velocity"].shape) == (4, 3)
    with pytest.raises(FileNotFoundError):
        read_geotaichi(str(tmp_path / "nothing"))
