"""Black-hole accretion solver and forecaster (issue #399)."""
import math

import numpy as np
import pytest
import torch

from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, bondi_pw, torus_state
from pinneapple_physics.blackhole.forecast import (
    DensityCodec,
    DuarteUNet,
    MassEnvelope,
    lead_time_scores,
    make_blocks,
    rollout,
    trust_horizon,
)
from pinneapple_physics.blackhole.twin import CutawayMesh, accretion_scene


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def test_mass_and_angular_momentum_budgets_close_to_round_off():
    flow = AccretionFlow(RIAFConfig(nr=48, ntheta=24, alpha=0.1, viscosity="SS"))
    flow.set_primitives(torus_state(flow))
    b0 = flow.budget()
    for _ in range(150):
        flow.step()
    assert flow.capped == 0                       # no safety cap acted, so nothing was added or removed
    b1 = flow.budget()
    for k in ("mass", "angmom"):
        assert abs(b1[k] - b0[k]) / b0[k] < 1e-12, k
    assert flow.boundary["mass_in"] > 0           # gas already falls into the hole


def test_uniform_pressure_at_rest_stays_at_rest_without_gravity_terms():
    """Geometric source terms balance a uniform pressure exactly (the classic spherical-grid pitfall)."""
    flow = AccretionFlow(RIAFConfig(nr=24, ntheta=16, viscosity="none"))
    W = np.zeros((5, 24, 16))
    W[0], W[4] = 1.0, 1.0
    flow.set_primitives(W)
    dU, _, _ = flow.rhs(flow.W)
    # momentum-theta and energy have no gravity, so they must vanish; r-momentum carries only -rho dPhi/dr
    assert float(dU[2].abs().max()) < 1e-12
    r = flow.r[2:-2]
    grav = -1.0 / (r - 2.0) ** 2
    assert float((dU[1] - grav).abs().max()) < 1e-12


def test_inviscid_torus_stays_in_equilibrium():
    cfg = RIAFConfig(nr=64, ntheta=32, viscosity="none", perturbation=0.0)
    flow = AccretionFlow(cfg)
    W0 = torus_state(flow)
    flow.set_primitives(W0)
    flow.run(t_end=150.0, every=150.0)
    rho = flow.primitives()[0]
    core = W0[0] > 0.3                             # torus body, away from its surface
    rel = np.abs(rho[core] / W0[0][core] - 1)
    # the steep torus surface diffuses at this resolution; the body holds its equilibrium
    assert np.median(rel) < 0.03 and np.percentile(rel, 90) < 0.1


def test_bondi_solution_is_transonic_and_conserves_mass_flux():
    r = np.geomspace(4.0, 400.0, 60)
    rho, v, p, mdot, rc = bondi_pw(r, cs_inf=0.1, gamma=1.4)
    flux = 4 * math.pi * r ** 2 * rho * (-v)
    assert np.allclose(flux, mdot, rtol=1e-8)
    mach = -v / np.sqrt(1.4 * p / rho)
    assert np.all(mach[r < rc * 0.99] > 1) and np.all(mach[r > rc * 1.01] < 1)


def test_bondi_accretion_is_steady_in_the_solver():
    gam = 1.4
    cfg = RIAFConfig(nr=96, ntheta=4, gamma=gam, viscosity="none", perturbation=0.0, outer_bc="fixed", t_cap=None)
    flow = AccretionFlow(cfg)
    r = flow.grid()["r"]
    rho, v, p, mdot, _ = bondi_pw(r, cs_inf=0.1, gamma=gam)
    W = np.zeros((5, len(r), 4))
    W[0], W[1], W[4] = rho[:, None], v[:, None], p[:, None]
    flow.set_primitives(W)
    out = flow.run(t_end=1000.0, every=1000.0)
    assert abs(out["mdot"][-1] / mdot - 1) < 0.02
    assert np.median(np.abs(flow.primitives()[0, :, 1] / rho - 1)) < 0.02


def test_unet_shapes_and_blocks():
    x = np.random.default_rng(0).random((20, 32, 16)).astype(np.float32)
    X, Y = make_blocks(x, 5)
    assert X.shape == (11, 5, 32, 16) and np.array_equal(Y[0], x[5:10])
    net = DuarteUNet(filters=4)
    out = net(torch.from_numpy(X[:2]))
    assert out.shape == (2, 5, 32, 16)
    roll = rollout(net, X[0], 3)
    assert roll.shape == (15, 32, 16)


def test_codec_roundtrip_and_crop():
    rho = 10.0 ** np.random.default_rng(1).uniform(-5, 0, (3, 40, 20))
    codec = DensityCodec.fit(rho, r_cells=32, theta_trim=2)
    z = codec.encode(rho)
    assert z.shape == (3, 32, 16) and z.min() >= 0 and z.max() <= 1
    assert np.allclose(codec.decode(z), rho[:, :32, 2:18], rtol=1e-5)


def test_trust_horizon_and_mass_envelope():
    truth = np.random.default_rng(2).random((10, 8, 8))
    mean = truth.mean(0)
    pred = truth.copy()
    pred[6:] = mean                                   # the forecast collapses to the mean after 6 frames
    s = lead_time_scores(pred, truth, truth[0], mean)
    h = trust_horizon(s, frame_dt=10.0, acc_min=0.6)
    assert h["acc"] == 60.0
    env = MassEnvelope.fit([np.array([1.0, 0.99, 0.98, 0.97])], frame_dt=1.0)
    assert env.first_violation(np.array([0.96, 0.95, 1.2]), 1.0, m0=0.97) == 2


def test_twin_scene_cutaway():
    flow = AccretionFlow(RIAFConfig(nr=32, ntheta=16))
    flow.set_primitives(torus_state(flow))
    g = flow.grid()
    lr = np.log10(flow.primitives()[0])[None].repeat(2, 0)
    mesh = CutawayMesh(g["r_faces"], g["theta_faces"], r_view=60.0, n_phi=12)
    assert mesh.values(lr).shape == (2, len(mesh.vertices))
    sc = accretion_scene(g, {"simulation": lr, "forecast": lr}, [0.0, 10.0], r_view=60.0,
                         sensors=[{"id": "err", "panel": "forecast", "series": [0.0, 0.1], "envelope": (0, 0.2)}])
    assert len(sc.parts) == 4 and sc.parts[0].fields["log10_density"].shape[0] == 2


def test_ray_tracer_shadow_has_the_schwarzschild_size():
    """Rays with impact parameter below 3 sqrt(3) M fall in; above it they escape."""
    from pinneapple_physics.blackhole.raytrace import SHADOW_B, Camera
    flow = AccretionFlow(RIAFConfig(nr=32, ntheta=16))
    cam = Camera(width=61, height=61, r_cam=1000.0, inclination_deg=90.0, fov_deg=1.6, ds_out=40.0)
    paths = cam.trace(flow.grid(), r_max=20.0)
    d = cam.directions()
    pos = cam.basis()[0]
    sin_a = np.linalg.norm(np.cross(d, pos / np.linalg.norm(pos)), axis=1)
    b = cam.r_cam * sin_a / math.sqrt(1 - 2 / cam.r_cam)
    clear = np.abs(b - SHADOW_B) > 0.15
    assert np.array_equal(paths.captured[clear], (b < SHADOW_B)[clear])


def test_ray_tracer_renders_a_lensed_thin_disc():
    """A thin equatorial disc seen nearly edge on also shows its far side above the shadow (lensing)."""
    from pinneapple_physics.blackhole.raytrace import Camera, render
    flow = AccretionFlow(RIAFConfig(nr=64, ntheta=64))
    g = flow.grid()
    r, th = g["r"], g["theta"]
    rho = np.where((np.abs(th - np.pi / 2)[None] < 0.03) & (r[:, None] > 6) & (r[:, None] < 30), 1.0, 0.0)
    cam = Camera(width=96, height=54, r_cam=150.0, inclination_deg=86.0, fov_deg=24.0)
    inten = render(cam.trace(g, r_max=35.0), rho, np.ones_like(rho), None, doppler=False, return_intensity=True)
    col = inten[:, 48]                                    # vertical line through the hole
    top = col[:20].max()                              # lensed image of the far side, above the shadow
    assert top > 0.05 * col.max()
    assert inten[18, 48] < 0.02 * col.max()               # the shadow, between the arc and the disc, is dark


def test_numba_backend_matches_the_torch_reference():
    pytest.importorskip("numba")
    out = {}
    for backend in ("torch", "numba"):
        flow = AccretionFlow(RIAFConfig(nr=48, ntheta=24, backend=backend))
        flow.set_primitives(torus_state(flow))
        for _ in range(60):
            flow.step()
        out[backend] = (flow.primitives(), flow.t, flow.boundary["mass_in"])
    a, b = out["torch"], out["numba"]
    assert abs(a[1] - b[1]) < 1e-12
    for q in range(5):
        assert np.max(np.abs(a[0][q] - b[0][q])) <= 1e-10 * (np.abs(a[0][q]).max() + 1e-30)
    assert abs(a[2] - b[2]) <= 1e-10 * abs(a[2])


def test_mode_perturbation_is_the_same_field_on_every_grid():
    """The resolution study needs one initial perturbation for all grids: the 'modes' field sampled on two grids
    agrees where the cells coincide in physical position, the per-cell white noise does not."""
    fields = {}
    for nr in (32, 64):
        cfg = RIAFConfig(nr=nr, ntheta=nr // 2, perturbation_kind="modes", seed=3, perturbation=0.05)
        flow = AccretionFlow(cfg)
        base = torus_state(AccretionFlow(RIAFConfig(nr=nr, ntheta=nr // 2, perturbation=0.0)))[0]
        fields[nr] = (torus_state(flow)[0], base)
    (pc, bc), (pf, bf) = fields[32], fields[64]
    torus = (bc > 1e-2)
    noise_c = np.where(torus, pc / np.where(torus, bc, 1) - 1, 0)
    torus_f = (bf > 1e-2)
    noise_f = np.where(torus_f, pf / np.where(torus_f, bf, 1) - 1, 0)
    # average 2 x 2 fine cells onto the coarse grid: a smooth field agrees to the averaging error
    avg = noise_f.reshape(32, 2, 16, 2).mean((1, 3))
    both = torus & (torus_f.reshape(32, 2, 16, 2).all((1, 3)))
    assert both.sum() > 20
    assert np.abs(avg[both] - noise_c[both]).max() < 0.25 * 0.05
    assert np.abs(noise_c[torus]).max() <= 0.05 + 1e-12


def test_rollout_loss_noise_and_mass_penalty_train(tmp_path):
    from pinneapple_physics.blackhole.forecast import TrainConfig, load_forecaster, train_forecaster
    rng = np.random.default_rng(0)
    x = (0.5 + 0.05 * np.cumsum(rng.normal(size=(40, 16, 16)), 0) / 10).astype(np.float32)
    X, Y = make_blocks(x, 5, out_blocks=2)
    assert Y.shape[1] == 10 and np.array_equal(Y[0], x[5:15])
    codec = DensityCodec(-6.0, 0.0, 16, 0)
    cfg = TrainConfig(filters=2, epochs=2, residual=True, rollout_blocks=2, noise=0.01, mass_weight=0.1,
                      mass_rate_weight=1.0, padding="replicate")
    hist = train_forecaster(X, Y, X[:4], Y[:4], cfg, codec, str(tmp_path), log=lambda s: None,
                            cell_volume=np.ones((16, 16)))
    assert len(hist["val"]) == 2 and all(np.isfinite(hist["val"]))
    model, c2, ck = load_forecaster(str(tmp_path / "forecaster.pt"))
    assert ck["train"]["rollout_blocks"] == 2 and c2 == codec
    with pytest.raises(ValueError, match="cell_volume"):
        train_forecaster(X, Y, X[:4], Y[:4], cfg, codec, str(tmp_path), log=lambda s: None)


def test_conservative_remap_keeps_mass_between_non_nested_grids():
    from pinneapple_physics.blackhole.convergence import cell_volumes, remap, remap_r
    grids = {}
    for nr in (45, 32):
        f = AccretionFlow(RIAFConfig(nr=nr, ntheta=nr // 2))
        grids[nr] = f.grid()
    rng = np.random.default_rng(0)
    a = rng.random((45, 22))
    faces = {n: (g["r_faces"], g["theta_faces"]) for n, g in grids.items()}
    b = remap(a, faces[45], faces[32])
    m_f = (a * cell_volumes(grids[45])).sum()
    m_c = (b * cell_volumes(grids[32])).sum()
    assert b.shape == (32, 16) and abs(m_c / m_f - 1) < 1e-12
    assert np.allclose(remap(np.full((45, 22), 3.0), faces[45], faces[32]), 3.0)
    assert np.allclose(remap_r(np.full(45, 2.0), grids[45]["r_faces"], grids[32]["r_faces"]), 2.0)


def test_dataset_codec_roundtrip():
    from pinneapple_physics.blackhole.dataset import Q_HI, Q_LO, decode, encode
    lg = np.random.default_rng(1).uniform(-7.5, 0.5, (4, 6, 5))
    assert np.abs(decode(encode(lg)) - lg).max() <= 0.5 * (Q_HI - Q_LO) / 65535 + 1e-6
    with pytest.raises(ValueError):
        encode(np.array([Q_LO - 1.0]))


def test_dmd_baseline_is_exact_on_a_linear_system():
    from pinneapple_physics.blackhole.skill import DMDBaseline
    rng = np.random.default_rng(2)
    modes = rng.normal(size=(2, 4, 4))
    t, w = np.arange(30), 2 * np.pi * 3 / 29         # whole periods over the 29 snapshot pairs: mean = 0.5 exactly
    traj = [(0.5 + 0.1 * np.cos(w * t + ph)[:, None, None] * modes[0] + 0.1 * np.sin(w * t + ph)[:, None, None]
             * modes[1]) for ph in (0.0, 1.0)]
    d = DMDBaseline(traj, rank=2)
    f = d.forecast(traj[0][10], 5)
    assert np.abs(f - traj[0][11:16]).max() < 1e-6


def test_forecast_skill_horizon_and_intervals():
    from pinneapple_analysis.verification.forecast_skill import batch_means_ci, skill_vs_lead
    rng = np.random.default_rng(3)
    ref = 1.0 + 0.05 * rng.random((200, 6))
    model = ref.copy()
    model[:, :4] *= 0.8                                   # better for 4 leads, then equal
    res = skill_vs_lead(model, ref, np.arange(1, 7) * 10.0, block=5)
    assert res.horizon == 40.0
    assert res.skill_score[0, 1] > 0.15 and res.diff[0, 2] < 0
    m, h = batch_means_ci(rng.normal(size=4000))
    assert abs(m) < 3 * h and 0 < h < 0.2


def test_simulate_run_resumes_from_its_checkpoint(tmp_path):
    """A run interrupted after its first chunk continues from state.pt (the flow is rebuilt without set_primitives)."""
    from pinneapple_physics.blackhole.runs import load_run, simulate_run
    cfg = RIAFConfig(nr=16, ntheta=8, backend="torch", perturbation_kind="modes")
    simulate_run(cfg, str(tmp_path / "a"), t_end=20.0, every=10.0, chunk=1, log=lambda m: None)
    simulate_run(cfg, str(tmp_path / "a"), t_end=40.0, every=10.0, chunk=1, log=lambda m: None)
    simulate_run(cfg, str(tmp_path / "b"), t_end=40.0, every=10.0, chunk=1, log=lambda m: None)
    a, b = load_run(str(tmp_path / "a")), load_run(str(tmp_path / "b"))
    assert len(a["t"]) == len(b["t"]) == 5
    assert np.array_equal(a["frames"], b["frames"])
