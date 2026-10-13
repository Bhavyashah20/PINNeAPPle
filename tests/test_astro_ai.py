"""PINNeAPPle models in the observatory, each checked against an independent reference:
orbit-decay PINN (RK integration, SGP4), small-body propagator (JPL Horizons, JPL close approaches), learned NEO
propagator (full integrator on unseen bodies), Kp forecaster (observed Kp, persistence), stellar ML (BSC5 MK types),
observing-conditions post-processing, thin-disc black-hole fields."""
import numpy as np
import pytest

pytest.importorskip("skyfield_data")
torch = pytest.importorskip("torch")

ISS_2018 = ("1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
            "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106")


def test_decay_pinn_matches_integrator():
    from pinneapple_physics.astro.decay import train_decay_pinn
    r = train_decay_pinn(415.0, 0.005, 30.0, f107=150, ap=15, iters=700, with_envelope=False)
    assert r["verdict"]["status"] == "trustworthy", r["checks"]
    assert r["checks"][0]["value"] < 50.0            # metres
    assert 20 < r["decay_rate_m_per_day_t0"] < 400   # ISS-like decay at moderate-high solar activity


def test_decay_vs_sgp4_independent_drag_model():
    pytest.importorskip("sgp4")
    from pinneapple_physics.astro.decay import compare_with_sgp4
    c = compare_with_sgp4(*ISS_2018, days=30, f107=70, ap=6)
    assert 0.5 < c["ratio"] < 2.0, c["ratio"]
    assert c["rate_sgp4_m_day"] > 0


def test_exponential_atmosphere_table():
    from pinneapple_physics.astro.decay import exponential_density
    assert abs(exponential_density(400.0) / 3.725e-12 - 1) < 1e-9
    assert exponential_density(500.0) < exponential_density(400.0)


def test_ceres_against_horizons():
    from pinneapple_physics.astro.ephemeris import AU_KM
    from pinneapple_physics.astro.smallbodies import (
        ECL_TO_ICRF,
        horizons_vectors,
        load_body,
        propagate,
    )
    jd, P, _ = horizons_vectors()
    r, _ = propagate(load_body("ceres"), jd)
    err = np.linalg.norm(r - (ECL_TO_ICRF @ P.T).T * AU_KM, axis=1)
    assert err.max() < 100.0, err


def test_apophis_2029_against_jpl_close_approach():
    from pinneapple_physics.astro.ephemeris import AU_KM
    from pinneapple_physics.astro.smallbodies import cad_reference, close_approaches, load_body
    ap = load_body("apophis")
    ref = cad_reference(ap, 2029)
    ca = close_approaches(ap, ref["jd_tdb"] - 2, ref["jd_tdb"] + 2, step_days=0.25)[0]
    assert abs(ca["dist_km"] - ref["dist_au"] * AU_KM) <= (ref["dist_max_au"] - ref["dist_min_au"]) / 2 * AU_KM
    assert abs(ca["jd_tdb"] - ref["jd_tdb"]) * 86400 < 120
    assert abs(ca["v_rel_km_s"] - ref["v_rel_km_s"]) < 0.1


def test_learned_propagator_is_judged_by_its_gate():
    """The shipped many-body network is evaluated honestly: it runs, and its verdict follows from its held-out
    errors (it does not beat Kepler by the required 5x, so it must not be marked trustworthy)."""
    from pinneapple_physics.astro import neural_propagator as npg
    model = npg.NeuralPropagator.load()
    b = npg._body(npg.sample_population(1, seed=12345)[0])
    r = model.predict(b, b.epoch_jd + np.linspace(0, 365, 5))
    assert r.shape == (5, 3) and np.all(np.isfinite(r))
    rep = npg.trust_report()
    good = rep["info"]["test_max_err_km_median"] <= rep["info"]["test_kepler_max_err_km_median"] / 5
    assert (rep["verdict"]["status"] == "trustworthy") == (good and rep["checks"][1]["status"] == "pass")


def test_smallbody_pinn_trustworthy_for_apophis():
    from pinneapple_physics.astro.smallbodies import load_body
    from pinneapple_physics.astro.surrogate import train_orbit_pinn
    rep = train_orbit_pinn(load_body("apophis"), 2461327.0, 365.0, iters=800)
    assert rep["verdict"]["status"] == "trustworthy", rep["verdict"]
    assert rep["summary"]["max_angle_arcsec_test"] < 1.0


def test_smallbody_validation_summary():
    from pinneapple_physics.astro.smallbodies import validation_summary
    v = validation_summary()
    assert v["verdict"]["status"] == "trustworthy", v["checks"]


def test_stellar_ml_beats_baselines():
    from pinneapple_physics.astro.stars_ml import analyse, luminosity_group, spectral_class
    assert luminosity_group("K0IIIbCN-0.5") == 2 and luminosity_group("A1Vm") == 0
    assert luminosity_group("M1-2Ia-Iab") == 3 and spectral_class("gG8III") == 4
    r = analyse()
    assert r["lum_cv"]["accuracy"] > r["lum_cv"]["baseline"] + 0.1
    assert r["spec_cv"]["accuracy"] > r["spec_cv"]["baseline"] + 0.3
    assert r["cluster_purity"] > 0.6


def test_kp_forecaster_against_observed_kp():
    from pinneapple_physics.astro import spaceweather as sw
    model, rep = sw.train_forecaster(8)
    assert rep["mae_model"] <= rep["mae_persistence"] + 1e-9
    assert rep["coverage_80"] >= 0.75
    h = sw.load_history()["kp"]
    lo, med, hi = sw.forecast(model, h[: 300 * 8], 150.0)
    assert 0 <= lo <= med <= hi <= 9


def test_aurora_geometry():
    from pinneapple_physics.astro import spaceweather as sw
    lat, plat, plon = sw.geomagnetic_latitude(80.65, -72.68)
    assert abs(plat - 80.65) < 0.1 and abs(plon + 72.68) < 0.1 and lat > 89.9
    assert abs(sw.oval_boundary(0) - 66.5) < 1e-9 and abs(sw.oval_boundary(9) - 48.1) < 0.1
    p = sw.aurora_probability(2, 4, 6, 60.0)
    assert 0 <= p["p_overhead"] <= p["p_horizon"] <= 1


def test_conditions_postprocessing_and_calibration():
    from pinneapple_physics.astro.conditions import calibrate, summarize
    rng = np.random.default_rng(0)
    hourly = {"time": [f"2026-10-14T{h:02d}:00" for h in range(24)]}
    for m in range(10):
        suf = "" if m == 0 else f"_member{m:02d}"
        hourly["cloud_cover" + suf] = list(rng.uniform(0, 100, 24))
        hourly["relative_humidity_2m" + suf] = list(rng.uniform(40, 95, 24))
        hourly["wind_speed_250hPa" + suf] = list(rng.uniform(30, 200, 24))
    s = summarize({"hourly": hourly})
    assert s["n_members"] == 10 and len(s["p_clear"]) == 24
    assert all(0 <= p <= 1 for p in s["p_clear"])
    assert all(a <= b <= c for a, b, c in zip(s["cloud_q10"], s["cloud_q50"], s["cloud_q90"], strict=True))
    # missing values (a variable absent for some hours, as the live API returns) become None, never NaN
    for m in range(10):
        suf = "" if m == 0 else f"_member{m:02d}"
        hourly["wind_speed_250hPa" + suf] = [None] * 12 + hourly["wind_speed_250hPa" + suf][12:]
        hourly["cloud_cover" + suf][5] = None
    import json
    s = summarize({"hourly": hourly})
    json.dumps(s, allow_nan=False)
    assert s["jet_wind_kmh"][0] is None and s["seeing_proxy"][0] == "unknown" and s["p_clear"][5] is None
    p = rng.uniform(0, 1, 2000)
    o = (rng.uniform(0, 1, 2000) < p).astype(float)               # perfectly reliable forecasts
    cal = calibrate(p, o)
    assert cal["brier"] < 0.2
    assert all(abs(f - (a + b) / 2) < 0.08 for f, a, b in zip(cal["observed_freq"], cal["bin_edges"][:-1],
                                                                cal["bin_edges"][1:], strict=True))


def test_thin_disc_geodesics_match_schwarzschild():
    """The disc tracer against closed-form Schwarzschild results: the shadow edge sits at the critical impact
    parameter 3 sqrt(3) GM/c^2, and face-on the disc's frequency shift is purely gravitational and transverse,
    g = sqrt(1 - 3/r) / sqrt(1 - 2/r_cam)."""
    from pinneapple_physics.astro.lensing import B_CRIT, disc_image, trace_thin_disc
    tr = trace_thin_disc(160, 90, inclination_deg=0.5, r_cam=60.0, fov_deg=30.0)
    b, cap = tr["b"], tr["captured"]
    assert cap[b < B_CRIT * 0.99].all() and not cap[b > B_CRIT * 1.01].any()
    r = tr["hit_r"][:, 0]
    seen = np.isfinite(r) & (r > 7) & (r < 30)
    assert seen.sum() > 1000
    assert np.allclose(tr["lz_photon"][seen], 0, atol=0.05 * B_CRIT)
    edge = trace_thin_disc(240, 135, inclination_deg=86.0)
    rgb, cover = disc_image(edge)
    H, W = edge["shape"]
    left, right = rgb[:, : W // 2].sum(), rgb[:, W // 2:].sum()
    assert max(left, right) > 1.5 * min(left, right)               # Doppler beaming: one side is brighter
    assert np.isfinite(edge["hit_r"][:, 1]).sum() > 500            # the far side's image wraps over the shadow


