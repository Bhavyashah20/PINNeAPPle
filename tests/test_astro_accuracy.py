"""Accuracy of pinneapple_physics.astro against independent references (USNO NOVAS, JPL Horizons, USNO almanac,
NASA eclipse predictions, an independent ISS pass finder) and the integrators' conservation laws."""
import numpy as np
import pytest

pytest.importorskip("skyfield_data", reason="the JPL DE421 kernel ships in the skyfield-data package")

from pinneapple_physics.astro import Observer, Time, apparent, validation  # noqa: E402
from pinneapple_physics.astro.frames import angular_separation, nutation  # noqa: E402
from pinneapple_physics.astro.spk import SPK  # noqa: E402


def _assert_rows(rows):
    bad = [r for r in rows if r["status"] != "pass"]
    assert not bad, bad


def test_spk_reader_matches_chebyshev_identity():
    from pinneapple_physics.astro.ephemeris import find_kernel
    k = SPK(find_kernel())
    p, v = k.state(301, 399, np.array([0.0, 1.0]))
    # velocity is the time derivative of position
    assert np.allclose(p[1] - p[0], 0.5 * (v[0] + v[1]), rtol=0, atol=1e-6)
    assert 356000 < np.linalg.norm(p[0]) < 407000


def test_novas_apparent_places():
    _assert_rows(validation.check_novas())


def test_horizons():
    _assert_rows(validation.check_horizons())


def test_usno_almanac():
    _assert_rows(validation.check_usno_almanac())


def test_nasa_eclipses_and_transits():
    _assert_rows(validation.check_eclipses())


def test_iss_passes_against_reference():
    pytest.importorskip("sgp4")
    _assert_rows(validation.check_iss_passes())


def test_integrators_conserve():
    _assert_rows(validation.check_integrators())


def test_against_skyfield_random():
    pytest.importorskip("skyfield")
    _assert_rows(validation.check_skyfield_live(n=12))


def test_nutation_truncation_bound():
    from skyfield.api import load
    from skyfield.nutationlib import iau2000a_radians
    pytest.importorskip("skyfield")
    ts = load.timescale()
    for jd in np.linspace(2433282.5, 2469807.5, 9):
        dpsi, deps = nutation(jd)
        d2, e2 = iau2000a_radians(ts.tt_jd(jd))
        assert abs(dpsi - d2) * 206264.806 < 0.03 and abs(deps - e2) * 206264.806 < 0.03


def test_refraction_and_airmass():
    from pinneapple_physics.astro import airmass, refraction_deg
    assert abs(refraction_deg(0.0) * 60 - 29.0) < 6          # ~29-35' at the horizon depending on the formula
    assert abs(refraction_deg(45.0) * 3600 - 59) < 3        # ~1' at 45 deg
    assert abs(airmass(90.0) - 1.0) < 1e-3 and abs(airmass(30.0) - 1.995) < 0.01 and airmass(0.0) < 40


def test_separation_is_symmetric_and_exact():
    assert abs(angular_separation(0, 0, 90, 0) - 90) < 1e-12
    assert abs(angular_separation(10, 89.9, 190, 89.9) - 0.2) < 1e-9


def test_topocentric_moon_parallax():
    t = Time.from_iso("2026-10-13T00:00:00Z")
    geo = apparent("moon", t)
    top = apparent("moon", t, Observer(0.0, 0.0, 0.0))
    d = angular_separation(geo.ra[0], geo.dec[0], top.ra[0], top.dec[0])
    assert 0.0 < d < 1.0                                      # horizontal parallax ~ 0.9-1.0 deg


def test_stars_and_blackbody():
    from pinneapple_physics.astro.stars import (
        bolometric_correction,
        load_bright_stars,
        star_properties,
        wien_peak_m,
    )
    cat = load_bright_stars()
    assert len(cat) > 9000
    s = star_properties(cat, cat.index_of(2491))             # Sirius A
    assert s["name"] == "Sirius" and abs(s["distance_ly"] - 8.6) < 0.3
    assert 15 < s["luminosity_lsun"] < 40 and 1.4 < s["radius_rsun"] < 2.1
    assert abs(bolometric_correction(5772.0) + 0.07) < 1e-9
    assert abs(wien_peak_m(5772.0) * 1e9 - 502.0) < 1.0


def test_lensing_closed_forms():
    from pinneapple_physics.astro.lensing import eht_comparison, sun_deflection_arcsec
    assert abs(sun_deflection_arcsec(0.26656) - 1.75) < 0.01
    for row in eht_comparison():
        assert row["sigma"] < 1.5, row


def test_orbit_elements_round_trip():
    from pinneapple_physics.astro.ephemeris import GM_SUN
    from pinneapple_physics.astro.orbits import (
        elements_from_state,
        planet_elements,
        state_from_elements,
    )
    el = planet_elements("mars", 2461327.0)
    assert abs(el["a_au"] - 1.5237) < 0.001 and abs(el["e"] - 0.0934) < 0.001
    r, v = state_from_elements(el, GM_SUN + 42828.375, 0.0)
    el2 = elements_from_state(r, v, GM_SUN + 42828.375)
    assert abs(el2["a_km"] - el["a_km"]) < 1.0 and abs(el2["M_deg"] - el["M_deg"]) < 1e-8


def test_pinn_orbit_surrogate_trustworthy_for_mars():
    pytest.importorskip("torch")
    from pinneapple_physics.astro.surrogate import train_orbit_pinn
    rep = train_orbit_pinn("mars", 2461327.0, 365.0, iters=800)
    assert rep["verdict"]["status"] == "trustworthy", rep["verdict"]
    assert rep["summary"]["max_angle_arcsec_test"] < 1.0
