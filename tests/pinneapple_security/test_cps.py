"""Physics-residual false-data-injection detection (pinneapple_security.cps) on the three-tank benchmark."""
import numpy as np
import pytest

from pinneapple_security import cps, cps_bench


@pytest.fixture(scope="module")
def traces():
    return cps_bench.simulate_three_tank(3, seed=0), cps_bench.simulate_three_tank(3, seed=1)


def test_grey_box_identifies_tank_areas_and_valve_laws(traces):
    train, _ = traces
    det = cps_bench.physics_detector().fit(train)
    areas = [c.area for c in det.components if isinstance(c, cps.MassBalance)]
    np.testing.assert_allclose(areas, [2.0, 1.5, 1.8], rtol=0.05)
    a, b = det.components[3].coef                                     # q^2 = c^2 (dz + dh): c = 0.0105, dz = 1.5
    assert b == pytest.approx(0.0105 ** 2, rel=0.05)
    assert a / b == pytest.approx(1.5, rel=0.05)


def test_clean_data_respects_the_false_alarm_budget_and_bias_is_found_and_localised(traces):
    train, calib = traces
    test = cps_bench.simulate_three_tank(3, seed=2)
    det = cps_bench.physics_detector().fit(train)
    det.calibrate(calib, false_alarms_per_day=1.0)
    assert len(det.detect(test)["alarms"]) <= 3 * 3                   # budget 1/day, generous margin
    eps = [(1500, 1740), (2500, 2740), (3500, 3740)]
    res = det.detect(cps_bench.inject(test, "bias", eps))
    m = cps.episode_metrics(res["alarms"], eps, len(test["h1"]))
    assert m["detection_rate"] == 1.0 and m["median_delay"] < 60
    blamed = [c for t, c in zip(res["alarms"], res["channels"], strict=True) if any(s <= t < e for s, e in eps)]
    assert all("F12" in c or "h1" in c or "h2" in c for c in blamed)  # the balances that contain the F12 meter


def test_roc_auc_and_cusum():
    assert cps.roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert cps.roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert cps.roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    z = np.r_[np.zeros(50), np.full(20, 2.0), np.zeros(30)]
    stat, alarms = cps.cusum(z, k=0.5, h=5.0)
    assert alarms and 50 <= alarms[0] <= 54
    assert not cps.cusum(-z, k=0.5, h=5.0, upper_only=True)[1]
    m = cps.episode_metrics([10, 55, 300], [(50, 100)], 1440, grace=10)
    assert m["detected"] == 1 and m["delays"] == [5] and m["false_alarms"] == 2


def test_inject_rejects_unknown_attacks_and_replays_without_history(traces):
    _, calib = traces
    with pytest.raises(ValueError):
        cps_bench.inject(calib, "teleport", [(2000, 2100)])
    with pytest.raises(ValueError):
        cps_bench.inject(calib, "delayed", [(100, 200)])
    X = cps_bench.inject(calib, "stuck", [(2000, 2100)])
    assert np.all(X["h3"][2000:2100] == calib["h3"][2000])
    assert np.array_equal(X["h1"], calib["h1"])


def test_physics_detector_survives_a_regime_shift_that_breaks_the_data_driven_twin():
    r = cps_bench.run_benchmark(train_days=3, calib_days=3, test_days=4, episodes=6, attacks=("bias", "scaling"),
                                detectors={"physics": cps_bench.physics_detector(),
                                           "ar_twin": cps_bench.ARResidualDetector()})
    phys, ar = r["results"]["physics"], r["results"]["ar_twin"]
    assert phys["shifted"]["false_alarms_per_day_clean"] < 3
    assert ar["shifted"]["false_alarms_per_day_clean"] > 10 * max(phys["shifted"]["false_alarms_per_day_clean"], 1)
    assert phys["shifted"]["bias"]["auc"] > 0.95 and phys["nominal"]["scaling"]["detection_rate"] >= 0.8
