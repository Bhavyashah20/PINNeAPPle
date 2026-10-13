"""Tests for the PCB stack-up physics and the PCB Hotspot app."""
from __future__ import annotations

import json
import os
import sys

import pytest

from pinneapple_physics.closed_form import pcb_thermal as pt

APP = os.path.join(os.path.dirname(__file__), "..", "apps", "pcb_hotspot")
sys.path.insert(0, os.path.abspath(APP))

pytest.importorskip("scipy")

from pinneapple_physics.pcb.engine import calibrate, evaluate, what_if  # noqa: E402
from pinneapple_physics.pcb.solver import Board, Component, overlaps, solve  # noqa: E402

ENV = pt.Environment(t_ambient_c=35.0)


def example():
    board = Board(width_mm=160, depth_mm=100, n_copper=6)
    comps = [Component("FPGA", 62, 52, 8.0, "BGA 35x35", tj_max_c=100),
             Component("DDR", 110, 60, 1.5, "BGA 17x17", tj_max_c=95),
             Component("VRM", 22, 80, 2.0, "QFN-32 5x5", vias=9),
             Component("LDO", 140, 20, 0.3, "SOIC-8")]
    return board, comps


# ── stack-up physics ─────────────────────────────────────────────────────────

def test_stackup_effective_conductivity_is_anisotropic():
    layers = pt.stackup(4, 1.6, 1.0, 1.0, 0.3, 0.9)
    k = pt.effective_conductivity(layers)
    assert sum(la.thickness_m for la in layers) == pytest.approx(1.6e-3, rel=1e-6)
    assert k["k_xy"] > 20 * k["k_z"]                 # copper planes spread in-plane
    k6 = pt.effective_conductivity(pt.stackup(6, 1.6, 1.0, 1.0, 0.3, 0.9))
    assert k6["k_xy"] > k["k_xy"]


def test_via_fraction_and_convection_trends():
    assert pt.via_copper_fraction(16, 0.3, 25, 25e-6) > pt.via_copper_fraction(4, 0.3, 25, 25e-6) > 0
    nat = pt.convection_coefficients(0.16, 0.1, (60.0, 60.0), ENV)
    hot = pt.convection_coefficients(0.16, 0.1, (90.0, 90.0), ENV)
    fan = pt.convection_coefficients(0.16, 0.1, (60.0, 60.0), pt.Environment(35.0, air_velocity_m_s=2.0))
    assert 3 < nat["h_top"] < 20 and hot["h_top"] > nat["h_top"] and fan["h_top"] > nat["h_top"]


@pytest.mark.parametrize("package", sorted(pt.JEDEC_2S2P_THETA_JA))
def test_default_packages_reproduce_published_jedec_theta_ja(package):
    """Board + convection + default package model on the JESD51-7 2s2p board
    (still air, horizontal) must give the published median theta_JA."""
    big = pt.PACKAGES[package]["w_mm"] > 27
    board = Board(114.3, 101.6 if big else 76.2, n_copper=4, thickness_mm=1.6, outer_oz=2,
                  inner_oz=1, outer_coverage=0.1, inner_coverage=0.95)
    vias = {"QFN-32 5x5": 9, "QFN-64 9x9": 16}.get(package, 0)
    c = Component("U", 57.15, board.depth_mm / 2, 1.0, package, vias=vias)
    r = solve(board, [c], pt.Environment(25.0, orientation="horizontal"), n_cells=60)
    theta_ja = r["components"][0]["t_junction_c"] - 25.0
    assert theta_ja == pytest.approx(pt.JEDEC_2S2P_THETA_JA[package], rel=0.08)


# ── board solver ─────────────────────────────────────────────────────────────

def test_full_board_component_matches_resistor_network():
    """One part covering the whole board with fixed h is 1-D: two parallel
    paths, junction->case->air and junction->board->stack-up->bottom air."""
    board = Board(width_mm=50, depth_mm=40, n_copper=4)
    c = Component("U", 25, 20, 5.0, "QFN-32 5x5", w_mm=50, d_mm=40, theta_jb=2.0, theta_jc=6.0)
    h = {"top": 15.0, "bottom": 9.0}
    r = solve(board, [c], ENV, n_cells=20, h_override=h)
    A = 0.05 * 0.04
    r_board = 2.0 + sum(la.thickness_m / (la.k_z() * A) for la in board.layers()) + 1 / (h["bottom"] * A)
    r_air = 6.0 + 1 / (h["top"] * A)
    expected = 35.0 + 5.0 * r_board * r_air / (r_board + r_air)
    assert r["components"][0]["t_junction_c"] == pytest.approx(expected, rel=1e-6)
    assert r["components"][0]["heat_to_board_pct"] == pytest.approx(100 * r_air / (r_board + r_air), rel=1e-5)


def test_energy_balance_and_second_law():
    board, comps = example()
    r = solve(board, comps, ENV, n_cells=40)
    assert r["energy_balance_rel_error"] < 1e-8
    assert sum(r["heat_split_w"].values()) == pytest.approx(sum(c.power_w for c in comps), rel=1e-8)
    assert r["t_top_c"].min() > ENV.t_ambient_c and r["converged"]


@pytest.mark.parametrize("change", [
    lambda b, c: (b, [Component(**{**x.__dict__, "vias": 64}) if x.name == "FPGA" else x for x in c], ENV),
    lambda b, c: (Board(**{**b.__dict__, "outer_oz": 2.0}), c, ENV),
    lambda b, c: (Board(**{**b.__dict__, "n_copper": 8}), c, ENV),
    lambda b, c: (b, c, pt.Environment(35.0, air_velocity_m_s=1.0)),
    lambda b, c: (Board(**{**b.__dict__, "chassis_edges": True}), c, ENV),
])
def test_every_standard_fix_cools_the_fpga(change):
    board, comps = example()
    base = solve(board, comps, ENV, n_cells=36)["components"][0]["t_junction_c"]
    b2, c2, e2 = change(board, comps)
    assert solve(b2, c2, e2, n_cells=36)["components"][0]["t_junction_c"] < base - 0.5


def test_validation_and_overlap_detection():
    board, comps = example()
    with pytest.raises(ValueError, match="beyond the board"):
        solve(board, [Component("X", 2, 2, 1.0, "BGA 35x35")], ENV)
    assert overlaps(comps) == []
    stacked = comps + [Component("CAP", 62, 52, 0.1, "SOIC-8")]
    assert any("FPGA and CAP" in w for w in overlaps(stacked))
    assert overlaps(comps + [Component("BOT", 62, 52, 0.1, "SOIC-8", side="bottom")]) == []


# ── engine ───────────────────────────────────────────────────────────────────

def test_evaluate_report_structure_and_band():
    board, comps = example()
    r = evaluate(board, comps, ENV)
    assert r["critical"] == "FPGA"
    for c in r["components"]:
        lo, hi = c["t_junction_band_c"]
        assert lo < c["t_junction_c"] < hi
    names = {c["name"] for c in r["checks"]}
    assert {"energy_balance", "grid_convergence", "temperatures_above_ambient"} <= names
    assert all(c["status"] != "fail" for c in r["checks"])
    assert r["verdict"]["status"] in ("meets", "marginal", "fails")
    json.dumps(r)


def test_what_if_ranks_fixes():
    board, comps = example()
    w = what_if(board, comps, ENV, n_cells=32)
    assert w["critical"] == "FPGA" and len(w["scenarios"]) >= 4
    assert all(s["delta_critical_c"] < 0 for s in w["scenarios"])
    d = [s["delta_critical_c"] for s in w["scenarios"]]
    assert d == sorted(d)


def _synthetic(board, comps, h_true=0.75, k_true=0.6):
    nom = solve(board, comps, ENV, n_cells=60)
    h = {s: v * h_true for s, v in nom["h"].items() if s in ("top", "bottom")}
    truth = solve(board, comps, ENV, n_cells=60, h_override=h, k_xy_scale=k_true)
    from pinneapple_physics.pcb.engine import _interp_map
    pts = [(20, 20), (60, 50), (100, 80), (140, 60), (30, 70), (120, 30)]
    meas = [{"x_mm": x, "y_mm": y, "t_c": _interp_map(truth["t_top_c"], 160, 100, x, y)} for x, y in pts]
    meas.append({"component": "FPGA", "t_c": truth["components"][0]["t_junction_c"]})
    return truth, meas


def test_calibration_recovers_known_parameters():
    board, comps = example()
    truth, meas = _synthetic(board, comps)
    c = calibrate(board, comps, ENV, meas, noise_std_c=0.3)
    assert c["parameters"]["h_scale"]["value"] == pytest.approx(0.75, rel=0.01)
    assert c["parameters"]["k_inplane_scale"]["value"] == pytest.approx(0.6, rel=0.02)
    assert c["fit"]["rms_after_c"] < 0.2 * c["fit"]["rms_before_c"]
    assert c["holdout"]["n"] >= 1 and c["holdout"]["rms_after_c"] < c["holdout"]["rms_before_c"]
    assert c["adequacy"]["status"] == "consistent" and not c["warnings"]
    fpga = c["junctions_after_calibration"][0]
    assert fpga["t_junction_c"] == pytest.approx(truth["components"][0]["t_junction_c"], abs=0.2)
    json.dumps(c)


def test_calibration_flags_inconsistent_data_and_inflates_uncertainty():
    board, comps = example()
    _, meas = _synthetic(board, comps)
    for i, m in enumerate(meas):                    # corrupt the readings
        m["t_c"] += 12.0 if i % 2 else -9.0
    c = calibrate(board, comps, ENV, meas, noise_std_c=0.3)
    assert c["adequacy"]["status"] != "consistent"
    assert c["adequacy"]["uncertainty_inflation"] > 3 and c["warnings"]



def test_calibration_with_two_readings_says_fit_is_untestable():
    board, comps = example()
    _, meas = _synthetic(board, comps)
    c = calibrate(board, comps, ENV, meas[:2], noise_std_c=0.3)
    assert c["adequacy"]["status"] == "not testable with 2 readings"
    assert any("third reading" in w for w in c["warnings"])
    assert c["holdout"]["n"] == 0

# ── API ──────────────────────────────────────────────────────────────────────

CASE = {"board": {"width_mm": 160, "depth_mm": 100, "n_copper": 6},
        "environment": {"t_ambient_c": 35},
        "components": [{"name": "FPGA", "package": "BGA 35x35", "power_w": 8, "x_mm": 62, "y_mm": 52, "tj_max_c": 100},
                       {"name": "LDO", "package": "SOIC-8", "power_w": 0.3, "x_mm": 140, "y_mm": 20}]}


@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from pcb_hotspot import api
    return TestClient(api.app), api


def test_api_endpoints(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert "BGA 35x35" in c.get("/api/meta").json()["packages"]
    assert c.get("/").status_code == 200 and c.get("/static/app.js").status_code == 200
    assert c.get("/static/board3d.js").status_code == 200
    assert c.get("/vendor/three/three.module.min.js").status_code == 200
    author = c.get("/shared/author.json").json()
    assert author["email"] and c.get("/shared/author.js").status_code == 200
    q = c.post("/api/solve?quick=true", json=CASE)
    assert q.status_code == 200 and q.json()["quick"] and q.json()["critical"] == "FPGA"
    full = c.post("/api/solve", json=CASE).json()
    assert "checks" in full and "field3d" in full
    w = c.post("/api/whatif", json=CASE)
    assert w.status_code == 200 and w.json()["scenarios"]


def test_api_validation(client):
    c, _ = client
    dup = {**CASE, "components": CASE["components"] + [CASE["components"][0]]}
    assert c.post("/api/solve", json=dup).status_code == 422
    off = {**CASE, "components": [{**CASE["components"][0], "x_mm": 5}]}
    r = c.post("/api/solve", json=off)
    assert r.status_code == 422 and "beyond the board" in r.json()["detail"]
    bad_pkg = {**CASE, "components": [{**CASE["components"][0], "package": "DIP-99"}]}
    assert c.post("/api/solve", json=bad_pkg).status_code == 422
    unk = {**CASE, "measurements": [{"component": "NOPE", "t_c": 80}, {"x_mm": 10, "y_mm": 10, "t_c": 50}]}
    assert c.post("/api/calibrate", json=unk).status_code == 422
    nowhere = {**CASE, "measurements": [{"t_c": 80}, {"t_c": 50}]}
    assert c.post("/api/calibrate", json=nowhere).status_code == 422


def test_api_basic_auth_and_busy_limit(client, monkeypatch):
    import base64
    import threading
    c, api = client
    monkeypatch.setattr(api.app.state, "auth_user", "eng")
    monkeypatch.setattr(api.app.state, "auth_password", "s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/api/meta").status_code == 401
    good = {"Authorization": "Basic " + base64.b64encode(b"eng:s3cret").decode()}
    assert c.get("/api/meta", headers=good).status_code == 200
    monkeypatch.setattr(api._HEAVY, "slots", threading.BoundedSemaphore(1))
    api._HEAVY.slots.acquire()
    assert c.post("/api/whatif", json=CASE, headers=good).status_code == 429
    api._HEAVY.slots.release()


def test_report_scope_lists_what_applies_to_this_case():
    board, comps = example()
    r = evaluate(board, comps, ENV, quick=False)
    topics = {i["topic"]: i for i in r["scope"]["items"]}
    assert "LDO" in topics["Package thermal data"]["detail"]
    assert topics["Enclosure"]["effect"] == "optimistic"
    assert "Air heating along the flow" not in topics                      # natural convection
    fan = evaluate(board, comps, pt.Environment(35.0, air_velocity_m_s=1.5))
    assert "Air heating along the flow" in {i["topic"] for i in fan["scope"]["items"]}
    sheet = [Component(**{**c.__dict__, "theta_jb": 5.0, "theta_jc": 5.0}) for c in comps]
    assert "Package thermal data" not in {i["topic"] for i in evaluate(board, sheet, ENV)["scope"]["items"]}
