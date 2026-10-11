"""Physics audit: assumptions, validity monitor, knowledge graph, physics-aware routing, failure attribution and
improvement recommendations."""
import math

import numpy as np
import pytest

from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.physics_audit import (
    ALARM,
    MARGINAL,
    OK,
    UNKNOWN,
    VALID,
    VIOLATED,
    WARN,
    Assumption,
    Condition,
    Envelope,
    ModelProfile,
    PhysicsAwareRouter,
    PhysicsValidityMonitor,
    attribute_failure,
    audit_assumptions,
    boundary_check,
    bounds_check,
    conservation_check,
    default_graph,
    design_points,
    diagnose,
    get_assumption,
    hydraulically_smooth,
    improvement_plan,
    laminar,
    recommend,
    regime_features,
    residual_check,
    roughness_reynolds,
    turbulent,
)
from pinneapple_physics.physics_audit.regime import mach, reynolds

# ------------------------------------------------------------------------------------------------ fixtures
EPS_D = 1e-3


def churchill(re, eps_d=EPS_D):
    a = (2.457 * np.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * eps_d))) ** 16
    b = (37530.0 / re) ** 16
    return 8.0 * ((8.0 / re) ** 12 + 1.0 / (a + b) ** 1.5) ** (1.0 / 12)


LAM = ModelProfile("laminar", "closed form", ("Hagen-Poiseuille / laminar correlations",), (laminar(2300), "incompressible"),
                   envelope=Envelope({"Re": (1.0, 2300.0)}), cost=1)
BLA = ModelProfile("blasius", "closed form", ("smooth-pipe friction correlations (Blasius, Prandtl-Karman)",),
                   (turbulent(4000), hydraulically_smooth(), "incompressible"), envelope=Envelope({"Re": (4000.0, 1e5)}))
SOL = ModelProfile("solver", "solver", ("RANS with eddy viscosity", "Colebrook-White correlation"), ("incompressible",),
                   cost=100)


def feats(re):
    return {"Re": float(re), "Ma": 0.05, "eps_plus": roughness_reynolds(re, EPS_D)}


# ------------------------------------------------------------------------------------------------ assumptions
def test_condition_status_score_and_distance():
    c = Condition("Re", hi=2300, band=0.12)
    assert c.status(500) == VALID and c.score(500) == 1.0
    assert c.status(2200) == MARGINAL and c.score(2200) == 1.0          # inside, near the edge
    assert c.status(2600) == MARGINAL and 0.5 < c.score(2600) < 1.0    # just outside
    assert c.status(10000) == VIOLATED and c.score(10000) < 1e-3
    assert c.distance(230) == pytest.approx(-1.0)                       # one decade inside
    lin = Condition("alpha_deg", lo=-12, hi=12, band=1.5, log=False)
    assert lin.status(5) == VALID and lin.status(13) == MARGINAL and lin.status(16) == VIOLATED
    assert "Re <= 2.3e+03" == c.describe()


def test_missing_feature_is_unknown_never_a_pass():
    chk = laminar().check({"Ma": 0.1})
    assert chk.status == UNKNOWN and math.isnan(chk.score)
    audit = audit_assumptions(LAM, {"Ma": 0.1})
    assert audit.status == UNKNOWN and audit.unknown


def test_audit_regime_change_and_envelope():
    a = audit_assumptions(LAM, feats(6000))
    assert a.status == VIOLATED and a.compatibility < 1e-3
    assert [c.name for c in a.violated] == ["laminar"]
    assert a.triggered == ["turbulence"] and a.missing_mechanisms == ["turbulent transport (eddy viscosity)"]
    assert not a.envelope["inside"] and a.envelope["outside"] == ["Re"]
    ok = audit_assumptions(LAM, feats(800))
    assert ok.status == VALID and ok.compatibility == 1.0 and not ok.triggered
    assert "laminar" in str(a)


def test_envelope_knn_detects_a_hole_inside_the_box():
    samples = [{"Re": r, "Ma": 0.1} for r in np.r_[np.geomspace(1e3, 3e3, 20), np.geomspace(3e5, 1e6, 20)]]
    env = Envelope.from_samples(samples, ["Re"])
    inside_data = env.check({"Re": 2e3})
    hole = env.check({"Re": 3e4})                                      # inside the range, far from every sample
    assert inside_data["inside"] and hole["inside"]
    assert hole["knn_distance"] > 3 > inside_data["knn_distance"]
    assert hole["score"] < 0.5 < inside_data["score"]


def test_get_assumption_and_custom():
    assert get_assumption("incompressible").conditions[0].hi == 0.3
    with pytest.raises(KeyError):
        get_assumption("nonsense")
    custom = Assumption("dilute", "dilute suspension", (Condition("phi", hi=0.05),), "particle interaction", "collisions")
    assert custom.check({"phi": 0.2}).status == VIOLATED


def test_regime_numbers():
    assert reynolds(1.0, 0.1, 1e-6) == pytest.approx(1e5)
    assert mach(34.3, T=293.15) == pytest.approx(0.1, rel=0.01)
    f = regime_features(velocity=2.0, length=0.05, nu=1e-6, diffusivity=1e-9, beta=2e-4, dT=10, T=300, alpha=1.4e-7)
    assert f["Re"] == pytest.approx(1e5) and f["Pe"] == pytest.approx(1e8) and f["Pr"] == pytest.approx(1e-6 / 1.4e-7)
    assert f["beta_dT"] == pytest.approx(2e-3) and "Ri" in f and "Ra" in f and f["dT_over_T"] == pytest.approx(1 / 30)
    # eps+ ~ Re eps/D sqrt(f/8): about 25 at Re 5e5 with eps/D 1e-3
    assert roughness_reynolds(5e5, 1e-3) == pytest.approx(25, rel=0.1)


# ------------------------------------------------------------------------------------------------ monitor
def test_monitor_checks_statuses_persistence_and_drift():
    pw = np.full(10, 0.1)
    mon = PhysicsValidityMonitor([
        residual_check(lambda p, q: q["res"], 0.1),
        conservation_check(lambda p, q: float(pw @ p), lambda q: 1.0, rel_tol=0.01),
        bounds_check(lo=0.0),
        boundary_check(lambda p, q: p[:1], lambda q: np.array([1.0]), rel_tol=0.05),
    ], persistence=3, drift_ratio=3.0, baseline_cases=3)
    good = np.ones(10)
    for _ in range(3):
        assert mon.check(good, {"res": 0.01}).status == OK
    r = mon.check(good * 1.1, {"res": 0.01})                         # mass and boundary off by 10 %
    assert r.status == WARN and {x.name for x in r.failed} == {"conservation", "boundary_condition"}
    neg = good.copy()
    neg[3] = -0.1
    neg[4] = 1.1                                                      # keeps the total
    assert mon.check(neg, {"res": 0.01}).status == ALARM               # bounds are critical by default
    mon2 = PhysicsValidityMonitor([residual_check(lambda p, q: q["res"], 0.1)], persistence=2, baseline_cases=3)
    for _ in range(3):
        mon2.check(good, {"res": 0.01})
    assert mon2.check(good, {"res": 0.2}).status == WARN
    assert mon2.check(good, {"res": 0.2}).status == ALARM              # two failures in a row
    mon3 = PhysicsValidityMonitor([residual_check(lambda p, q: q["res"], 1.0)], baseline_cases=3, drift_alpha=1.0)
    for _ in range(3):
        mon3.check(good, {"res": 0.01})
    rep = mon3.check(good, {"res": 0.05})                              # 5x its baseline, still under tolerance
    assert rep.status == ALARM and "drifting" in rep.alarms[0]
    assert mon.summary()["alarm"] == 1 and mon.summary()["first_alarm"] == 4


def test_monitor_non_finite_and_broken_check():
    mon = PhysicsValidityMonitor([residual_check(lambda p, q: 1 / 0, 0.1)])
    rep = mon.check(np.array([1.0, np.nan]))
    assert rep.status == ALARM
    assert any("failed to run" in r.detail for r in rep.results)
    with pytest.raises(ValueError):
        PhysicsValidityMonitor([residual_check(lambda p, q: 0, 1), residual_check(lambda p, q: 0, 1)])


# ------------------------------------------------------------------------------------------------ knowledge graph
def test_knowledge_graph_unrepresented_and_alternatives():
    g = default_graph([LAM, BLA, SOL])
    assert g.has_model("laminar") and "laminar" in g.nodes            # model and assumption share a name
    u = g.unrepresented("laminar", feats(6000))
    assert u[0]["phenomenon"] == "turbulence" and u[0]["status"] == "ACTIVE"
    assert "solver" in g.models_for("turbulent transport (eddy viscosity)")
    assert "wall roughness" in g.mechanisms_of("solver")
    ub = g.unrepresented("blasius", feats(5e5))
    assert [x["phenomenon"] for x in ub] == ["rough-wall turbulence"]
    assert not g.unrepresented("blasius", feats(2e4))
    lines = g.explain("laminar", feats(6000))
    assert "turbulence" in lines[0] and "solver" in lines[0]
    assert {p["phenomenon"] for p in g.phenomena_triggered({"Ma": 0.6})} >= {"compressibility"}
    mm = g.to_mermaid()
    assert mm.startswith("graph LR") and "assumes" in mm


def test_knowledge_graph_generic_triggers_need_declared_mechanisms():
    g = default_graph([ModelProfile("black_box", "FNO")])
    assert g.unrepresented("black_box", {"Ma": 0.8}) == []             # nothing declared: nothing claimed
    g.add_model(ModelProfile("rans", "solver", ("RANS with eddy viscosity",)))
    assert [x["phenomenon"] for x in g.unrepresented("rans", {"Ma": 0.8, "Re": 1e6})] == ["compressibility"]


# ------------------------------------------------------------------------------------------------ routing
def test_router_routes_by_physics_sleeps_and_escalates():
    r = PhysicsAwareRouter([LAM, BLA, SOL], abstain_below=0.05, escalate_below=0.9, escalate_to="solver")
    d = r.route(feats(800))
    assert d.chosen == "laminar" and not d.awake["blasius"] and not d.escalate
    assert r.route(feats(2e4)).chosen == "blasius"
    d = r.route(feats(3000))                                           # transition: nobody fits
    assert d.escalate and d.chosen == "solver" and "escalated" in str(d)
    lp = r.prior(feats(6000), ["laminar", "blasius", "solver"])
    assert lp["laminar"] < -5 and lp["solver"] == 0.0
    with pytest.raises(ValueError):
        PhysicsAwareRouter([LAM], escalate_to="solver")


def _pipe_stream(seed=0):
    rng = np.random.default_rng(seed)
    segs = [(300, 2000, 25), (5000, 6e4, 30), (400, 1800, 25)]
    re = np.concatenate([np.exp(rng.uniform(np.log(a), np.log(b), n)) for a, b, n in segs])
    return [feats(x) for x in re], [np.array([churchill(x)]) for x in re]


def _pipe_experts():
    coef = np.polyfit(np.linspace(np.log10(4000), np.log10(2e5), 30),
                      np.log10(churchill(np.geomspace(4000, 2e5, 30))), 3)
    return [from_callable("laminar", lambda q: np.array([64.0 / q["Re"]])),
            from_callable("blasius", lambda q: np.array([0.316 * q["Re"] ** -0.25])),
            from_callable("surrogate", lambda q: np.array([10 ** np.polyval(coef, np.log10(q["Re"]))]))]


def test_router_in_ensemble_beats_history_and_keeps_sleeping_weights():
    cases, refs = _pipe_stream()
    sur = ModelProfile("surrogate", "learned", (), (turbulent(4000),), mechanisms=("wall roughness",),
                       envelope=Envelope({"Re": (4000.0, 2e5)}))
    router = PhysicsAwareRouter([LAM, BLA, sur])
    plain = PhysicsEnsemble(_pipe_experts(), mode="select", explore_every=1).run(cases, refs)
    ens = PhysicsEnsemble(_pipe_experts(), mode="select", explore_every=1, router=router)
    w_before = None
    errs = []
    for k, (q, ref) in enumerate(zip(cases, refs, strict=True)):
        out = ens.predict(q)
        assert set(out["awake"]) == {"laminar", "blasius", "surrogate"} and "prior" in out
        errs.append(abs(out["prediction"][0] - ref[0]) / ref[0])
        if k == 25:
            w_before = ens.weights()["laminar"]
        ens.update(q, ref)
        if k == 54:                                                  # end of the turbulent segment
            assert ens.weights()["laminar"] >= 0.5 * w_before       # asleep: not punished for turbulent cases
    back = slice(55, 60)                                              # first laminar cases after the turbulent ones
    assert np.mean(errs[back]) < 0.01 < np.mean(plain.ensemble_errors[back])
    assert np.mean(errs) < np.nanmean(plain.ensemble_errors)


# ------------------------------------------------------------------------------------------------ attribution
def test_attribution_regime_change_vs_extrapolation():
    a = attribute_failure(audit=audit_assumptions(LAM, feats(6000)), error=0.7)
    assert a.failed and a.top.code == "regime_change" and a.notes
    in_physics_out_of_data = ModelProfile("s", "", (), (turbulent(4000),), envelope=Envelope({"Re": (4000.0, 2e4)}))
    b = attribute_failure(audit=audit_assumptions(in_physics_out_of_data, feats(2e5)), error=0.2)
    assert b.top.code == "extrapolation"


def test_attribution_low_residual_high_error_is_missing_physics():
    mon = PhysicsValidityMonitor([residual_check(lambda p, q: 0.003, 0.03, name="smooth_law")])
    a = attribute_failure(validity=mon.check(np.array([0.02])), audit=audit_assumptions(BLA, feats(6e4)), error=0.15,
                          missing_mechanisms=["wall roughness"])
    assert a.top.code == "missing_physics" and a.top.probability > 0.6
    assert any("wall roughness" in e for e in a.top.evidence)
    # the same low residual without a reference: no failure claimed, and the caveat is stated
    c = attribute_failure(validity=mon.check(np.array([0.02])), audit=audit_assumptions(BLA, feats(6e4)))
    assert not c.failed and any("does not prove" in n for n in c.notes)


def test_attribution_other_causes():
    a = attribute_failure(error=0.3, measurement_suspect=True)
    assert a.top.code == "measurement"
    a = attribute_failure(error=0.3, resolution={"gci": 0.2, "gci_tol": 0.05})
    assert a.top.code == "numerical_resolution"
    mon = PhysicsValidityMonitor([
        boundary_check(lambda p, q: p[:1], lambda q: np.array([1.0]), rel_tol=0.01),
        conservation_check(lambda p, q: float(p.sum()), lambda q: 2.0, rel_tol=0.01)])
    rep = mon.check(np.array([0.5, 1.5]))                               # boundary wrong, total right
    assert attribute_failure(validity=rep).top.code == "boundary_condition"
    rep = mon.check(np.array([1.0, 1.5]))                               # boundary right, total wrong
    assert attribute_failure(validity=rep).top.code == "conservation"
    mon_r = PhysicsValidityMonitor([residual_check(lambda p, q: 1.0, 0.01)])
    assert attribute_failure(validity=mon_r.check(np.zeros(2))).top.code == "pde_violation"


def test_attribution_no_failure_and_confidence_tracks_evidence():
    a = attribute_failure(audit=audit_assumptions(LAM, feats(800)), error=0.001)
    assert not a.failed and a.top.code == "unexplained"
    full = attribute_failure(audit=audit_assumptions(LAM, feats(6000)), error=0.7, spread=0.01,
                             resolution={"gci": 0.01}, validity=PhysicsValidityMonitor([]).check(np.ones(2)))
    thin = attribute_failure(audit=audit_assumptions(LAM, feats(6000)))
    assert full.top.code == thin.top.code == "regime_change"
    assert full.confidence > thin.confidence
    assert "evidence not available" in str(thin)


# ------------------------------------------------------------------------------------------------ improvement
def test_recommendations_name_alternatives_and_design_points():
    g = default_graph([LAM, BLA, SOL])
    audit = audit_assumptions(BLA, feats(5e5))
    att = attribute_failure(audit=audit, error=0.4)
    recs = recommend(att, audit, g, feats(5e5), "blasius")
    ids = [r.action_id for r in recs]
    assert ids[0] == "switch_formulation"
    sw = recs[0]
    assert sw.details["missing_mechanisms"] == ["wall roughness"] and "solver" in sw.details["alternatives"]["wall roughness"]
    hifi = next(r for r in recs if r.action_id == "hifi_new_regime")
    pts = hifi.details["design_points"]
    assert len(pts) == 5 and pts[0] == feats(5e5)
    assert all(p["Ma"] == 0.05 for p in pts)                             # only the offending features are varied
    assert len({round(p["eps_plus"], 6) for p in pts}) == 5


def test_design_points_and_plan():
    pts = design_points({"Re": 1e4, "x": -1.0}, spread=0.2, n=6, seed=1)
    assert len(pts) == 6 and all(10 ** -0.2 * 1e4 - 1 <= p["Re"] <= 10 ** 0.2 * 1e4 + 1 for p in pts)
    a1 = attribute_failure(error=0.3, resolution={"gci": 0.2})
    a2 = attribute_failure(error=0.3, resolution={"gci": 0.3})
    a3 = attribute_failure(error=0.3, measurement_suspect=True)
    plan = improvement_plan([a1, a2, a3])
    assert plan[0].cause == "numerical_resolution" and plan[0].cases == 2


# ------------------------------------------------------------------------------------------------ diagnose
def test_diagnose_report():
    d = diagnose(LAM, feats(6000), error=0.7)
    assert not d.trusted and d.attribution.top.code == "regime_change"
    rep = d.report()
    for section in ("## Assumptions", "## Physics the model does not represent", "## Failure attribution",
                    "## Recommended next steps"):
        assert section in rep
    ok = diagnose(LAM, feats(800), error=0.001)
    assert ok.trusted and not ok.recommendations
