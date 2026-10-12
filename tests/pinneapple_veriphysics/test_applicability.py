"""Applicability map: pure classification of already-computed facts -- no
training, no pinneapple imports needed for the map itself."""
from pinneapple_veriphysics.applicability import (
    FAIL, INFERRED, NOT_RUN, PASS, build_applicability_map,
)
from pinneapple_veriphysics.decision import DecisionRecord
from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf

CHECKS = [{"name": "pde_residual", "passed": True, "detail": "res=1e-3", "value": 1e-3, "threshold": 1e-2}]


def _map(**kw):
    base = dict(
        architecture="vanilla_pinn", coords=["x", "t"], fields=["u"],
        domain_bounds={"x": (0.0, 1.0), "t": (0.0, 1.0)}, physical_parameters={"nu": 0.01},
        guardrail_checks=CHECKS, confidence_components=[], final_loss=1e-3,
        loss_history=[1.0] * 50 + [1e-3] * 50, epochs=100, n_collocation=64,
    )
    base.update(kw)
    return build_applicability_map(**base)


def _item(m, key):
    return next(i for i in m.checklist if i.key == key)


def test_all_eight_items_present_and_absent_checks_are_not_run():
    m = _map()
    assert [i.key for i in m.checklist] == [
        "surrogate", "error", "convergence", "generalization", "variables", "weights", "uq", "physics_laws"]
    assert _item(m, "uq").status == NOT_RUN
    assert _item(m, "generalization").status == NOT_RUN
    assert _item(m, "physics_laws").status == NOT_RUN and _item(m, "physics_laws").tier == INFERRED
    assert _item(m, "error").status == PASS
    assert any("nu other than 0.01" in u for u in m.unsupported)


def test_unconverged_training_and_failed_conservation_are_reported_as_fail():
    m = _map(loss_history=[1.0 - i * 0.005 for i in range(100)],
             guardrail_checks=CHECKS + [{"name": "conservation_mass_continuity", "passed": False, "detail": "div!=0"}])
    assert _item(m, "convergence").status == FAIL
    assert _item(m, "physics_laws").status == FAIL


def test_uq_and_heldout_become_verified_only_when_supplied():
    m = _map(confidence_components=[{"name": "uq_calibration", "score": 0.8, "source_summary": "coverage 0.9"}],
             calibration={"ece": 0.05, "coverage": 0.91, "target_coverage": 0.9},
             heldout_error={"metric": "rel_L2", "value": 0.02, "threshold": 0.05, "description": "unseen nu=0.02"})
    assert _item(m, "uq").status == PASS
    assert _item(m, "generalization").status == PASS


def test_record_renders_and_pdf_builds_from_dict_form():
    rec = DecisionRecord(
        run_id="r", problem_description="p", recommended_strategy="s", reasoning="r",
        trust_score=50.0, trust_coverage=0.5, trustworthy=True, applicability=_map().to_dict())
    text = rec.render()
    assert "Applicability map" in text and "[NOT_RUN] Uncertainty quantification?" in text
    assert render_evidence_report_pdf(rec).startswith(b"%PDF")


def test_dimensional_check_alone_does_not_count_as_conservation():
    m = _map(guardrail_checks=CHECKS + [{"name": "dimensional_analysis", "passed": True, "detail": "ok"}])
    assert _item(m, "physics_laws").status == NOT_RUN


def test_overconservative_uq_is_not_reported_as_calibrated():
    m = _map(confidence_components=[{"name": "uq_calibration", "score": 0.76, "source_summary": "ECE=0.24"}],
             calibration={"ece": 0.24, "coverage": 1.0, "target_coverage": 0.9})
    assert _item(m, "uq").status == FAIL and "OVER-conservative" in _item(m, "uq").summary


def test_noisy_but_plateaued_loss_counts_as_converged():
    import random
    random.seed(0)
    hist = [1.0 / (1 + i) for i in range(200)] + [1e-3 * (1 + 0.3 * random.random()) for i in range(800)]
    assert _item(_map(loss_history=hist), "convergence").status == PASS


def test_plateau_ignores_resampling_noise_but_catches_real_decrease():
    """#5: a flat loss with per-step scatter is converged; a clear downward trend is not."""
    import random

    from pinneapple_veriphysics.applicability import PLATEAU_REL_TOL, _tail_plateau
    rnd = random.Random(1)
    flat_fails = sum(_tail_plateau([1e-3 * (1 + 0.3 * rnd.gauss(0, 1)) for _ in range(300)]) >= PLATEAU_REL_TOL
                     for _ in range(100))
    assert flat_fails <= 3                                   # was about 50 of 100 with plain window means
    falling = [1e-3 * (2.718 ** (-3 * i / 300)) * (1 + 0.05 * rnd.gauss(0, 1)) for i in range(300)]
    assert _tail_plateau(falling) >= PLATEAU_REL_TOL         # e-fold 3 over the run: about 26 % between windows
    assert _tail_plateau([1.0] * 10) is None


def test_tolerances_from_the_request_change_the_verdict_and_are_recorded():
    """#6: a stricter calibration tolerance turns PASS into FAIL; the map lists what was used and its source."""
    import pytest

    from pinneapple_veriphysics.applicability import Tolerances
    cal = {"calibration": {"ece": 0.08, "coverage": 0.9, "target_coverage": 0.9},
           "confidence_components": [{"name": "uq_calibration", "score": 0.9, "source_summary": "ECE=0.08"}]}
    default = _map(**cal)
    assert _item(default, "uq").status == PASS
    strict = _map(**cal, tolerances={"ece": 0.05})
    assert _item(strict, "uq").status == FAIL
    assert strict.tolerances["from_request"] == {"ece": 0.05} and strict.tolerances["used"]["ece"] == 0.05
    assert "ece=0.05 (request)" in strict.render() and "coverage=0.05 (default heuristic)" in strict.render()
    with pytest.raises(ValueError, match="unknown tolerance"):
        Tolerances.from_any({"plateau": 0.1})
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        Tolerances.from_any({"ece": 2})


def test_fast_robustness_mode_is_stated_in_the_map():
    """#7: a map built from fast-mode studies says the studies were shorter and parameter shifts were skipped."""
    m = _map(robustness_mode="fast", study_epochs=250)
    assert any("FAST mode" in r and "250 epochs" in r for r in m.inferred)
    assert not any("FAST mode" in r for r in _map().inferred)
