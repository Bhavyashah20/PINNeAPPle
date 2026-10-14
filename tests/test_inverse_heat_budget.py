"""run_plate under a ComputeBudget (equal-budget comparisons) keeps the default behaviour without one."""
from pinneapple_analysis.cost import ComputeBudget
from pinneapple_analysis.inverse_problems.heat_transfer import run_plate


def test_run_plate_with_a_step_budget_reports_its_accounting():
    r = run_plate(power=10.0, h_true=25.0, noise=0.0, budget=ComputeBudget(steps=30), width=16, depth=2, n_colloc=300)
    assert r["steps"] == 30 and r["train"]["steps"] == 30 and r["train"]["stopped_by"] == "steps"
    assert r["train"]["flops_per_step"] > 0 and r["h_pinn"] > 0
    assert len(r["field_pinn"]) == len(r["field_reference"])


def test_run_plate_without_a_budget_has_no_accounting():
    r = run_plate(power=10.0, h_true=25.0, noise=0.0, steps=5)
    assert "train" not in r and r["steps"] == 5
