"""Equal-budget comparison toolkit (pinneapple_analysis.cost.equal_budget)."""
import math

import numpy as np
import pytest
import torch

from pinneapple_analysis.cost import (
    AmortizedCost,
    ComputeBudget,
    cosine_lr,
    count_flops,
    equal_budget_search,
    hardware_record,
    relative_l2,
    time_per_query,
    train_within_budget,
)


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def test_count_flops_is_exact_for_a_linear_layer_forward_and_backward():
    lin = torch.nn.Linear(8, 4, bias=False)
    x = torch.rand(32, 8)
    fwd, per_op, _ = count_flops(lambda: lin(x))
    assert fwd == 2 * 32 * 8 * 4
    both, _, _ = count_flops(lambda: lin(x).sum().backward())
    assert both == 2 * fwd                       # weight gradient only: x needs no gradient
    assert any("mm" in k for k in per_op)


def test_count_flops_runs_around_autograd_grad_and_second_derivatives_cost_more():
    net = _mlp()
    x = torch.rand(64, 2, requires_grad=True)

    def residual(order):
        def f():
            u = net(x)
            g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            if order == 2:
                g = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0]
            (g ** 2).mean().backward()
        return f

    f1, f2 = count_flops(residual(1))[0], count_flops(residual(2))[0]
    assert 0 < f1 < f2


def test_count_flops_includes_ffts():
    x = torch.rand(4, 3, 64)
    n, per_op, _ = count_flops(lambda: torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=64, dim=-1))
    expected = 2 * int(5 * 64 * math.log2(64) * 12 / 2)
    assert n == expected
    assert any("fft" in k for k in per_op)


def test_budget_validation_fraction_and_scaling():
    with pytest.raises(ValueError):
        ComputeBudget()
    with pytest.raises(ValueError):
        ComputeBudget(wall_s=-1)
    b = ComputeBudget(wall_s=10, flops=100)
    assert b.fraction(5, 80, 0) == 0.8
    assert b.scaled(0.5).wall_s == 5 and b.scaled(0.5).flops == 50


def test_train_within_budget_stops_at_steps_and_flops_and_measures_flops():
    net = _mlp()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, y = torch.rand(32, 2), torch.rand(32, 1)
    seen = []

    def step(progress):
        seen.append(progress)
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        return loss.item()

    run = train_within_budget(step, ComputeBudget(steps=25))
    assert run.steps == 25 and run.stopped_by == "steps"
    assert run.flops_per_step > 0 and run.flops == pytest.approx(25 * run.flops_per_step)
    assert seen[0] == 0.0 and all(0 <= p < 1 for p in seen) and seen == sorted(seen)
    run2 = train_within_budget(step, ComputeBudget(flops=10.5 * run.flops_per_step))
    assert run2.steps == 11 and run2.stopped_by == "flops"


def test_train_within_budget_wall_clock_pauses_for_evaluation():
    calls = []

    def step(progress):
        x = sum(range(2000))
        return float(x)

    def evaluate():
        import time
        time.sleep(0.05)
        calls.append(1)
        return {"err": 0.0}

    run = train_within_budget(step, ComputeBudget(wall_s=0.2), flops_per_step=0.0, evaluate=evaluate,
                              eval_at=(0.25, 0.5, 0.75))
    assert run.stopped_by == "wall_s"
    assert 0.2 <= run.wall_s < 0.3                        # evaluation time not charged
    assert len(calls) == 4 and run.eval_wall_s >= 0.2
    assert run.history[-1]["final"] and "eval" in run.history[-1]


def test_train_within_budget_converged_stop():
    n = {"i": 0}

    def step(progress):
        n["i"] += 1
        if n["i"] > 3:
            raise StopIteration
        return 0.0

    run = train_within_budget(step, ComputeBudget(steps=100), flops_per_step=1.0)
    assert run.steps == 3 and run.stopped_by == "converged"


def test_cosine_lr_endpoints():
    assert cosine_lr(0.0, 1.0) == pytest.approx(1.0)
    assert cosine_lr(1.0, 1.0, final_fraction=0.1) == pytest.approx(0.1)
    assert cosine_lr(2.0, 1.0, final_fraction=0.1) == pytest.approx(0.1)


def test_equal_budget_search_same_number_of_trials_and_best():
    space = {"a": [1, 2, 3], "b": [10, 20]}
    res = equal_budget_search(space, lambda c: (c["a"] - 2) ** 2 + c["b"] / 100, n_trials=6, seed=1)
    assert res.n_trials == 6 and res.best_config == {"a": 2, "b": 10}
    small = equal_budget_search({"a": [1, 2]}, lambda c: c["a"], n_trials=5)
    assert small.n_trials == 2
    nan = equal_budget_search({"a": [1, 2]}, lambda c: float("nan") if c["a"] == 1 else 5.0, n_trials=2)
    assert nan.best_config == {"a": 2}
    again = equal_budget_search(space, lambda c: c["a"], n_trials=3, seed=1)
    assert [t.config for t in again.trials] == [t.config for t in res.trials[:3]]


def test_amortized_cost_and_break_even():
    pinn = AmortizedCost(per_query_train_s=300.0, infer_s=1e-3)
    op = AmortizedCost(data_s=3000.0, train_s=300.0, infer_s=1e-2)
    assert pinn.per_query(1) == pytest.approx(300.001)
    assert op.per_query(10) == pytest.approx(330.01)
    n = op.break_even(pinn)
    assert n == pytest.approx(3300.0 / (300.001 - 0.01))
    assert pinn.break_even(op) is None
    assert AmortizedCost().break_even(op) == 0.0


def test_relative_l2_per_sample():
    t = np.ones((3, 4, 4))
    p = t.copy()
    p[1] *= 1.1
    e = relative_l2(p, t)
    assert e.shape == (3,) and e[0] == 0 and e[1] == pytest.approx(0.1)
    assert relative_l2(torch.tensor(p), torch.tensor(t))[1] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        relative_l2(p, t[:2])


def test_time_per_query_and_hardware_record():
    r = time_per_query(lambda: sum(range(100)), repeats=5, warmup=1, n_per_call=10)
    assert r["median_s"] > 0 and r["n_per_call"] == 10
    x = torch.rand(4, 1, requires_grad=True)
    g = time_per_query(lambda: torch.autograd.grad((x ** 2).sum(), x)[0], repeats=2, warmup=0, no_grad=False)
    assert g["median_s"] > 0
    hw = hardware_record()
    assert hw["logical_cpus"] and "torch" in hw and "numpy" in hw


def test_train_within_budget_checkpoint_and_resume_continue_the_same_budget():
    saved = []
    n = {"i": 0}

    def step(progress):
        n["i"] += 1
        return float(n["i"])

    budget = ComputeBudget(steps=50)
    first = train_within_budget(step, budget, flops_per_step=2.0, checkpoint=saved.append, checkpoint_every_s=0.0)
    assert first.steps == 50 and saved and saved[-1]["steps"] == 50
    mid = next(s for s in saved if s["steps"] == 20)
    n["i"] = 0
    rest = train_within_budget(step, budget, resume=mid)
    assert n["i"] == 30 and rest.steps == 50 and rest.flops == 100.0 and rest.flops_per_step == 2.0
    assert any(h.get("resumed") for h in rest.history)
