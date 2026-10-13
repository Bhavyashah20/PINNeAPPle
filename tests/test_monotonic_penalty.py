"""Hinge penalty on input-output derivatives (pinneapple_neural.trainer.monotonic)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.trainer.monotonic import monotone_violation_share, monotonicity_penalty  # noqa: E402


def test_penalty_sign_convention_and_margin():
    lin = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        lin.weight[:] = torch.tensor([[-0.5, 0.2]])           # decreasing in x0, increasing in x1
    z = torch.randn(64, 2)
    assert float(monotonicity_penalty(lin, z, [0, 1], [-1, +1]).detach()) == 0.0
    assert float(monotonicity_penalty(lin, z, [0], [+1]).detach()) == pytest.approx(0.25)     # wrong sign: (0.5)^2
    assert float(monotonicity_penalty(lin, z, [1], [+1], margin=0.3).detach()) == pytest.approx(0.01)   # 0.2 < margin
    assert float(monotonicity_penalty(lin, z, [0], [-1], scale=[0.1], margin=0.1).detach()) == pytest.approx(0.0025)
    assert monotone_violation_share(lin, z, [0, 1], [-1, -1]) == [0.0, 1.0]


def test_penalty_makes_a_weak_monotone_effect_come_out_right():
    """y = sin(3 x1) - 0.01 x0 + noise: the x0 effect is below the noise, so an unconstrained fit gets its sign wrong
    on part of the domain; the penalty (on training and random points) cuts the violations on fresh points."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(512, 2, generator=g) * 2 - 1
    y = torch.sin(3 * x[:, 1:]) - 0.01 * x[:, :1] + 0.02 * torch.randn(512, 1, generator=g)
    zt = torch.rand(2000, 2, generator=g) * 2 - 1

    def fit(lam):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(600):
            loss = torch.mean((m(x) - y) ** 2)
            if lam:
                zr = torch.rand(256, 2, generator=g) * 2 - 1        # random in-range points, as in the recipes
                loss = loss + lam * (monotonicity_penalty(m, x, [0], [-1], margin=1e-3) +
                                     monotonicity_penalty(m, zr, [0], [-1], margin=1e-3))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return monotone_violation_share(m, zt, [0], [-1])[0], float(torch.mean((m(x) - y) ** 2))

    free, _ = fit(0.0)
    constrained, mse = fit(10.0)
    # a soft penalty does not guarantee the sign everywhere; it must cut the violations several-fold without
    # spoiling the fit (the zoo gate then measures the share on held-out designs)
    assert free > 0.1 and constrained < 0.25 * free and np.isfinite(mse) and mse < 2e-3
