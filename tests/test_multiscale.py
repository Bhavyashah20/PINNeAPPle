"""Multiscale physics: filters, closures, multiscale residuals, cross-scale consistency and scale-aware routing."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.multiscale import (
    DynamicSmagorinsky,
    Filter,
    GradientModel,
    ScaleAwareRouter,
    ScaleProfile,
    ScaleQuery,
    Smagorinsky,
    VorticityNS2D,
    apply_filter,
    apriori_test,
    closure_health,
    commutation_error,
    cross_scale_consistency,
    germano_error,
    multiscale_residuals,
    ns2d_multiscale_residuals,
    prolong,
    random_vorticity,
    realizability_mismatch,
    refilter,
    restrict,
    scale_compatibility,
    transfer,
)
from pinneapple_physics.multiscale.closures import strain, stress_tendency
from pinneapple_physics.multiscale.ns2d import vorticity_residual

L = 2 * np.pi


def grid(n):
    x = np.arange(n) * L / n
    return np.meshgrid(x, x, indexing="ij")


# ------------------------------------------------------------------------------------------ filters
@pytest.mark.parametrize("kind", ["box", "gaussian", "sharp"])
@pytest.mark.parametrize("m", [1, 3, 7, 12])
def test_transfer_function_exact_on_fourier_modes(kind, m):
    n, delta = 64, 0.37
    x = np.arange(n) * L / n
    u = np.cos(m * x + 0.3)
    out = apply_filter(u, Filter(kind, delta), L, axes=[-1])
    np.testing.assert_allclose(out, transfer(kind, np.array(m), delta) * u, atol=1e-12)


def test_transfer_functions_match_closed_forms():
    k, d = np.linspace(0.1, 30, 50), 0.25
    np.testing.assert_allclose(transfer("box", k, d), np.sin(k * d / 2) / (k * d / 2), rtol=1e-12)
    np.testing.assert_allclose(transfer("gaussian", k, d), np.exp(-k ** 2 * d ** 2 / 24), rtol=1e-12)
    np.testing.assert_array_equal(transfer("sharp", k, d), (k <= np.pi / d).astype(float))


@pytest.mark.parametrize("kind", ["box", "gaussian", "sharp"])
def test_2d_filter_is_tensor_product(kind):
    X, Y = grid(32)
    u = np.sin(3 * X) * np.cos(5 * Y)
    f = Filter(kind, 0.5)
    np.testing.assert_allclose(f(u), transfer(kind, np.array(3), 0.5) * transfer(kind, np.array(5), 0.5) * u,
                               atol=1e-12)


def test_gaussian_filters_compose_and_refilter_is_exact():
    rng = np.random.default_rng(0)
    w = random_vorticity(rng, 64)
    a, b = Filter("gaussian", 0.2), Filter("gaussian", 0.5)
    composed = apply_filter(apply_filter(w, a), Filter("gaussian", np.sqrt(0.5 ** 2 - 0.2 ** 2)))
    np.testing.assert_allclose(composed, apply_filter(w, b), atol=1e-10)
    np.testing.assert_allclose(refilter(apply_filter(w, a), a, b), apply_filter(w, b), atol=1e-10)
    assert realizability_mismatch(a, b, 40.0) < 1e-12
    assert realizability_mismatch(b, a, 40.0) > 0.1                 # a finer filter cannot be recovered
    assert realizability_mismatch(Filter("sharp", 0.3), Filter("gaussian", 0.3), 40.0) > 0.05
    # box of twice the width: its first zero sits at the sharp cut-off, so it is nearly reachable
    assert realizability_mismatch(Filter("sharp", 0.3), Filter("box", 0.6), 40.0) < 0.1


def test_restriction_preserves_integral_and_prolongation_round_trips():
    rng = np.random.default_rng(1)
    w = random_vorticity(rng, 64) + 0.7
    c = restrict(w, 24)
    assert abs(c.mean() - w.mean()) < 1e-13
    smooth = apply_filter(w, Filter("sharp", L / 22))               # band-limited below the coarse Nyquist
    np.testing.assert_allclose(prolong(restrict(smooth, 24), 64), smooth, atol=1e-10)


def test_commutation_error_vanishes_for_uniform_filter_only():
    rng = np.random.default_rng(2)
    w = random_vorticity(rng, 48)
    for kind in ("box", "gaussian", "sharp"):
        assert np.abs(commutation_error(w, kind, 0.4, axis=-2)).max() < 1e-10
    X, _ = grid(48)
    delta = 0.25 + 0.2 * np.sin(X)                                  # width varying along x
    e = commutation_error(w, "gaussian", delta, axis=-2)
    assert np.sqrt(np.mean(e ** 2)) > 1e-2 * np.sqrt(np.mean(w ** 2))


# ------------------------------------------------------------------------------------------ closures
def test_smagorinsky_on_taylor_green_field():
    X, Y = grid(32)
    w = 2 * np.sin(X) * np.sin(Y)                                   # psi = sin x sin y
    s = VorticityNS2D(32, 0.0)
    st = strain(w, s)
    np.testing.assert_allclose(st["S11"], np.cos(X) * np.cos(Y), atol=1e-12)
    np.testing.assert_allclose(st["S12"], 0.0, atol=1e-12)
    cs, d = 0.17, 0.3
    a, b = Smagorinsky(cs=cs, delta=d).stress(w, s)
    nu_t = (cs * d) ** 2 * 2 * np.abs(np.cos(X) * np.cos(Y))
    np.testing.assert_allclose(a, -2 * nu_t * np.cos(X) * np.cos(Y), atol=1e-12)
    np.testing.assert_allclose(b, 0.0, atol=1e-12)


def test_constant_eddy_viscosity_tendency_is_laplacian():
    rng = np.random.default_rng(3)
    s = VorticityNS2D(48, 0.0)
    w = random_vorticity(rng, 48)
    st = strain(w, s)
    t = stress_tendency(-2 * 0.1 * st["S11"], -2 * 0.1 * st["S12"], s)
    np.testing.assert_allclose(t, 0.1 * s.laplacian(w), atol=1e-10 * np.abs(s.laplacian(w)).max())


def test_apriori_2d_gradient_model_beats_eddy_viscosity_and_backscatter():
    rng = np.random.default_rng(4)
    w = VorticityNS2D(96, 1e-3).simulate(random_vorticity(rng, 96, k0=6), [0.3], dt=5e-3)[0]
    f = Filter("gaussian", L / 24)
    smag = apriori_test(w, f, Smagorinsky(delta=f.delta), 48)
    clark = apriori_test(w, f, GradientModel(delta=f.delta), 48)
    assert clark.stress_correlation > 0.9                            # exact to leading order for a Gaussian filter
    assert abs(smag.stress_correlation) < 0.2                        # 2-D: tau ~ omega x rotated strain, not ~ S
    assert smag.backscatter_true > 0.2 and smag.backscatter_model == 0.0


def test_closure_health_flags_overdissipative_and_unstable_closures():
    rng = np.random.default_rng(5)
    nu = 2.5e-4
    w0 = VorticityNS2D(96, nu).simulate(random_vorticity(rng, 96, k0=6), [0.8], dt=5e-3)[0]
    wb = restrict(apply_filter(w0, Filter("sharp", np.pi / 16)), 48)
    s48 = VorticityNS2D(48, nu)
    assert germano_error(wb, Smagorinsky(), s48) > 1.1               # fixed Cs = 0.17 too large in 2-D
    assert germano_error(wb, DynamicSmagorinsky(), s48) < 1.05
    les = VorticityNS2D(48, nu, closure=GradientModel())
    try:
        wl = les.simulate(wb, [3.0], dt=5e-3)
        h = closure_health(wl, wb, 16)
        assert not h.ok and any("enstrophy" in r for r in h.reasons)
    except FloatingPointError:
        pass                                                          # blew up: also a failure
    ok = closure_health(VorticityNS2D(48, nu, closure=DynamicSmagorinsky()).simulate(wb, [0.2], dt=5e-3), wb, 16,
                        closure=DynamicSmagorinsky(), solver=s48)
    assert ok.ok, ok.reasons


# ------------------------------------------------------------------------------------------ residuals
def test_multiscale_residual_of_exact_steady_taylor_green_is_zero():
    X, Y = grid(32)
    nu, k = 0.01, 2
    w = 2 * k * np.sin(k * X) * np.sin(k * Y)                       # steady with forcing f = 2 nu k^2 w
    f = 2 * nu * k ** 2 * w
    filters = [Filter(kind, d) for kind in ("box", "gaussian", "sharp") for d in (0.2, 0.5, 1.0)]
    r = ns2d_multiscale_residuals(w, w, 0.1, nu, filters, forcing=f)
    assert np.all(r.raw < 1e-12) and np.all(r.model < 1e-12)


def test_noise_residual_decays_as_filter_transfer_predicts():
    rng = np.random.default_rng(6)
    n, sigma, dt = 64, 1e-3, 1.0
    a, b = sigma * rng.standard_normal((n, n)), sigma * rng.standard_normal((n, n))
    k = 2 * np.pi * np.fft.fftfreq(n, L / n)
    filters = [Filter("gaussian", d) for d in (0.1, 0.3, 0.6)]
    res = multiscale_residuals(np.stack([a, b]), lambda s, f: ((s[1] - s[0]) / dt, 1.0), filters)
    for i, f in enumerate(filters):
        G2 = (f.transfer(k)[:, None] * f.transfer(k)[None, :]) ** 2
        expected = np.sqrt(2) * sigma / dt * np.sqrt(G2.mean())
        assert abs(res.raw[i] / expected - 1) < 0.1


def test_filtered_dns_residual_is_the_subfilter_term():
    rng = np.random.default_rng(7)
    nu, dt = 1e-3, 2e-3
    traj = VorticityNS2D(64, nu).simulate(random_vorticity(rng, 64, k0=3), [0.2, 0.2 + dt], dt=dt / 2)
    r = ns2d_multiscale_residuals(traj[0], traj[1], dt, nu, [Filter("gaussian", d) for d in (0.2, 0.4)])
    assert np.all(r.subfilter > 5 * r.model)                         # the raw residual is the closure, not an error
    _, scale = vorticity_residual(traj[0], traj[1], dt, nu)
    assert scale > 0


# ------------------------------------------------------------------------------------------ consistency
def test_cross_scale_consistency_passes_for_consistent_solvers_and_fails_with_biased_source():
    rng = np.random.default_rng(8)
    nu = 5e-3
    w0 = apply_filter(random_vorticity(rng, 64, k0=3), Filter("sharp", L / 20))
    fine = VorticityNS2D(64, nu).simulate(w0, [0.3], dt=5e-3)[0]
    coarse = VorticityNS2D(32, nu).simulate(restrict(w0, 32), [0.3], dt=5e-3)[0]
    f = Filter("sharp", L / 16)
    rep = cross_scale_consistency(fine, apply_filter(coarse, f), f, tol=0.05)
    assert rep.passed, str(rep)
    X, Y = grid(32)
    biased = VorticityNS2D(32, nu, forcing=0.5 + 0 * X).simulate(restrict(w0, 32), [0.3], dt=5e-3)[0]
    bad = cross_scale_consistency(fine, apply_filter(biased, f), f, tol=0.05)
    assert not bad.passed and "circulation (mean)" in bad.failed


def test_restricting_an_exact_field_preserves_integrals():
    rng = np.random.default_rng(9)
    w = random_vorticity(rng, 64) + 0.3
    f = Filter("sharp", L / 16)
    rep = cross_scale_consistency(w, restrict(apply_filter(w, f), 32), f, tol=1e-10, integral_tol=1e-10)
    assert rep.passed and rep.field_error < 1e-12


# ------------------------------------------------------------------------------------------ routing
H = L / 64
PROFILES = [
    ScaleProfile("surrogate_coarse", Filter("gaussian", 8 * H), H, re_range=(1e3, 1e4), fidelity=0, cost=0.01),
    ScaleProfile("surrogate_fine", Filter("gaussian", 2 * H), H, re_range=(1e3, 1e4), fidelity=0, cost=0.01),
    ScaleProfile("les", Filter("sharp", np.pi / 16), L / 48, closure="dynamic Smagorinsky", fidelity=1, cost=0.1),
    ScaleProfile("dns", Filter("sharp", 0.0), L / 128, fidelity=2, cost=1.0),
]


def test_scale_compatibility_respects_resolved_scales_and_filters():
    q = ScaleQuery(Filter("gaussian", 4 * H), re=3e3, kmax=32)
    c = {p.name: scale_compatibility(p, q) for p in PROFILES}
    assert not c["surrogate_coarse"].covered and c["surrogate_coarse"].score == 0.0
    assert c["surrogate_fine"].score > 0.99 and c["dns"].score > 0.99
    # a box filter keeps more high wavenumbers (sinc ~ 1/k) than a Gaussian of the same width: not reachable from it
    wide = ScaleProfile("g4", Filter("gaussian", 4 * H), H, fidelity=0)
    assert realizability_mismatch(wide.filter, Filter("box", 4 * H), 32, slope=0.0) > 0.05
    assert scale_compatibility(wide, ScaleQuery(Filter("box", 4 * H), kmax=32)).mismatch < 0.1   # little energy there
    assert scale_compatibility(wide, ScaleQuery(Filter("gaussian", 6 * H), kmax=32)).mismatch < 1e-9
    out = scale_compatibility(PROFILES[1], ScaleQuery(Filter("gaussian", 4 * H), re=1e5))
    assert out.score < 0.01 and any("Re" in r for r in out.reasons)


def test_router_ladder_escalates_with_scale_and_reports_not_covered():
    r = ScaleAwareRouter(PROFILES)
    names = [p.name for p in PROFILES]
    aw = r.awake(ScaleQuery(Filter("gaussian", 8 * H), re=3e3), names)
    assert aw["surrogate_coarse"] and aw["surrogate_fine"] and not aw["les"] and not aw["dns"]
    aw = r.awake(ScaleQuery(Filter("sharp", 2.5 * H), re=3e3), names)   # mild deconvolution of the 2h Gaussian
    assert aw["surrogate_fine"] and not aw["surrogate_coarse"] and not aw["les"]
    aw = r.awake(ScaleQuery(Filter("sharp", 2.5 * H), re=5e4), names)   # surrogates outside their Re range
    assert aw["les"] and not aw["surrogate_fine"] and not aw["dns"]
    aw = r.awake(ScaleQuery(Filter("sharp", H), re=3e3), names)
    assert aw == {"surrogate_coarse": False, "surrogate_fine": False, "les": False, "dns": True}
    route = r.route(ScaleQuery(Filter("sharp", H / 4), re=3e3), names=names)
    assert route.chosen is None and "not covered" in route.reasons[0]
    assert r.escalate_from(ScaleQuery(Filter("sharp", 2.5 * H), re=5e4), names, ["les"]) == \
        {"surrogate_coarse": False, "surrogate_fine": False, "les": False, "dns": True}


def test_ensemble_with_scale_router_and_failing_check_escalates_to_solver():
    names = [p.name for p in PROFILES]
    truth = lambda q: np.full(4, 1.0)  # noqa: E731
    experts = [from_callable(n, (lambda n: lambda q: truth(q) + (0.5 if n == "les" else 0.01 * i))(n))  # noqa: B023
               for i, n in enumerate(names)]
    ens = PhysicsEnsemble(experts, mode="select", router=ScaleAwareRouter(PROFILES),
                          prediction_checks={"les": lambda p, q: (0.0, "closure check failed")})
    q = ScaleQuery(Filter("sharp", 2.5 * H), re=5e4)
    out = ens.predict(q)
    assert out["active"] == "dns" and out["escalated_by_checks"] and out["failed_checks"] == ["les"]
    ens.update(q, truth(q))
    out = ens.predict(ScaleQuery(Filter("gaussian", 8 * H), re=3e3))
    assert out["active"] in ("surrogate_coarse", "surrogate_fine")


# ------------------------------------------------------------------------------------------ learned (torch)
def test_differentiable_les_matches_numpy_solver_and_closure_conserves_circulation():
    torch = pytest.importorskip("torch")
    from pinneapple_physics.multiscale.learned import (
        LearnedClosure,
        TorchVorticityLES2D,
        as_numpy_closure,
    )

    rng = np.random.default_rng(10)
    w0 = apply_filter(random_vorticity(rng, 32, k0=3), Filter("sharp", L / 20))
    ref = VorticityNS2D(32, 5e-3).simulate(w0, [0.1], dt=2.5e-3)[0]
    les = TorchVorticityLES2D(32)
    out = les.rollout(torch.tensor(w0)[None], torch.tensor([5e-3]), 2.5e-3, [40])[0, 0].numpy()
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-4
    cl = LearnedClosure(hidden=8, layers=2)
    torch.nn.init.normal_(cl.net[-1].weight, std=0.1)
    t = as_numpy_closure(cl, 32)(w0)
    assert abs(t.mean()) < 1e-10 and np.abs(t).max() > 0          # a flux divergence: zero mean


def test_scale_conditioned_fno_and_consistency_loss():
    torch = pytest.importorskip("torch")
    from pinneapple_physics.multiscale.learned import (
        ScaleConditionedFNO2d,
        filtering_consistency_loss,
        torch_filter,
        torch_refilter,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32, dtype=torch.float64)
    d1, d2 = torch.tensor([0.2, 0.2], dtype=torch.float64), torch.tensor([0.5, 0.6], dtype=torch.float64)
    np.testing.assert_allclose(torch_refilter(torch_filter(x, "gaussian", d1), "gaussian", d1, d2).numpy(),
                               torch_filter(x, "gaussian", d2).numpy(), atol=1e-10)
    np.testing.assert_allclose(torch_filter(x, "box", 0.4).numpy(), apply_filter(x.numpy(), Filter("box", 0.4)),
                               atol=1e-12)

    class ExactFilter(torch.nn.Module):                               # a perfectly scale-consistent "model"
        conditioned = True

        def forward(self, x, d):
            return torch_filter(x, "gaussian", d)

    assert float(filtering_consistency_loss(ExactFilter(), x, d1, d2)) < 1e-20
    m = ScaleConditionedFNO2d(1, 1, width=8, modes=4, layers=2).double()
    torch.nn.init.normal_(m.film[-1].weight, std=0.5)
    assert float(filtering_consistency_loss(m, x, d1, d2).detach()) > 1e-6
    blind = ScaleConditionedFNO2d(1, 1, width=8, modes=4, layers=2, conditioned=False).double()
    torch.testing.assert_close(blind(x, 0.1), blind(x, 0.9))


# ------------------------------------------------------------------------------------------ benchmark helpers
def test_fourier_coefficients_round_trip():
    from pinneapple_physics.multiscale.filters import fourier_basis, fourier_coefficients, low_modes

    rng = np.random.default_rng(11)
    modes = low_modes(4)
    c = rng.standard_normal(2 * len(modes))
    w = (c @ fourier_basis(modes, 32)).reshape(32, 32)
    np.testing.assert_allclose(fourier_coefficients(w, modes), c, atol=1e-12)
    np.testing.assert_allclose(apply_filter(w, Filter("sharp", np.pi / 4)), w, atol=1e-12)   # all modes |k| <= 4


def test_turbulence_benchmark_streams_and_pairs():
    from pinneapple_physics.multiscale.turbulence import DecayingTurbulence2D

    b = DecayingTurbulence2D(n_out=16, n_train=32, train_times=(0.02, 0.04), t_out=(0.02, 0.04), spinup=0.02,
                             dt_train=1e-2)
    q1, q2 = b.query_stream(3, 9), b.query_stream(3, 9)
    assert [q["re"] for q in q1] == [q["re"] for q in q2]                  # reproducible
    assert [q["band"] for q in q1] == ["low"] * 3 + ["high"] * 3 + ["mid"] * 3
    assert all(q["filter"].delta == q["width"] * b.h for q in q1)
    data = {"w": np.stack([b._map([(s, 0.01, 32, 1e-2, 0.02, b.train_times, 3.0, 0.5, 0)], 1)[0] for s in range(4)]),
            "nu": np.full(4, 0.01), "band": np.array(["low"] * 4)}
    tr, te = b.split(data, "low", test_frac=0.25)
    assert len(te) == 1 and len(tr) == 3
    x, y = b.filtered_pairs(data, tr, Filter("gaussian", 2 * b.h))
    assert x.shape == (3, 16, 16) and y.shape == (3, 2, 16, 16)
    assert abs(b.les_filter.cutoff - b.les_n // 3) < 1e-9


def test_adapt_output_and_closed_residual():
    from pinneapple_physics.multiscale.comparison import (
        adapt_output,
        closed_multiscale_residual,
        raw_residual,
    )

    X, Y = grid(32)
    w = 2 * np.sin(X) * np.sin(Y)
    g2, g4 = Filter("gaussian", 0.2), Filter("gaussian", 0.4)
    src = apply_filter(w, g2)
    np.testing.assert_allclose(adapt_output(src, g2, g4, aware=True), apply_filter(w, g4), atol=1e-12)
    blind = adapt_output(src, g2, g4, aware=False)                         # applies g4 on top: wider than asked
    assert np.abs(blind - apply_filter(w, g4)).max() > 1e-4
    nu, dt = 0.01, 0.01
    pair = np.stack([w * np.exp(-2 * nu * 0.0), w * np.exp(-2 * nu * dt)])  # exact decaying Taylor-Green
    assert raw_residual(pair, nu, dt) < 1e-5
    assert closed_multiscale_residual(apply_filter(pair, g2), nu, dt, g2, 0.1) < 1e-4


def test_run_ensembles_on_a_synthetic_table():
    from pinneapple_physics.multiscale.comparison import (
        EnsembleConfig,
        markdown_tables,
        run_ensembles,
        summarise_runs,
    )

    X, Y = grid(16)
    truth = np.stack([2 * np.sin(X) * np.sin(Y)] * 2)
    names = ["coarse", "dns"]
    prof = {"coarse": ScaleProfile("coarse", Filter("gaussian", 0.8), 0.4, fidelity=0),
            "dns": ScaleProfile("dns", Filter("sharp", 0.0), 0.1, fidelity=2)}
    qs = [{"id": k, "nu": 0.01, "re": 100.0, "width": w, "filter": Filter("gaussian", w * 0.4)}
          for k, w in enumerate([1, 4, 4, 1])]
    raw = np.stack([np.stack([apply_filter(truth, prof["coarse"].filter) * 0.9, truth]) for _ in qs])
    table = {"raw": raw, "secs": np.tile([0.01, 1.0], (4, 1)), "check": np.full((4, 2), np.nan),
             "reasons": np.empty((4, 2), dtype=object), "names": np.array(names)}
    refs = [apply_filter(truth, q["filter"]) for q in qs]
    cfgs = {"blind": EnsembleConfig(False, False, False, tuple(names)),
            "aware": EnsembleConfig(True, True, False, tuple(names))}
    res = run_ensembles(qs, refs, table, prof, cfgs, dt=0.01, h=0.4, lookahead=0.0)
    assert res["aware"]["active"][0] == "dns" and res["aware"]["active"][1] == "coarse"   # ladder by scale
    rows = summarise_runs({0: (qs, res)}, prof, list(cfgs), (1, 4), solver="dns")
    assert rows[1]["scale_violations"] == 0
    assert "| aware |" in markdown_tables(rows, (1, 4))


def test_learned_benchmark_models():
    torch = pytest.importorskip("torch")
    from pinneapple_physics.multiscale.filters import fourier_basis, low_modes
    from pinneapple_physics.multiscale.learned import (
        FourierGalerkinPINN,
        periodic_cnn2d,
        torus_graph,
    )

    assert periodic_cnn2d(2, 3, width=8)(torch.randn(1, 2, 16, 16)).shape == (1, 3, 16, 16)
    e, a = torus_graph(8, (1, 2))
    assert e.shape == (2, 8 * 8 * 8) and a.shape == (8 * 8 * 8, 2)
    pinn = FourierGalerkinPINN(kmax=2, scale=1.0)
    rng = np.random.default_rng(12)
    c = rng.standard_normal((1, 2 * len(low_modes(2))))
    nu = 0.01
    w = (c @ fourier_basis(low_modes(2), 32)).reshape(32, 32)
    s = VorticityNS2D(32, nu, dealias=False)
    expected = apply_filter(-s.advection(w) + nu * s.laplacian(w), Filter("sharp", np.pi / 2))
    got = (pinn.tendency(torch.tensor(c, dtype=torch.float32), torch.tensor([[nu]])).numpy().astype(float)
           @ fourier_basis(low_modes(2), 32)).reshape(32, 32)
    np.testing.assert_allclose(got, expected, atol=1e-4 * np.abs(expected).max())


def test_error_tolerance_escalates_and_returns():
    names = [p.name for p in PROFILES]
    bad = {"v": 0.5}
    experts = [from_callable(n, (lambda n: lambda q: np.full(4, 1.0) + (bad["v"] if n.startswith("surrogate") else 0.0))(n))  # noqa: B023
               for n in names]
    ens = PhysicsEnsemble(experts, mode="select", router=ScaleAwareRouter(PROFILES), error_tolerance=0.2,
                          error_context=lambda q: q.filter.delta)
    q = lambda: ScaleQuery(Filter("gaussian", 8 * H), re=3e3)  # noqa: E731
    assert ens.predict(q())["active"].startswith("surrogate")      # no error known yet: cheapest rung
    ens.update(reference=np.ones(4))
    out = ens.predict(q())
    assert out["active"] in ("les", "dns") and out["escalated_by_checks"]
    ens.update(reference=np.ones(4))
    bad["v"] = 0.0                                                  # surrogates good again: they are still scored
    ens.predict(q())
    ens.update(reference=np.ones(4))
    assert ens.predict(q())["active"].startswith("surrogate")


# ------------------------------------------------------------------------------------------ issue acceptance checks
def _steady_adv_diff_central(n: int, pe: float) -> tuple[np.ndarray, np.ndarray]:
    """u' = u'' / Pe on [0, 1], u(0) = 0, u(1) = 1, central differences on n intervals; exact solution too."""
    x = np.linspace(0, 1, n + 1)
    h = 1.0 / n
    a, b = 1.0 / pe / h ** 2, 1.0 / (2 * h)
    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)
    for i in range(n - 1):
        A[i, i] = -2 * a
        if i > 0:
            A[i, i - 1] = a + b
        if i < n - 2:
            A[i, i + 1] = a - b
    rhs[-1] = -(a - b) * 1.0
    u = np.r_[0.0, np.linalg.solve(A, rhs), 1.0]
    exact = np.expm1(pe * x) / np.expm1(pe)
    return u, exact


def test_issue_599_router_picks_fine_model_exactly_where_coarse_grid_peclet_exceeds_two():
    n_c, n_f = 20, 400
    profiles = [ScaleProfile("coarse", Filter("sharp", 0.0), 1.0 / n_c, re_range=(0.0, 2.0 * n_c), fidelity=0),
                ScaleProfile("fine", Filter("sharp", 0.0), 1.0 / n_f, re_range=(0.0, 2.0 * n_f), fidelity=0)]
    r = ScaleAwareRouter(profiles, re_band=1e-3)
    for pe in (5.0, 20.0, 38.0, 42.0, 80.0, 200.0):
        aw = r.awake(ScaleQuery(Filter("sharp", 1.0 / n_c), re=pe), ["coarse", "fine"])
        chosen = r.route(ScaleQuery(Filter("sharp", 1.0 / n_c), re=pe), {"coarse": 0.9, "fine": 0.1}).chosen
        pe_grid = pe / n_c
        assert (chosen == "fine") == (pe_grid > 2), (pe, chosen)
        assert aw["coarse"] == (pe_grid <= 2)
        u, exact = _steady_adv_diff_central(n_c, pe)                  # the analytic reason: monotone iff Pe_h <= 2
        monotone = bool(np.all(np.diff(u) >= -1e-12))
        assert monotone == (pe_grid <= 2)
        uf, ef = _steady_adv_diff_central(n_f, pe)
        assert np.abs(uf - ef).max() < np.abs(u - exact).max() + 1e-12
    route = r.route(ScaleQuery(Filter("sharp", 1.0 / (4 * n_f)), re=10.0), names=["coarse", "fine"])
    assert route.chosen is None and "not covered" in route.reasons[0]


def test_issue_600_truncated_field_residual_rises_only_below_its_cutoff():
    rng = np.random.default_rng(13)
    nu, dt = 1e-3, 2e-3
    traj = VorticityNS2D(64, nu).simulate(random_vorticity(rng, 64, k0=5), [0.2, 0.2 + dt], dt=dt / 2)
    kc = 8                                                          # an operator that only carries |k| <= 8
    trunc = apply_filter(traj, Filter("sharp", np.pi / kc))
    widths = [Filter("sharp", np.pi / k) for k in (24, 16, 8, 4)]
    true = ns2d_multiscale_residuals(traj[0], traj[1], dt, nu, widths, subfilter=None)
    cut = ns2d_multiscale_residuals(trunc[0], trunc[1], dt, nu, widths, subfilter=None)
    ratio = cut.raw / true.raw
    assert ratio[0] > 1.5 and ratio[1] > 1.2                          # finer than the cut-off: residual rises
    np.testing.assert_allclose(ratio[2:], 1.0, rtol=1e-6)             # at and above the cut-off: same as the truth


def _heat_fd(n: int, t: float = 0.1, nu: float = 1.0) -> np.ndarray:
    """Periodic 1-D heat equation, second-order central differences, RK4 with a small step (space error dominates)."""
    x = np.arange(n) * 2 * np.pi / n
    u = np.sin(x) + 0.5 * np.sin(3 * x) + 0.2 * np.cos(5 * x)
    h = 2 * np.pi / n
    lap = lambda v: (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / h ** 2  # noqa: E731
    steps = 4000
    k = t / steps
    for _ in range(steps):
        k1 = nu * lap(u)
        k2 = nu * lap(u + 0.5 * k * k1)
        k3 = nu * lap(u + 0.5 * k * k2)
        k4 = nu * lap(u + k * k3)
        u = u + k / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def test_issue_602_grid_levels_pass_with_mismatch_at_the_observed_order():
    u32, u64, u128 = _heat_fd(32), _heat_fd(64), _heat_fd(128)
    f = Filter("sharp", 2 * np.pi / 16)                              # compare on the shared large scales
    mean = {"mean": lambda v: float(np.mean(v)), "energy": lambda v: float(np.mean(v ** 2))}
    r1 = cross_scale_consistency(u64, apply_filter(u32, f), f, length=2 * np.pi, tol=0.01, integrals=mean,
                                 spectrum=False)
    r2 = cross_scale_consistency(u128, apply_filter(u64, f), f, length=2 * np.pi, tol=0.01, integrals=mean,
                                 spectrum=False)
    assert r1.passed and r2.passed
    order = np.log2(r1.field_error / r2.field_error)
    assert abs(order - 2.0) < 0.2                                    # second-order scheme
