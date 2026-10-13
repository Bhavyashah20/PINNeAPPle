"""Scale-aware adaptive ensemble on decaying 2-D turbulence: which model can answer a question at the scale it asks?

Test case (:class:`pinneapple_physics.multiscale.turbulence.DecayingTurbulence2D`): 2-D decaying turbulence on the
periodic square ``[0, 2 pi)^2`` at three Reynolds numbers (``Re = 1 / nu`` with an rms velocity of 1: 150-300,
700-1400 and 2800-5000). Every case starts from a random field (energy spectrum peaked at ``k = 6``) evolved for 0.5
time units at its own viscosity, so the cascade is developed. A query gives that state and asks for the vorticity at
``t = 0.28`` and ``0.30`` later, *filtered at a requested width* ``Delta`` (box, Gaussian or sharp spectral filter,
``Delta / h`` in {1, 2, 4, 6, 8} with ``h = 2 pi / 64``), on a 64^2 grid. The independent reference is a
pseudo-spectral DNS at 256^2 (dt = 2.5e-3) filtered at that width; training data come from a DNS at 128^2.

Experts (all predict the same 2 x 64 x 64 output; each declares the scales it represents):

==================  ============================  =================  =============================================
expert              filter of its output          Re range           how
==================  ============================  =================  =============================================
FNO                 Gaussian 2h                   150-300            supervised on filtered DNS
CNN                 Gaussian 4h, on 32^2          2800-5000          supervised; periodic dilated 2-D CNN
DeepONet            sharp 8h (k <= 4)             700-1400           supervised; branch = low Fourier modes
GNN                 box 4h, on a 24^2 graph       700-1400           supervised; MeshGraphNet on a multiscale torus
PINN                sharp 8h (k <= 4)             150-300            physics loss only (Fourier-Galerkin NS)
LES-Smagorinsky     sharp, k <= 16 (LES 48^2)     any                Cs = 0.17
LES-dynamic         sharp, k <= 16                any                dynamic Smagorinsky (Germano-Lilly)
LES-gradient        sharp, k <= 16                any                Clark gradient model
DNS-128             resolved (k <= 42)            <= 6000            pseudo-spectral DNS 128^2, dt = 5e-3
SC-FNO              Gaussian 2h..8h (Delta in)    700-5000           scale-conditioned FNO + consistency loss
LES-learned         sharp, k <= 16                700-5000           CNN closure trained a posteriori
==================  ============================  =================  =============================================

Ensembles compared on the same query streams (3 seeds, 60 queries each; Re regime low -> high -> mid, requested
width and filter drawn per query):

A. original, scale-blind: the nine base experts; past errors + residual look-ahead (as in
   ``five_model_families_gif.py``); every output filtered with the requested filter, whatever it already carries;
B. + scale-aware selection: :class:`~pinneapple_physics.multiscale.ScaleAwareRouter` (declared scales, filter
   realizability, Re range, fidelity ladder surrogate -> LES -> DNS, escalating also when a rung's last error at
   the requested width exceeds 25 %); outputs post-filtered only where realizable;
C. + closure and consistency checks: closure health of each LES (enstrophy growth, spectral tail, Germano identity)
   as prediction checks with escalation, and the residual of the filtered equations with the subfilter term removed;
D. + scale-linking models: SC-FNO and LES-learned.

Everything here is configuration: the solvers, filters, closures, checks, routing, data generation, models,
training loops, metrics and figures are in :mod:`pinneapple_physics.multiscale`. ``python multiscale_turbulence.py``
runs all stages with caching in ``runs/multiscale_ensemble`` (``--stage data`` / ``train`` for one); a full run
takes a few hours on a 4-core CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pinneapple_physics.multiscale import (
    DynamicSmagorinsky,
    Filter,
    GradientModel,
    ScaleProfile,
    Smagorinsky,
    VorticityNS2D,
    closure_health,
    cross_scale_consistency,
    prolong,
    restrict,
)
from pinneapple_physics.multiscale.comparison import (
    DEFAULT_CONFIGS,
    EnsembleConfig,
    closure_flag_table,
    evaluate_experts,
    frontier_figure,
    markdown_tables,
    run_ensembles,
    summarise_runs,
    summary_figure,
)
from pinneapple_physics.multiscale.filters import fourier_coefficients, low_modes
from pinneapple_physics.multiscale.turbulence import (
    DecayingTurbulence2D,
    benchmark_closures,
    scale_conditioning_study,
)

BENCH = DecayingTurbulence2D()
H, L, N_OUT = BENCH.h, BENCH.length, BENCH.n_out
OUT = Path(os.environ.get("MULTISCALE_RUNS", "runs/multiscale_ensemble"))
SEEDS, N_QUERIES = (0, 1, 2), 60
SCALE = 5.0                                   # vorticity scale for the networks
SURROGATES = {                                # name: (Re band, filter of its output, grid)
    "FNO": ("low", Filter("gaussian", 2 * H), 64),
    "CNN": ("high", Filter("gaussian", 4 * H), 32),
    "DeepONet": ("mid", Filter("sharp", 8 * H), 64),
    "GNN": ("mid", Filter("box", 4 * H), 24),
    "PINN": ("low", Filter("sharp", 8 * H), 64),
}
SC_WIDTHS = (2, 4, 8)                         # trained widths of the scale-linking FNOs (6h is held out)
BASE = ["FNO", "CNN", "DeepONet", "GNN", "PINN", "LES-Smagorinsky", "LES-dynamic", "LES-gradient", "DNS-128"]
LINK = ["SC-FNO", "LES-learned"]
TOLERANCE = 0.25                              # accepted relative error before the ladder escalates
CONFIGS = dict(zip(DEFAULT_CONFIGS, (
    EnsembleConfig(aware=False, router=False, checks=False, experts=tuple(BASE)),
    EnsembleConfig(aware=True, router=True, checks=False, experts=tuple(BASE), error_tolerance=TOLERANCE),
    EnsembleConfig(aware=True, router=True, checks=True, experts=tuple(BASE), error_tolerance=TOLERANCE),
    EnsembleConfig(aware=True, router=True, checks=True, experts=tuple(BASE + LINK), error_tolerance=TOLERANCE),
), strict=True))


def re_feature(nu):
    return np.log(1.0 / np.asarray(nu, float) / 1000.0) / np.log(4.0)


def profiles() -> dict[str, ScaleProfile]:
    les = BENCH.les_filter
    mid_high = (BENCH.re_bands["mid"][0], BENCH.re_bands["high"][1])
    p = [ScaleProfile(n, f, L / g, re_range=BENCH.re_bands[b], fidelity=0) for n, (b, f, g) in SURROGATES.items()]
    p += [ScaleProfile(n, les, L / BENCH.les_n, closure=c, fidelity=1) for n, c in (
        ("LES-Smagorinsky", "Smagorinsky"), ("LES-dynamic", "dynamic Smagorinsky"), ("LES-gradient", "gradient"))]
    p += [ScaleProfile("LES-learned", les, L / BENCH.les_n, closure="learned", re_range=mid_high, fidelity=1),
          ScaleProfile("DNS-128", Filter("sharp", 0.0), L / 128, re_range=(0.0, 6000.0), fidelity=2),
          ScaleProfile("SC-FNO", Filter("gaussian", 2 * H), H, re_range=mid_high, fidelity=0)]
    return {x.name: x for x in p}


# ----------------------------------------------------------------------------------------------- models
def build_surrogates(data, scale=1.0):
    """The five families, each trained on its own band and filter; returns {name: predict(w0, nu) -> (2, 64, 64)}."""
    import torch

    from pinneapple_neural.architectures.graphnn.base import GraphBatch
    from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
    from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d
    from pinneapple_physics.multiscale.learned import (
        FourierGalerkinPINN,
        fit_supervised,
        periodic_cnn2d,
        torus_graph,
    )

    torch.set_num_threads(4)
    steps = lambda n: max(20, int(n * scale))  # noqa: E731
    modes = low_modes(4)
    fno = FNO2d(2, 2, width=24, modes1=16, modes2=16, layers=4, use_grid=False)
    cnn = periodic_cnn2d(2, 2, width=32)
    don = DeepONet(2 * len(modes) + 1, 16, 2, hidden=128, modes=96, depth=3)
    gnn = MeshGraphNet(node_in_dim=2, out_dim=2, edge_in_dim=2, hidden_dim=48, n_message_passing=5)
    pinn = FourierGalerkinPINN(kmax=4, scale=SCALE, t_ref=BENCH.t_out[-1], length=L)
    edges, eattr = torus_graph(SURROGATES["GNN"][2])
    x = np.arange(N_OUT) * L / N_OUT
    X, Y = np.meshgrid(x, x, indexing="ij")
    trunk = torch.tensor(np.stack([f(m * A) for m in (1, 2, 3, 4) for A in (X, Y) for f in (np.sin, np.cos)], -1)
                         .reshape(-1, 16), dtype=torch.float32)

    def don_forward(b):
        return don(b, trunk).y.transpose(1, 2).reshape(-1, 2, N_OUT, N_OUT)

    def gnn_forward(X):
        n = SURROGATES["GNN"][2]
        g = GraphBatch(x=X.flatten(2).transpose(1, 2), edge_index=edges, edge_attr=eattr.expand(X.shape[0], -1, -1))
        return gnn(g).y.transpose(1, 2).reshape(X.shape[0], 2, n, n)

    def grid_input(x, nu):
        return torch.tensor(np.stack([x, np.broadcast_to(re_feature(nu)[:, None, None], x.shape)], 1),
                            dtype=torch.float32)

    def branch(x, nu):
        return torch.tensor(np.concatenate([fourier_coefficients(x, modes, L), re_feature(nu)[:, None]], 1),
                            dtype=torch.float32)

    forwards = {"FNO": lambda X: fno(X).y, "CNN": cnn, "GNN": gnn_forward, "DeepONet": don_forward}
    mods = {"FNO": fno, "CNN": cnn, "DeepONet": don, "GNN": gnn}
    path = OUT / "surrogates.pt"
    if path.exists():
        st = torch.load(path)
        for k, m in mods.items():
            m.load_state_dict(st[k])
        pinn.load_state_dict(st["PINN"])
    else:
        plan = {"FNO": (800, 8, 2e-3), "CNN": (800, 8, 2e-3), "DeepONet": (2000, 16, 1e-3), "GNN": (400, 8, 1e-3)}
        for seed, (nm, (n_steps, batch, lr)) in enumerate(plan.items()):
            band, filt, n = SURROGATES[nm]
            tr, _ = BENCH.split(data, band, test_frac=0.0)
            xs, ys = BENCH.filtered_pairs(data, tr, filt, n, SCALE)
            Xin = branch(xs, data["nu"][tr]) if nm == "DeepONet" else grid_input(xs, data["nu"][tr])
            fit_supervised(forwards[nm], mods[nm].parameters(), Xin, torch.tensor(ys, dtype=torch.float32),
                           steps(n_steps), seed, lr, batch, log=nm)
        tr, _ = BENCH.split(data, "low", test_frac=0.0)                # PINN: initial states only, no solutions
        c0 = fourier_coefficients(BENCH.filtered_input(data["w"][tr, 0], SURROGATES["PINN"][1]), modes, L)
        pinn.fit(c0, data["nu"][tr], re_feature(data["nu"][tr]), steps(2000), seed=4)
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({**{k: m.state_dict() for k, m in mods.items()}, "PINN": pinn.state_dict()}, path)
    for m in (*mods.values(), pinn):
        m.eval()

    def make(nm):
        band, filt, n = SURROGATES[nm]

        def predict(w0, nu):
            x = BENCH.filtered_input(w0, filt, n)[None] / SCALE
            nu1 = np.array([nu])
            with torch.no_grad():
                if nm == "PINN":
                    c = pinn.predict(fourier_coefficients(x * SCALE, modes, L), re_feature(nu1), BENCH.t_out)[0]
                    return pinn.field(c, N_OUT)
                inp = branch(x, nu1) if nm == "DeepONet" else grid_input(x, nu1)
                y = forwards[nm](inp)[0].numpy().astype(float) * SCALE
            return prolong(y, N_OUT) if n < N_OUT else y
        return predict

    return {nm: make(nm) for nm in SURROGATES}


def build_scale_linking(data, scale=1.0):
    """Scale-conditioned FNO, scale-blind FNO, fixed-width FNOs (2h, 4h, 8h) on the mid and high bands, and the LES
    closure trained a posteriori."""
    import torch

    from pinneapple_physics.multiscale.learned import (
        LearnedClosure,
        ScaleConditionedFNO2d,
        TorchVorticityLES2D,
        fit_scale_conditioned,
        train_closure_aposteriori,
    )

    torch.set_num_threads(4)
    steps = lambda n: max(20, int(n * scale))  # noqa: E731
    tr, _ = BENCH.split(data, ("mid", "high"))
    kw = dict(in_channels=2, out_channels=2, width=24, modes=16, layers=4, delta_ref=H)
    models = {"SC-FNO": ScaleConditionedFNO2d(conditioned=True, **kw),
              "FNO-blind": ScaleConditionedFNO2d(conditioned=False, **kw),
              **{f"FNO-{d}h": ScaleConditionedFNO2d(conditioned=False, **kw) for d in SC_WIDTHS}}
    closure = LearnedClosure()
    path, cpath = OUT / "scale_linking.pt", OUT / "closure.pt"
    if path.exists():
        st = torch.load(path)
        for k, m in models.items():
            m.load_state_dict(st[k])
    else:
        w = np.asarray(data["w"][tr], float)
        X = torch.tensor(np.stack([restrict(w[:, 0], N_OUT) / SCALE,
                                   np.broadcast_to(re_feature(data["nu"][tr])[:, None, None], (len(tr), N_OUT, N_OUT))],
                                  1), dtype=torch.float32)
        Y = {d * H: torch.tensor(BENCH.filtered_input(w[:, BENCH.out_index()], Filter("gaussian", d * H)) / SCALE,
                                 dtype=torch.float32) for d in SC_WIDTHS}
        for k, m in models.items():
            ys = Y if k in ("SC-FNO", "FNO-blind") else {int(k[4:-1]) * H: Y[int(k[4:-1]) * H]}
            print(f"training {k}", flush=True)
            fit_scale_conditioned(m, X, ys, "gaussian", steps=steps(1200 if len(ys) > 1 else 700), batch=8,
                                  consistency_weight=0.5 if k == "SC-FNO" else 0.0, seed=11, log_every=200)
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({k: m.state_dict() for k, m in models.items()}, path)
    if cpath.exists():
        closure.load_state_dict(torch.load(cpath))
    else:
        les_f = BENCH.les_filter
        sub = tr[:: max(1, len(tr) // 96)]
        ws = np.asarray(data["w"][sub], float)
        idx = [1 + BENCH.train_times.index(t) for t in (0.1, 0.2, 0.3)]
        W0 = torch.tensor(BENCH.filtered_input(ws[:, 0], les_f, BENCH.les_n))
        REF = torch.tensor(BENCH.filtered_input(ws[:, idx], les_f, BENCH.les_n))
        train_closure_aposteriori(closure, TorchVorticityLES2D(BENCH.les_n), W0, torch.tensor(data["nu"][sub]), REF,
                                  1e-2, [10, 20, 30], steps=steps(300), batch=4, lr=1e-3, seed=5, log_every=50)
        torch.save(closure.state_dict(), cpath)
    for m in models.values():
        m.eval()
    closure.eval()

    def predict(name):
        def f(w0, nu, delta):
            x = restrict(np.asarray(w0, float), N_OUT) if np.shape(w0)[-1] > N_OUT else np.asarray(w0, float)
            from pinneapple_physics.multiscale.learned import torch_filter
            X = torch.tensor(np.stack([x / SCALE, np.full_like(x, float(re_feature(nu)))])[None], dtype=torch.float32)
            d = torch.tensor([delta], dtype=torch.float32)
            with torch.no_grad():
                return models[name](torch_filter(X, "gaussian", d), d)[0].numpy().astype(float) * SCALE
        return f

    return {k: predict(k) for k in models}, closure


# ----------------------------------------------------------------------------------------------- experts
def experts_and_checks(surrogates, sc, closure):
    """{name: fn(w0, query)} for every expert, and the label-free checks of those that have one."""
    from pinneapple_physics.multiscale.learned import as_numpy_closure

    les_f, n_les = BENCH.les_filter, BENCH.les_n
    s_les = VorticityNS2D(n_les, 0.0, L)
    closures = {"LES-Smagorinsky": Smagorinsky(), "LES-dynamic": DynamicSmagorinsky(), "LES-gradient": GradientModel(),
                "LES-learned": as_numpy_closure(closure, n_les)}
    experts = {nm: (lambda nm: lambda w0, q: surrogates[nm](w0, q["nu"]))(nm) for nm in SURROGATES}  # noqa: B023

    def les(nm):
        def run(w0, q):
            w0b = BENCH.filtered_input(w0, les_f, n_les)
            wl = VorticityNS2D(n_les, q["nu"], L, closure=closures[nm]).simulate(w0b, list(BENCH.t_out), dt=5e-3)
            return prolong(wl, N_OUT), (wl, w0b)
        return run

    def les_check(nm):
        cl = closures[nm]

        def check(internals, w0, q):
            wl, w0b = internals
            h = closure_health(wl, w0b, n_les // 3, L, closure=cl if hasattr(cl, "stress") else None, solver=s_les)
            return h.score, "; ".join(h.reasons)
        return check

    for nm in closures:
        experts[nm] = les(nm)
    experts["DNS-128"] = lambda w0, q: restrict(VorticityNS2D(128, q["nu"], L).simulate(
        restrict(w0, 128), list(BENCH.t_out), dt=5e-3), N_OUT)

    def sc_width(q):                       # Gaussian queries at their width (2h..8h); others from the finest, 2h
        return float(np.clip(q["filter"].delta, 2 * H, 8 * H)) if q["kind"] == "gaussian" else 2 * H

    def sc_run(w0, q):
        d = sc_width(q)
        y = sc["SC-FNO"](w0, q["nu"], d)
        return y, (y, sc["SC-FNO"](w0, q["nu"], 2 * d), d)

    def sc_check(internals, w0, q):
        y, y2, d = internals
        rep = cross_scale_consistency(y, y2, Filter("gaussian", 2 * d), Filter("gaussian", d), length=L, tol=0.1,
                                      integral_tol={"circulation (mean)": 0.02, "energy": 0.1, "enstrophy": 0.1})
        return rep.score, "; ".join(rep.reasons)

    experts["SC-FNO"] = sc_run
    checks = {nm: les_check(nm) for nm in closures}
    checks["SC-FNO"] = sc_check
    return experts, checks


# ----------------------------------------------------------------------------------------------- run
def main(scale=1.0, workers=4, assets="docs/assets/physics_ensemble", stage="all"):
    data = BENCH.generate_training_data(OUT / "train.npz", workers=workers)
    streams = {s: BENCH.query_stream(s, N_QUERIES) for s in SEEDS}
    refs = {s: BENCH.generate_reference(OUT / f"reference_seed{s}.npz", streams[s], workers) for s in SEEDS}
    if stage == "data":
        return None
    surrogates = build_surrogates(data, scale)
    sc, closure = build_scale_linking(data, scale)
    if stage == "train":
        return None
    from pinneapple_physics.multiscale.learned import as_numpy_closure
    from pinneapple_physics.multiscale.viz import animate_multiscale_ensemble

    _, te = BENCH.split(data, ("mid", "high"))
    sc_rows = scale_conditioning_study(BENCH, data, te, sc["SC-FNO"],
                                       {d * H: (lambda d: lambda w0, nu: sc[f"FNO-{d}h"](w0, nu, d * H))(d)  # noqa: B023
                                        for d in SC_WIDTHS}, sc["FNO-blind"])
    bench_rows = benchmark_closures(BENCH, data, {"no model": None, "Smagorinsky (Cs 0.17)": Smagorinsky(),
                                                  "dynamic Smagorinsky": DynamicSmagorinsky(),
                                                  "gradient (Clark)": GradientModel(),
                                                  "learned (a posteriori)": as_numpy_closure(closure, BENCH.les_n)})
    experts, checks = experts_and_checks(surrogates, sc, closure)
    prof = profiles()
    tables, runs = {}, {}
    for s in SEEDS:
        tables[s] = evaluate_experts(OUT / f"experts_seed{s}.npz", streams[s], refs[s]["w0"], experts, checks)
        out_refs = [BENCH.reference_output(refs[s]["traj"][k], q["filter"]) for k, q in enumerate(streams[s])]
        runs[s] = (streams[s], run_ensembles(streams[s], out_refs, tables[s], prof, CONFIGS, dt=BENCH.dt_pair, h=H,
                                             length=L, cost_weight=0.1))
    rows = summarise_runs(runs, prof, list(CONFIGS), BENCH.widths, solver="DNS-128", target=TOLERANCE)
    les_targets = {s: [BENCH.reference_output(refs[s]["traj"][k], BENCH.les_filter) for k in range(N_QUERIES)]
                   for s in SEEDS}
    flags = closure_flag_table(tables, les_targets, [n for n in BASE + LINK if n.startswith("LES")], BENCH.les_n // 3, L)
    md = markdown_tables(rows, BENCH.widths, flags, sc_rows, bench_rows)
    print(md)
    (OUT / "results.md").write_text(md)
    (OUT / "results.json").write_text(json.dumps(
        {"configs": rows, "closure_flags": flags, "scale_conditioning": sc_rows, "closure_benchmark": bench_rows,
         "choices": {str(s): {c: runs[s][1][c]["active"] for c in CONFIGS} for s in SEEDS}}, indent=1, default=str))
    summary_figure(rows, BENCH.widths, Path(assets) / "multiscale_summary.png", sc_rows)
    qs, res = runs[SEEDS[0]]
    d, a = res[DEFAULT_CONFIGS[-1]], res[DEFAULT_CONFIGS[0]]
    first = [k for k in range(len(qs)) if k == 0 or qs[k]["band"] != qs[k - 1]["band"]]
    animate_multiscale_ensemble(
        [BENCH.reference_output(refs[SEEDS[0]]["traj"][k], q["filter"])[-1] for k, q in enumerate(qs)], d["pred"],
        d["active"], BASE + LINK, d["weights"], [q["width"] for q in qs],
        [f"{q['kind']} filter, width {q['width']}h, Re {q['re']:.0f}" for q in qs],
        [min(q["filter"].cutoff, N_OUT / 2) for q in qs], Path(assets) / "multiscale_ensemble.gif",
        compare_active=a["active"], compare_label="A (scale-blind)", flags=d["failed"],
        regimes=[(k, f"Re {qs[k]['band']}") for k in first], title="Scale-aware ensemble (D)")
    return rows


SWEEP_COST = (0.1, 0.5, 2.0, 5.0)              # A: weight of the cost in the loss
SWEEP_TOL = (0.1, 0.25, 0.5)                   # B-D: accepted error before the ladder escalates


def sweep_configs():
    out = {}
    for cw in SWEEP_COST:
        out[f"A|cost weight {cw:g}"] = EnsembleConfig(False, False, False, tuple(BASE), cost_weight=cw)
    for tol in SWEEP_TOL:
        for name, cfg in list(CONFIGS.items())[1:]:
            out[f"{name.split('.')[0]}|tolerance {tol:g}"] = EnsembleConfig(cfg.aware, cfg.router, cfg.checks,
                                                                            cfg.experts, error_tolerance=tol)
    return out


def sweep(seed: int):
    """Every trade-off setting on one cached stream (run after the main stage); writes sweep_seed<s>.json."""
    qs = BENCH.query_stream(seed, N_QUERIES)
    refs = BENCH.generate_reference(OUT / f"reference_seed{seed}.npz", qs)
    table = evaluate_experts(OUT / f"experts_seed{seed}.npz", qs, refs["w0"], {}, {})
    out_refs = [BENCH.reference_output(refs["traj"][k], q["filter"]) for k, q in enumerate(qs)]
    cfgs = sweep_configs()
    res = run_ensembles(qs, out_refs, table, profiles(), cfgs, dt=BENCH.dt_pair, h=H, length=L, cost_weight=0.1)
    keep = {c: {k: res[c][k] for k in ("active", "error", "spec_err", "expert_errors", "policy_cost")} for c in cfgs}
    keep["_cost"] = res["_cost"]
    (OUT / f"sweep_seed{seed}.json").write_text(json.dumps(keep, default=float))


def sweep_report(assets="docs/assets/physics_ensemble"):
    prof = profiles()
    runs = {}
    for s in SEEDS:
        res = json.loads((OUT / f"sweep_seed{s}.json").read_text())
        runs[s] = (BENCH.query_stream(s, N_QUERIES), res)
    cfgs = list(sweep_configs())
    rows = summarise_runs(runs, prof, cfgs, BENCH.widths, solver="DNS-128", oracle_config=cfgs[-1], target=TOLERANCE)
    md = markdown_tables(rows, BENCH.widths)
    (OUT / "sweep.md").write_text(md)
    print(md)
    pts: dict[str, list] = {}
    for r in rows:
        if "|" in r["config"]:
            group, setting = r["config"].split("|")
            label = {"A": "A. original (scale-blind)"}.get(group, next(c for c in CONFIGS if c.startswith(group)))
            pts.setdefault(label, []).append((r["cost_chosen"], r["error"], setting.split()[-1]))
    base = [(r["cost_chosen"], r["error"], r["config"]) for r in rows if "|" not in r["config"]]
    frontier_figure(pts, Path(assets) / "multiscale_frontier.png", base,
                    title="Error against cost: cost weight (A) or error tolerance (B-D)")
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=1, default=float))
    # the default operating points (A: cost weight 0.1; B-D: tolerance 0.25) as the main table and figure
    main_cfg = {f"A|cost weight {SWEEP_COST[0]:g}": DEFAULT_CONFIGS[0],
                **{f"{c.split('.')[0]}|tolerance {TOLERANCE:g}": c for c in DEFAULT_CONFIGS[1:]}}
    main_runs = {s: (qs, {**{v: res[k] for k, v in main_cfg.items()}, "_cost": res["_cost"]})
                 for s, (qs, res) in runs.items()}
    main_rows = summarise_runs(main_runs, prof, list(DEFAULT_CONFIGS), BENCH.widths, solver="DNS-128",
                               target=TOLERANCE)
    previous = json.loads((OUT / "results.json").read_text())
    md = markdown_tables(main_rows, BENCH.widths, previous["closure_flags"], previous["scale_conditioning"],
                         previous["closure_benchmark"])
    (OUT / "results.md").write_text(md)
    print(md)
    summary_figure(main_rows, BENCH.widths, Path(assets) / "multiscale_summary.png", previous["scale_conditioning"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "train", "all", "sweep", "sweep-report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the training steps")
    a = ap.parse_args()
    if a.stage == "sweep":
        sweep(a.seed)
    elif a.stage == "sweep-report":
        sweep_report()
    else:
        main(a.scale, a.workers, stage=a.stage)
