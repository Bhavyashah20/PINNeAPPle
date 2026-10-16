"""PhysicsNeMo trains, PINNeAPPle says when to trust: FNO and MeshGraphNet on 2-D Darcy, then the trust layer.

    python run.py data                       # shifted test sets (reference solver), ~3 min on 4 cores
    python run.py train --model fno --budget-s 3600
    python run.py train --model meshgraphnet --budget-s 3600
    python run.py evaluate                   # checks, trust cards, results.json, figure, GIF, `pinneapple check` files
    python run.py datapipe                   # the same checks on a batch of PhysicsNeMo's Darcy2D datapipe

Both models are PhysicsNeMo's own (``physicsnemo.models.fno.FNO``, ``physicsnemo.models.meshgraphnet.MeshGraphNet``),
trained on ``darcy_2d_li2021`` (1000 train / 200 test, 128 x 128) with the same wall-clock budget
(``ComputeBudget``). PINNeAPPle trains nothing: it checks every prediction against the physics
(``pinneapple_physics.darcy_checks``) without the reference solution, and gives a deploy decision per model and
regime. Everything here is composition of library calls; outputs go to ``examples/_out/physicsnemo_darcy``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pinneapple as pp
from pinneapple_analysis.cost.equal_budget import ComputeBudget, hardware_record, time_per_query
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    GridFieldPredictor,
    physicsnemo_fno,
    physicsnemo_meshgraphnet,
    train_grid_operator,
)
from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers.operator_datasets import DARCY, darcy_set
from pinneapple_tools.visualization.trust_panels import Panel, Row, trust_panels, trust_panels_gif

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "_out" / "physicsnemo_darcy"
CHECK = HERE / "check"
#: Regimes: the dataset's test split, and two shifts of the input distribution solved with the same reference solver.
SHIFTS = {"contrast": dict(high=20.0, low=1.0, seed=7701, label="contrast a in {1, 20}"),
          "length_scale": dict(tau=9.0, seed=7702, label="3x shorter length scale (tau 9)")}
MODELS = {"fno": dict(build=lambda: physicsnemo_fno(128, latent_channels=32, num_fno_layers=4, num_fno_modes=12,
                                                    decoder_layer_size=64), batch_size=8, lr=1e-3),
          "meshgraphnet": dict(build=lambda: physicsnemo_meshgraphnet(128, processor_size=15, hidden_dim=32),
                               batch_size=2, lr=1e-3)}
LABEL = {"fno": "FNO", "meshgraphnet": "MeshGraphNet"}
N_CAL = 100           # first half of the test split: calibration inputs of the drift monitor (labels not used)


def load_regimes(n_shift: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Test split halves: ``calibration`` (inputs only are used, to calibrate the drift limits) and
    ``in_distribution``; then the shifted sets."""
    test = pp.datasets.load("darcy_2d_li2021", split="test")
    a, u = test["a"].astype(np.float64), test["u"].astype(np.float64)
    reg = {"calibration": {"a": a[:N_CAL], "u": u[:N_CAL]}, "in_distribution": {"a": a[N_CAL:], "u": u[N_CAL:]}}
    for k in SHIFTS:
        with np.load(OUT / f"shift_{k}.npz") as z:
            reg[k] = {"a": z["a"][:n_shift], "u": z["u"][:n_shift]}
    return reg


def cmd_data(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for k, s in SHIFTS.items():
        t0 = time.perf_counter()
        a, u = darcy_set(args.n_shift, s["seed"], tau=s.get("tau"), high=s.get("high"), low=s.get("low"),
                         workers=args.workers)
        np.savez_compressed(OUT / f"shift_{k}.npz", a=a, u=u)
        print(f"{k}: {len(a)} samples in {time.perf_counter() - t0:.0f} s")


def cmd_train(args) -> None:
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    spec = MODELS[args.model]
    model = spec["build"]()
    budget = ComputeBudget(wall_s=args.budget_s)
    t0 = time.perf_counter()
    res = train_grid_operator(model, train["a"], train["u"], budget, batch_size=spec["batch_size"], lr=spec["lr"],
                              threads=args.threads)
    model.save(OUT / "models" / args.model)
    rec = {"model": args.model, "parameters": model.n_parameters(), "config": model.config,
           "budget": budget.to_dict(), "elapsed_s": time.perf_counter() - t0, "hardware": hardware_record(),
           **res.to_dict()}
    rec.pop("history", None)
    (OUT / "models" / f"{args.model}_training.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{args.model}: {res.run.steps} steps, {res.epochs:.2f} epochs, final loss {res.final_loss:.4f}")


def _predictions(models, regimes, recompute: bool) -> dict:
    cache = OUT / "predictions.npz"
    if cache.exists() and not recompute:
        with np.load(cache) as z:
            return {tuple(k.split("__")): z[k] for k in z.files}
    preds = {(r, m): model.predict(data["a"]) for r, data in regimes.items() for m, model in models.items()}
    np.savez_compressed(cache, **{f"{r}__{m}": v for (r, m), v in preds.items()})
    return preds


def cmd_evaluate(args) -> None:
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    envelope = dc.CoefficientEnvelope.fit(train["a"])
    regimes = load_regimes(args.n_shift)
    models = {m: GridFieldPredictor.load(OUT / "models" / m) for m in MODELS}
    preds = _predictions(models, regimes, args.recompute)
    results: dict = {"envelope": envelope.to_dict(), "limits": dc.DEFAULT_TOLERANCES, "regimes": {}, "models": {}}
    calibrated = {}
    for m, model in models.items():
        a0 = regimes["in_distribution"]["a"][:8]
        tq = time_per_query(lambda model=model, a0=a0: model.predict(a0), repeats=3, warmup=1, n_per_call=len(a0))
        train_rec = json.loads((OUT / "models" / f"{m}_training.json").read_text())
        calibrated[m] = dc.calibrate_tolerances(regimes["calibration"]["a"], preds[("calibration", m)], DARCY["f"])
        results["models"][m] = {"parameters": model.n_parameters(), "inference_s_per_sample": tq,
                                "drift_limits": calibrated[m],
                                "training": {k: train_rec[k] for k in ("steps", "epochs", "wall_s", "final_loss",
                                                                        "stopped_by")}}
    for r, data in regimes.items():
        if r == "calibration":
            continue
        out = results["regimes"][r] = {"n": len(data["a"])}
        out["reference_solution"] = {c.name: c.value for c in dc.darcy_checks(data["a"], data["u"], DARCY["f"])}
        for m in models:
            u = preds[(r, m)]
            fixed = dc.darcy_checks(data["a"], u, DARCY["f"], reference=data["u"], envelope=envelope)
            drift = dc.darcy_checks(data["a"], u, DARCY["f"], tolerances=calibrated[m], physics_kind="drift")
            card = dc.trust_report(fixed + drift, model=m, kinds=("physics", "drift", "applicability"))
            verdict = {k: dc.trust_report(fixed + drift, kinds=(k,)).decisao for k in ("physics", "drift",
                                                                                       "applicability")}
            err = fixed[[c.kind for c in fixed].index("reference")].values
            out[m] = {"decision": card.decisao, "verdicts": verdict, "rel_l2_median": float(np.median(err)),
                      "rel_l2_p90": float(np.quantile(err, 0.9)),
                      "checks": [c.to_dict() for c in fixed + drift], "trust_card": card.to_dict()}
            print(f"[{r:15s}] {LABEL[m]:12s} rel L2 median {out[m]['rel_l2_median']:.4f}  decision {card.decisao:8s} "
                  + "  ".join(f"{k} {v}" for k, v in verdict.items()))
            for c in fixed + drift:
                print(f"      {'PASS' if c.passed else 'FAIL'}  [{c.kind}] {c.name}: p90 {c.value:.4g} "
                      f"(limit {c.limit:.4g}), {100 * c.failing_fraction:.0f}% of samples fail")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    _export_for_check(regimes, preds, calibrated)
    _figures(regimes, preds, results)


def _export_for_check(regimes, preds, calibrated) -> None:
    """Two predictions per model and regime as grid results for `pinneapple check`, and its configuration: the
    tolerances of each target are its model's validated limits (``calibrate_tolerances``), the bounds of ``a`` the
    training values, so a prediction fails when the model or its input leaves the validated range."""
    keys = {"energy_bound": "darcy-energy", "flux": "darcy-flux", "bc": "darcy-bc", "positivity": "darcy-positivity"}
    lines = ["# `pinneapple check` on predictions of the two PhysicsNeMo models (written by ../run.py evaluate).",
             "# One predicted field u per file, for one coefficient a on the 128 x 128 cell centres of the unit square.",
             "# physics = \"darcy\": -div(a grad u) = 1, u = 0 on the boundary, checked with the reference solver's",
             "# finite volumes. tolerances: the model's own 99th percentiles on held-out in-distribution inputs (no",
             "# labels); bounds on a: the values seen in training. With the default tolerances every prediction of",
             "# both models fails PC101 and PC301 (see docs/guides/physicsnemo_trust.md).", ""]
    for (r, m), u in preds.items():
        if r == "calibration":
            continue
        tol = ", ".join(f'{keys[k]} = {v:.4g}' for k, v in calibrated[m].items())
        for i in range(2):
            rel = f"predictions/{r}/{m}_{i}.npz"
            dc.save_darcy_npz(CHECK / rel, regimes[r]["a"][i], u[i])
            lines += ["[[result]]", f'name = "{r}/{m}_{i}"', f'path = "{rel}"', 'physics = "darcy"',
                      "parameters = { f = 1.0 }", 'units = { a = "1", u = "1" }', "bounds = { a = [3, 12] }",
                      f"tolerances = {{ {tol} }}", ""]
    (CHECK / "pinneapple-check.toml").write_text("\n".join(lines))


def _figures(regimes, preds, results) -> None:
    def rows(i):
        out = []
        for r, data in regimes.items():
            if r == "calibration":
                continue
            panels = [Panel("reference solver", data["u"][i])]
            for m in MODELS:
                res = results["regimes"][r][m]
                err = np.linalg.norm(preds[(r, m)][i] - data["u"][i]) / np.linalg.norm(data["u"][i])
                v = res["verdicts"]
                panels.append(Panel(f"PhysicsNeMo {LABEL[m]}", preds[(r, m)][i], note=f"error {100 * err:.1f}%",
                                    verdicts=[f"physics: {v['physics']}", f"vs validation: {v['drift']}",
                                              f"inputs: {v['applicability']}"]))
            for m in MODELS:
                panels.append(Panel(f"{LABEL[m]} PDE residual", dc.darcy_residual(preds[(r, m)][i], data["a"][i],
                                                                                    DARCY["f"]), kind="residual"))
            label = {"in_distribution": "test set (as trained)"}.get(r, SHIFTS.get(r, {}).get("label", r))
            out.append(Row(label, panels))
        return out
    title = "PhysicsNeMo trains, PINNeAPPle says when to trust: 2-D Darcy, -div(a grad u) = 1"
    footer = ("Verdicts per model and regime, with no reference solution. physics: residual error bound, u = 0, flux "
              "balance, u >= 0 against fixed limits.\nvs validation: the same checks against the model's own values "
              "on held-out in-distribution inputs. inputs: a inside the training envelope. "
              "Error vs the reference solver shown for reading only.")
    rng = (1e-2, 3e2)
    fig = trust_panels(rows(0), title=title, footer=footer, residual_range=rng)
    fig.savefig(OUT / "physicsnemo_darcy_trust.png", facecolor=fig.get_facecolor())
    trust_panels_gif([rows(i) for i in range(6)], OUT / "physicsnemo_darcy_trust.gif", title=title, footer=footer,
                     residual_range=rng, dpi=80)
    print("figure:", OUT / "physicsnemo_darcy_trust.png", "GIF:", OUT / "physicsnemo_darcy_trust.gif")


def cmd_datapipe(args) -> None:
    from physicsnemo import __version__ as pnemo_version
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
    pipe = Darcy2D(resolution=args.resolution, batch_size=args.batch, device="cpu", max_iterations=args.iterations,
                   nr_multigrids=args.multigrids)
    t0 = time.perf_counter()
    it = iter(pipe)
    ks, us, fouriers = [], [], []
    for _ in range(args.batches):                           # each batch is drawn with a new seed
        batch = next(it)
        ks.append(batch["permeability"][:, 0].numpy().copy())
        us.append(batch["darcy"][:, 0].numpy().copy())
        fouriers.append(pipe.rand_fourier.numpy().copy())   # (4, batch, n_freq, n_freq) amplitudes of this batch
    k, u = np.concatenate(ks), np.concatenate(us)
    # residual of the documented equation; for comparison, of the stencil with grad k . grad u divided by 2 dx
    # (the Jacobi kernel of this version, NVIDIA/physicsnemo#2036)
    report = dc.check_nodal_darcy_batch(k, u, f=1.0, dx=pipe.dx,
                                        alt_cross_denominators={"grad k . grad u over 2 dx": 2 * pipe.dx})
    per_sample = np.concatenate([fo.transpose(1, 0, 2, 3).reshape(args.batch, -1) for fo in fouriers])
    report["fourier_amplitudes"] = {
        "distinct_values_per_sample": [int(np.unique(row).size) for row in per_sample],
        "distinct_samples": len({row.tobytes() for row in per_sample}),
        "values_per_sample": int(per_sample.shape[1])}
    report.update({"physicsnemo_version": pnemo_version, "resolution": args.resolution, "batch": args.batch,
                   "batches": args.batches,
                   "generation_s": time.perf_counter() - t0})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datapipe_check.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n-shift", type=int, default=100)
    d.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--model", choices=list(MODELS), required=True)
    t.add_argument("--budget-s", type=float, default=3600.0)
    t.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("--n-shift", type=int, default=None)
    e.add_argument("--recompute", action="store_true", help="predict again instead of reading predictions.npz")
    p = sub.add_parser("datapipe")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--multigrids", type=int, default=4)
    a = ap.parse_args(argv)
    {"data": cmd_data, "train": cmd_train, "evaluate": cmd_evaluate, "datapipe": cmd_datapipe}[a.cmd](a)


if __name__ == "__main__":
    main()
