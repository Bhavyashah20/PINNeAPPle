"""Compare ensembles on a stream of multiscale queries: scale-blind against scale-aware selection, with and without
closure / consistency checks and scale-linking models.

* :func:`evaluate_experts`: run every expert once per query (its raw output on the common grid, with the filter it
  carries), time it, and run the label-free checks that need its internals (an LES's closure health on its own
  grid, a scale-conditioned model's cross-scale self-consistency); the table is cached in an ``.npz`` file;
* :func:`adapt_output`: bring a raw output to the requested filter, blindly (apply the requested filter on top of
  whatever the output already carries) or scale-aware (:func:`~.filters.refilter`, only what is realizable);
* :func:`raw_residual` / :func:`closed_multiscale_residual`: label-free residuals of a predicted pair of snapshots,
  at the grid scale or of the filtered equations with the subfilter term removed;
* :func:`run_ensembles`: one :class:`~pinneapple_physics.ensemble.PhysicsEnsemble` per configuration on the same
  stream, replaying the cached outputs;
* :func:`summarise_runs`, :func:`closure_flag_table`, :func:`markdown_tables`, :func:`summary_figure`.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .closures import GradientModel, spectrum_error
from .filters import Filter, apply_filter, refilter
from .ns2d import VorticityNS2D, vorticity_residual

__all__ = ["EnsembleConfig", "DEFAULT_CONFIGS", "evaluate_experts", "adapt_output", "raw_residual",
           "closed_multiscale_residual", "run_ensembles", "summarise_runs", "closure_flag_table", "markdown_tables",
           "summary_figure", "frontier_figure"]


@dataclass(frozen=True)
class EnsembleConfig:
    """``aware``: scale-aware post-filtering (else blind); ``router``: :class:`~.scales.ScaleAwareRouter` with the
    fidelity ladder; ``checks``: prediction checks and the closed multiscale residual (else the grid-scale residual);
    ``experts``: the experts it may use; ``error_tolerance``: escalate up the fidelity ladder when the awake
    experts' last error at the requested width is above it (see ``PhysicsEnsemble(error_tolerance=...)``)."""
    aware: bool
    router: bool
    checks: bool
    experts: tuple[str, ...]
    error_tolerance: float | None = None
    cost_weight: float | None = None          # None: the cost_weight given to run_ensembles


DEFAULT_CONFIGS = ("A. original (scale-blind)", "B. + scale-aware selection", "C. + closure & consistency checks",
                   "D. + scale-linking models")


def evaluate_experts(path: str | Path, queries: Sequence[dict], initial_states: np.ndarray,
                     experts: Mapping[str, Callable[[np.ndarray, dict], Any]],
                     checks: Mapping[str, Callable[[Any, np.ndarray, dict], tuple[float, str]]] | None = None,
                     n_out: int | None = None) -> dict[str, np.ndarray]:
    """Run every expert on every query once. ``experts[name](w0, query)`` returns the output on the common grid, or a
    pair ``(output, internals)`` whose internals are handed to ``checks[name](internals, w0, query) -> (score,
    reason)``. A failure (exception, non-finite values) is stored as NaN. Returns (and caches in ``path``) ``raw``
    (queries, experts, ...), ``secs``, ``check`` (NaN where no check), ``reasons``, ``names``."""
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    names = list(experts)
    checks = dict(checks or {})
    raw, secs = None, np.zeros((len(queries), len(names)))
    check = np.full((len(queries), len(names)), np.nan)
    reasons = np.empty((len(queries), len(names)), dtype=object)
    for k, q in enumerate(queries):
        w0 = np.asarray(initial_states[k], float)
        for j, nm in enumerate(names):
            t0 = time.perf_counter()
            internals = None
            try:
                out = experts[nm](w0, q)
                if isinstance(out, tuple):
                    out, internals = out
                y = np.asarray(out, float)
            except FloatingPointError as e:
                y, internals = None, e
            secs[k, j] = time.perf_counter() - t0
            if raw is None and y is not None:
                raw = np.full((len(queries), len(names), *y.shape), np.nan)
            if y is not None:
                raw[k, j] = y
            if nm in checks:
                if isinstance(internals, Exception):
                    check[k, j], reasons[k, j] = 0.0, f"failed: {internals}"
                else:
                    check[k, j], reasons[k, j] = checks[nm](internals, w0, q)
        print(f"query {k}: " + " ".join(f"{nm} {secs[k, j]:.2f}s" for j, nm in enumerate(names)), flush=True)
    out = {"raw": raw, "secs": secs, "check": check, "reasons": reasons, "names": np.array(names)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def adapt_output(raw: np.ndarray, source: Filter, target: Filter, aware: bool, length: float = 2 * np.pi
                 ) -> np.ndarray:
    """The answer to a query for ``target`` from an output that carries ``source``: blind (``target`` applied on top)
    or scale-aware (:func:`~.filters.refilter`)."""
    if not np.all(np.isfinite(raw)):
        return raw
    return refilter(raw, source, target, length) if aware else apply_filter(raw, target, length)


def _rms(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def raw_residual(pred: np.ndarray, nu: float, dt: float, length: float = 2 * np.pi) -> float:
    """Relative residual of the vorticity equation between the two snapshots of ``pred`` at the grid scale."""
    R, sc = vorticity_residual(pred[0], pred[1], dt, nu, length)
    return _rms(R) / sc


def closed_multiscale_residual(pred: np.ndarray, nu: float, dt: float, filt: Filter, h: float,
                               length: float = 2 * np.pi) -> float:
    """Residual of the filtered vorticity equation at the requested width and at twice it, with the subfilter
    tendency estimated by the gradient model at each width removed; mean of the two relative residuals."""
    s = VorticityNS2D(pred.shape[-1], 0.0, length)
    d0 = max(filt.delta, h)
    out = []
    for d in (d0, 2 * d0):
        f = pred if d == d0 else apply_filter(pred, Filter(filt.kind, d), length)
        g = GradientModel(delta=d)
        sub = 0.5 * (g(f[0], s) + g(f[1], s))
        R, sc = vorticity_residual(f[0], f[1], dt, nu, length, closure_tendency=sub)
        out.append(_rms(R) / sc)
    return float(np.mean(out))


def run_ensembles(queries: Sequence[dict], references: Sequence[np.ndarray], table: Mapping[str, np.ndarray],
                  profiles: Mapping[str, Any], configs: Mapping[str, EnsembleConfig], *, dt: float, h: float,
                  length: float = 2 * np.pi, cost_weight: float = 0.1, lookahead: float = 3.0,
                  check_names: Sequence[str] | None = None, kmax_out: float | None = None) -> dict[str, dict]:
    """Every configuration on the same stream, replaying the outputs of :func:`evaluate_experts`. Queries carry
    ``id``, ``nu``, ``re`` and ``filter``. Costs are the median measured wall times. ``check_names``: experts whose
    stored checks are used as prediction checks (default: all that have one)."""
    from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

    from .scales import ScaleAwareRouter, ScaleQuery

    names_all = [str(x) for x in table["names"]]
    cost = {nm: float(np.median(table["secs"][:, j])) for j, nm in enumerate(names_all)}
    has_check = {nm for j, nm in enumerate(names_all) if np.any(np.isfinite(table["check"][:, j]))}
    check_names = set(check_names) if check_names is not None else has_check
    kmax_out = kmax_out if kmax_out is not None else references[0].shape[-1] / 2
    results = {}
    for cname, cfg in configs.items():
        names = list(cfg.experts)
        idx = {nm: names_all.index(nm) for nm in names}

        def predict(nm, q, cfg=cfg, idx=idx):
            return adapt_output(table["raw"][q["id"], idx[nm]], profiles[nm].filter, q["filter"], cfg.aware, length)

        experts = [from_callable(nm, (lambda nm: lambda q: predict(nm, q))(nm), cost=cost[nm]) for nm in names]  # noqa: B023
        router = ScaleAwareRouter([profiles[nm] for nm in names],
                                  lambda q: ScaleQuery(q["filter"], re=q["re"], kmax=kmax_out)) if cfg.router else None
        checks = None
        if cfg.checks:
            checks = {nm: (lambda j: lambda p, q: (float(table["check"][q["id"], j]),
                                                   str(table["reasons"][q["id"], j] or "")))(idx[nm])
                      for nm in names if nm in check_names}
        if cfg.checks:
            res_fn = lambda p, q: closed_multiscale_residual(p, q["nu"], dt, q["filter"], h, length)  # noqa: E731
        else:
            res_fn = lambda p, q: raw_residual(p, q["nu"], dt, length)  # noqa: E731
        ens = PhysicsEnsemble(experts, mode="select", residual_fn=res_fn, residual_lookahead=lookahead,
                              cost_weight=cost_weight if cfg.cost_weight is None else cfg.cost_weight,
                              router=router, prediction_checks=checks,
                              error_tolerance=cfg.error_tolerance, error_context=lambda q: q["filter"])
        rec: dict[str, list] = {k: [] for k in ("active", "error", "spec_err", "expert_errors", "weights", "awake",
                                                "failed", "escalated", "policy_cost", "pred")}
        for k, q in enumerate(queries):
            p = ens.predict(q)
            info = ens.update(q, references[k])
            aw = p.get("awake", {nm: True for nm in names})
            rec["active"].append(p["active"])
            rec["error"].append(float(info["ensemble_error"]))
            kc = min(q["filter"].cutoff * length / (2 * np.pi), kmax_out - 1)
            rec["spec_err"].append(spectrum_error(p["prediction"][-1], references[k][-1], kc, length))
            rec["expert_errors"].append(dict(zip(names, np.asarray(info["errors"], float).tolist(), strict=True)))
            rec["weights"].append(dict(p["weights"]))
            rec["awake"].append(dict(aw))
            rec["failed"].append(list(p.get("failed_checks", [])))
            rec["escalated"].append(bool(p.get("escalated_by_checks", False)))
            run_set = [nm for nm in names if aw.get(nm, True) or nm in p.get("failed_checks", [])]
            rec["policy_cost"].append(sum(cost[nm] for nm in run_set))
            rec["pred"].append(p["prediction"][-1])
        results[cname] = rec
        print(f"{cname}: mean error {np.nanmean(rec['error']):.4f}", flush=True)
    results["_cost"] = cost
    return results


def summarise_runs(runs: Mapping[Any, tuple[Sequence[dict], Mapping[str, dict]]], profiles: Mapping[str, Any],
                   configs: Sequence[str], widths: Sequence[int], *, wrong_factor: float = 1.5,
                   wrong_margin: float = 0.03, solver: str | None = None,
                   oracle_config: str | None = None, target: float | None = None) -> list[dict[str, Any]]:
    """Per configuration, pooled over the streams in ``runs`` (``{seed: (queries, results)}``): mean error, error
    per requested width (with n), spectrum error, wrong selections (error above ``max(wrong_factor * best,
    best + wrong_margin)``, best = best expert available to the configuration on that query), scale violations
    (chosen model coarser than the request), cost of the chosen model and of the experts the policy ran, share of
    queries sent to ``solver``; with ``target``, the queries whose error is above that accuracy target (the fair count
    for policies that pick a cheaper model on purpose). Then two baselines: always ``solver`` and the oracle (best expert in hindsight, of
    the experts of ``oracle_config``)."""
    rows = []
    for cname in configs:
        acc: dict[str, list] = {k: [] for k in ("err", "spec", "wrong", "viol", "cc", "cp", "solver", "over")}
        byw: dict[int, list] = {d: [] for d in widths}
        for _, (qs, res) in runs.items():
            r, cost = res[cname], res["_cost"]
            for k, q in enumerate(qs):
                fin = [v for v in r["expert_errors"][k].values() if np.isfinite(v)]
                best = min(fin)
                e = r["error"][k]
                acc["err"].append(e)
                byw[q["width"]].append(e)
                acc["spec"].append(r["spec_err"][k])
                acc["wrong"].append(float(e > max(wrong_factor * best, best + wrong_margin)))
                acc["viol"].append(float(profiles[r["active"][k]].resolved_width > q["filter"].delta * (1 + 1e-9)))
                acc["cc"].append(cost[r["active"][k]])
                acc["cp"].append(r["policy_cost"][k])
                acc["solver"].append(float(r["active"][k] == solver))
                acc["over"].append(float(target is not None and e > target))
        rows.append({"config": cname, "n": len(acc["err"]), "error": float(np.mean(acc["err"])),
                     **{f"error_{d}h": float(np.mean(v)) if v else float("nan") for d, v in byw.items()},
                     **{f"n_{d}h": len(v) for d, v in byw.items()},
                     "spectrum_error": float(np.mean(acc["spec"])), "wrong": int(sum(acc["wrong"])),
                     "scale_violations": int(sum(acc["viol"])), "cost_chosen": float(np.mean(acc["cc"])),
                     "cost_policy": float(np.mean(acc["cp"])), "solver_share": float(np.mean(acc["solver"])),
                     **({"over_target": int(sum(acc["over"]))} if target is not None else {})})
    oc = oracle_config or configs[-1]
    base = {f"always {solver}": ([], []), "oracle (best expert in hindsight)": ([], [])}
    for _, (qs, res) in runs.items():
        r, cost = res[oc], res["_cost"]
        for k in range(len(qs)):
            ee = r["expert_errors"][k]
            if solver is not None:
                base[f"always {solver}"][0].append(ee[solver])
                base[f"always {solver}"][1].append(cost[solver])
            b = min((v, nm) for nm, v in ee.items() if np.isfinite(v))
            base["oracle (best expert in hindsight)"][0].append(b[0])
            base["oracle (best expert in hindsight)"][1].append(cost[b[1]])
    for nm, (e, c) in base.items():
        if e:
            rows.append({"config": nm, "n": len(e), "error": float(np.mean(e)), "cost_chosen": float(np.mean(c))})
    return rows


def closure_flag_table(tables: Mapping[Any, Mapping[str, np.ndarray]], targets: Mapping[Any, Sequence[np.ndarray]],
                       names: Sequence[str], kc: float, length: float = 2 * np.pi) -> list[dict[str, Any]]:
    """How the label-free checks flagged each expert in ``names``: count, and the expert's field and spectrum error
    against ``targets`` (``{seed: [reference at the expert's own filter, per query]}``) when flagged and when not,
    with the reasons given."""
    out = []
    for nm in names:
        fl, ef, en, sf, sn, reasons = [], [], [], [], [], {}
        for seed, tab in tables.items():
            j = [str(x) for x in tab["names"]].index(nm)
            for k, tgt in enumerate(targets[seed]):
                flagged = bool(tab["check"][k, j] < 0.5)
                fl.append(flagged)
                y = tab["raw"][k, j]
                ok = np.all(np.isfinite(y))
                e = float(np.linalg.norm(y - tgt) / np.linalg.norm(tgt)) if ok else float("inf")
                se = spectrum_error(y[-1], tgt[-1], kc, length) if ok else float("inf")
                (ef if flagged else en).append(e)
                (sf if flagged else sn).append(se)
                if flagged:
                    for rr in str(tab["reasons"][k, j] or "").split("; "):
                        if rr:
                            key = rr.split(" (")[0].split(" by ")[0].split(" error ")[0].split(" grew")[0]
                            reasons[key] = reasons.get(key, 0) + 1
        m = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
        out.append({"expert": nm, "n": len(fl), "flagged": int(sum(fl)), "field_error_flagged": m(ef),
                    "field_error_not_flagged": m(en), "spectrum_error_flagged": m(sf),
                    "spectrum_error_not_flagged": m(sn), "reasons": reasons})
    return out


def _pct(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.1f} %"


def markdown_tables(rows: Sequence[dict], widths: Sequence[int], flags: Sequence[dict] = (),
                    sc_rows: Sequence[dict] = (), bench_rows: Sequence[dict] = ()) -> str:
    """Markdown of :func:`summarise_runs`, :func:`closure_flag_table`, the scale-conditioning study and the closure
    benchmark."""
    cfg_rows = [r for r in rows if "error_" + f"{widths[0]}h" in r]
    tgt = bool(cfg_rows) and "over_target" in cfg_rows[0]
    head = ("| configuration | error (all) | " + " | ".join(f"{d}h" for d in widths) +
            " | spectrum error | wrong model | " + ("above target | " if tgt else "") +
            "coarser than asked | cost of chosen model (s) | cost of policy (s) | solver share |")
    out = [head, "|---|---|" + "---|" * len(widths) + "---|---|" + ("---|" if tgt else "") + "---|---|---|---|"]
    for r in rows:
        if r in cfg_rows:
            out.append(f"| {r['config']} | {_pct(r['error'])} | " + " | ".join(_pct(r[f'error_{d}h']) for d in widths) +
                       f" | {r['spectrum_error']:.3f} | {r['wrong']} | " + (f"{r['over_target']} | " if tgt else "") +
                       f"{r['scale_violations']} | "
                       f"{r['cost_chosen']:.3f} | {r['cost_policy']:.2f} | {100 * r['solver_share']:.0f} % |")
        else:
            out.append(f"| {r['config']} | {_pct(r['error'])} | " + " | ".join("" for _ in widths) +
                       " | | | " + ("| " if tgt else "") + f"| {r['cost_chosen']:.3f} | | |")
    if cfg_rows:
        out.append(f"| n (queries) | {cfg_rows[0]['n']} | " + " | ".join(str(cfg_rows[0][f'n_{d}h']) for d in widths)
                   + " | | | " + ("| " if tgt else "") + "| | | |")
    if flags:
        out += ["", "| expert | flagged | field error flagged / not | spectrum error flagged / not | reasons |",
                "|---|---|---|---|---|"]
        for f in flags:
            out.append(f"| {f['expert']} | {f['flagged']} of {f['n']} | {_pct(f['field_error_flagged'])} / "
                       f"{_pct(f['field_error_not_flagged'])} | {f['spectrum_error_flagged']:.3f} / "
                       f"{f['spectrum_error_not_flagged']:.3f} | "
                       + ", ".join(f"{k} ({v})" for k, v in f["reasons"].items()) + " |")
    if sc_rows:
        out += ["", "| Gaussian width | n | scale-conditioned | scale-blind | best fixed-width | spectrum error |"
                " consistency (Delta vs 2 Delta) |", "|---|---|---|---|---|---|---|"]
        for r in sc_rows:
            out.append(f"| {r['width']}h | {r['n']} | {_pct(r['conditioned'])} | {_pct(r['blind'])} | "
                       f"{_pct(r['best fixed'])} | {r['spectrum']:.3f} | {_pct(r['consistency'])} |")
    if bench_rows:
        out += ["", "| Re band | closure | stress corr. | tendency corr. | backscatter true / model | Germano error |"
                " spectrum error | field error | dissipation LES / DNS | stable to t = 2 | flagged |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        f2 = lambda v: "n/a" if not np.isfinite(v) else f"{v:.2f}"  # noqa: E731
        for r in bench_rows:
            out.append(f"| {r['band']} (n={r['n']}) | {r['closure']} | {f2(r['stress_corr'])} | {f2(r['tend_corr'])} | "
                       f"{f2(r['backscatter_true'])} / {f2(r['backscatter_model'])} | {f2(r['germano'])} | "
                       f"{r['spec_err']:.3f} | {_pct(r['field_err'])} | {f2(r['diss_ratio'])} | "
                       f"{100 * r['stable']:.0f} % | {100 * r['flagged']:.0f} % |")
    return "\n".join(out)


def summary_figure(rows: Sequence[dict], widths: Sequence[int], path: str | Path, sc_rows: Sequence[dict] = (),
                   colors: Sequence[str] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), unit: str = "h") -> Path:
    """Four panels: error per requested width and configuration; error against cost (with the baselines); wrong
    selections and scale violations; the scale-conditioning study."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = [r for r in rows if f"error_{widths[0]}h" in r]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))
    a = ax[0, 0]
    x = np.arange(len(widths))
    bw = 0.8 / max(len(cfg), 1)
    for i, r in enumerate(cfg):
        a.bar(x + (i - (len(cfg) - 1) / 2) * bw, [100 * r[f"error_{d}h"] for d in widths], bw * 0.9,
              color=colors[i % len(colors)], label=r["config"])
    a.set_xticks(x, [f"{d}{unit}\n(n={cfg[0][f'n_{d}h']})" for d in widths] if cfg else [])
    a.set_ylabel("relative L2 error (%)")
    a.set_title("Error at each requested filter width")
    a.legend(fontsize=7)
    a = ax[0, 1]
    for i, r in enumerate(cfg):
        a.scatter(r["cost_chosen"], 100 * r["error"], s=70, color=colors[i % len(colors)], zorder=3)
        a.annotate(r["config"].split(".")[0], (r["cost_chosen"], 100 * r["error"]), textcoords="offset points",
                   xytext=(6, 4), fontsize=9)
    base: dict[tuple, list[str]] = {}
    for r in rows:
        if r not in cfg:
            base.setdefault((round(r["cost_chosen"], 6), round(r["error"], 6)), []).append(r["config"])
    for (x, y), labs in base.items():                                  # coinciding baselines share one label
        a.scatter(x, 100 * y, s=50, marker="s", color="0.45", zorder=3)
        a.annotate(" = ".join(labs), (x, 100 * y), textcoords="offset points", xytext=(-6, -12), fontsize=7,
                   color="0.3", ha="right")
    a.set_xscale("log")
    a.set_xlabel("wall time of the chosen model per query (s, CPU)")
    a.set_ylabel("mean relative L2 error (%)")
    a.set_title("Error against cost")
    a = ax[1, 0]
    lab = [r["config"].split(".")[0] for r in cfg]
    a.bar(np.arange(len(cfg)) - 0.2, [r["wrong"] for r in cfg], 0.38, color="0.3", label="wrong model")
    a.bar(np.arange(len(cfg)) + 0.2, [r["scale_violations"] for r in cfg], 0.38, color="0.7",
          label="model coarser than the requested scale")
    a.set_xticks(range(len(cfg)), lab)
    a.set_ylabel(f"queries (of {cfg[0]['n']})" if cfg else "queries")
    a.set_title("Wrong selections")
    a.legend(fontsize=8)
    a = ax[1, 1]
    if sc_rows:
        ds = [r["width"] for r in sc_rows]
        for key, c, m in (("conditioned", colors[3 % len(colors)], "o"), ("blind", "0.55", "s"),
                          ("best fixed", "0.2", "^")):
            a.plot(ds, [100 * r[key] for r in sc_rows], marker=m, color=c, lw=2, label=key)
        a.set_xlabel(f"Gaussian filter width ({unit})")
        a.set_ylabel("relative L2 error (%)")
        a.set_title(f"Scale-conditioned FNO on held-out cases (n={sc_rows[0]['n']})")
        a.legend(fontsize=8)
    else:
        a.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def frontier_figure(points: Mapping[str, Sequence[tuple[float, float, str]]], path: str | Path,
                    baselines: Sequence[tuple[float, float, str]] = (),
                    colors: Sequence[str] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
                    title: str = "Error against cost over the trade-off setting") -> Path:
    """One line per configuration through its operating points ``(cost, error, label)`` (e.g. a sweep of the cost
    weight or of the error tolerance); ``baselines`` as grey squares. Lower-left is better."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, a = plt.subplots(figsize=(7.5, 5))
    for i, (name, pts) in enumerate(points.items()):
        pts = sorted(pts)
        c = colors[i % len(colors)]
        a.plot([p[0] for p in pts], [100 * p[1] for p in pts], "-o", color=c, lw=2, ms=6, label=name)
        for x, y, lab in pts:
            a.annotate(lab, (x, 100 * y), textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.3")
    merged: dict[tuple, list[str]] = {}
    for x, y, lab in baselines:
        merged.setdefault((round(x, 6), round(y, 6)), []).append(lab)
    for (x, y), labs in merged.items():
        a.scatter([x], [100 * y], marker="s", s=45, color="0.45", zorder=3)
        a.annotate(" = ".join(labs), (x, 100 * y), textcoords="offset points", xytext=(-6, -12), fontsize=7,
                   color="0.3", ha="right")
    a.set_xscale("log")
    a.set_xlabel("wall time of the chosen model per query (s, CPU)")
    a.set_ylabel("mean relative L2 error (%)")
    a.set_title(title)
    a.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
