"""Diagnostics and suggestions: what in a benchmark's setup is likely to hurt its result, with the evidence and a fix.

Every item is measured from the run (the training histories, the error-versus-budget levels, the checks, the data);
nothing is a generic tip. Each has a severity (error / warning / info), the evidence, the suggested change and, where
possible, the exact configuration edit (``apply`` writes a new configuration with them for review).
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfig
from .registry import Check

__all__ = ["Diagnosis", "diagnose", "apply_suggestions"]


@dataclass
class Diagnosis:
    severity: str                # error | warning | info
    topic: str                   # budget | pinn | data | method | solver | protocol
    subject: str
    finding: str
    evidence: str
    suggestion: str
    edit: dict[str, Any] = field(default_factory=dict)   # {"budget.wall_s": [300, 900]} (old, new)

    def to_dict(self):
        return asdict(self)

    def line(self) -> str:
        e = "; ".join(f"`{k}: {v[0]} → {v[1]}`" for k, v in self.edit.items())
        return f"**{self.severity}** [{self.subject}] {self.finding} Evidence: {self.evidence} Suggestion: " \
               f"{self.suggestion}" + (f" Edit: {e}." if e else "")


def _hist(rec) -> list[dict]:
    return [h for h in rec["result"].get("train", {}).get("history", []) if h.get("loss") is not None]


def _budget_key(cfg: BenchmarkConfig, pid: str) -> tuple[str, dict]:
    if pid in cfg.problem_budgets:
        return f"problem_budgets.{pid}", cfg.problem_budgets[pid]
    return "budget", cfg.budget


def diagnose(cfg: BenchmarkConfig, plan, extracted: dict, checks: list[Check], arch_cfgs: dict | None = None,
             loss_terms: dict | None = None) -> list[Diagnosis]:
    """``extracted``: {(problem, method): {level: {seed: {set: {metric: {"values": [...]}}}}}} (report.collect)."""
    D: list[Diagnosis] = []
    ws = plan.ws
    # ------------------------------------------------------------------ protocol
    if cfg.seeds < 3:
        D.append(Diagnosis("warning", "protocol", "all", "Fewer than 3 seeds: the spread is not reliable.",
                           f"seeds = {cfg.seeds}", "Use at least 3 seeds.", {"seeds": [cfg.seeds, 3]}))
    # ------------------------------------------------------------------ budget: still improving at the full budget
    for (pid, mid), lv in extracted.items():
        prob = plan.P[pid]
        main = prob.main_metric_for(mid) if hasattr(prob, "main_metric") else None
        if main is None or len(lv) < 2:
            continue
        s, m = main
        levels = sorted(lv)
        means = []
        for L in levels[-2:]:
            vals = [np.mean(d[s][m]["values"]) for d in lv[L].values() if s in d and m in d[s] and d[s][m]["values"]]
            means.append(float(np.mean(vals)) if vals else np.nan)
        if all(np.isfinite(means)) and means[0] > 0:
            gain = 1 - means[1] / means[0]
            ratio = levels[-1] / levels[-2]
            if gain >= 0.2:
                key, b = _budget_key(cfg, pid)
                edit = {f"{key}.wall_s": [b.get("wall_s"), round(b["wall_s"] * ratio)]} if b.get("wall_s") else {}
                D.append(Diagnosis("warning", "budget", f"{pid}/{mid}",
                                   "Still improving at the full budget: the comparison is budget-limited for this method.",
                                   f"{m} on {s}: {means[0]:.3g} at {levels[-2]:.3g}x -> {means[1]:.3g} at 1x "
                                   f"({100 * gain:.0f} % lower for {ratio:.1f}x the budget)",
                                   f"Train longer (the next {ratio:.1f}x) or report it as budget-limited.", edit))
    # ------------------------------------------------------------------ did not converge / learn
    for c in checks:
        if c.stage == "training" and c.name.startswith("learned") and c.passed is False:
            pid, mid = c.subject.split("/")
            if mid not in plan.M:
                continue
            hint = _pinn_hint(pid, mid, loss_terms or {})
            # a loss that falls by less than half is a warning; a row whose error is no better than a trivial
            # prediction is the error item ("did not converge", below)
            D.append(Diagnosis("warning", "pinn" if plan.M[mid].per_instance else "method", c.subject,
                               "The loss fell by less than half in full-budget runs: the optimisation is slow or "
                               "stalled.", f"{c.value}; {c.detail[:160]}",
                               hint or "Check the learning rate and the loss scaling."))
    for (pid, mid), lv in extracted.items():
        prob = plan.P[pid]
        main = prob.main_metric_for(mid) if hasattr(prob, "main_metric") else None
        if main is None or 1.0 not in lv:
            continue
        s, m = main
        vals = [np.mean(d[s][m]["values"]) for d in lv[1.0].values() if s in d and m in d[s] and d[s][m]["values"]]
        if vals and float(np.mean(vals)) >= 0.9:
            D.append(Diagnosis("error", "pinn" if plan.M[mid].per_instance else "method", f"{pid}/{mid}",
                               "Did not converge: the error is no better than a trivial prediction.",
                               f"mean {m} on {s} = {np.mean(vals):.3g} (>= 0.9)",
                               _pinn_hint(pid, mid, loss_terms or {}) or
                               "Treat this row as a failure of the setup, not as a measured accuracy."))
    # ------------------------------------------------------------------ learning rate: spikes or no progress early
    for p in sorted(ws.runs.glob("final__*__L1.000__s0*.json")):
        rec = json.loads(p.read_text())
        if not rec.get("ok") or rec["spec"]["method"] not in plan.M or rec["spec"]["problem"] not in plan.P:
            continue
        sp = rec["spec"]
        H = _hist(rec)
        if len(H) < 6:
            continue
        L = np.array([h["loss"] for h in H], float)
        prog = np.array([h.get("progress", 0.0) for h in H], float)
        run_min = np.minimum.accumulate(L)
        spikes = int(np.sum(L[len(L) // 2:] > 3 * run_min[len(L) // 2:]))
        lr = sp["cfg"].get("lr")
        if spikes >= 2 and lr:
            D.append(Diagnosis("warning", "method", f"{sp['problem']}/{sp['method']}",
                               "Loss spikes in the second half of training: the learning rate is likely too high.",
                               f"{spikes} logged losses above 3x the running minimum ({sp['id']})",
                               "Halve the learning rate (add it to the method's grid).",
                               {f"method_options.{sp['method']}.lr": [lr, sorted({lr / 2, lr / 4})]}))
        early = L[prog <= 0.1]
        if len(early) >= 2 and early[-1] > 0.9 * early[0] and lr:
            D.append(Diagnosis("warning", "method", f"{sp['problem']}/{sp['method']}",
                               "No progress in the first 10 % of the budget: the learning rate may be too low.",
                               f"loss {early[0]:.3g} -> {early[-1]:.3g} over the first 10 % ({sp['id']})",
                               "Try a 3x larger learning rate.",
                               {f"method_options.{sp['method']}.lr": [lr, sorted({lr, lr * 3})]}))
    # ------------------------------------------------------------------ data and problem setup (from the checks)
    for c in checks:
        if c.passed is not False:
            continue
        if "shared between splits" in c.name or "duplicate" in c.name:
            D.append(Diagnosis("error", "data", c.subject, "Test leakage: inputs repeated across or within splits.",
                               f"{c.value}", "Regenerate the splits with disjoint seeds; never reuse a case."))
        elif "OOD" in c.name or "outside the training envelope" in c.name:
            D.append(Diagnosis("warning", "data", c.subject, "The OOD set is not clearly outside the training data.",
                               f"{c.name}: {c.value} (threshold {c.threshold})",
                               "Move the OOD inputs further out (amplitude, length scale, geometry, angle)."))
        elif "normalisation" in c.name:
            D.append(Diagnosis("error", "data", c.subject, "Normalisation statistics include non-training data.",
                               f"{c.value}", "Compute means and scales on the training split only."))
        elif c.name.startswith("retained Fourier modes"):
            D.append(Diagnosis("error", "method", c.subject, "FNO modes above the Nyquist limit of the grid.",
                               f"modes {c.value} > {c.threshold}", "Reduce the modes in the grid.",
                               {"method_options.fno.modes": [c.value, sorted({c.threshold, max(1, c.threshold // 2)})]}))
        elif c.name.startswith("p >= rank"):
            D.append(Diagnosis("warning", "method", c.subject, "DeepONet basis smaller than the field's POD rank.",
                               f"p = {c.value} < rank(99 % energy) = {c.threshold}",
                               "Raise p to at least the rank.",
                               {"method_options.deeponet.p": [c.value, [int(c.threshold), int(2 * c.threshold)]]}))
        elif "grid convergence" in c.name or "y+" in c.name:
            D.append(Diagnosis("warning", "solver", c.subject, "The reference solution may not be mesh-converged.",
                               f"{c.name}: {c.value}", "Refine the solver grid (or report the GCI as the reference "
                                                       "uncertainty) before comparing methods below it."))
        elif "solver converged" in c.name or "campaign" in c.name:
            D.append(Diagnosis("warning", "solver", c.subject, "Part of the reference data is not converged.",
                               f"{c.value}", "Drop or re-run the unconverged cases, or report them."))
        elif "CFL" in c.name:
            D.append(Diagnosis("error", "solver", c.subject, "Time step above the stability (CFL) limit.",
                               f"{c.value} (limit {c.threshold})", "Reduce the time step."))
    # n = 1 OOD sets
    for pid, prob in plan.P.items():
        for col in prob.columns():
            if "ood" in col.set and "(n = 1)" in col.label:
                D.append(Diagnosis("warning", "data", pid, f"Column '{col.label}' has a single case.",
                                   "n = 1", "Add OOD cases (at least 8) before drawing conclusions from it."))
    # FNO modes vs the output spectrum
    for pid, prob in plan.P.items():
        if "fno" in plan.M and plan.M["fno"].supports(prob) and (ws.prepared(pid) / "data.npz").exists():
            best = plan.best(pid, "fno").best_config
            y = np.load(ws.prepared(pid) / "data.npz")["train_y"][:64].astype(float)
            spec = np.abs(np.fft.rfftn(y - y.mean(0), axes=tuple(range(1, y.ndim)))) ** 2
            k = np.abs(np.fft.fftfreq(y.shape[1], 1 / y.shape[1])) if y.ndim == 3 else np.fft.rfftfreq(y.shape[1], 1 / y.shape[1])
            kk = (np.maximum.outer(k, np.fft.rfftfreq(y.shape[2], 1 / y.shape[2])) if y.ndim == 3 else k)
            beyond = float(spec[:, kk > best["modes"]].sum() / max(spec.sum(), 1e-30))
            if beyond > 0.01:
                D.append(Diagnosis("warning", "method", f"{pid}/fno", "FNO keeps fewer modes than the output needs.",
                                   f"{100 * beyond:.1f} % of the training outputs' energy is above mode {best['modes']}",
                                   "Add larger modes to the grid.",
                                   {"method_options.fno.modes": [best["modes"], [best["modes"], 2 * best["modes"]]]}))
    return D


def _pinn_hint(pid: str, mid: str, loss_terms: dict) -> str:
    t = loss_terms.get((pid, mid))
    if not t:
        return ""
    parts = {k: v for k, v in t.items() if k != "total" and v and v > 0}
    if len(parts) >= 2:
        hi, lo = max(parts.values()), min(parts.values())
        if hi / lo > 100:
            big = max(parts, key=parts.get)
            return (f"Loss terms out of balance at the end ({', '.join(f'{k} {v:.2g}' for k, v in parts.items())}): "
                    f"re-weight '{big}' or use adaptive weights (Wang, Teng & Perdikaris 2021); check the "
                    "nondimensionalisation of inputs and outputs, and the collocation density near walls.")
    return ("Check the nondimensionalisation (inputs and outputs O(1)), the loss weights and the collocation density "
            "near walls and leading edges; consider more budget or a formulation with fewer stiff terms.")


def apply_suggestions(cfg: BenchmarkConfig, diagnoses: list[Diagnosis], path: str | Path) -> BenchmarkConfig:
    """Write a new configuration with every suggested edit applied (for the user to review and run)."""
    d = copy.deepcopy(cfg.to_dict())
    for dg in diagnoses:
        for key, (_, new) in dg.edit.items():
            node = d
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = new
    new = BenchmarkConfig.from_dict(d)
    new.to_yaml(path)
    return new
