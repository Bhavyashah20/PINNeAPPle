"""``pp.zoo.benchmark``: a zoo model inside ``pp.benchmark`` (``pinneapple_analysis.benchmark``).

The problem is the model's own dataset (train / val / held-out test, in the model's input and output columns); the
methods are the shipped model (``zoo_pretrained``: no training, evaluated on the same splits) and the same
architecture trained from scratch on the train split under the benchmark's ``ComputeBudget`` (``zoo_retrain``). The
table therefore answers "what does the pre-trained model buy over training it yourself for N seconds", with
``pp.benchmark``'s gates, determinism check, report and trust card.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets


def _bench():
    try:
        import pinneapple_analysis.benchmark as B
    except ImportError as e:  # pragma: no cover - depends on the installed version
        raise ImportError("pp.zoo.benchmark needs pp.benchmark (pinneapple_analysis.benchmark), which is not in this "
                          "installation of PINNeAPPle; evaluate() and validate() work without it") from e
    return B


def _arrays(m, offline=None) -> dict[str, np.ndarray]:
    """train / val / test arrays of the model's dataset (val carved from train, seed 0, when there is none)."""
    name = m.card.data_lineage["dataset"]
    out = {}
    for split in ("train", "val", "test"):
        try:
            d = datasets.load(name, split=split, offline=offline)
        except KeyError:
            continue
        X, Y = d.xy(m.input_names, m.reference_outputs)
        keep = m.replay_mask(d) if split == "train" else np.ones(len(X), bool)
        out[f"{split}_x"], out[f"{split}_y"] = X[keep], Y[keep]
    if "val_x" not in out:
        idx = np.random.default_rng(0).permutation(len(out["train_x"]))
        nv = max(1, len(idx) // 10)
        out["val_x"], out["val_y"] = out["train_x"][idx[:nv]], out["train_y"][idx[:nv]]
        out["train_x"], out["train_y"] = out["train_x"][idx[nv:]], out["train_y"][idx[nv:]]
    return out


def _make_classes(B):
    Check, Column = B.Check, B.Column

    class ZooProblem(B.Problem):
        kind = "tabular"

        def __init__(self, model=None, **options):
            self.model = model
            self.id = f"zoo_{model.name}" if model is not None else "zoo"
            self.title = f"{model.name} (zoo) -- {model.card.task}" if model is not None else "zoo"
            super().__init__(**options)

        def default_options(self):
            m = self.model
            return {"model": m.name, "version": m.card.version, "dataset": m.card.data_lineage["dataset"],
                    "metrics": list(m.spec.get("guard_metrics", []))}

        @property
        def metrics(self) -> list[str]:
            return [g.split("/", 1)[1] for g in self.options["metrics"]]

        def prepare(self, out: Path, n_workers: int = 1) -> dict[str, Any]:
            np.savez(Path(out) / "data.npz", **_arrays(self.model))
            (Path(out) / "cost.json").write_text(json.dumps({"train_set_core_s": None, "timing": "dataset shipped "
                                                             "with the zoo; generation cost not re-measured"}))
            return {"files": ["data.npz", "cost.json"], "cost": None}

        def artifact_files(self, out):
            return ["data.npz", "cost.json"]

        def verify_solver(self):
            ds = datasets.info(self.options["dataset"])
            return [Check("reference solver verification", "problem", self.id, None,
                          detail=f"documented with the dataset (not re-run here): {ds.verification}")]

        def data_checks(self, out):
            z = np.load(Path(out) / "data.npz")
            fin = all(np.isfinite(z[f"{s}_x"]).all() for s in ("train", "val", "test"))
            rows = {s: {r.tobytes() for r in np.ascontiguousarray(z[f"{s}_x"])} for s in ("train", "val", "test")}
            disjoint = not (rows["test"] & rows["train"]) and not (rows["test"] & rows["val"])
            inside = float(self.model.in_range(z["test_x"]).mean())
            return [Check("inputs finite", "data", self.id, fin),
                    Check("held-out test disjoint from train and val", "data", self.id, disjoint),
                    Check("test inputs inside the model's validity range", "data", self.id, inside == 1.0,
                          value=round(inside, 4), severity="warning")]

        def columns(self):
            n = len(np.load(self._prepared / "data.npz")["test_y"]) if hasattr(self, "_prepared") and \
                (self._prepared / "data.npz").exists() else "?"
            return [Column(k.replace("/", "_"), f"{k.split('/', 1)[1]}, test (n = {n})", "test", k.split("/", 1)[1],
                           kind="integral") for k in self.options["metrics"]]

        def extract(self, method_id, results):
            r = results.get("")
            if not r:
                return {}
            return {s: {k: {"values": [v], "ids": [0]} for k, v in r[s].items()} for s in ("val", "test") if s in r}

        def search_score(self, method_id, result):
            return result["val"][self.metrics[0]]

        def main_metric(self):
            return "test", self.metrics[0]

        def ood_metric(self):
            return None

        def summary_metrics(self):
            return [(k, "integral", ("test", k), None, False) for k in self.metrics]

        def space_zoo_pretrained(self):
            return {"weights": ["shipped"]}

        def space_zoo_retrain(self):
            return {"lr": [1e-3, 3e-3], "batch": [64, 256]}

    def _score(m, X, Y):
        return m.evaluate_xy(X, Y)

    class ZooPretrained(B.Method):
        id, title, per_instance, outputs = "zoo_pretrained", "Zoo model (pre-trained)", False, "integrals"

        def supports(self, problem):
            return isinstance(problem, ZooProblem)

        def space(self, problem):
            return problem.space_zoo_pretrained()

        def train(self, problem, prepared, cfg, seed, budget, final=True, **job):
            from pinneapple_analysis.cost.equal_budget import time_per_query
            z = np.load(Path(prepared) / "data.npz")
            m = problem.model
            out = {"train": {"steps": 0, "wall_s": 0.0, "flops": 0.0, "flops_per_step": 0.0,
                             "stopped_by": "pre-trained", "budget": {}, "history": [], "eval_wall_s": 0.0},
                   "n_params": int(sum(p.numel() for p in m.module.parameters()))}
            for s in (("val", "test") if final else ("val",)):
                out[s] = _score(m, z[f"{s}_x"], z[f"{s}_y"])
            x1 = z["val_x"][:1]
            out["infer_batch1"] = time_per_query(lambda: m._forward(x1), repeats=30, warmup=5)
            return out

        def architecture_checks(self, problem, prepared, cfg):
            return [Check("shipped weights verified (SHA-256)", "architecture", f"{problem.id}/{self.id}", True,
                          detail="checked by the zoo loader against manifest.json")]

    class ZooRetrain(B.Method):
        id, title, per_instance, outputs = "zoo_retrain", "Same architecture, trained from scratch", False, \
            "integrals"

        def supports(self, problem):
            return isinstance(problem, ZooProblem)

        def space(self, problem):
            return problem.space_zoo_retrain()

        def train(self, problem, prepared, cfg, seed, budget, final=True, **job):
            import torch
            from pinneapple_analysis.cost.equal_budget import (
                cosine_lr,
                set_lr,
                time_per_query,
                train_within_budget,
            )

            from ._models import copy_model
            z = np.load(Path(prepared) / "data.npz")
            torch.manual_seed(seed)
            m = copy_model(problem.model)
            m.module = m._build(m.config)                          # fresh initialisation, same architecture
            Xn = torch.as_tensor(m.train_inputs(z["train_x"]), dtype=torch.float32)
            T = torch.as_tensor(m.train_targets(z["train_y"]), dtype=torch.float32)
            mask = torch.isfinite(T)
            T = torch.nan_to_num(T)
            opt = torch.optim.Adam(m.module.parameters(), lr=cfg["lr"])
            rng = np.random.default_rng(seed)
            bs = min(int(cfg["batch"]), len(Xn))
            m.module.train()

            def step(progress):
                set_lr(opt, cosine_lr(progress, cfg["lr"]))
                idx = torch.from_numpy(rng.integers(0, len(Xn), bs))
                o = m.train_outputs(Xn[idx])
                k = o.shape[0] if o.dim() == 3 else 1
                loss = ((o - T[idx]) ** 2 * mask[idx]).sum() / (mask[idx].sum() * k).clamp(min=1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return loss.detach()

            t0 = time.time()
            run = train_within_budget(step, budget)
            m.module.eval()
            out = {"train": run.to_dict(), "n_params": int(sum(p.numel() for p in m.module.parameters())),
                   "train_wall_measured_s": time.time() - t0}
            for s in (("val", "test") if final else ("val",)):
                out[s] = _score(m, z[f"{s}_x"], z[f"{s}_y"])
            x1 = z["val_x"][:1]
            out["infer_batch1"] = time_per_query(lambda: m._forward(x1), repeats=30, warmup=5)
            return out

    return ZooProblem, ZooPretrained, ZooRetrain


def run_benchmark(model, *, budget_s: float, seeds: int, baselines: Sequence[str], out_dir, force: bool, log,
                  **kwargs):
    B = _bench()
    ZooProblem, ZooPretrained, ZooRetrain = _make_classes(B)
    methods = [ZooRetrain()] if "retrain" in baselines else []        # first: the determinism spot check trains
    methods.append(ZooPretrained())
    opts = {"budget": B.ComputeBudget(wall_s=float(budget_s)), "seeds": int(seeds), "n_trials": 2,
            "budget_levels": [1.0], "n_workers": 1}
    opts.update(kwargs)
    return B.run(problems=[ZooProblem(model)], methods=methods, out_dir=out_dir, force=force, log=log, **opts)
