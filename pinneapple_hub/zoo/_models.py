"""Zoo model adapters: a ready-to-use predictor per shipped model, with its validity range, metrics against the
stored reference, physics checks (feeding a ``TrustReport``) and the hooks fine-tuning needs.

Weights are plain ``state_dict`` tensors loaded with ``torch.load(weights_only=True)`` (no pickled code), the
normalisation and architecture live in ``config.json``.
"""
from __future__ import annotations

import copy
import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_hub.model_card import ModelCard

from . import _artifacts as art


class OutOfRangeWarning(UserWarning):
    """Inputs outside the model's validity range (the prediction is an extrapolation)."""


ArrayLike = np.ndarray | Sequence | float


def _flat_metrics(m: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {f"{s}/{k}": float(v) for s, d in m.items() for k, v in d.items()}


class ZooModel:
    """Base class: a loaded zoo model. Subclasses implement ``_build``, ``_encode``/``_decode`` and the metrics."""
    adapter = ""

    def __init__(self, name: str, card: ModelCard, config: dict[str, Any], state: dict[str, Any], spec: dict):
        self.name, self.card, self.config, self.spec = name, card, config, spec
        self.input_names = [d["name"] for d in card.inputs]
        self.output_names = [d["name"] for d in card.outputs]
        self.module = self._build(config)
        self.module.load_state_dict(state)
        self.module.eval()
        self.finetuned_from: str | None = None

    # ------------------------------------------------------------------ hooks for subclasses
    def _build(self, config):  # pragma: no cover - abstract
        raise NotImplementedError

    def _forward(self, X: np.ndarray) -> dict[str, np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Box bounds per input column (physical units) used for the out-of-range check."""
        vr = self.card.validity_range
        return {k: tuple(vr[k]) for k in self.input_names if k in vr}

    def extra_range_mask(self, X: np.ndarray) -> np.ndarray:
        """Additional validity constraints beyond the box (derived quantities). True = inside."""
        return np.ones(len(X), bool)

    def metrics(self, Y_pred: dict[str, np.ndarray], Y_true: np.ndarray) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def physics_checks(self) -> list[Any]:  # pragma: no cover
        raise NotImplementedError

    def train_targets(self, Y: np.ndarray) -> np.ndarray:  # pragma: no cover - fine-tuning hook
        raise NotImplementedError

    def train_inputs(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def train_outputs(self, Xn) -> Any:  # pragma: no cover
        """Network output for normalised inputs (torch); same layout as ``train_targets``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ inputs
    def as_array(self, inputs: Mapping[str, ArrayLike] | np.ndarray | Sequence) -> np.ndarray:
        """(n, n_inputs) float64 array from a mapping of named columns or an array in ``input_names`` order."""
        if isinstance(inputs, Mapping):
            missing = [k for k in self.input_names if k not in inputs]
            if missing:
                raise KeyError(f"{self.name} needs inputs {self.input_names}; missing {missing}")
            cols = np.broadcast_arrays(*[np.asarray(inputs[k], float) for k in self.input_names])
            return np.stack([c.ravel() for c in cols], 1)
        X = np.atleast_2d(np.asarray(inputs, float))
        if X.shape[1] != len(self.input_names):
            raise ValueError(f"{self.name} expects {len(self.input_names)} input columns {self.input_names}, "
                             f"got shape {X.shape}")
        return X

    def in_range(self, inputs) -> np.ndarray:
        """Boolean mask: True where every input is inside the validity range."""
        X = self.as_array(inputs)
        ok = np.ones(len(X), bool)
        for j, k in enumerate(self.input_names):
            if k in self.bounds():
                lo, hi = self.bounds()[k]
                tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
                ok &= (X[:, j] >= lo - tol) & (X[:, j] <= hi + tol)
        return ok & self.extra_range_mask(X)

    # ------------------------------------------------------------------ prediction
    def predict(self, inputs=None, *, warn: bool = True, **named) -> dict[str, np.ndarray]:
        """Predict in physical units. ``inputs``: a mapping of named columns, an ``(n, n_inputs)`` array, or the
        columns as keyword arguments. Returns ``{output name: array}``; warns (:class:`OutOfRangeWarning`) for rows
        outside the validity range."""
        X = self.as_array(named if inputs is None else inputs)
        if warn:
            bad = ~self.in_range(X)
            if bad.any():
                warnings.warn(f"{self.name}: {int(bad.sum())} of {len(X)} inputs are outside the validity range "
                              f"(see info('{self.name}').validity_range); those predictions are extrapolations",
                              OutOfRangeWarning, stacklevel=2)
        return self._forward(X)

    __call__ = predict

    # ------------------------------------------------------------------ evaluation
    def reference(self, split: str = "test", *, offline: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(inputs, outputs) of the stored reference cases of a split of this model's dataset."""
        from . import datasets
        d = datasets.load(self.card.data_lineage["dataset"], split=split, offline=offline)
        return d.xy(self.input_names, self.reference_outputs)

    reference_outputs: list[str] = []

    def evaluate_xy(self, X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
        return self.metrics(self._forward(self.as_array(X)), np.asarray(Y, float))

    def evaluate_splits(self, splits: Sequence[str] = ("test",), *, offline=None) -> dict[str, float]:
        out = {}
        for s in splits:
            X, Y = self.reference(s, offline=offline)
            out.update({f"{s}/{k}": v for k, v in self.evaluate_xy(X, Y).items()})
        return out

    eval_splits: tuple[str, ...] = ("test",)

    def replay_mask(self, train_split) -> np.ndarray:
        """Rows of the dataset's train split mixed into a fine-tuning run (``finetune(replay=True)``)."""
        return np.ones(train_split.n_rows, bool)

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def save(self, directory: str | Path) -> Path:
        """Write ``weights.pt`` + ``config.json`` + ``model_card.json`` (reloadable with :func:`load_dir`)."""
        import torch
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu().clone() for k, v in self.state_dict().items()}, d / "weights.pt")
        (d / "config.json").write_text(json.dumps({**self.config, "adapter": self.adapter}, indent=1))
        self.card.save(str(d / "model_card.json"))
        return d

    def __repr__(self) -> str:
        return f"<zoo model {self.name} v{self.card.version}: {', '.join(self.input_names)} -> " \
               f"{', '.join(self.output_names)}>"


# ====================================================================== airfoil
class AirfoilMLP(ZooModel):
    adapter = "airfoil_mlp"
    reference_outputs = ["Cl", "Cd", "Cm"]
    eval_splits = ("test", "naca_reference", "xfoil_naca2412")

    def _build(self, config):
        from pinneapple_design.aero.surrogate import MLPEnsemble
        return MLPEnsemble(config["k"], config["width"])

    def _enc(self, X):
        from pinneapple_design.aero.surrogate import MLPEnsemble
        return MLPEnsemble.inputs(X[:, :6], X[:, 6])

    def _forward(self, X):
        import torch
        m, s = np.asarray(self.config["y_mean"]), np.asarray(self.config["y_std"])
        with torch.no_grad():
            y = self.module(torch.from_numpy(self._enc(X))).numpy() * s + m       # (K, n, 3)
        cl, cd, cm = y[..., 0], np.exp(y[..., 1]), y[..., 2]
        return {"Cl": cl.mean(0), "Cd": cd.mean(0), "Cm": cm.mean(0),
                "Cl_std": cl.std(0), "Cd_std": cd.std(0), "Cm_std": cm.std(0)}

    def metrics(self, P, Y):
        out = {}
        for j, k in enumerate(("Cl", "Cd", "Cm")):
            t = Y[:, j]
            ok = np.isfinite(t)
            if not ok.any():
                continue
            e = P[k][ok] - t[ok]
            out[f"{k}_mae"] = float(np.abs(e).mean())
            out[f"{k}_max_abs"] = float(np.abs(e).max())
            if k == "Cd":
                rel = 100 * np.abs(e) / t[ok]
                out["Cd_mape_pct"] = float(rel.mean())
                out["Cd_p90_pct"] = float(np.percentile(rel, 90))
        out["n"] = float(len(Y))
        return out

    def replay_mask(self, train_split):
        # the campaign table records final-iteration values; unsteady runs were averaged (or dropped in deep stall)
        # for training, so only steady runs are replayed
        return np.asarray(train_split["status"]) != "unsteady"

    # fine-tuning in the standardised (Cl, log Cd, Cm) space of every ensemble member
    def train_inputs(self, X):
        return self._enc(X)

    def train_targets(self, Y):
        t = np.stack([Y[:, 0], np.log(Y[:, 1]), Y[:, 2]], 1)
        return (t - np.asarray(self.config["y_mean"])) / np.asarray(self.config["y_std"])

    def train_outputs(self, Xn):
        return self.module(Xn)                                     # (K, n, 3): every member is trained

    def physics_checks(self):
        from pinneapple_analysis.trust.trust_report import checagem_booleana, checagem_erro_relativo
        X, _ = self.reference("test")
        shapes = np.unique(X[:, :6], axis=0)
        a = np.linspace(-2.0, 6.0, 9)
        grid = np.array([[*s, ai] for s in shapes for ai in a])
        cl = self._forward(grid)["Cl"].reshape(len(shapes), len(a))
        slope = np.polyfit(a, cl.T, 1)[0]                          # per degree, linear range
        thin = 2 * math.pi * math.pi / 180                         # thin-airfoil 2 pi per rad
        dev = float(np.max(np.abs(slope / thin - 1)))
        a2 = np.linspace(-2.0, 10.0, 25)
        g2 = np.array([[*s, ai] for s in shapes for ai in a2])
        mono = bool(np.all(np.diff(self._forward(g2)["Cl"].reshape(len(shapes), len(a2)), axis=1) > 0))
        naca0012 = np.array([0.168848, 0.145789, 0.130731, 0.0, 0.0, 0.0])
        sym = self._forward(np.array([[*naca0012, x] for x in (-2.0, 0.0, 2.0)]))["Cl"]
        anti = float(max(abs(sym[1]), abs(sym[0] + sym[2])))
        rng = np.random.default_rng(0)
        b = np.array([self.card.validity_range[k] for k in self.input_names[:6]])
        R = np.c_[b[:, 0] + rng.random((2000, 6)) * (b[:, 1] - b[:, 0]), rng.uniform(-2, 14, 2000)]
        P = self._forward(R)
        positive = bool(np.all(P["Cd"] > 0) and np.all(np.isfinite(P["Cl"])) and np.all(np.isfinite(P["Cm"])))
        return [
            checagem_erro_relativo("lift slope vs thin-airfoil 2 pi/rad (alpha -2..6, held-out shapes)", dev, 0.15,
                                   fonte="thin-airfoil theory; viscous RANS slopes sit ~5-10 % below", critica=True),
            checagem_booleana("Cl increases with alpha below stall (alpha -2..10, held-out shapes)", mono,
                              fonte="attached-flow lift curve", critica=True),
            checagem_erro_relativo("symmetric airfoil: |Cl(0)| and |Cl(2)+Cl(-2)| (NACA 0012)", anti, 0.01,
                                   fonte="symmetry", critica=True),
            checagem_booleana("Cd > 0 and finite outputs on 2000 random in-range inputs", positive,
                              fonte="second law (drag is dissipative)", critica=True),
        ]


# ====================================================================== heat sink (forced)
class HeatsinkForced(ZooModel):
    adapter = "heatsink_forced"
    reference_outputs = ["r_base_k_w", "pressure_drop_pa"]

    def _build(self, config):
        import pinneapple_neural.architectures  # noqa: F401  (registers the architectures)
        from pinneapple_neural.architectures.registry import ModelRegistry
        a = config["arch"]
        return ModelRegistry.build(a["name"], in_dim=config["in_dim"], out_dim=config["out_dim"], width=a["width"],
                                   depth=a["depth"], act=a["act"])

    @staticmethod
    def _features(X):
        from pinneapple_design.thermal.heatsink.surrogate import features
        W, D, tb, N, t, H, k, sw, sd, Ta, V, Q = X.T
        return features("forced", W, D, tb, N, t, H, k, sw, sd, Ta, Q, V)

    def extra_range_mask(self, X):
        vr = self.card.validity_range
        W, D, N, t = X[:, 0], X[:, 1], X[:, 3], X[:, 4]
        gap = (W - N * t) / np.maximum(N - 1, 1)
        fw, fd = X[:, 7] / W, X[:, 8] / D
        inside = lambda v, k: (v >= vr[k][0] - 1e-9) & (v <= vr[k][1] + 1e-9)  # noqa: E731
        return inside(gap, "fin_gap_mm") & inside(fw, "source_frac_w") & inside(fd, "source_frac_d") & (N >= 2)

    def _log(self, X):
        import torch
        c = self.config
        with torch.no_grad():
            z = torch.as_tensor((self._features(X) - np.asarray(c["x_mean"])) / np.asarray(c["x_std"]),
                                dtype=torch.float32)
            y = self.module(z)
            y = (y.y if hasattr(y, "y") else y).numpy()
        return y * np.asarray(c["y_std"]) + np.asarray(c["y_mean"])

    def _forward(self, X):
        y = self._log(X)
        r = np.exp(y[:, 0])
        return {"r_base_k_w": r, "pressure_drop_pa": np.exp(y[:, 1]),
                "r_base_upper95_k_w": r * math.exp(self.config["conformal_log_q"]["95"])}

    def metrics(self, P, Y):
        out = {}
        for j, (k, lab) in enumerate((("r_base_k_w", "R"), ("pressure_drop_pa", "dP"))):
            e = 100 * np.abs(P[k] / Y[:, j] - 1)
            out.update({f"{lab}_mean_abs_pct": float(e.mean()), f"{lab}_p95_abs_pct": float(np.percentile(e, 95)),
                        f"{lab}_max_abs_pct": float(e.max())})
        out["R_upper95_coverage_pct"] = float(100 * np.mean(Y[:, 0] <= P["r_base_upper95_k_w"]))
        out["n"] = float(len(Y))
        return out

    def train_inputs(self, X):
        c = self.config
        return ((self._features(X) - np.asarray(c["x_mean"])) / np.asarray(c["x_std"])).astype(np.float32)

    def train_targets(self, Y):
        t = np.log(np.stack([Y[:, 0], np.maximum(Y[:, 1], 1e-6)], 1))
        return (t - np.asarray(self.config["y_mean"])) / np.asarray(self.config["y_std"])

    def train_outputs(self, Xn):
        y = self.module(Xn)
        return y.y if hasattr(y, "y") else y

    def physics_checks(self):
        from pinneapple_analysis.trust.trust_report import checagem_booleana, checagem_erro_relativo
        X, _ = self.reference("test")
        X = X[self.in_range(X)]

        def frac_sign(col, factor, out, sign):
            X2 = X.copy()
            X2[:, col] *= factor
            ok = self.in_range(X2)
            d = np.log(self._forward(X2[ok])[out]) - np.log(self._forward(X[ok])[out])
            return float(np.mean(np.sign(d) == sign)) if ok.any() else None

        iv, ik, isw = (self.input_names.index(n) for n in ("air_velocity_m_s", "k_w_mk", "source_width_mm"))
        checks = [
            ("more airflow never raises the resistance", frac_sign(iv, 1.05, "r_base_k_w", -1)),
            ("a more conductive material never raises the resistance", frac_sign(ik, 1.05, "r_base_k_w", -1)),
            ("a wider heat source spreads less (lower resistance)", frac_sign(isw, 1.05, "r_base_k_w", -1)),
            ("pressure drop grows with air velocity", frac_sign(iv, 1.05, "pressure_drop_pa", 1)),
        ]
        out = [checagem_erro_relativo(f"monotonic: {n} (share of held-out designs violating, +5 % step)",
                                      None if v is None else 1 - v, 0.01, fonte="physics of convection/conduction",
                                      critica=True)
               for n, v in checks]
        P = self._forward(X)
        out.append(checagem_booleana("R > 0 and dP > 0 on the held-out designs",
                                     bool(np.all(P["r_base_k_w"] > 0) and np.all(P["pressure_drop_pa"] > 0)),
                                     fonte="second law", critica=True))
        return out


# ====================================================================== rover contact
class RoverContact(ZooModel):
    adapter = "rover_contact"
    reference_outputs = ["Fx_N", "Fz_N", "My_Nm"]

    def _build(self, config):
        from pinneapple_simulation.terramechanics.surrogate import contact_net
        a = config["arch"]
        return contact_net(a["hidden"], a["depth"], a["n_fourier"], a["scale"], a["seed"])

    def _norm(self, X):
        lo, hi = np.asarray(self.config["x_lo"]), np.asarray(self.config["x_hi"])
        return (2.0 * (X - lo) / np.maximum(hi - lo, 1e-12) - 1.0).astype(np.float32)

    def _forward(self, X):
        import torch
        lo, hi = np.asarray(self.config["y_lo"]), np.asarray(self.config["y_hi"])
        with torch.no_grad():
            yn = self.module(torch.from_numpy(self._norm(X))).double().numpy()
        y = (yn + 1.0) * (hi - lo) / 2.0 + lo
        return {"Fx_N": y[:, 0], "Fz_N": y[:, 1], "My_Nm": y[:, 2]}

    def metrics(self, P, Y):
        out = {}
        for j, (k, u) in enumerate((("Fx", "N"), ("Fz", "N"), ("My", "Nm"))):
            e = P[f"{k}_{u}"] - Y[:, j]
            out[f"{k}_rmse_{u}"] = float(np.sqrt(np.mean(e ** 2)))
            out[f"{k}_max_abs_{u}"] = float(np.max(np.abs(e)))
            out[f"{k}_r2"] = float(1 - np.sum(e ** 2) / max(np.sum((Y[:, j] - Y[:, j].mean()) ** 2), 1e-30))
        out["n"] = float(len(Y))
        return out

    def train_inputs(self, X):
        return self._norm(X)

    def train_targets(self, Y):
        lo, hi = np.asarray(self.config["y_lo"]), np.asarray(self.config["y_hi"])
        return 2 * (Y - lo) / np.maximum(hi - lo, 1e-12) - 1

    def train_outputs(self, Xn):
        return self.module(Xn)

    def physics_checks(self):
        from pinneapple_analysis.trust.trust_report import checagem_erro_relativo
        c = self.config["physics"]
        s, z = np.meshgrid(np.linspace(-0.4, 0.9, 66), np.linspace(0.002, 0.06, 59))
        P = self._forward(np.c_[s.ravel(), z.ravel()])
        Fx, Fz, My = (P[k].reshape(s.shape) for k in ("Fx_N", "Fz_N", "My_Nm"))
        span = lambda a: float(np.ptp(a)) or 1.0  # noqa: E731
        noise = 2e-3                                               # 0.2 % of the output range: model noise level
        dfz = np.diff(Fz, axis=0) / span(Fz)
        mfx = s[0] <= 0.3
        dfx = np.diff(Fx[:, mfx], axis=1) / span(Fx)
        theta = np.arccos(np.clip(1 - z / c["R_m"], -1, 1))
        mc = (Fx - (c["c_Pa"] * c["b_m"] * c["R_m"] * theta + Fz * math.tan(math.radians(c["phi_deg"])))) / span(Fx)
        tq = (c["R_m"] * Fx - My) / span(My)
        return [
            checagem_erro_relativo("dFz/dz >= 0 (share of grid steps violating beyond 0.2 % of range)",
                                   float(np.mean(dfz < -noise)), 0.01, fonte="pressure-sinkage", critica=True),
            checagem_erro_relativo("dFx/ds >= 0 for slip <= 0.3 (share violating)", float(np.mean(dfx < -noise)),
                                   0.01, fonte="shear mobilisation", critica=True),
            checagem_erro_relativo("Mohr-Coulomb limit Fx <= c A + Fz tan(phi) (share violating)",
                                   float(np.mean(mc > noise)), 0.01, fonte="Mohr-Coulomb", critica=True),
            checagem_erro_relativo("torque coupling My >= R Fx (share violating)", float(np.mean(tq > noise)), 0.01,
                                   fonte="moment balance", critica=True),
        ]


ADAPTERS = {c.adapter: c for c in (AirfoilMLP, HeatsinkForced, RoverContact)}


def load_dir(directory: str | Path) -> ZooModel:
    """Load a model saved with :meth:`ZooModel.save` (e.g. a fine-tuned copy)."""
    import torch
    d = Path(directory)
    config = json.loads((d / "config.json").read_text())
    card = ModelCard.load(str(d / "model_card.json"))
    state = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
    cls = ADAPTERS[config["adapter"]]
    return cls(card.name, card, config, state, {})


def load_model(name: str, *, offline: bool | None = None) -> ZooModel:
    import torch
    spec = art.entry("models", name)
    card = ModelCard.load(str(art.fetch("models", name, "model_card.json", offline=offline)))
    config = json.loads(art.fetch("models", name, "config.json", offline=offline).read_text())
    state = torch.load(art.fetch("models", name, "weights.pt", offline=offline), map_location="cpu",
                       weights_only=True)
    return ADAPTERS[spec["adapter"]](name, card, config, state, spec)


def copy_model(m: ZooModel) -> ZooModel:
    new = copy.copy(m)
    new.module = copy.deepcopy(m.module)
    new.card = ModelCard(**copy.deepcopy(m.card.to_dict()))
    new.config = copy.deepcopy(m.config)
    return new


def fit(m: ZooModel, X: np.ndarray, Y: np.ndarray, *, epochs: int = 300, lr: float = 1e-4, batch: int = 256,
        weight_decay: float = 0.0, seed: int = 0, progress: Callable[[int, float], None] | None = None) -> list[float]:
    """Adam on the model's own normalised targets (NaN targets are masked); returns the loss per epoch."""
    import torch
    torch.manual_seed(seed)
    Xn = torch.as_tensor(m.train_inputs(np.asarray(X, float)), dtype=torch.float32)
    T = torch.as_tensor(m.train_targets(np.asarray(Y, float)), dtype=torch.float32)
    mask = torch.isfinite(T)
    T = torch.nan_to_num(T)
    opt = torch.optim.Adam(m.module.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    hist = []
    m.module.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xn), generator=g)
        tot = 0.0
        for i in range(0, len(Xn), batch):
            idx = perm[i:i + batch]
            out = m.train_outputs(Xn[idx])
            err = (out - T[idx]) ** 2 * mask[idx]
            loss = err.sum() / (mask[idx].sum() * (out.shape[0] if out.dim() == 3 else 1)).clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        hist.append(tot / len(Xn))
        if progress:
            progress(ep, hist[-1])
    m.module.eval()
    return hist
