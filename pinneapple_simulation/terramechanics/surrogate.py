"""PINN surrogate of a wheel-soil contact model: (slip, sinkage) -> (drawbar pull, normal load, soil torque).

    from pinneapple_simulation.terramechanics import WongReeceModel, get_soil
    from pinneapple_simulation.terramechanics.surrogate import train_contact_surrogate
    sur, report = train_contact_surrogate(WongReeceModel(get_soil("GRC-1"), wheel), sinkage_range=(0.002, 0.06))
    report["test"]["Fx"]["rmse_N"], sur.predict(slip, z)           # held-out error, fast evaluation
    sur.save("outputs/viper_contact")                              # TorchScript + JSON (SurrogateContact format)

Training data: the reference model on a grid + Latin hypercube over the slip x sinkage box. A disjoint Latin
hypercube (different seed, never seen in training) is the held-out test set; its errors are the ones to report.

Physics terms (soft, one-sided, evaluated with autograd at random collocation points):

* ``dFz/dz >= 0`` (more sinkage carries more load),
* ``dFx/ds >= 0`` up to ``monotone_slip`` (drawbar pull grows with slip before its peak),
* Mohr-Coulomb traction limit ``Fx <= c A + Fz tan(phi)`` with ``A = b R theta_f``,
* torque coupling ``My >= R Fx``.

Each is first checked on the reference model over the training box; a constraint the reference itself violates there
is dropped (and listed in the report) -- a surrogate must not be pushed away from its own reference.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .wong_reece import WongReeceModel


def _lhs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    d = len(lo)
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return np.asarray(lo) + u * (np.asarray(hi) - np.asarray(lo))


@dataclass
class ContactSurrogate:
    """A trained network with its normalisation and training box."""
    net: object
    x_lo: np.ndarray
    x_hi: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray
    slip_range: tuple
    sinkage_range: tuple
    meta: dict = field(default_factory=dict)

    def predict(self, slip, z) -> np.ndarray:
        import torch
        x = np.stack([np.asarray(slip, float).ravel(), np.asarray(z, float).ravel()], 1)
        xn = 2.0 * (x - self.x_lo) / np.maximum(self.x_hi - self.x_lo, 1e-12) - 1.0
        with torch.no_grad():
            yn = self.net(torch.tensor(xn, dtype=torch.float32)).double().numpy()
        return (yn + 1.0) * (self.y_hi - self.y_lo) / 2.0 + self.y_lo

    def in_range(self, slip, z) -> np.ndarray:
        slip, z = np.asarray(slip, float), np.asarray(z, float)
        return ((slip >= self.slip_range[0]) & (slip <= self.slip_range[1]) & (z >= self.sinkage_range[0])
                & (z <= self.sinkage_range[1]))

    def save(self, stem: str) -> tuple[str, str]:
        """``<stem>.pt`` (TorchScript) and ``<stem>.json``; loadable with
        :meth:`pinneapple_simulation.rover.SurrogateContact.from_torchscript` and :func:`load_surrogate`."""
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        self.net.eval()
        ts = torch.jit.trace(self.net, torch.zeros(1, 2))
        ts.save(stem + ".pt")
        meta = {**self.meta, "inputs": ["slip [-]", "sinkage [m]"], "outputs": ["Fx [N]", "Fz [N]", "My [N m]"],
                "norm_x_lo": self.x_lo.tolist(), "norm_x_hi": self.x_hi.tolist(), "norm_y_lo": self.y_lo.tolist(),
                "norm_y_hi": self.y_hi.tolist(), "slip_range": list(self.slip_range),
                "sinkage_range_m": list(self.sinkage_range)}
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return stem + ".pt", stem + ".json"


def load_surrogate(stem: str) -> ContactSurrogate:
    import torch
    meta = json.load(open(stem + ".json"))
    net = torch.jit.load(stem + ".pt", map_location="cpu").eval()
    return ContactSurrogate(net, np.array(meta["norm_x_lo"]), np.array(meta["norm_x_hi"]), np.array(meta["norm_y_lo"]),
                            np.array(meta["norm_y_hi"]), tuple(meta["slip_range"]), tuple(meta["sinkage_range_m"]),
                            meta)


def contact_net(hidden: int = 96, depth: int = 4, n_fourier: int = 16, scale: float = 2.0, seed: int = 0):
    """Random-Fourier-feature residual MLP with tanh activations, 2 inputs -> 3 outputs (all in [-1, 1])."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("B", torch.randn(n_fourier, 2, generator=g) * scale)
            self.stem = nn.Linear(2 * n_fourier + 2, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
                                         for _ in range(depth)])
            self.head = nn.Linear(hidden, 3)

        def forward(self, x):
            p = x @ self.B.T
            h = torch.tanh(self.stem(torch.cat([x, torch.sin(p), torch.cos(p)], -1)))
            for b in self.blocks:
                h = torch.tanh(h + b(h))
            return self.head(h)

    torch.manual_seed(seed)
    return Net()


def _check_constraints(model: WongReeceModel, slip_range, sinkage_range, monotone_slip) -> dict:
    """Which physics terms the reference model satisfies on the box (fraction of violating grid points)."""
    S, Z = np.meshgrid(np.linspace(*slip_range, 41), np.linspace(*sinkage_range, 41))
    r = model.forces(S, Z)
    R, b = model.wheel.radius, model.wheel.width
    dfz = np.diff(r.Fz, axis=0)
    dfx = np.diff(r.Fx, axis=1)[:, : int(np.searchsorted(S[0], monotone_slip))]
    A = b * R * r.theta_f
    mc = r.Fx - (model.soil.c * A + r.Fz * model.soil.tan_phi)
    tq = R * r.Fx - r.My
    tol = 1e-9
    return {"fz_monotone": float(np.mean(dfz < -tol * np.abs(r.Fz).max())),
            "fx_monotone": float(np.mean(dfx < -tol * np.abs(r.Fx).max())) if dfx.size else 1.0,
            "mohr_coulomb": float(np.mean(mc > tol * np.abs(r.Fz).max())),
            "torque_coupling": float(np.mean(tq > tol * np.abs(r.My).max()))}


def _errors(pred, ref, names=("Fx", "Fz", "My"), units=("N", "N", "N m")) -> dict:
    out = {}
    for j, (nm, u) in enumerate(zip(names, units, strict=True)):
        e = pred[:, j] - ref[:, j]
        span = float(np.ptp(ref[:, j])) or 1.0
        out[nm] = {f"rmse_{u.replace(' ', '')}": float(np.sqrt(np.mean(e ** 2))),
                   f"max_abs_{u.replace(' ', '')}": float(np.max(np.abs(e))),
                   "rmse_over_range": float(np.sqrt(np.mean(e ** 2)) / span),
                   "r2": float(1 - np.sum(e ** 2) / max(np.sum((ref[:, j] - ref[:, j].mean()) ** 2), 1e-30))}
    return out


def train_contact_surrogate(model: WongReeceModel, slip_range=(0.0, 0.8), sinkage_range=(0.002, 0.06), *,
                            n_grid: int = 40, n_lhs: int = 2000, n_test: int = 2000, epochs: int = 3000,
                            lr: float = 2e-3, batch: int = 1024, w_phys: float = 0.1, monotone_slip: float = 0.3,
                            seed: int = 0, hidden: int = 96, depth: int = 4, log_every: int = 0,
                            progress=None) -> tuple[ContactSurrogate, dict]:
    """Train a surrogate of ``model`` (precompaction and slip angle zero) on the box; returns it and a report with
    the held-out test errors, the physics terms used / dropped and the training history."""
    import torch
    lo, hi = np.array([slip_range[0], sinkage_range[0]]), np.array([slip_range[1], sinkage_range[1]])
    S, Z = np.meshgrid(np.linspace(*slip_range, n_grid), np.linspace(*sinkage_range, n_grid))
    X = np.concatenate([np.stack([S.ravel(), Z.ravel()], 1), _lhs(n_lhs, lo, hi, seed + 1)])
    Xt = _lhs(n_test, lo, hi, seed + 1000)
    ref = lambda A: model.forces(A[:, 0], A[:, 1]).as_array()  # noqa: E731
    Y, Yt = ref(X), ref(Xt)
    ylo, yhi = Y.min(0), Y.max(0)
    nx = lambda A: 2 * (A - lo) / (hi - lo) - 1  # noqa: E731
    ny = lambda A: 2 * (A - ylo) / np.maximum(yhi - ylo, 1e-12) - 1  # noqa: E731
    checks = _check_constraints(model, slip_range, sinkage_range, monotone_slip)
    use = {k: v == 0.0 for k, v in checks.items()}
    net = contact_net(hidden, depth, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
    Xn = torch.tensor(nx(X), dtype=torch.float32)
    Yn = torch.tensor(ny(Y), dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    tlo, thi = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
    ylo_t, yhi_t = torch.tensor(ylo, dtype=torch.float32), torch.tensor(yhi, dtype=torch.float32)
    R, b, c, tphi = model.wheel.radius, model.wheel.width, model.soil.c, model.soil.tan_phi
    hist = []
    for ep in range(1, epochs + 1):
        idx = torch.randint(0, len(Xn), (min(batch, len(Xn)),), generator=g)
        loss_d = torch.mean((net(Xn[idx]) - Yn[idx]) ** 2)
        loss_p = torch.zeros(())
        if w_phys > 0 and any(use.values()):
            xp = (torch.rand(256, 2, generator=g) * (thi - tlo) + tlo).requires_grad_(True)
            yp = (net(2 * (xp - tlo) / (thi - tlo) - 1) + 1) * (yhi_t - ylo_t) / 2 + ylo_t
            scale = (yhi_t - ylo_t)
            if use["fz_monotone"] or use["fx_monotone"]:
                gfz = torch.autograd.grad(yp[:, 1].sum(), xp, create_graph=True)[0]
                gfx = torch.autograd.grad(yp[:, 0].sum(), xp, create_graph=True)[0]
                if use["fz_monotone"]:
                    loss_p = loss_p + torch.mean(torch.relu(-gfz[:, 1] * (thi[1] - tlo[1]) / scale[1]) ** 2)
                if use["fx_monotone"]:
                    m = (xp[:, 0] <= monotone_slip).float()
                    loss_p = loss_p + torch.mean(m * torch.relu(-gfx[:, 0] * (thi[0] - tlo[0]) / scale[0]) ** 2)
            if use["mohr_coulomb"]:
                th = torch.acos(torch.clamp(1 - xp[:, 1] / R, -1, 1))
                lim = c * b * R * th + yp[:, 1].detach() * tphi
                loss_p = loss_p + torch.mean(torch.relu((yp[:, 0] - lim) / scale[0]) ** 2)
            if use["torque_coupling"]:
                loss_p = loss_p + torch.mean(torch.relu((R * yp[:, 0] - yp[:, 2]) / scale[2]) ** 2)
        loss = loss_d + w_phys * loss_p
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 100 == 0 or ep == epochs:
            hist.append((ep, float(loss_d.detach()), float(loss_p.detach())))
            if progress:
                progress(ep, epochs, float(loss_d))
    net.eval()
    sur = ContactSurrogate(net, lo, hi, ylo, yhi, tuple(slip_range), tuple(sinkage_range),
                           {"model": "PINN contact surrogate (pinneapple_simulation.terramechanics.surrogate)",
                            "reference": "Wong-Reece (WongReeceModel)", "soil": model.soil.name,
                            "soil_params": {"c_Pa": model.soil.c, "phi_deg": model.soil.phi_deg, "K_m": model.soil.K,
                                            "k_c": model.soil.k_c, "k_phi": model.soil.k_phi, "n": model.soil.n0},
                            "wheel_params": {"R_m": model.wheel.radius, "b_m": model.wheel.width,
                                             "grousers": model.wheel.grousers.count,
                                             "grouser_height_m": model.wheel.grousers.height}})
    pred_t = sur.predict(Xt[:, 0], Xt[:, 1])
    report = {"n_train": int(len(X)), "n_test": int(len(Xt)), "epochs": epochs,
              "box": {"slip": list(slip_range), "sinkage_m": list(sinkage_range)},
              "physics_terms_used": [k for k, v in use.items() if v],
              "physics_terms_dropped": {k: f"reference violates it at {100 * v:.1f} % of grid points"
                                        for k, v in checks.items() if not use[k]},
              "test": _errors(pred_t, Yt), "train": _errors(sur.predict(X[:, 0], X[:, 1]), Y),
              "history": hist}
    sur.meta["test_errors"] = report["test"]
    return sur, report


class ZooContactSurrogate:
    """A model-zoo wheel-contact model (e.g. ``pp.zoo.load("rover_contact_grc1_viper")``) with the
    :class:`ContactSurrogate` interface: ``predict(slip, z) -> (N, 3)`` [Fx N, Fz N, My N m], ``in_range`` on its
    validity box, and ``meta["test_errors"]`` from its model card (held-out errors against its own reference)."""

    def __init__(self, zoo_model):
        self.zoo = zoo_model
        card = zoo_model.card
        vm = card.validation_metrics or {}
        te = {}
        for k, u in (("Fx", "N"), ("Fz", "N"), ("My", "Nm")):
            if f"test/{k}_rmse_{u}" in vm:
                te[k] = {f"rmse_{u}": float(vm[f"test/{k}_rmse_{u}"]),
                         f"max_abs_{u}": float(vm.get(f"test/{k}_max_abs_{u}", math.nan))}
        self.meta = {"name": f"{card.name} v{card.version}", "test_errors": te, "source": "model zoo",
                     "validity_range": dict(card.validity_range)}

    def predict(self, slip, z) -> np.ndarray:
        X = np.column_stack([np.asarray(slip, float).ravel(), np.asarray(z, float).ravel()])
        p = self.zoo.predict(X, warn=False)
        return np.column_stack([p["Fx_N"], p["Fz_N"], p["My_Nm"]])

    def in_range(self, slip, z) -> np.ndarray:
        X = np.column_stack([np.asarray(slip, float).ravel(), np.asarray(z, float).ravel()])
        return self.zoo.in_range(X)


def compare_on_states(sur: ContactSurrogate, model: WongReeceModel, slip, z) -> dict:
    """Surrogate and reference at given (slip, sinkage) states (e.g. every wheel along a run): both predictions,
    the errors and the share of states outside the training box."""
    slip, z = np.asarray(slip, float).ravel(), np.asarray(z, float).ravel()
    ok = z > 0
    ref = np.zeros((len(slip), 3))
    pred = np.zeros((len(slip), 3))
    ref[ok] = model.forces(slip[ok], z[ok]).as_array()
    pred[ok] = sur.predict(slip[ok], z[ok])
    inside = sur.in_range(slip, z) & ok
    return {"reference": ref, "surrogate": pred, "inside": inside,
            "errors_inside": _errors(pred[inside], ref[inside]) if inside.sum() > 2 else {},
            "fraction_outside": float(1 - inside[ok].mean()) if ok.any() else math.nan}


__all__ = ["ContactSurrogate", "ZooContactSurrogate", "train_contact_surrogate", "load_surrogate", "contact_net",
           "compare_on_states"]
