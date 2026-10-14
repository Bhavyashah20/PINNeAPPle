"""Paper benchmarks for the self-scaled quasi-Newton optimizers (BFGS / SSBFGS / SSBroyden vs Adam).

Problems are the ones distributed with CrunchOptimizer/PINNs (MIT) for Jnini, Kiyani, Shukla et
al., "Curvature-Aware Optimization for High-Accuracy PINNs", arXiv:2604.05230, and used by
Urban, Stefanou & Pons, J. Comput. Phys. 523 (2025), arXiv:2405.04230. Each problem has an
independent reference solution computed here (no data copied from the upstream repo):

- ``helmholtz``: 2D Helmholtz, u = sin(a1 pi x) sin(a2 pi y), a1=1, a2=4, k=1 (closed form).
- ``pkpd``: stiff PK-PD tumour-growth-inhibition model (Simeoni et al., Cancer Res. 64 (2004)
  1094-1101) with 4 transit compartments, forced by a 2-compartment IV-bolus PK with a closed
  form; parameters as in the upstream script. Reference: scipy Radau, split at the dose time.
- ``burgers``: inviscid Burgers u_t + u u_x = 0, u0 = -sin(pi x) on [-1, 1], t in [0, 1]; a
  shock forms at t = 1/pi and stays at x = 0. Reference: exact entropy solution by characteristics.
- ``sod``: Sod shock tube for 1D Euler (gamma = 1.4), t in [0, 0.2]. Reference: exact Riemann
  solver (Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., ch. 4).

The upstream Stokes case is not included: it is a wedge-corner (Moffatt eddies) problem solved
with Gauss-Newton, not a quasi-Newton benchmark.

These runs use a plain residual-MSE PINN and a reduced, laptop-sized budget. They measure the
optimizers against each other on the same model and budget; they are NOT a reproduction of the
papers' final numbers (which use their own architectures, RAD resampling, shock-capturing
residuals for Burgers/Euler and up to 1e5 iterations).

    python -m benchmarks.ssqn_paper_benchmarks --problems pkpd,burgers --adam-iters 2000 --qn-iters 1500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from pinneapple_neural.trainer.self_scaled_qn import SelfScaledQuasiNewton
from pinneapple_physics.closed_form.burgers import burgers_sine_inviscid_exact
from pinneapple_physics.metrics import relative_l2
from pinneapple_physics.pinn_solver.compiler.autograd_ops import grad
from pinneapple_simulation.numerical_solvers.compressible_fv import sod_exact


def mlp(n_in, n_out, width, depth, seed):
    torch.manual_seed(seed)
    layers, d = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers).double()


# ── Helmholtz ───────────────────────────────────────────────────────────
class Helmholtz:
    a1, a2, k = 1.0, 4.0, 1.0

    def __init__(self, seed, n=2000):
        self.net = mlp(2, 1, 30, 4, seed)
        g = torch.Generator().manual_seed(seed)
        self.xy = (2 * torch.rand(n, 2, generator=g, dtype=torch.float64) - 1)
        self.q = (self.k ** 2 - (self.a1 * math.pi) ** 2 - (self.a2 * math.pi) ** 2) * self.exact(self.xy)

    def exact(self, xy):
        return torch.sin(self.a1 * math.pi * xy[:, :1]) * torch.sin(self.a2 * math.pi * xy[:, 1:2])

    def u(self, xy):
        return (1 - xy[:, :1] ** 2) * (1 - xy[:, 1:2] ** 2) * self.net(xy)

    def loss(self):
        xy = self.xy.clone().requires_grad_(True)
        u = self.u(xy)
        g = grad(u, xy)
        lap = grad(g[:, :1], xy)[:, :1] + grad(g[:, 1:2], xy)[:, 1:2]
        return ((lap + self.k ** 2 * u - self.q) ** 2).mean()

    def error(self):
        s = torch.linspace(-1, 1, 200, dtype=torch.float64)
        X, Y = torch.meshgrid(s, s, indexing="ij")
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        with torch.no_grad():
            return relative_l2(self.u(xy).numpy().ravel(), self.exact(xy).numpy().ravel())


# ── PK-PD (Simeoni tumour growth inhibition + 2-compartment PK) ─────────
class PKPD:
    V1, dose, tdose = 810.0, 3e7, 2.0
    k10, k12, k21 = 0.868 * 24, 0.006 * 24, 0.0838 * 24
    psi, k1, k2, l0, l1 = 20.0, 0.968, 6.29e-4, 0.273, 0.814
    y0 = np.array([1.082, 0.191, 0.36, 0.35])
    T = 17.0

    def __init__(self, seed, n=1001):
        s = self.k10 + self.k12 + self.k21
        disc = math.sqrt(s * s - 4 * self.k10 * self.k21)
        self.alpha, self.beta = (s + disc) / 2, (s - disc) / 2
        self.A = self.dose / self.V1 * (self.k21 - self.alpha) / (self.beta - self.alpha)
        self.B = self.dose / self.V1 * (self.k21 - self.beta) / (self.alpha - self.beta)
        self.t_ref = np.linspace(0, self.T, 1701)
        self.y_ref = self._reference(self.t_ref)
        self.lo, self.hi = self.y_ref.min(0), self.y_ref.max(0)
        self.net = mlp(1, 4, 40, 4, seed)
        self.t = torch.linspace(0, 1, n, dtype=torch.float64).reshape(-1, 1)
        self.c = torch.as_tensor(self.conc(self.t.numpy().ravel() * self.T)).reshape(-1, 1)

    def conc(self, t):
        tau = np.asarray(t) - self.tdose
        return np.where(tau >= 0, self.A * np.exp(-self.alpha * np.clip(tau, 0, None))
                        + self.B * np.exp(-self.beta * np.clip(tau, 0, None)), 0.0)

    def rhs(self, t, y, c=None):
        c = self.conc(t) if c is None else c
        x1, x2, x3, x4 = y
        w = x1 + x2 + x3 + x4
        return [self.l0 * x1 * (1 + (self.l0 / self.l1 * w) ** self.psi) ** (-1 / self.psi) - self.k2 * c * x1,
                self.k2 * c * x1 - self.k1 * x2, self.k1 * (x2 - x3), self.k1 * (x3 - x4)]

    def _reference(self, t):
        from scipy.integrate import solve_ivp
        out = []
        y = self.y0
        for a, b in [(0.0, self.tdose), (self.tdose, self.T)]:
            mask = (t >= a) & (t <= b) if a == 0 else (t > a) & (t <= b)
            sol = solve_ivp(self.rhs, (a, b), y, method="Radau", rtol=1e-11, atol=1e-13,
                            t_eval=t[mask] if mask.any() else None, dense_output=True)
            out.append(sol.y.T[: mask.sum()])
            y = sol.sol(b)
        return np.concatenate(out)

    def x(self, tn):  # physical states from the network (min-max scaled outputs)
        lo, hi = (torch.as_tensor(v).reshape(1, -1) for v in (self.lo, self.hi))
        return lo + (hi - lo) * self.net(tn)

    def loss(self):
        tn = self.t.clone().requires_grad_(True)
        x = self.x(tn)
        dx = torch.cat([grad(x[:, i:i + 1], tn) for i in range(4)], 1) / self.T
        x1, x2, x3, x4 = (x[:, i:i + 1] for i in range(4))
        w = x1 + x2 + x3 + x4
        f = torch.cat([self.l0 * x1 * (1 + (self.l0 / self.l1 * w) ** self.psi) ** (-1 / self.psi) - self.k2 * self.c * x1,
                       self.k2 * self.c * x1 - self.k1 * x2, self.k1 * (x2 - x3), self.k1 * (x3 - x4)], 1)
        scale = torch.as_tensor(self.hi - self.lo).reshape(1, -1)
        r = (dx - f) / scale
        ic = (self.x(torch.zeros(1, 1, dtype=torch.float64)) - torch.as_tensor(self.y0).reshape(1, -1)) / scale
        return (r ** 2).mean() + 10 * (ic ** 2).mean()

    def error(self):
        tn = torch.as_tensor(self.t_ref / self.T).reshape(-1, 1)
        with torch.no_grad():
            return relative_l2(self.x(tn).numpy().ravel(), self.y_ref.ravel())   # all four states pooled


# ── Inviscid Burgers ────────────────────────────────────────────────────
class Burgers:
    def __init__(self, seed, n=4000):
        self.net = mlp(2, 1, 40, 4, seed)
        g = torch.Generator().manual_seed(seed)
        self.tx = torch.cat([torch.rand(n, 1, generator=g), 2 * torch.rand(n, 1, generator=g) - 1], 1).double()
        self.x0 = torch.linspace(-1, 1, 256, dtype=torch.float64).reshape(-1, 1)

    @staticmethod
    def exact(t, x):
        """Entropy solution at one (t, x): :func:`pinneapple_physics.closed_form.burgers.burgers_sine_inviscid_exact`."""
        return float(burgers_sine_inviscid_exact(x, t))

    def u(self, tx):
        t, x = tx[:, :1], tx[:, 1:2]
        return -torch.sin(math.pi * x) + t * (1 - x ** 2) * self.net(tx)  # IC and u(+-1)=0 exactly

    def loss(self):
        tx = self.tx.clone().requires_grad_(True)
        u = self.u(tx)
        g = grad(u, tx)
        return ((g[:, :1] + u * g[:, 1:2]) ** 2).mean()

    def error(self):
        ts, xs = np.linspace(0, 1, 101), np.linspace(-1, 1, 201)
        T, X = np.meshgrid(ts, xs, indexing="ij")
        ref = burgers_sine_inviscid_exact(X, T)
        tx = torch.as_tensor(np.stack([T.ravel(), X.ravel()], 1))
        with torch.no_grad():
            return relative_l2(self.u(tx).numpy().ravel(), ref.ravel())


# ── Sod shock tube ──────────────────────────────────────────────────────
class Sod:
    gamma, T = 1.4, 0.2

    def __init__(self, seed, n=4000):
        self.net = mlp(2, 3, 40, 4, seed)
        g = torch.Generator().manual_seed(seed)
        self.tx = torch.cat([self.T * torch.rand(n, 1, generator=g), torch.rand(n, 1, generator=g)], 1).double()
        x0 = torch.linspace(0, 1, 400, dtype=torch.float64).reshape(-1, 1)
        self.ic_tx = torch.cat([torch.zeros_like(x0), x0], 1)
        left = (x0 < 0.5).double()
        self.ic = torch.cat([left * 1.0 + (1 - left) * 0.125, torch.zeros_like(x0), left * 1.0 + (1 - left) * 0.1], 1)

    def prim(self, tx):
        o = self.net(tx)
        return torch.nn.functional.softplus(o[:, :1]) + 1e-3, o[:, 1:2], torch.nn.functional.softplus(o[:, 2:3]) + 1e-3

    def loss(self):
        tx = self.tx.clone().requires_grad_(True)
        rho, u, p = self.prim(tx)
        E = p / (self.gamma - 1) + 0.5 * rho * u ** 2
        U = [rho, rho * u, E]
        F = [rho * u, rho * u ** 2 + p, u * (E + p)]
        r = sum(((grad(Ui, tx)[:, :1] + grad(Fi, tx)[:, 1:2]) ** 2).mean() for Ui, Fi in zip(U, F, strict=True))
        rho0, u0, p0 = self.prim(self.ic_tx)
        ic = ((torch.cat([rho0, u0, p0], 1) - self.ic) ** 2).mean()
        return r + 10 * ic

    def error(self):
        xs = np.linspace(0, 1, 401)
        ref_rho = sod_exact(xs, self.T, gamma=self.gamma)[0]
        tx = torch.as_tensor(np.stack([np.full_like(xs, self.T), xs], 1))
        with torch.no_grad():
            rho, u, p = self.prim(tx)
        return relative_l2(rho.numpy().ravel(), ref_rho)  # density error at t = T


PROBLEMS = {"helmholtz": Helmholtz, "pkpd": PKPD, "burgers": Burgers, "sod": Sod}


def run(name, variant, args):
    prob = PROBLEMS[name](args.seed)
    t0 = time.time()
    opt = torch.optim.Adam(prob.net.parameters(), lr=1e-3)
    for _ in range(args.adam_iters):
        opt.zero_grad()
        loss = prob.loss()
        loss.backward()
        opt.step()
    rec = {"problem": name, "variant": variant, "adam_iters": args.adam_iters, "error_after_adam": prob.error()}
    if variant == "adam":
        for _ in range(args.qn_iters):
            opt.zero_grad()
            loss = prob.loss()
            loss.backward()
            opt.step()
        rec["iters"] = args.qn_iters
    else:
        qn = SelfScaledQuasiNewton(prob.net.parameters(), variant=variant, max_iter=args.qn_iters,
                                   tolerance_grad=0.0, tolerance_change=0.0)

        def closure():
            qn.zero_grad()
            loss = prob.loss()
            loss.backward()
            return loss

        qn.step(closure)
        rec["iters"] = qn.state["n_iter"]
        taus = qn.state["taus"]
        rec["fraction_tau_below_1"] = sum(t < 1 for t in taus) / max(len(taus), 1)
    rec.update(loss=float(prob.loss().detach()), rel_l2=prob.error(), seconds=round(time.time() - t0, 1),
               n_params=sum(p.numel() for p in prob.net.parameters()))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", default="pkpd,burgers,sod,helmholtz")
    p.add_argument("--variants", default="adam,bfgs,ssbfgs,ssbroyden")
    p.add_argument("--adam-iters", type=int, default=2000)
    p.add_argument("--qn-iters", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "ssqn_paper_benchmarks.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results = json.load(open(out))["results"] if os.path.exists(out) else []
    for name in args.problems.split(","):
        for v in args.variants.split(","):
            r = run(name, v, args)
            print(json.dumps(r), flush=True)
            results = [x for x in results if not (x["problem"] == name and x["variant"] == v)] + [r]
            with open(out, "w") as f:
                json.dump({"budget": "reduced (see module docstring)", "args": vars(args), "results": results}, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
