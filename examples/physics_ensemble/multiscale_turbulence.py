"""Scale-aware adaptive ensemble on decaying 2-D turbulence: which model can answer a question at the scale it asks?

Test case: 2-D decaying turbulence on the periodic square ``[0, 2 pi)^2`` at three Reynolds numbers (``Re = 1 / nu``
with an rms velocity of 1: about 200, 1000 and 4000). Every case starts from a random field (energy spectrum peaked at
``k = 6``) evolved for 0.5 time units at its own viscosity, so the cascade is developed. A query gives that state and
asks for the vorticity at ``t = 0.28`` and ``0.30`` later, *filtered at a requested width* ``Delta`` (box, Gaussian or
sharp spectral filter, ``Delta / h`` in {1, 2, 4, 6, 8} with ``h = 2 pi / 64``), on a 64^2 grid. The independent
reference is a pseudo-spectral DNS at 256^2 (dt = 2.5e-3) filtered at that width.

Experts (all predict the same 2 x 64 x 64 output; each declares the scales it represents):

==================  ============================  ===========================  ==========================================
expert              scales (filter of its output)  Re range                     how
==================  ============================  ===========================  ==========================================
FNO                 Gaussian 2h                    150-300 (low)                supervised, filtered DNS 128^2
CNN                 Gaussian 4h                    2800-5000 (high)             supervised; periodic dilated 2-D CNN
DeepONet            sharp 8h (k <= 4)              700-1400 (mid)               supervised; branch = low Fourier modes
GNN                 box 4h, on a 24^2 graph        700-1400 (mid)               supervised; MeshGraphNet, multiscale torus
PINN                sharp 8h (k <= 4)              150-300 (low)                physics loss only (Fourier-Galerkin NS)
LES-Smagorinsky     sharp, k <= 16 (48^2 LES)      any                          Cs = 0.17
LES-dynamic         sharp, k <= 16                 any                          dynamic Smagorinsky (Germano-Lilly)
LES-gradient        sharp, k <= 16                 any                          Clark gradient model
DNS-128             resolved (k <= 42)             <= 5000                      pseudo-spectral DNS 128^2, dt = 5e-3
SC-FNO              Gaussian 2h..8h (Delta input)  700-5000                     scale-conditioned FNO + consistency loss
LES-learned         sharp, k <= 16                 700-5000                     CNN closure trained a posteriori
==================  ============================  ===========================  ==========================================

Ensembles compared on the same query streams (3 seeds, 60 queries each, Re regime and requested scale shifting):

A. original, scale-blind: the five families, the LES and the DNS; past errors + current residual (as in
   ``five_model_families_gif.py``); every output filtered with the requested filter, whatever it already was;
B. + scale-aware selection: :class:`~pinneapple_physics.multiscale.ScaleAwareRouter` (declared scales, filter
   realizability, Re range, fidelity ladder surrogate -> LES -> DNS); outputs post-filtered only where realizable;
C. + closure and consistency checks: closure health of each LES (enstrophy growth, spectral tail, Germano identity)
   as prediction checks with escalation, and the multiscale residual with the subfilter term removed;
D. + scale-linking models: SC-FNO and LES-learned.

``python multiscale_turbulence.py`` runs every stage (data, training, closure benchmark, SC-FNO study, ensembles,
figures) with caching in ``runs/multiscale_ensemble``; ``--stage`` runs one. A full run takes a few hours on a
4-core CPU, most of it in the reference DNS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from pinneapple_physics.multiscale import (
    DynamicSmagorinsky,
    Filter,
    GradientModel,
    Smagorinsky,
    VorticityNS2D,
    apply_filter,
    random_vorticity,
    restrict,
)

L = 2 * math.pi
N_OUT = 64
H = L / N_OUT
T_OUT = (0.28, 0.30)
DT_PAIR = T_OUT[1] - T_OUT[0]
SPINUP = 0.5
RE_BANDS = {"low": (150.0, 300.0), "mid": (700.0, 1400.0), "high": (2800.0, 5000.0)}
WIDTHS = (1, 2, 4, 6, 8)                     # requested Delta / h
KINDS = ("gaussian", "sharp", "box")
LES_N = 48
DELTA_LES = math.pi / (LES_N // 3)            # implicit sharp filter of the dealiased 48^2 LES (k <= 16)
OUT = Path(os.environ.get("MULTISCALE_RUNS", "runs/multiscale_ensemble"))


def initial_state(seed: int, nu: float, n: int, dt: float) -> np.ndarray:
    """Developed state: random field at resolution ``n`` evolved for SPINUP time units."""
    w0 = random_vorticity(np.random.default_rng(seed), n, k0=6.0, energy=0.5)
    return VorticityNS2D(n, nu).simulate(w0, [SPINUP], dt=dt)[0]


# ----------------------------------------------------------------------------------------------- data
def _train_case(args):
    seed, nu = args
    os.nice(5)
    w = initial_state(seed, nu, 128, 5e-3)
    traj = VorticityNS2D(128, nu).simulate(w, [0.1, 0.2, *T_OUT], dt=5e-3)
    return np.concatenate([w[None], traj]).astype(np.float32)        # t = 0, 0.1, 0.2, 0.28, 0.30


def _query_case(args):
    seed, nu = args
    os.nice(5)
    w = initial_state(seed, nu, 256, 2.5e-3)
    traj = VorticityNS2D(256, nu).simulate(w, list(T_OUT), dt=2.5e-3)
    return w.astype(np.float32), traj.astype(np.float32)


def generate_training_data(n_per_band: int = 200, workers: int = 4):
    from multiprocessing import Pool

    path = OUT / "train.npz"
    if path.exists():
        return dict(np.load(path))
    rng = np.random.default_rng(12345)
    jobs, band = [], []
    for b, (lo, hi) in RE_BANDS.items():
        for i in range(n_per_band):
            re = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            jobs.append((100000 + len(jobs), 1.0 / re))
            band.append(b)
    t0 = time.time()
    with Pool(workers) as p:
        res = p.map(_train_case, jobs, chunksize=4)
    print(f"training DNS: {len(jobs)} cases in {time.time() - t0:.0f} s", flush=True)
    data = {"w": np.stack(res), "nu": np.array([j[1] for j in jobs]), "band": np.array(band),
            "seed": np.array([j[0] for j in jobs])}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def query_stream(seed: int, n: int = 60) -> list[dict]:
    """Queries of one stream: the Re regime shifts low -> high -> mid (20 queries each); the requested width and
    filter kind are drawn at random for every query. Initial states are new (never in training)."""
    rng = np.random.default_rng(1000 + seed)
    order = ["low", "high", "mid"]
    qs = []
    for k in range(n):
        band = order[min(k * len(order) // n, len(order) - 1)]
        lo, hi = RE_BANDS[band]
        re = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        qs.append({"id": k, "seed": 500000 + 1000 * seed + k, "nu": 1.0 / re, "re": re, "band": band,
                   "width": int(rng.choice(WIDTHS, p=[0.15, 0.2, 0.25, 0.15, 0.25])),
                   "kind": str(rng.choice(KINDS))})
    for q in qs:
        q["filter"] = Filter(q["kind"], q["width"] * H)
    return qs


def generate_reference(seeds=(0, 1, 2), n: int = 60, workers: int = 4):
    from multiprocessing import Pool

    out = {}
    for s in seeds:
        path = OUT / f"reference_seed{s}.npz"
        if path.exists():
            out[s] = dict(np.load(path))
            continue
        qs = query_stream(s, n)
        t0 = time.time()
        with Pool(workers) as p:
            res = p.map(_query_case, [(q["seed"], q["nu"]) for q in qs], chunksize=1)
        print(f"reference DNS 256^2, seed {s}: {n} cases in {time.time() - t0:.0f} s", flush=True)
        out[s] = {"w0": np.stack([r[0] for r in res]), "traj": np.stack([r[1] for r in res])}
        np.savez(path, **out[s])
    return out


def reference_output(traj256: np.ndarray, filt: Filter) -> np.ndarray:
    """Filtered reference on the 64^2 output grid: (2, 64, 64)."""
    return restrict(apply_filter(np.asarray(traj256, float), filt, L), N_OUT)


# ----------------------------------------------------------------------------------------------- surrogates
VORT_SCALE = 5.0
SURROGATES = {                               # name: (band, filter of its output)
    "FNO": ("low", Filter("gaussian", 2 * H)),
    "CNN": ("high", Filter("gaussian", 4 * H)),
    "DeepONet": ("mid", Filter("sharp", 8 * H)),
    "GNN": ("mid", Filter("box", 4 * H)),
    "PINN": ("low", Filter("sharp", 8 * H)),
}


def re_feature(nu):
    return np.log(1.0 / np.asarray(nu) / 1000.0) / np.log(4.0)


def filtered_input(w0: np.ndarray, filt: Filter) -> np.ndarray:
    """Initial state filtered with ``filt`` on the 64^2 grid (from any finer grid)."""
    return restrict(apply_filter(np.asarray(w0, float), filt, L), N_OUT)


def training_pairs(data: dict, band, filt: Filter, n_grid: int = N_OUT):
    """Inputs (B, 2, n, n): filtered initial vorticity and the Re feature; targets (B, 2, n, n): filtered vorticity at
    the two output times. ``band``: one Re band or a tuple of bands."""
    import torch

    bands = (band,) if isinstance(band, str) else tuple(band)
    sel = np.isin(data["band"], bands)
    w = data["w"][sel].astype(float)
    x = restrict(apply_filter(w[:, 0], filt, L), n_grid) / VORT_SCALE
    y = restrict(apply_filter(w[:, 3:5], filt, L), n_grid) / VORT_SCALE
    nu = np.broadcast_to(re_feature(data["nu"][sel])[:, None, None], x.shape)
    X = torch.tensor(np.stack([x, nu], 1), dtype=torch.float32)
    return X, torch.tensor(y, dtype=torch.float32), data["nu"][sel]


def _fit(forward, params, X, Y, steps, seed, lr=2e-3, batch=16, log=""):
    import torch

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = ((forward(X[idx]) - Y[idx]) ** 2).mean() / (Y[idx] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  {log} step {step}: relative mse {float(loss):.4f} ({time.time() - t0:.0f} s)", flush=True)


def _torch():
    import torch
    from torch import nn
    return torch, nn


class PeriodicCNN:
    """Dilated 2-D CNN with circular padding (built lazily so the module imports without torch)."""

    @staticmethod
    def build(width=48):
        torch, nn = _torch()
        layers, c = [], 2
        for d in (1, 2, 4, 8, 4, 2, 1):
            layers += [nn.Conv2d(c, width, 3, padding=d, dilation=d, padding_mode="circular"), nn.GELU()]
            c = width
        layers.append(nn.Conv2d(c, 2, 1))
        return nn.Sequential(*layers)


def torus_graph(n: int, offsets=(1, 2, 4)):
    """Edges of an n x n torus to the neighbours 1, 2 and 4 cells away along each axis, attribute (dx, dy) / 8."""
    torch, _ = _torch()
    idx = np.arange(n * n).reshape(n, n)
    src, dst, attr = [], [], []
    for o in offsets:
        for dx, dy in ((o, 0), (-o, 0), (0, o), (0, -o)):
            src.append(idx.ravel())
            dst.append(np.roll(np.roll(idx, -dx, 0), -dy, 1).ravel())
            attr.append(np.tile([[dx / 8, dy / 8]], (n * n, 1)))
    return (torch.tensor(np.stack([np.concatenate(src), np.concatenate(dst)])),
            torch.tensor(np.concatenate(attr), dtype=torch.float32))


GNN_N = 24
LOW_MODES = [(kx, ky) for kx in range(0, 5) for ky in range(-4, 5) if (kx > 0 or ky > 0)]   # half plane, |k| <= 4


def mode_basis(n: int) -> np.ndarray:
    """(80, n*n) real Fourier basis cos / sin(kx x + ky y) of the modes in LOW_MODES on an n x n grid."""
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rows = []
    for kx, ky in LOW_MODES:
        rows += [np.cos(kx * X + ky * Y).ravel(), np.sin(kx * X + ky * Y).ravel()]
    return np.array(rows)


def low_mode_coefficients(w: np.ndarray) -> np.ndarray:
    """Coefficients of a field (any n >= 9) on the LOW_MODES basis (sharp filter k <= 4)."""
    n = w.shape[-1]
    B = mode_basis(n)
    return np.asarray(w, float).reshape(*w.shape[:-2], -1) @ B.T * (2.0 / (n * n))


def train_surrogates(data: dict, scale: float = 1.0):
    """The five families, each on its own Re band and filter. Returns {name: predict(w0_any_grid, nu) -> (2, 64, 64)}."""
    torch, nn = _torch()
    from pinneapple_neural.architectures.graphnn.base import GraphBatch
    from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
    from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d

    torch.set_num_threads(4)
    steps = lambda n: max(20, int(n * scale))  # noqa: E731
    path = OUT / "surrogates.pt"
    fno = FNO2d(2, 2, width=32, modes1=16, modes2=16, layers=4, use_grid=False)
    cnn = PeriodicCNN.build()
    don = DeepONet(2 * len(LOW_MODES) + 1, 8 * 2 + 0, 2, hidden=128, modes=96, depth=3)
    gnn = MeshGraphNet(node_in_dim=2, out_dim=2, edge_in_dim=2, hidden_dim=48, n_message_passing=5)
    pinn = PINNGalerkin()
    mods = {"FNO": fno, "CNN": cnn, "DeepONet": don, "GNN": gnn, "PINN": pinn}
    edges, eattr = torus_graph(GNN_N)
    xg = np.arange(N_OUT) * L / N_OUT
    XG, YG = np.meshgrid(xg, xg, indexing="ij")
    trunk = torch.tensor(np.stack([f(m * A) for m in (1, 2, 3, 4) for A in (XG, YG) for f in (np.sin, np.cos)], -1)
                         .reshape(-1, 16), dtype=torch.float32)

    def don_forward(X):                      # X: (B, 2*80 + 1) branch features
        return don(X, trunk).y.transpose(1, 2).reshape(-1, 2, N_OUT, N_OUT)

    def gnn_forward(X):                      # X: (B, 2, 32, 32)
        B = X.shape[0]
        g = GraphBatch(x=X.flatten(2).transpose(1, 2), edge_index=edges, edge_attr=eattr.expand(B, -1, -1))
        return gnn(g).y.transpose(1, 2).reshape(B, 2, GNN_N, GNN_N)

    def don_features(x_field, nu_feat):      # x_field (B, n, n) normalised filtered input
        c = low_mode_coefficients(x_field)
        return np.concatenate([c, nu_feat[:, None]], 1)

    if path.exists():
        state = torch.load(path)
        for k, m in mods.items():
            m.load_state_dict(state[k])
    else:
        t0 = time.time()
        X, Y, _ = training_pairs(data, "low", SURROGATES["FNO"][1])
        _fit(lambda x: fno(x).y, fno.parameters(), X, Y, steps(1500), 0, log="FNO")
        X, Y, _ = training_pairs(data, "high", SURROGATES["CNN"][1])
        _fit(cnn, cnn.parameters(), X, Y, steps(1500), 1, log="CNN")
        X, Y, _ = training_pairs(data, "mid", SURROGATES["DeepONet"][1])
        Xb = torch.tensor(don_features(X[:, 0].numpy(), X[:, 1, 0, 0].numpy()), dtype=torch.float32)
        _fit(don_forward, don.parameters(), Xb, Y, steps(4000), 2, lr=1e-3, log="DeepONet")
        X, Y, _ = training_pairs(data, "mid", SURROGATES["GNN"][1], n_grid=GNN_N)
        _fit(gnn_forward, gnn.parameters(), X, Y, steps(800), 3, lr=1e-3, batch=8, log="GNN")
        sel = data["band"] == "low"
        c0 = low_mode_coefficients(data["w"][sel, 0].astype(float))
        pinn.fit(c0, data["nu"][sel], steps(4000), seed=4)
        print(f"surrogates trained in {time.time() - t0:.0f} s", flush=True)
        torch.save({k: m.state_dict() for k, m in mods.items()}, path)
    for m in mods.values():
        m.eval()
    B64 = mode_basis(N_OUT)

    def make(name):
        filt = SURROGATES[name][1]

        def predict(w0, nu):
            with torch.no_grad():
                z = float(re_feature(nu))
                if name == "PINN":
                    c = pinn.predict(low_mode_coefficients(filtered_input(w0, filt))[None], np.array([nu]))[0]
                    return (c @ B64).reshape(2, N_OUT, N_OUT)
                n = GNN_N if name == "GNN" else N_OUT
                x = restrict(apply_filter(np.asarray(w0, float), filt, L), n) / VORT_SCALE
                X = torch.tensor(np.stack([x, np.full_like(x, z)])[None], dtype=torch.float32)
                if name == "FNO":
                    y = fno(X).y
                elif name == "CNN":
                    y = cnn(X)
                elif name == "GNN":
                    y = gnn_forward(X)
                else:
                    y = don_forward(torch.tensor(don_features(x[None], np.array([z])), dtype=torch.float32))
                y = y[0].numpy().astype(float) * VORT_SCALE
                from pinneapple_physics.multiscale import prolong
                return prolong(y, N_OUT) if n < N_OUT else y
        return predict

    return {n: make(n) for n in SURROGATES}


class PINNGalerkin:
    """Physics-informed network for the large scales (|k| <= 4, sharp filter 8h), trained on the physics only: the
    coefficients c(t) of the low Fourier modes are ``c0 + t * net(c0, Re, t)`` (exact at t = 0), and the loss is the
    residual of the Fourier-Galerkin truncated vorticity equation, ``dc/dt - P[-div(u w) + nu lap w]``, with dc/dt from
    forward-mode automatic differentiation in t. No solution data is used."""

    def __init__(self, hidden: int = 256):
        torch, nn = _torch()
        d = 2 * len(LOW_MODES)
        self.net = nn.Sequential(nn.Linear(d + 2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d))
        n = 18
        self.n = n
        B = mode_basis(n)
        kx = np.array([k for k, _ in LOW_MODES for _ in (0, 1)], float)
        ky = np.array([k for _, k in LOW_MODES for _ in (0, 1)], float)
        self.B = torch.tensor(B, dtype=torch.float32)                 # (80, n*n)
        self.k2 = torch.tensor(kx ** 2 + ky ** 2, dtype=torch.float32)
        k = np.fft.fftfreq(n, 1.0 / n)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        self.KX, self.KY = torch.tensor(KX), torch.tensor(KY)
        self.inv = torch.tensor(np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0))

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, s):
        self.net.load_state_dict(s)

    def eval(self):
        self.net.eval()

    def coeffs(self, c0, z, t):
        torch, _ = _torch()
        return c0 + t * self.net(torch.cat([c0 / VORT_SCALE, z, t / T_OUT[1]], -1)) * VORT_SCALE

    def tendency(self, c, nu):
        """Galerkin-projected right-hand side for coefficients c (B, 80)."""
        torch, _ = _torch()
        n = self.n
        w = (c @ self.B).reshape(-1, n, n).double()
        wh = torch.fft.fft2(w)
        psih = wh * self.inv
        u = torch.fft.ifft2(1j * self.KY * psih).real
        v = torch.fft.ifft2(-1j * self.KX * psih).real
        adv = torch.fft.ifft2(1j * self.KX * torch.fft.fft2(u * w) + 1j * self.KY * torch.fft.fft2(v * w)).real
        proj = (-adv.reshape(-1, n * n).float() @ self.B.T) * (2.0 / (n * n))
        return proj - nu * self.k2 * c

    def fit(self, c0: np.ndarray, nu: np.ndarray, steps: int, seed: int = 0, batch: int = 64, lr: float = 1e-3):
        torch, _ = _torch()
        from torch.func import jvp

        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        C0 = torch.tensor(c0, dtype=torch.float32)
        NU = torch.tensor(nu, dtype=torch.float32)[:, None]
        Z = torch.tensor(re_feature(nu), dtype=torch.float32)[:, None]
        opt = torch.optim.Adam(self.net.parameters(), lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        t0 = time.time()
        for step in range(steps):
            idx = torch.randint(0, C0.shape[0], (batch,), generator=g)
            t = torch.rand(batch, 1, generator=g) * T_OUT[1]
            c, dcdt = jvp(lambda tt: self.coeffs(C0[idx], Z[idx], tt), (t,), (torch.ones_like(t),))
            r = dcdt - self.tendency(c, NU[idx])
            loss = (r ** 2).mean() / (self.tendency(C0[idx], NU[idx]) ** 2).mean().detach()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % max(1, steps // 5) == 0 or step == steps - 1:
                print(f"  PINN step {step}: relative Galerkin residual {float(loss):.4f} ({time.time() - t0:.0f} s)",
                      flush=True)

    def predict(self, c0: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """(B, 2, 80): coefficients at the two output times."""
        torch, _ = _torch()
        C0 = torch.tensor(c0, dtype=torch.float32)
        Z = torch.tensor(re_feature(nu), dtype=torch.float32)[:, None]
        with torch.no_grad():
            out = [self.coeffs(C0, Z, torch.full((C0.shape[0], 1), t)).numpy() for t in T_OUT]
        return np.stack(out, 1).astype(float)


# ----------------------------------------------------------------------------------------------- scale-linking models
SC_DELTAS = (2, 4, 8)                        # trained widths / h; 6 is held out


def _split(data, bands, test_frac=0.2):
    idx = np.flatnonzero(np.isin(data["band"], bands))
    n_test = int(round(test_frac * len(idx)))
    return idx[n_test:], idx[:n_test]


def _sc_tensors(data, idx, deltas):
    import torch

    w = data["w"][idx].astype(float)
    x = restrict(w[:, 0], N_OUT) / VORT_SCALE
    z = np.broadcast_to(re_feature(data["nu"][idx])[:, None, None], x.shape)
    X = torch.tensor(np.stack([x, z], 1))
    Y = {float(d * H): torch.tensor(restrict(apply_filter(w[:, 3:5], Filter("gaussian", d * H), L), N_OUT) / VORT_SCALE)
         for d in deltas}
    return X, Y


def train_scale_linking(data: dict, scale: float = 1.0):
    """Scale-conditioned FNO (Delta input, filtering-consistency loss), the scale-blind FNO on the same data, separate
    fixed-width FNOs at 2h, 4h and 8h, and the LES closure trained a posteriori. Trained on the mid and high Re bands
    (80 %); the remaining 20 % are held out."""
    torch, _ = _torch()
    from pinneapple_physics.multiscale.learned import (
        LearnedClosure,
        ScaleConditionedFNO2d,
        TorchVorticityLES2D,
        fit_scale_conditioned,
        train_closure_aposteriori,
    )

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    steps = lambda n: max(20, int(n * scale))  # noqa: E731
    tr, _ = _split(data, ("mid", "high"))
    path = OUT / "scale_linking.pt"
    kw = dict(in_channels=2, out_channels=2, width=32, modes=16, layers=4, delta_ref=H)
    models = {"SC-FNO": ScaleConditionedFNO2d(conditioned=True, **kw),
              "FNO-blind": ScaleConditionedFNO2d(conditioned=False, **kw),
              **{f"FNO-{d}h": ScaleConditionedFNO2d(conditioned=False, **kw) for d in SC_DELTAS}}
    closure = LearnedClosure()
    if path.exists():
        st = torch.load(path)
        for k, m in models.items():
            m.load_state_dict(st[k])
        closure.load_state_dict(st["closure"])
    else:
        X, Y = _sc_tensors(data, tr, SC_DELTAS)
        t0 = time.time()
        for k, m in models.items():
            ys = Y if k in ("SC-FNO", "FNO-blind") else {float(int(k[4:-1]) * H): Y[float(int(k[4:-1]) * H)]}
            print(f"training {k}", flush=True)
            fit_scale_conditioned(m, X, ys, "gaussian", steps=steps(2500), batch=16, lr=2e-3,
                                  consistency_weight=0.5 if k == "SC-FNO" else 0.0, seed=11, log_every=500)
        print(f"FNOs trained in {time.time() - t0:.0f} s", flush=True)
        # a-posteriori closure: LES 48^2 rolled out 60 steps (t = 0.3) against the filtered DNS at t = 0.1, 0.2, 0.3
        sharp = Filter("sharp", DELTA_LES)
        sub = tr[:: max(1, len(tr) // 96)]
        w = data["w"][sub].astype(float)
        W0 = torch.tensor(restrict(apply_filter(w[:, 0], sharp, L), LES_N))
        REF = torch.tensor(restrict(apply_filter(w[:, [1, 2, 4]], sharp, L), LES_N))
        les = TorchVorticityLES2D(LES_N)
        t0 = time.time()
        train_closure_aposteriori(closure, les, W0, torch.tensor(data["nu"][sub]), REF, 5e-3, [20, 40, 60],
                                  steps=steps(300), batch=4, lr=1e-3, seed=5, log_every=50)
        print(f"closure trained a posteriori in {time.time() - t0:.0f} s", flush=True)
        torch.save({**{k: m.state_dict() for k, m in models.items()}, "closure": closure.state_dict()}, path)
    for m in models.values():
        m.eval()
    closure.eval()
    torch.set_default_dtype(torch.float32)
    return models, closure


def sc_predict(model, w0: np.ndarray, nu: float, delta: float) -> np.ndarray:
    """(2, 64, 64) prediction of a (scale-conditioned) FNO at Gaussian width ``delta``."""
    torch, _ = _torch()
    x = restrict(apply_filter(np.asarray(w0, float), Filter("gaussian", delta), L), N_OUT) / VORT_SCALE
    X = torch.tensor(np.stack([x, np.full_like(x, float(re_feature(nu)))])[None], dtype=torch.float64)
    with torch.no_grad():
        return model(X, torch.tensor([delta], dtype=torch.float64))[0].numpy() * VORT_SCALE


def scale_conditioning_study(data, models):
    """#604: errors at Delta in {1, 2, 4, 6 (held out), 8} h on held-out cases, against fixed-width models
    (post-filtered from a narrower width where possible) and the scale-blind network."""
    from pinneapple_physics.multiscale import refilter, spectrum_error

    _, te = _split(data, ("mid", "high"))
    rows = []
    for d in (1, 2, 4, 6, 8):
        tgt = Filter("gaussian", d * H)
        errs = {k: [] for k in ("SC-FNO", "FNO-blind", "best fixed-width FNO", "SC-FNO spectrum", "consistency")}
        for i in te:
            w = data["w"][i].astype(float)
            ref = restrict(apply_filter(w[3:5], tgt, L), N_OUT)
            rel = lambda p: float(np.linalg.norm(p - ref) / np.linalg.norm(ref))  # noqa: E731
            p = sc_predict(models["SC-FNO"], w[0], data["nu"][i], d * H)
            errs["SC-FNO"].append(rel(p))
            errs["SC-FNO spectrum"].append(spectrum_error(p[1], ref[1], min(tgt.cutoff, 31.0)))
            errs["FNO-blind"].append(rel(sc_predict(models["FNO-blind"], w[0], data["nu"][i], d * H)))
            fixed = [rel(refilter(sc_predict(models[f"FNO-{e}h"], w[0], data["nu"][i], e * H),
                                  Filter("gaussian", e * H), tgt)) for e in SC_DELTAS if e <= d]
            errs["best fixed-width FNO"].append(min(fixed) if fixed else float("nan"))
            if d < 8:                                                     # prediction at d, filtered to 2d, vs at 2d
                p2 = sc_predict(models["SC-FNO"], w[0], data["nu"][i], 2 * d * H)
                errs["consistency"].append(float(np.linalg.norm(refilter(p, tgt, Filter("gaussian", 2 * d * H)) - p2)
                                                 / np.linalg.norm(p2)))
        rows.append({"delta_over_h": d, "n": len(te), **{k: float(np.nanmean(v)) if v else float("nan")
                                                          for k, v in errs.items()}})
        print(rows[-1], flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- closure benchmark
def les_expert(nu: float, closure, n: int = LES_N):
    return VorticityNS2D(n, nu, closure=closure)


def closure_benchmark(data, learned):
    """A priori (filtered DNS 128^2 -> LES grid, sharp k <= 16) and a posteriori (LES 48^2 to t = 0.3 against the
    filtered DNS; long run to t = 2 for stability) for every closure, on held-out cases of each Re band."""
    from pinneapple_physics.multiscale import (
        NoModel,
        aposteriori_test,
        apriori_test,
        closure_health,
        germano_error,
    )
    from pinneapple_physics.multiscale.learned import as_numpy_closure

    sharp = Filter("sharp", DELTA_LES)
    rows = []
    s48 = VorticityNS2D(LES_N, 0.0)
    for band in RE_BANDS:
        _, te = _split(data, (band,))
        te = te[:8]
        closures = {"no model": NoModel(), "Smagorinsky (Cs 0.17)": Smagorinsky(), "dynamic Smagorinsky":
                    DynamicSmagorinsky(), "gradient (Clark)": GradientModel(), "learned (a posteriori)":
                    as_numpy_closure(learned, LES_N)}
        for name, cl in closures.items():
            acc = {k: [] for k in ("stress_corr", "tend_corr", "backscatter_true", "backscatter_model", "spec_err",
                                   "field_err", "diss_ratio", "stable", "z_growth", "germano", "flagged")}
            for i in te:
                w = data["w"][i].astype(float)
                nu = float(data["nu"][i])
                w0b = restrict(apply_filter(w[0], sharp, L), LES_N)
                if hasattr(cl, "stress"):
                    a = apriori_test(w[0], sharp, cl, LES_N)
                    acc["stress_corr"].append(a.stress_correlation)
                    acc["backscatter_model"].append(a.backscatter_model)
                    acc["germano"].append(germano_error(w0b, cl, s48))
                from pinneapple_physics.multiscale import true_subfilter
                tr = true_subfilter(w[0], sharp, LES_N)
                t_m = cl(tr["w_bar"], s48)
                if np.std(t_m) > 0:
                    acc["tend_corr"].append(float(np.corrcoef(tr["tendency"].ravel(), t_m.ravel())[0, 1]))
                st = __import__("pinneapple_physics.multiscale.closures", fromlist=["strain"]).strain(tr["w_bar"], s48)
                acc["backscatter_true"].append(float(np.mean(-2 * (tr["a"] * st["S11"] + tr["b"] * st["S12"]) < 0)))
                ref = restrict(apply_filter(w[[1, 2, 4]], sharp, L), LES_N)
                les = les_expert(nu, None if name == "no model" else cl)
                ap = aposteriori_test(les, w0b, ref, [0.1, 0.2, 0.3], dt=5e-3, t_long=2.0)
                acc["spec_err"].append(ap.spectrum_error[-1])
                acc["field_err"].append(ap.field_error[-1])
                acc["diss_ratio"].append(ap.dissipation_les / ap.dissipation_ref if ap.dissipation_ref else np.nan)
                acc["stable"].append(float(ap.stable))
                acc["z_growth"].append(ap.enstrophy_growth)
                if ap.stable:
                    wl = les.simulate(w0b, list(T_OUT), dt=5e-3)
                    h = closure_health(wl, w0b, LES_N // 3, closure=cl if hasattr(cl, "stress") else None, solver=s48)
                    acc["flagged"].append(float(not h.ok))
                else:
                    acc["flagged"].append(1.0)
            rows.append({"band": band, "closure": name, "n": len(te),
                         **{k: (float(np.nanmean(v)) if len(v) else float("nan")) for k, v in acc.items()}})
            print(rows[-1], flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- expert outputs
def profiles():
    from pinneapple_physics.multiscale import ScaleProfile

    out = [ScaleProfile(n, f, L / GNN_N if n == "GNN" else H, re_range=RE_BANDS[b], fidelity=0)
           for n, (b, f) in SURROGATES.items()]
    for n, cl in (("LES-Smagorinsky", "Smagorinsky"), ("LES-dynamic", "dynamic Smagorinsky"),
                  ("LES-gradient", "gradient (Clark)")):
        out.append(ScaleProfile(n, Filter("sharp", DELTA_LES), L / LES_N, closure=cl, fidelity=1))
    out.append(ScaleProfile("LES-learned", Filter("sharp", DELTA_LES), L / LES_N, closure="learned (a posteriori)",
                            re_range=(RE_BANDS["mid"][0], RE_BANDS["high"][1]), fidelity=1))
    out.append(ScaleProfile("DNS-128", Filter("sharp", 0.0), L / 128, re_range=(0.0, 6000.0), fidelity=2))
    out.append(ScaleProfile("SC-FNO", Filter("gaussian", 2 * H), H, re_range=(RE_BANDS["mid"][0], RE_BANDS["high"][1]),
                            fidelity=0, notes="takes Delta as input: answers Gaussian widths 2h..8h directly"))
    return {p.name: p for p in out}


BASE_EXPERTS = ["FNO", "CNN", "DeepONet", "GNN", "PINN", "LES-Smagorinsky", "LES-dynamic", "LES-gradient", "DNS-128"]
LINK_EXPERTS = ["SC-FNO", "LES-learned"]


def sc_native_width(q) -> float:
    """Width at which the SC-FNO is asked: the requested Gaussian width (2h..8h), else its finest, 2h."""
    d = q["filter"].delta
    return float(np.clip(d, 2 * H, 8 * H)) if q["kind"] == "gaussian" else 2 * H


def evaluate_experts(seed: int, surrogates, sc_models, learned, n: int = 60):
    """Run every expert on every query of a stream once (cached): raw outputs on the 64^2 grid with the filter they
    carry, wall time, and the label-free checks that need the expert's internals (closure health of each LES on its
    own grid, cross-scale self-consistency of the SC-FNO)."""
    from pinneapple_physics.multiscale import closure_health, cross_scale_consistency, prolong
    from pinneapple_physics.multiscale.learned import as_numpy_closure

    path = OUT / f"experts_seed{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    ref = generate_reference((seed,), n)[seed]
    qs = query_stream(seed, n)
    names = BASE_EXPERTS + LINK_EXPERTS
    raw = np.zeros((n, len(names), 2, N_OUT, N_OUT))
    filt = {}
    secs = np.zeros((n, len(names)))
    check = np.full((n, len(names)), np.nan)
    reasons = np.empty((n, len(names)), dtype=object)
    sharp = Filter("sharp", DELTA_LES)
    s48 = VorticityNS2D(LES_N, 0.0)
    les_closures = {"LES-Smagorinsky": Smagorinsky(), "LES-dynamic": DynamicSmagorinsky(),
                    "LES-gradient": GradientModel(), "LES-learned": as_numpy_closure(learned, LES_N)}
    for k, q in enumerate(qs):
        w0 = ref["w0"][k].astype(float)
        for j, name in enumerate(names):
            t0 = time.perf_counter()
            if name in surrogates:
                y = surrogates[name](w0, q["nu"])
            elif name.startswith("LES"):
                cl = les_closures[name]
                w0b = restrict(apply_filter(w0, sharp, L), LES_N)
                try:
                    wl = VorticityNS2D(LES_N, q["nu"], closure=cl).simulate(w0b, list(T_OUT), dt=5e-3)
                    y = prolong(wl, N_OUT)
                except FloatingPointError:
                    wl, y = None, np.full((2, N_OUT, N_OUT), np.nan)
            elif name == "DNS-128":
                y = restrict(VorticityNS2D(128, q["nu"]).simulate(restrict(w0, 128), list(T_OUT), dt=5e-3), N_OUT)
            else:                                                       # SC-FNO
                d = sc_native_width(q)
                y = sc_predict(sc_models["SC-FNO"], w0, q["nu"], d)
            secs[k, j] = time.perf_counter() - t0
            raw[k, j] = y
            if name.startswith("LES"):                                  # label-free closure health
                if wl is None:
                    check[k, j], reasons[k, j] = 0.0, "LES blew up"
                else:
                    h = closure_health(wl, w0b, LES_N // 3, closure=cl if hasattr(cl, "stress") else None, solver=s48)
                    check[k, j], reasons[k, j] = h.score, "; ".join(h.reasons)
            if name == "SC-FNO":                                        # self-consistency across widths
                d = sc_native_width(q)
                p2 = sc_predict(sc_models["SC-FNO"], w0, q["nu"], 2 * d)
                rep = cross_scale_consistency(y, p2, Filter("gaussian", 2 * d), Filter("gaussian", d), tol=0.1,
                                              integral_tol={"circulation (mean)": 0.02, "energy": 0.1,
                                                            "enstrophy": 0.1})
                check[k, j], reasons[k, j] = rep.score, "; ".join(rep.reasons)
        print(f"seed {seed} query {k}: Re {q['re']:.0f} {q['kind']} {q['width']}h | "
              + " ".join(f"{nm} {secs[k, j]:.2f}s" for j, nm in enumerate(names)), flush=True)
    out = {"raw": raw, "secs": secs, "check": check, "reasons": reasons, "names": np.array(names)}
    np.savez(path, **out)
    return out


# ----------------------------------------------------------------------------------------------- ensembles
CONFIGS = {
    "A. original (scale-blind)": dict(aware=False, router=False, checks=False, link=False),
    "B. + scale-aware selection": dict(aware=True, router=True, checks=False, link=False),
    "C. + closure & consistency checks": dict(aware=True, router=True, checks=True, link=False),
    "D. + scale-linking models": dict(aware=True, router=True, checks=True, link=True),
}
COST_WEIGHT = 0.1


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def raw_residual(pred, q):
    """Grid-scale residual of the vorticity equation between the two output snapshots (blind to the filter)."""
    from pinneapple_physics.multiscale import vorticity_residual

    R, sc = vorticity_residual(pred[0], pred[1], DT_PAIR, q["nu"])
    return _rms(R) / sc


def closed_multiscale_residual(pred, q):
    """Residual of the *filtered* equations at the requested width and at twice it, with the subfilter term
    estimated by the gradient model at each width removed (the part left is blamed on the model); mean over the two."""
    from pinneapple_physics.multiscale import vorticity_residual

    s64 = VorticityNS2D(N_OUT, 0.0)
    d0 = max(q["filter"].delta, H)
    out = []
    for d in (d0, 2 * d0):
        f = pred if d == d0 else apply_filter(pred, Filter(q["kind"], d), L)
        g = GradientModel(delta=d)
        sub = 0.5 * (g(f[0], s64) + g(f[1], s64))
        R, sc = vorticity_residual(f[0], f[1], DT_PAIR, q["nu"], closure_tendency=sub)
        out.append(_rms(R) / sc)
    return float(np.mean(out))


def run_configs(seed: int, ev: dict, n: int = 60):
    from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
    from pinneapple_physics.multiscale import ScaleAwareRouter, ScaleQuery, refilter, spectrum_error

    qs = query_stream(seed, n)
    ref = generate_reference((seed,), n)[seed]
    refs = [reference_output(ref["traj"][k].astype(float), q["filter"]) for k, q in enumerate(qs)]
    names_all = [str(x) for x in ev["names"]]
    prof = profiles()
    cost = {nm: float(np.median(ev["secs"][:, j])) for j, nm in enumerate(names_all)}

    def adapted(j, name, q, aware):
        y = ev["raw"][q["id"], j]
        if not np.all(np.isfinite(y)):
            return y
        return refilter(y, prof[name].filter, q["filter"], L) if aware else apply_filter(y, q["filter"], L)

    results = {}
    for cname, cfg in CONFIGS.items():
        names = BASE_EXPERTS + (LINK_EXPERTS if cfg["link"] else [])
        idx = {nm: names_all.index(nm) for nm in names}
        experts = [from_callable(nm, (lambda nm: lambda q: adapted(idx[nm], nm, q, cfg["aware"]))(nm), cost=cost[nm])
                   for nm in names]
        router = ScaleAwareRouter([prof[nm] for nm in names],
                                  lambda q: ScaleQuery(q["filter"], re=q["re"], kmax=N_OUT / 2)) if cfg["router"] else None
        checks = None
        if cfg["checks"]:
            checks = {nm: (lambda j: lambda p, q: (float(ev["check"][q["id"], j]), str(ev["reasons"][q["id"], j])))(idx[nm])
                      for nm in names if nm.startswith("LES") or nm == "SC-FNO"}
        ens = PhysicsEnsemble(experts, mode="select", residual_fn=closed_multiscale_residual if cfg["checks"]
                              else raw_residual, residual_lookahead=3.0, cost_weight=COST_WEIGHT, router=router,
                              prediction_checks=checks)
        rec = {k: [] for k in ("active", "error", "spec_err", "expert_errors", "weights", "awake", "failed",
                               "escalated", "policy_cost", "pred")}
        for k, q in enumerate(qs):
            p = ens.predict(q)
            info = ens.update(q, refs[k])
            aw = p.get("awake", {nm: True for nm in names})
            rec["active"].append(p["active"])
            rec["error"].append(info["ensemble_error"])
            kc = min(q["filter"].cutoff, N_OUT / 2 - 1)
            rec["spec_err"].append(spectrum_error(p["prediction"][1], refs[k][1], kc, L))
            rec["expert_errors"].append(dict(zip(names, np.asarray(info["errors"], float).tolist(), strict=True)))
            rec["weights"].append(dict(p["weights"]))
            rec["awake"].append(dict(aw))
            rec["failed"].append(list(p.get("failed_checks", [])))
            rec["escalated"].append(bool(p.get("escalated_by_checks", False)))
            run_set = [nm for nm in names if aw.get(nm, True) or nm in p.get("failed_checks", [])]
            rec["policy_cost"].append(sum(cost[nm] for nm in run_set))
            rec["pred"].append(p["prediction"][1])
        results[cname] = rec
        err = np.array(rec["error"])
        print(f"seed {seed} {cname}: mean error {np.nanmean(err):.4f}", flush=True)
    return qs, refs, results, cost


def summarise(all_runs, closure_rows=None):
    """Tables over seeds: error per requested width (with n), spectrum error, wrong selections, scale violations,
    cost; and how the closure checks flagged."""
    prof = profiles()
    rows = []
    for cname in CONFIGS:
        per = {"err": [], "spec": [], "wrong": [], "viol": [], "cost_chosen": [], "cost_policy": [], "dns": [],
               "by_width": {d: [] for d in WIDTHS}}
        for seed, (qs, refs, res, cost) in all_runs.items():
            r = res[cname]
            for k, q in enumerate(qs):
                ee = r["expert_errors"][k]
                fin = [v for v in ee.values() if np.isfinite(v)]
                best = min(fin)
                e = r["error"][k]
                per["err"].append(e)
                per["by_width"][q["width"]].append(e)
                per["spec"].append(r["spec_err"][k])
                per["wrong"].append(float(e > max(1.5 * best, best + 0.03)))
                per["viol"].append(float(prof[r["active"][k]].resolved_width > q["filter"].delta * (1 + 1e-9)))
                per["cost_chosen"].append(cost[r["active"][k]])
                per["cost_policy"].append(r["policy_cost"][k])
                per["dns"].append(float(r["active"][k] == "DNS-128"))
        n_q = len(per["err"])
        rows.append({"config": cname, "n": n_q, "error": float(np.mean(per["err"])),
                     **{f"error_{d}h": float(np.mean(v)) for d, v in per["by_width"].items()},
                     **{f"n_{d}h": len(v) for d, v in per["by_width"].items()},
                     "spectrum_error": float(np.mean(per["spec"])), "wrong": int(sum(per["wrong"])),
                     "scale_violations": int(sum(per["viol"])), "cost_chosen": float(np.mean(per["cost_chosen"])),
                     "cost_policy": float(np.mean(per["cost_policy"])), "dns_fraction": float(np.mean(per["dns"]))})
    # baselines: always DNS, oracle (best expert of each query among all, aware adaptation)
    base = {"always DNS-128": [], "oracle (best expert in hindsight)": []}
    costs = {"always DNS-128": [], "oracle (best expert in hindsight)": []}
    for seed, (qs, refs, res, cost) in all_runs.items():
        r = res["D. + scale-linking models"]
        for k in range(len(qs)):
            ee = r["expert_errors"][k]
            base["always DNS-128"].append(ee["DNS-128"])
            costs["always DNS-128"].append(cost["DNS-128"])
            b = min((v, nm) for nm, v in ee.items() if np.isfinite(v))
            base["oracle (best expert in hindsight)"].append(b[0])
            costs["oracle (best expert in hindsight)"].append(cost[b[1]])
    for nm, v in base.items():
        rows.append({"config": nm, "n": len(v), "error": float(np.mean(v)), "cost_chosen": float(np.mean(costs[nm]))})
    return rows


def closure_flags(all_runs, evs):
    """For each LES: how often its label-free check failed, and its error (at its own scale, against the filtered
    reference) when flagged and when not."""
    out = []
    names = [str(x) for x in next(iter(evs.values()))["names"]]
    for nm in [x for x in names if x.startswith("LES")]:
        j = names.index(nm)
        fl, ef, en, sf, sn = [], [], [], [], []
        reasons = {}
        for seed, (qs, refs, res, cost) in all_runs.items():
            ev = evs[seed]
            ref = generate_reference((seed,), len(qs))[seed]
            for k, q in enumerate(qs):
                flagged = bool(ev["check"][k, j] < 0.5)
                fl.append(flagged)
                tgt = reference_output(ref["traj"][k].astype(float), Filter("sharp", DELTA_LES))
                y = ev["raw"][k, j]
                e = float(np.linalg.norm(y - tgt) / np.linalg.norm(tgt)) if np.all(np.isfinite(y)) else float("inf")
                from pinneapple_physics.multiscale import spectrum_error
                se = spectrum_error(y[1], tgt[1], LES_N // 3) if np.all(np.isfinite(y)) else float("inf")
                (ef if flagged else en).append(e)
                (sf if flagged else sn).append(se)
                for rr in str(ev["reasons"][k, j] or "").split("; "):
                    if rr and flagged:
                        key = rr.split(" (")[0].split(" by ")[0].split(" error ")[0]
                        reasons[key] = reasons.get(key, 0) + 1
        out.append({"expert": nm, "n": len(fl), "flagged": int(sum(fl)),
                    "field_error_flagged": float(np.mean(ef)) if ef else float("nan"),
                    "field_error_not_flagged": float(np.mean(en)) if en else float("nan"),
                    "spectrum_error_flagged": float(np.mean(sf)) if sf else float("nan"),
                    "spectrum_error_not_flagged": float(np.mean(sn)) if sn else float("nan"),
                    "reasons": reasons})
        print(out[-1], flush=True)
    return out


# ----------------------------------------------------------------------------------------------- figures
CONFIG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed categorical order


def summary_figure(rows, flags, sc_rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = [r for r in rows if r["config"] in CONFIGS]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))
    a = ax[0, 0]
    x = np.arange(len(WIDTHS))
    bw = 0.2
    for i, r in enumerate(cfg):
        a.bar(x + (i - 1.5) * bw, [100 * r[f"error_{d}h"] for d in WIDTHS], bw * 0.9, color=CONFIG_COLORS[i],
              label=r["config"])
    a.set_xticks(x, [f"{d}h\n(n={cfg[0][f'n_{d}h']})" for d in WIDTHS])
    a.set_ylabel("relative L2 error (%)")
    a.set_title("Error at each requested filter width (3 seeds)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    for i, r in enumerate(cfg):
        a.scatter(r["cost_chosen"], 100 * r["error"], s=70, color=CONFIG_COLORS[i], zorder=3)
        a.annotate(r["config"].split(".")[0], (r["cost_chosen"], 100 * r["error"]), textcoords="offset points",
                   xytext=(6, 4), fontsize=8)
    for r in rows:
        if r["config"] not in CONFIGS:
            a.scatter(r["cost_chosen"], 100 * r["error"], s=50, marker="s", color="0.45", zorder=3)
            a.annotate(r["config"], (r["cost_chosen"], 100 * r["error"]), textcoords="offset points", xytext=(6, -10),
                       fontsize=7, color="0.3")
    a.set_xscale("log")
    a.set_xlabel("cost of the chosen model per query (s, CPU)")
    a.set_ylabel("mean relative L2 error (%)")
    a.set_title("Error against cost")
    a = ax[1, 0]
    lab = [r["config"].split(".")[0] for r in cfg]
    a.bar(np.arange(len(cfg)) - 0.2, [r["wrong"] for r in cfg], 0.38, color="0.35", label="wrong model")
    a.bar(np.arange(len(cfg)) + 0.2, [r["scale_violations"] for r in cfg], 0.38, color="0.7",
          label="model coarser than the requested scale")
    a.set_xticks(range(len(cfg)), lab)
    a.set_ylabel(f"queries (of {cfg[0]['n']})")
    a.set_title("Wrong selections")
    a.legend(fontsize=8)
    a = ax[1, 1]
    if sc_rows:
        ds = [r["delta_over_h"] for r in sc_rows]
        for key, c, m in (("SC-FNO", CONFIG_COLORS[3], "o"), ("FNO-blind", "0.55", "s"),
                          ("best fixed-width FNO", "0.25", "^")):
            a.plot(ds, [100 * r[key] for r in sc_rows], marker=m, color=c, lw=2, label=key)
        a.axvspan(5.5, 6.5, color="0.92", zorder=0)
        a.text(6, a.get_ylim()[1] * 0.95 if a.get_ylim()[1] else 1, "held out", ha="center", fontsize=8, color="0.4")
        a.set_xlabel("Gaussian filter width (h)")
        a.set_ylabel("relative L2 error (%)")
        a.set_title(f"Scale-conditioned FNO, held-out cases (n={sc_rows[0]['n']})")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown_tables(rows, flags, sc_rows, bench_rows):
    out = ["| configuration | error (all) | " + " | ".join(f"{d}h" for d in WIDTHS) +
           " | spectrum error | wrong model | coarser than asked | cost chosen (s) | cost of policy (s) | DNS share |",
           "|---|---|" + "---|" * len(WIDTHS) + "---|---|---|---|---|---|"]
    for r in rows:
        if r["config"] in CONFIGS:
            out.append(f"| {r['config']} | {100 * r['error']:.1f} % | " +
                       " | ".join(f"{100 * r[f'error_{d}h']:.1f} %" for d in WIDTHS) +
                       f" | {r['spectrum_error']:.3f} | {r['wrong']} | {r['scale_violations']} | {r['cost_chosen']:.3f} |"
                       f" {r['cost_policy']:.2f} | {100 * r['dns_fraction']:.0f} % |")
        else:
            out.append(f"| {r['config']} | {100 * r['error']:.1f} % | " + " | ".join("" for _ in WIDTHS) +
                       f" | | | | {r['cost_chosen']:.3f} | | |")
    n_row = "| n (queries) | " + str(rows[0]["n"]) + " | " + " | ".join(str(rows[0][f"n_{d}h"]) for d in WIDTHS) + " |"
    out.append(n_row)
    out += ["", "| LES | flagged (of n) | field error flagged / not | spectrum error flagged / not | reasons |",
            "|---|---|---|---|---|"]
    for f in flags:
        out.append(f"| {f['expert']} | {f['flagged']} / {f['n']} | {100 * f['field_error_flagged']:.1f} % / "
                   f"{100 * f['field_error_not_flagged']:.1f} % | {f['spectrum_error_flagged']:.3f} / "
                   f"{f['spectrum_error_not_flagged']:.3f} | " + ", ".join(f"{k}: {v}" for k, v in f["reasons"].items())
                   + " |")
    if sc_rows:
        out += ["", "| Gaussian width | SC-FNO | scale-blind FNO | best fixed-width FNO | SC-FNO spectrum error |"
                " SC-FNO consistency (d vs 2d) |", "|---|---|---|---|---|---|"]
        for r in sc_rows:
            out.append(f"| {r['delta_over_h']}h{' (held out)' if r['delta_over_h'] in (1, 6) else ''} | "
                       f"{100 * r['SC-FNO']:.1f} % | {100 * r['FNO-blind']:.1f} % | {100 * r['best fixed-width FNO']:.1f} % |"
                       f" {r['SC-FNO spectrum']:.3f} | {100 * r['consistency']:.1f} % |")
    if bench_rows:
        out += ["", "| Re band | closure | stress corr. | tendency corr. | backscatter (true / model) | Germano error |"
                " spectrum error t=0.3 | field error t=0.3 | dissipation LES / DNS | stable to t=2 | flagged |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in bench_rows:
            out.append(f"| {r['band']} | {r['closure']} | {r['stress_corr']:.2f} | {r['tend_corr']:.2f} | "
                       f"{r['backscatter_true']:.2f} / {r['backscatter_model']:.2f} | {r['germano']:.2f} | "
                       f"{r['spec_err']:.3f} | {100 * r['field_err']:.1f} % | {r['diss_ratio']:.2f} | "
                       f"{100 * r['stable']:.0f} % | {100 * r['flagged']:.0f} % |")
    return "\n".join(out)


def make_gif(seed, qs, refs, res, path):
    from pinneapple_physics.multiscale.viz import animate_multiscale_ensemble

    d = res["D. + scale-linking models"]
    a = res["A. original (scale-blind)"]
    names = BASE_EXPERTS + LINK_EXPERTS
    regimes = [(k, f"Re {qs[k]['band']}") for k in range(len(qs)) if k == 0 or qs[k]["band"] != qs[k - 1]["band"]]
    labels = [f"{q['kind']} filter, width {q['width']}h, Re {q['re']:.0f}" for q in qs]
    cut = [min(q["filter"].cutoff, N_OUT / 2) for q in qs]
    return animate_multiscale_ensemble([r[1] for r in refs], d["pred"], d["active"], names, d["weights"],
                                       [q["width"] for q in qs], labels, cut, path, compare_active=a["active"],
                                       compare_label="A (scale-blind)", flags=d["failed"], regimes=regimes,
                                       title="Scale-aware ensemble (D)")


def main(seeds=(0, 1, 2), n=60, workers=4, scale=1.0, assets="docs/assets/physics_ensemble"):
    data = generate_training_data(workers=workers)
    generate_reference(seeds, n, workers)
    surrogates = train_surrogates(data, scale)
    sc_models, learned = train_scale_linking(data, scale)
    res_path = OUT / "results.json"
    sc_rows = scale_conditioning_study(data, sc_models)
    bench_rows = closure_benchmark(data, learned)
    evs = {s: evaluate_experts(s, surrogates, sc_models, learned, n) for s in seeds}
    all_runs = {s: run_configs(s, evs[s], n) for s in seeds}
    rows = summarise(all_runs)
    flags = closure_flags(all_runs, evs)
    md = markdown_tables(rows, flags, sc_rows, bench_rows)
    print(md)
    Path(assets).mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text(md)
    res_path.write_text(json.dumps({"configs": rows, "closure_flags": flags, "scale_conditioning": sc_rows,
                                    "closure_benchmark": bench_rows,
                                    "choices": {str(s): {c: all_runs[s][2][c]["active"] for c in CONFIGS}
                                                for s in seeds}}, indent=1, default=str))
    summary_figure(rows, flags, sc_rows, Path(assets) / "multiscale_summary.png")
    qs, refs, res, _ = all_runs[seeds[0]]
    make_gif(seeds[0], qs, refs, res, Path(assets) / "multiscale_ensemble.gif")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "train", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the training steps")
    a = ap.parse_args()
    if a.stage == "data":
        generate_training_data(workers=a.workers)
        generate_reference(workers=a.workers)
    elif a.stage == "train":
        d = generate_training_data(workers=a.workers)
        train_surrogates(d, a.scale)
        train_scale_linking(d, a.scale)
    else:
        main(workers=a.workers, scale=a.scale)
