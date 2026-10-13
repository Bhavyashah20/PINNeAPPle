"""A learned correction of the initial guess: a small graph network on the matrix graph (PyTorch).

The network sees, per cell, the right-hand side, the projection guess (:class:`ProjectionWarmStart`) and its
residual, all scaled by the matrix diagonal; it passes messages along the matrix's off-diagonal couplings
(the mesh faces) and outputs a correction ``x = x_proj + s * net(features)``, ``s`` the size of a Jacobi step
on the projection's residual. It is trained to minimise the
residual ``||b - A x|| / ||b - A x_proj||`` on systems written by an earlier run. At use, the guess is kept
only when its residual is below the projection's (a safeguard), so a bad prediction costs one product.

A purely local network can remove high-frequency residual (it learns something close to a smoother); the
low-frequency error that sets the Krylov iteration count is global. Expect a modest effect; the benchmark
reports whatever it measures.
"""

from __future__ import annotations

import numpy as np

from .ldu import LduSystem
from .warm_start import ProjectionWarmStart


def _torch():
    import torch

    return torch


class GraphGuessNet:
    """Message-passing network ``h <- tanh(W_self h + W_nb (Â h))`` on the normalised matrix graph."""

    def __init__(self, hidden: int = 32, layers: int = 4, seed: int = 0):
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        self.hidden, self.layers = hidden, layers
        self.inp = nn.Linear(4, hidden)
        self.self_lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.nb_lin = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1)
        self.modules = nn.ModuleList([self.inp, self.self_lin, self.nb_lin, self.out]).double()
        with torch.no_grad():  # start as the identity map (zero correction)
            self.out.weight.zero_()
            self.out.bias.zero_()

    def parameters(self):
        return self.modules.parameters()

    def forward(self, feats, adj):
        torch = _torch()
        h = torch.tanh(self.inp(feats))
        for ls, ln in zip(self.self_lin, self.nb_lin, strict=True):
            h = torch.tanh(ls(h) + ln(torch.sparse.mm(adj, h)))
        return self.out(h).squeeze(-1)


class _Graph:
    """Tensors of one system for the network: normalised adjacency, A, features."""

    def __init__(self, system: LduSystem, x_proj: np.ndarray):
        torch = _torch()
        n = system.n_cells
        lo, up = system.lower_addr, system.upper_addr
        d = np.abs(system.diag)
        w = system.upper / np.sqrt(d[lo] * d[up])
        idx = torch.as_tensor(
            np.stack([np.concatenate([lo, up]), np.concatenate([up, lo])]), dtype=torch.int64
        )
        self.adj = torch.sparse_coo_tensor(
            idx, torch.as_tensor(np.concatenate([w, w])), (n, n)
        ).coalesce()
        A = system.to_csr().tocoo()
        self.A = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([A.row, A.col]), dtype=torch.int64),
            torch.as_tensor(A.data),
            A.shape,
        ).coalesce()
        self.b = torch.as_tensor(system.source)
        self.x_proj = torch.as_tensor(x_proj)
        r = system.source - system.amul(x_proj)
        self.r_proj = torch.as_tensor(r)
        self.scale = float(np.std(x_proj)) + 1e-30
        bs = system.source / system.diag
        rs = r / system.diag
        # the correction is measured in units of a Jacobi step on the projection's residual
        self.corr_scale = float(np.std(rs)) + 1e-300
        cols = [
            bs / (np.std(bs) + 1e-30),
            (x_proj - np.mean(x_proj)) / self.scale,
            rs / self.corr_scale,
            d / np.mean(d),
        ]
        self.feats = torch.as_tensor(np.stack(cols, axis=1))

    def guess(self, net):
        return self.x_proj + self.corr_scale * net.forward(self.feats, self.adj)

    def residual_ratio(self, x):
        torch = _torch()
        r = self.b - torch.sparse.mm(self.A, x[:, None]).squeeze(-1)
        return torch.linalg.norm(r) / (torch.linalg.norm(self.r_proj) + 1e-300)


def projection_inputs(systems: list[LduSystem], n_snapshots: int = 8) -> list[np.ndarray]:
    """The projection guess of every system of a run, from the run's own converged solutions (``psi``)."""
    ws = ProjectionWarmStart(n_snapshots)
    out = []
    for s in systems:
        out.append(ws.guess(s, s.psi0))
        ws.update(s.psi)
    return out


class LearnedWarmStart:
    """Projection guess + learned correction (safeguarded). Same interface as the other warm starts."""

    def __init__(self, n_snapshots: int = 8, hidden: int = 32, layers: int = 4, seed: int = 0):
        self.projection = ProjectionWarmStart(n_snapshots)
        self.net = GraphGuessNet(hidden, layers, seed)
        self.n_used = 0
        self.n_calls = 0

    def fit(
        self, systems: list[LduSystem], epochs: int = 200, lr: float = 3e-3, verbose: bool = False
    ) -> list[float]:
        """Train on the systems of an earlier run (in solve order, with their solutions)."""
        torch = _torch()
        graphs = [
            _Graph(s, xp)
            for s, xp in zip(
                systems, projection_inputs(systems, self.projection.n_snapshots), strict=True
            )
        ]
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses = []
        for ep in range(epochs):
            opt.zero_grad()
            loss = sum(g.residual_ratio(g.guess(self.net)) for g in graphs) / len(graphs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if verbose and ep % 20 == 0:
                print(f"epoch {ep}: mean residual ratio {float(loss):.4f}")
        return losses

    def guess(self, system: LduSystem, x_prev: np.ndarray) -> np.ndarray:
        torch = _torch()
        x_proj = self.projection.guess(system, x_prev)
        g = _Graph(system, x_proj)
        with torch.no_grad():
            x = g.guess(self.net).numpy()
        self.n_calls += 1
        if np.linalg.norm(system.source - system.amul(x)) < np.linalg.norm(
            system.source - system.amul(x_proj)
        ):
            self.n_used += 1
            return x
        return x_proj

    def update(self, x: np.ndarray) -> None:
        self.projection.update(x)
