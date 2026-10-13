"""Learned models that connect scales (PyTorch).

* :class:`ScaleConditionedFNO2d`: a 2-D Fourier neural operator that takes the filter width ``Delta`` as an input
  (FiLM: every Fourier layer is scaled and shifted by an embedding of ``log(Delta / delta_ref)``, Perez et al. 2018),
  trained on one fine dataset filtered at several widths. :func:`filtering_consistency_loss` ties its predictions at
  different widths together: the prediction at ``Delta_1``, filtered to ``Delta_2 > Delta_1``, must equal the
  prediction at ``Delta_2``. With ``conditioned=False`` it is the scale-blind version (same network, ``Delta``
  ignored), for comparison.
* :class:`TorchVorticityLES2D`: the 2-D vorticity LES of :class:`~.ns2d.VorticityNS2D`, differentiable (integrating
  factor RK4, dealiased), so a closure can be trained *a posteriori*, through the solver: the loss compares the
  LES trajectory with the filtered reference, and its gradient flows through every time step (as in the
  differentiable-physics training of subgrid models, e.g. Um et al. 2020; Kochkov et al. 2021; List et al. 2022).
* :class:`LearnedClosure`: a small periodic CNN that returns a subfilter vorticity *flux* ``sigma``; the tendency is
  ``-div(sigma)``, so the closure conserves circulation exactly whatever the network does.
  :func:`as_numpy_closure` plugs it into :class:`~.ns2d.VorticityNS2D`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .filters import Filter

__all__ = ["torch_transfer", "torch_filter", "torch_refilter", "ScaleConditionedFNO2d", "filtering_consistency_loss",
           "fit_scale_conditioned", "TorchVorticityLES2D", "LearnedClosure", "train_closure_aposteriori",
           "as_numpy_closure"]


def _k(n: int, length: float, device=None, dtype=torch.float64) -> torch.Tensor:
    return 2 * math.pi * torch.fft.fftfreq(n, length / n, device=device, dtype=dtype)


def torch_transfer(kind: str, k: torch.Tensor, delta: torch.Tensor | float) -> torch.Tensor:
    """Transfer function of :func:`~.filters.transfer` in torch (``delta`` may be a tensor broadcastable to ``k``)."""
    delta = torch.as_tensor(delta, dtype=k.dtype, device=k.device)
    if kind == "box":
        return torch.sinc(k * delta / (2 * math.pi))
    if kind == "gaussian":
        return torch.exp(-(k * delta) ** 2 / 24.0)
    if kind == "sharp":
        return torch.where(delta > 0, (k.abs() <= math.pi / torch.clamp(delta, min=1e-30) * (1 + 1e-12)).to(k.dtype),
                           torch.ones_like(k))
    raise ValueError(kind)


def _transfer2d(kind: str, n: int, delta: torch.Tensor, length: float, dtype) -> torch.Tensor:
    """(B, 1, n, n) transfer for a batch of widths ``delta`` (B,)."""
    k = _k(n, length, delta.device, dtype)
    d = delta.reshape(-1, 1, 1, 1).to(dtype)
    return torch_transfer(kind, k.view(1, 1, n, 1), d) * torch_transfer(kind, k.view(1, 1, 1, n), d)


def torch_filter(x: torch.Tensor, kind: str, delta: torch.Tensor | float, length: float = 2 * math.pi) -> torch.Tensor:
    """Filter ``x`` (B, C, n, n) with width ``delta`` (scalar or (B,))."""
    B, n = x.shape[0], x.shape[-1]
    d = torch.as_tensor(delta, dtype=x.dtype, device=x.device).expand(B) if torch.as_tensor(delta).ndim == 0 else \
        torch.as_tensor(delta, dtype=x.dtype, device=x.device)
    G = _transfer2d(kind, n, d, length, x.dtype)
    return torch.fft.ifft2(torch.fft.fft2(x) * G).real


def torch_refilter(x: torch.Tensor, kind: str, d_from: torch.Tensor, d_to: torch.Tensor, length: float = 2 * math.pi,
                   max_gain: float = 1.0 + 1e-9) -> torch.Tensor:
    """Bring ``x`` filtered at ``d_from`` to ``d_to`` (per sample), as :func:`~.filters.refilter`."""
    n = x.shape[-1]
    gs = _transfer2d(kind, n, d_from, length, x.dtype)
    gt = _transfer2d(kind, n, d_to, length, x.dtype)
    r = torch.where(gs.abs() > 1e-12, gt / torch.where(gs.abs() > 1e-12, gs, torch.ones_like(gs)),
                    torch.full_like(gs, float("inf")))
    r = torch.where(r.abs() <= max_gain, r, torch.zeros_like(r))
    return torch.fft.ifft2(torch.fft.fft2(x) * r).real


class _SpectralConv2d(nn.Module):
    def __init__(self, c: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c * c)
        self.w1 = nn.Parameter(scale * torch.randn(c, c, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(c, c, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        m = self.modes
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class ScaleConditionedFNO2d(nn.Module):
    """FNO on a periodic grid whose layers are modulated by the filter width: ``x -> gelu(gamma(Delta) * (K x + W x)
    + beta(Delta))``. ``forward(x, delta)`` with ``x`` (B, C, n, n) and ``delta`` (B,) or a float. ``conditioned=False``
    gives the scale-blind network (``gamma = 1``, ``beta = 0``)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32, modes: int = 12, layers: int = 4,
                 conditioned: bool = True, delta_ref: float = 2 * math.pi / 64):
        super().__init__()
        self.conditioned, self.delta_ref, self.layers = bool(conditioned), float(delta_ref), int(layers)
        self.inp = nn.Conv2d(in_channels, width, 1)
        self.convs = nn.ModuleList([_SpectralConv2d(width, modes) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.out = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, out_channels, 1))
        if self.conditioned:
            self.film = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 2 * width * layers))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, delta: torch.Tensor | float | None = None) -> torch.Tensor:
        B = x.shape[0]
        h = self.inp(x)
        if self.conditioned:
            d = torch.as_tensor(delta, dtype=x.dtype, device=x.device).reshape(-1).expand(B) \
                if torch.as_tensor(delta).numel() == 1 else torch.as_tensor(delta, dtype=x.dtype, device=x.device)
            z = torch.log(torch.clamp(d, min=1e-6) / self.delta_ref)
            gb = self.film(torch.stack([z, z ** 2], -1)).view(B, self.layers, 2, -1)
        for j, (k, w) in enumerate(zip(self.convs, self.ws, strict=True)):
            y = k(h) + w(h)
            if self.conditioned:
                y = y * (1 + gb[:, j, 0, :, None, None]) + gb[:, j, 1, :, None, None]
            h = nn.functional.gelu(y) if j < self.layers - 1 else y
        return self.out(h)


def filtering_consistency_loss(model: nn.Module, x: torch.Tensor, d_small: torch.Tensor, d_large: torch.Tensor,
                               kind: str = "gaussian", length: float = 2 * math.pi) -> torch.Tensor:
    """Mean squared mismatch between ``filter(model(x, d_small)) -> d_large`` and ``model(x, d_large)``, relative to
    the mean square of the latter."""
    p_s = model(x, d_small)
    p_l = model(x, d_large)
    p_sf = torch_refilter(p_s, kind, d_small, d_large, length)
    return ((p_sf - p_l) ** 2).mean() / (p_l.detach() ** 2).mean().clamp_min(1e-12)


def fit_scale_conditioned(model: nn.Module, inputs: torch.Tensor, targets: dict[float, torch.Tensor],
                          kind: str = "gaussian", steps: int = 2000, batch: int = 16, lr: float = 2e-3,
                          consistency_weight: float = 0.5, seed: int = 0, input_filter: bool = True,
                          length: float = 2 * math.pi, log_every: int = 0) -> list[float]:
    """Train on one dataset seen at several widths: ``inputs`` (N, C, n, n) unfiltered, ``targets`` {delta: (N, C', n, n)
    filtered at delta}. Each step draws a batch and a width; the input is filtered at that width when
    ``input_filter``; the loss is the relative MSE plus ``consistency_weight`` times
    :func:`filtering_consistency_loss` between two random widths (only for conditioned models)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    deltas = sorted(targets)
    N = inputs.shape[0]
    hist = []
    conditioned = getattr(model, "conditioned", True)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g)
        dsel = torch.tensor([deltas[int(i)] for i in torch.randint(0, len(deltas), (batch,), generator=g)],
                            dtype=inputs.dtype)
        x = inputs[idx]
        y = torch.stack([targets[float(d)][int(i)] for d, i in zip(dsel, idx, strict=True)])
        xin = torch_filter(x, kind, dsel, length) if input_filter else x
        p = model(xin, dsel)
        loss = ((p - y) ** 2).mean() / (y ** 2).mean().clamp_min(1e-12)
        if conditioned and consistency_weight > 0 and len(deltas) > 1:
            d1 = torch.tensor([deltas[0]] * batch, dtype=inputs.dtype)
            d2 = torch.tensor([deltas[int(i)] for i in torch.randint(1, len(deltas), (batch,), generator=g)],
                              dtype=inputs.dtype)
            xs = torch_filter(x, kind, d1, length) if input_filter else x
            ps, pl = model(xs, d1), model(torch_filter(x, kind, d2, length) if input_filter else x, d2)
            cons = ((torch_refilter(ps, kind, d1, d2, length) - pl) ** 2).mean() / (pl.detach() ** 2).mean().clamp_min(1e-12)
            loss = loss + consistency_weight * cons
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  step {step}: loss {float(loss):.4f}", flush=True)
    return hist


class TorchVorticityLES2D(nn.Module):
    """Differentiable dealiased pseudo-spectral 2-D vorticity solver (integrating-factor RK4) with an optional
    closure ``closure(w (B, n, n), solver) -> tendency``. ``nu`` per sample in :meth:`rollout`."""

    def __init__(self, n: int, length: float = 2 * math.pi, dtype=torch.float64):
        super().__init__()
        self.n, self.length, self.dtype = n, length, dtype
        k = _k(n, length, dtype=dtype)
        KX, KY = torch.meshgrid(k, k, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        kint = torch.fft.fftfreq(n, 1.0 / n, dtype=dtype).abs()
        KXi, KYi = torch.meshgrid(kint, kint, indexing="ij")
        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("K2", K2)
        self.register_buffer("inv_k2", torch.where(K2 > 0, 1.0 / torch.where(K2 > 0, K2, torch.ones_like(K2)),
                                                   torch.zeros_like(K2)))
        self.register_buffer("mask", ((KXi <= n / 3) & (KYi <= n / 3)).to(dtype))
        self.kmax = (n // 3) * 2 * math.pi / length

    @property
    def implicit_filter_width(self) -> float:
        return math.pi / self.kmax

    def velocity(self, w: torch.Tensor):
        psih = torch.fft.fft2(w) * self.inv_k2
        return torch.fft.ifft2(1j * self.KY * psih).real, torch.fft.ifft2(-1j * self.KX * psih).real

    def ddx(self, f):
        return torch.fft.ifft2(1j * self.KX * torch.fft.fft2(f)).real

    def ddy(self, f):
        return torch.fft.ifft2(1j * self.KY * torch.fft.fft2(f)).real

    def nonlinear_hat(self, wh: torch.Tensor, closure=None) -> torch.Tensor:
        w = torch.fft.ifft2(wh).real
        u, v = self.velocity(w)
        nl = -self.mask * (1j * self.KX * torch.fft.fft2(u * w) + 1j * self.KY * torch.fft.fft2(v * w))
        if closure is not None:
            nl = nl + self.mask * torch.fft.fft2(closure(w, self))
        return nl

    def rollout(self, w0: torch.Tensor, nu: torch.Tensor, dt: float, save_steps: Sequence[int], closure=None
                ) -> torch.Tensor:
        """Integrate ``w0`` (B, n, n) with viscosities ``nu`` (B,) and return the states after each number of steps
        in ``save_steps`` (sorted): (B, len(save_steps), n, n)."""
        nu = torch.as_tensor(nu, dtype=self.dtype).reshape(-1, 1, 1)
        E = torch.exp(-nu * self.K2 * dt)
        E2 = torch.exp(-nu * self.K2 * dt / 2)
        wh = torch.fft.fft2(w0.to(self.dtype))
        out, step = [], 0
        for target in save_steps:
            while step < target:
                N1 = self.nonlinear_hat(wh, closure)
                a = E2 * (wh + 0.5 * dt * N1)
                N2 = self.nonlinear_hat(a, closure)
                b = E2 * wh + 0.5 * dt * N2
                N3 = self.nonlinear_hat(b, closure)
                c = E * wh + dt * E2 * N3
                N4 = self.nonlinear_hat(c, closure)
                wh = E * wh + dt / 6 * (E * N1 + 2 * E2 * (N2 + N3) + N4)
                step += 1
            out.append(torch.fft.ifft2(wh).real)
        return torch.stack(out, 1)


class LearnedClosure(nn.Module):
    """Periodic CNN closure: inputs ``omega / rms``, ``u``, ``v`` (scaled) and the strain components; output the
    subfilter vorticity flux ``sigma`` (2 channels), scaled by ``delta^2 * rms(omega)^2``; tendency ``-div(sigma)``."""

    def __init__(self, hidden: int = 32, layers: int = 4, kernel: int = 5):
        super().__init__()
        mods, c = [], 5
        for _ in range(layers):
            mods += [nn.Conv2d(c, hidden, kernel, padding=kernel // 2, padding_mode="circular"), nn.GELU()]
            c = hidden
        mods.append(nn.Conv2d(c, 2, kernel, padding=kernel // 2, padding_mode="circular"))
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.delta: float | None = None

    def flux(self, w: torch.Tensor, solver: TorchVorticityLES2D) -> tuple[torch.Tensor, torch.Tensor]:
        w = w.to(torch.float64)
        rms = w.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        u, v = solver.velocity(w)
        s11, s12 = solver.ddx(u), 0.5 * (solver.ddy(u) + solver.ddx(v))
        feats = torch.stack([w / rms, u * solver.kmax / rms, v * solver.kmax / rms, s11 / rms, s12 / rms], 1).float()
        out = self.net(feats).to(w.dtype)
        d = self.delta if self.delta is not None else solver.implicit_filter_width
        scale = d ** 2 * rms ** 2
        return out[:, 0] * scale, out[:, 1] * scale

    def forward(self, w: torch.Tensor, solver: TorchVorticityLES2D) -> torch.Tensor:
        sx, sy = self.flux(w, solver)
        return -(solver.ddx(sx) + solver.ddy(sy))


def train_closure_aposteriori(closure: LearnedClosure, solver: TorchVorticityLES2D, w0: torch.Tensor,
                              nu: torch.Tensor, reference: torch.Tensor, dt: float, save_steps: Sequence[int],
                              steps: int = 300, batch: int = 8, lr: float = 1e-3, seed: int = 0,
                              spectrum_weight: float = 0.5, log_every: int = 0) -> list[float]:
    """A-posteriori training: roll the LES out from the filtered initial fields ``w0`` (N, n, n) with ``closure`` and
    match the filtered reference ``reference`` (N, len(save_steps), n, n): relative MSE of the fields plus
    ``spectrum_weight`` times the mean squared log ratio of the energy spectra (shells 1..kmax), the gradient going
    through every solver step."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(closure.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = solver.n
    kint = torch.fft.fftfreq(n, 1.0 / n, dtype=torch.float64)
    shell = torch.sqrt(kint.view(-1, 1) ** 2 + kint.view(1, -1) ** 2).round().long()
    kmax = n // 3

    def spec(w):
        wh = torch.fft.fft2(w) / n ** 2
        e = 0.5 * wh.abs() ** 2 * solver.inv_k2 * (2 * math.pi / solver.length) ** 0
        flat = e.reshape(*e.shape[:-2], -1)
        out = torch.zeros(*e.shape[:-2], kmax + 1, dtype=e.dtype)
        return out.index_add_(-1, shell.reshape(-1).clamp(max=kmax), flat * (shell.reshape(-1) <= kmax))[..., 1:]

    hist = []
    for step in range(steps):
        idx = torch.randint(0, w0.shape[0], (batch,), generator=g)
        traj = solver.rollout(w0[idx], nu[idx], dt, save_steps, closure)
        ref = reference[idx].to(traj.dtype)
        loss = ((traj - ref) ** 2).mean() / (ref ** 2).mean()
        if spectrum_weight > 0:
            loss = loss + spectrum_weight * (torch.log(spec(traj).clamp_min(1e-30)) -
                                             torch.log(spec(ref).clamp_min(1e-30))).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(closure.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  closure step {step}: loss {float(loss):.4f}", flush=True)
    return hist


def as_numpy_closure(closure: LearnedClosure, n: int, length: float = 2 * math.pi):
    """Wrap a learned closure as ``closure(w, solver) -> tendency`` for :class:`~.ns2d.VorticityNS2D` (numpy, no
    gradient). ``stress`` is not defined for it (it models the vorticity flux directly)."""
    solver = TorchVorticityLES2D(n, length)

    class _Wrapped:
        name = "learned (a posteriori)"
        delta = None

        def __call__(self, w, _solver=None):
            single = np.ndim(w) == 2
            with torch.no_grad():
                t = closure(torch.as_tensor(np.asarray(w)[None] if single else np.asarray(w)), solver).numpy()
            return t[0] if single else t

    return _Wrapped()
