"""Figures for PT-symmetric dimers and exceptional-point tuning (matplotlib)."""
from __future__ import annotations

import numpy as np

from .ep_tuning import route_a_sqrt_coupling, route_c_asymmetric
from .pt_dimer import PTChain, PTDimer

__all__ = ["plot_spectrum", "plot_ep_laws", "plot_coupling_plane", "plot_chain_threshold", "ep_tuning_figure"]


def plot_spectrum(ax, dimer: PTDimer, lam: float = 0.0, gamma_max: float | None = None, n: int = 400):
    """Real and imaginary parts of the eigenvalues against gamma, the EP, and the eigenvector overlap (twin axis)."""
    g_ep = dimer.ep_gamma(lam)
    gm = gamma_max or 2.0 * g_ep
    gs = np.linspace(0.0, gm, n)
    ab, ba = dimer.b(lam)
    s = np.sqrt((ab * ba - gs ** 2).astype(complex))               # the two branches E = eps +/- s, continuous
    for sign in (1, -1):
        e = dimer.onsite + sign * s
        ax.plot(gs, e.real, color="#1f77b4", lw=2, label="Re E" if sign == 1 else None)
        ax.plot(gs, e.imag, color="#d62728", lw=2, ls="--", label="Im E" if sign == 1 else None)
    ax.axvline(g_ep, color="0.3", lw=0.8, ls=":")
    ax.text(g_ep, ax.get_ylim()[1] * 0.85, "  EP: γ = |b|", fontsize=8)
    ax2 = ax.twinx()
    ov = [dimer.eigenvector_overlap(g, lam) for g in gs]
    ax2.plot(gs, ov, color="0.55", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("eigenvector overlap", color="0.45", fontsize=8)
    ax.set_xlabel("gain/loss γ")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("PT dimer: unbroken → EP → broken", fontsize=10)
    return ax


def plot_ep_laws(ax, J: float = 1.0, G: float = 0.5, lam=None):
    """gamma_EP^2(lam) of each coupling class against the linear target J^2 + lam J G."""
    lam = np.linspace(-1.5, 1.5, 200) if lam is None else np.asarray(lam)
    target = J ** 2 + lam * J * G
    c = G / 2                                           # the best reciprocal linear attempt: same slope at lam = 0
    lin = PTDimer(lambda x: J + c * x)
    phase = PTDimer(lambda x: J * np.exp(1j * x))
    ra = PTDimer(route_a_sqrt_coupling(J, G))
    ab, ba = route_c_asymmetric(J, G)
    rc = PTDimer(ab, ba)
    ax.plot(lam, target, color="k", lw=3, alpha=0.25, label="target J² + λJ⟨G⟩")
    ax.plot(lam, [lin.ep_gamma_squared(x).real for x in lam], color="#d62728", lw=1.8,
            label="reciprocal b = J + cλ (quadratic: no-go)")
    ax.plot(lam, [phase.ep_gamma_squared(x).real for x in lam], color="#9467bd", lw=1.5, label="phase-only (EP fixed)")
    ok = target >= 0
    ax.plot(lam[ok], [ra.ep_gamma_squared(x).real for x in lam[ok]], color="#2ca02c", lw=1.5, ls="--",
            label="route A √(J² + λJ⟨G⟩) (exact, PT)")
    ax.plot(lam, [rc.ep_gamma_squared(x).real for x in lam], color="#ff7f0e", lw=1.2, ls=":",
            label="route C non-reciprocal (exact, PT broken)")
    ax.set_xlabel("control parameter λ")
    ax.set_ylabel("γ_EP²")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Where the EP goes as λ changes", fontsize=10)
    return ax


def plot_coupling_plane(ax, J: float = 1.0, G: float = 0.5, lam=None):
    """The coupling b(lam) in the complex plane: gamma_EP is its distance to the origin. A straight (polynomial
    degree 1) path has a squared distance quadratic in lam; a circle keeps it constant."""
    lam = np.linspace(-1.5, 1.5, 61) if lam is None else np.asarray(lam)
    c = G / 2 + 0.3j * G
    path = J + c * lam
    ax.plot(path.real, path.imag, "-", color="#d62728", lw=1.8, label="b = J + cλ (line)")
    for k in (0, len(lam) // 2, len(lam) - 1):
        ax.plot([0, path[k].real], [0, path[k].imag], color="#d62728", lw=0.6, alpha=0.6)
    t = np.linspace(-np.pi, np.pi, 200)
    ax.plot(J * np.cos(t), J * np.sin(t), color="#9467bd", lw=1.2, label="phase-only: |b| = J (circle)")
    ra = np.sqrt(np.maximum(J ** 2 + lam * J * G, 0))
    ax.plot(ra, 0 * ra, color="#2ca02c", lw=3, alpha=0.6, label="route A: √(J² + λJ⟨G⟩)")
    ax.plot(0, 0, "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel("Re b")
    ax.set_ylabel("Im b")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("γ_EP = distance from b to the origin", fontsize=10)
    return ax


def plot_chain_threshold(ax, sizes=(2, 3, 4, 5), b_values=None):
    """PT-breaking threshold gamma_EP^2 of N-site chains with uniform coupling against |b|^2. For a uniform chain
    H(gamma, b) = |b| H(gamma / |b|, 1) after a gauge transformation, so gamma_EP = kappa_N |b| exactly and the
    dimer's obstruction carries over to every N. Chains where the control parameter changes only some bonds are not
    covered by this argument (the paper's Conjecture 2 / open question Q2)."""
    b_values = np.linspace(0.2, 1.5, 14) if b_values is None else np.asarray(b_values)
    markers = "osd^v<>"
    for k, n in enumerate(sizes):
        g = np.array([PTChain(n, b).ep_gamma_numeric() for b in b_values])
        kappa = float(np.median(g ** 2 / b_values ** 2))
        ax.plot(b_values ** 2, g ** 2, markers[k % 7] + "-", ms=6 - k, lw=1, mfc="none" if k % 2 else None,
                label=f"N = {n}: γ_EP² = {kappa:.3g} |b|²")
    ax.set_xlabel("|b|²")
    ax.set_ylabel("γ_EP² (numerical)")
    ax.legend(fontsize=8)
    ax.set_title("Uniform chains: γ_EP² = κ_N |b|² (scaling)", fontsize=10)
    return ax


def ep_tuning_figure(path: str, J: float = 1.0, G: float = 0.5) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    plot_spectrum(ax[0, 0], PTDimer(J + 0.3j))
    plot_ep_laws(ax[0, 1], J, G)
    plot_coupling_plane(ax[1, 0], J, G)
    plot_chain_threshold(ax[1, 1])
    fig.suptitle("Exceptional-point tuning in PT-symmetric dimers: the no-go theorem and the routes around it",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
