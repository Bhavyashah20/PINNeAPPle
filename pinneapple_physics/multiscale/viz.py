"""Animation of a scale-aware ensemble run: the scale each query asks for, the model chosen, the filtered reference
against the prediction, their energy spectra up to the requested cut-off, the weights and the timeline of choices."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .ns2d import energy_spectrum

__all__ = ["animate_multiscale_ensemble"]


def animate_multiscale_ensemble(references: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                                active: Sequence[str], names: Sequence[str], weights: Sequence[Mapping[str, float]],
                                widths: Sequence[float], labels: Sequence[str], cutoffs: Sequence[float],
                                path: str | Path = "multiscale_ensemble.gif", *,
                                compare_active: Sequence[str] | None = None, compare_label: str = "scale-blind",
                                flags: Sequence[Sequence[str]] | None = None, regimes: Sequence[tuple[int, str]] = (),
                                title: str = "Scale-aware adaptive ensemble", fps: int = 3, dpi: int = 70,
                                length: float = 2 * np.pi, width_unit: str = "h") -> Path:
    """One frame per query. ``references`` / ``predictions``: 2-D fields at the requested scale; ``active``: model
    chosen per query; ``weights``: per query ``{name: weight}``; ``widths``: requested filter width per query (in
    ``width_unit``); ``labels``: text describing each query; ``cutoffs``: requested cut-off wavenumber (shells);
    ``compare_active``: the choices of another ensemble, drawn on the timeline; ``flags``: experts whose checks
    failed per query; ``regimes``: ``[(first_query, label)]`` drawn on the timeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    names = list(names)
    n = len(references)
    cmap = plt.get_cmap("tab20")
    color = {nm: cmap(i % 20) for i, nm in enumerate(names)}
    ypos = {nm: i for i, nm in enumerate(names)}
    vmax = float(np.percentile(np.abs(np.stack(references)), 99.5))
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.28)
    ax_ref, ax_pred, ax_spec = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_w, ax_t = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    im_r = ax_ref.imshow(references[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im_p = ax_pred.imshow(predictions[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for a in (ax_ref, ax_pred):
        a.set_xticks([])
        a.set_yticks([])
    ax_ref.set_title("reference (filtered DNS 256²)")
    sup = fig.suptitle("", fontsize=12)
    ax_t.set_yticks(range(len(names)))
    ax_t.set_yticklabels(names, fontsize=7)
    ax_t.set_xlim(-0.5, n - 0.5)
    ax_t.set_ylim(-0.8, len(names) - 0.2)
    ax_t.set_xlabel("query")
    for first, lab in regimes:
        ax_t.axvline(first - 0.5, color="0.6", ls="--", lw=0.8)
        ax_t.text(first, len(names) - 0.4, lab, fontsize=7, color="0.35", va="bottom")
    if compare_active is not None:
        ax_t.scatter(range(n), [ypos[a] for a in compare_active], marker="x", s=14, color="0.55",
                     label=f"{compare_label} choice", zorder=2)
    ax_t2 = ax_t.twinx()
    ax_t2.step(range(n), widths, where="mid", color="0.2", lw=1.0, alpha=0.6)
    ax_t2.set_ylabel(f"requested width ({width_unit})", fontsize=8)
    ax_t2.set_ylim(0, max(widths) * 1.25)
    sc = ax_t.scatter([], [], s=28, zorder=3, label="chosen")
    cur = ax_t.axvline(0, color="k", lw=0.8)
    ax_t.legend(loc="lower left", fontsize=7, ncol=2)

    def frame(i):
        im_r.set_data(references[i].T)
        im_p.set_data(predictions[i].T)
        ax_pred.set_title(f"prediction: {active[i]}", color=color[active[i]], fontweight="bold")
        for s in ax_pred.spines.values():
            s.set_edgecolor(color[active[i]])
            s.set_linewidth(3)
        ax_spec.clear()
        k, Er = energy_spectrum(references[i], length)
        _, Ep = energy_spectrum(predictions[i], length)
        m = (k >= 1) & (Er > 0)
        ax_spec.loglog(k[m], Er[m], "k-", lw=2, label="reference")
        mp = (k >= 1) & (Ep > 0)
        ax_spec.loglog(k[mp], Ep[mp], "-", color=color[active[i]], lw=1.5, label=active[i])
        ax_spec.axvline(cutoffs[i], color="0.5", ls=":", lw=1)
        ax_spec.text(cutoffs[i], 1e-1, " cut-off", fontsize=7, color="0.4")
        ax_spec.set_ylim(1e-9, 1)
        ax_spec.set_xlim(1, length / (2 * np.pi) * len(references[i]) / 2)
        ax_spec.set_xlabel("k")
        ax_spec.set_title("energy spectrum E(k)")
        ax_spec.legend(fontsize=7, loc="lower left")
        ax_w.clear()
        w = np.array([weights[i].get(nm, 0.0) for nm in names])
        ax_w.barh(range(len(names)), w, color=[color[nm] for nm in names])
        bad = set(flags[i]) if flags is not None else set()
        ax_w.set_yticks(range(len(names)))
        ax_w.set_yticklabels([nm + ("  ✗ check" if nm in bad else "") for nm in names], fontsize=7)
        ax_w.set_xlim(0, 1)
        ax_w.set_title("weights for this query", fontsize=9)
        sc.set_offsets(np.c_[np.arange(i + 1), [ypos[a] for a in active[:i + 1]]])
        sc.set_color([color[a] for a in active[:i + 1]])
        cur.set_xdata([i, i])
        sup.set_text(f"{title}   |   query {i + 1}/{n}: {labels[i]}")
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
