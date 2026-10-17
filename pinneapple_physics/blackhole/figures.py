"""Figures for the black-hole weather study: skill against lead time and the resolution study.

    from pinneapple_physics.blackhole.figures import plot_skill, plot_convergence
    plot_skill(skill_report, "docs/assets/blackhole/skill_vs_lead.png")
    plot_convergence(convergence_report, "docs/assets/blackhole/convergence.png")

One y-axis per panel, series in a fixed colour order with direct labels and line styles as a second encoding
(two of the colours are below 3:1 contrast on white, so they are never the only cue).
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_skill", "plot_convergence"]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, in order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _label_ends(ax, x_end, ends, min_gap=0.06):
    """Direct labels at the line ends, nudged apart vertically (``min_gap`` in axis fraction) so none overlap."""
    lo, hi = ax.get_ylim()
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    items = sorted(ends, key=lambda e: e[0])
    frac = [(y - lo) / (hi - lo) for y, _, _ in items]
    pos = list(frac)
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    for (y, text, color), f0, f in zip(items, frac, pos, strict=True):
        ax.plot([x_end], [y], "o", ms=4, color=color)
        ax.annotate(text, (x_end, y), xytext=(8, (f - f0) * height_pt), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False)


def plot_skill(report: dict, path: str, model: str | None = None, title: str = "") -> str:
    """Left: MAE against lead for the model and the baselines. Right: skill score against persistence with its 95 %
    interval and the skill horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = model or next(iter(report["models"]))
    leads = np.asarray(report["leads_M"])
    mae = report["mean_mae"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ends = []
    for i, (name, ls) in enumerate(((model, "-"), ("persistence", "--"), ("dmd", "-."), ("climatology", ":"))):
        y = np.asarray(mae[name])
        a1.plot(leads, y, ls, lw=2, color=SERIES[i])
        ends.append((y[-1], {"dmd": f"DMD (rank {report['dmd_rank']})"}.get(name, name), SERIES[i]))
    _label_ends(a1, leads[-1], ends)
    a1.set_xlabel("lead time (GM/c³)", color=INK)
    a1.set_ylabel("MAE of normalised log density", color=INK)
    a1.set_xlim(0, leads[-1] * 1.28)
    _style(a1)
    vp = report["models"][model]["vs_persistence"]
    ss = np.asarray(vp["skill_score"])
    a2.fill_between(leads, ss[:, 1], ss[:, 2], color=SERIES[0], alpha=0.18, lw=0)
    a2.plot(leads, ss[:, 0], lw=2, color=SERIES[0])
    a2.axhline(0, color=MUTED, lw=1)
    h = vp["horizon"]
    if h > 0:
        a2.axvline(h, color=INK, lw=1, ls="--")
        a2.annotate(f"skill horizon {h:.0f} M", (h, ss[:, 2].max()), xytext=(4, -4), textcoords="offset points",
                    fontsize=9, color=INK, va="top")
    a2.set_xlabel("lead time (GM/c³)", color=INK)
    a2.set_ylabel("skill vs persistence, 1 − MAE/MAE_pers", color=INK)
    _style(a2)
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(report: dict, path: str) -> str:
    """Time-mean shell density and specific angular momentum on every grid, and the mean accretion rate per grid
    with its 95 % interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [n for n, _ in report["levels"]]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150)
    for ax, key, ylab, logy in ((axs[0], "rho_shell", "shell-averaged density", True),
                                (axs[1], "ell_shell", "specific angular momentum l", False)):
        p = report["profiles"][key]
        r = np.asarray(p["r"])
        for i, n in enumerate(levels):
            y = np.asarray(p["by_level"][str(n)] if str(n) in p["by_level"] else p["by_level"][n])
            ax.plot(r, y, lw=1.6, color=SERIES[i], ls=["-", "--", "-.", ":"][i % 4])
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("r (GM/c²)", color=INK)
        ax.set_ylabel(ylab, color=INK)
        _style(ax)
    axs[0].legend([f"{n} × {n // 2}" for n in levels], frameon=False, fontsize=9)
    s = report["scalars"]["mdot_mean"]["by_level"]
    m = [s[str(n)]["mean"] if str(n) in s else s[n]["mean"] for n in levels]
    c = [s[str(n)]["ci95"] if str(n) in s else s[n]["ci95"] for n in levels]
    axs[2].errorbar(levels, m, yerr=c, fmt="o", color=SERIES[0], ms=6, capsize=4, lw=1.5)
    axs[2].set_xscale("log", base=2)
    axs[2].set_xticks(levels, [str(n) for n in levels])
    axs[2].set_xlabel("radial cells", color=INK)
    axs[2].set_ylabel("mean accretion rate (code units), 95 % CI", color=INK)
    _style(axs[2])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
