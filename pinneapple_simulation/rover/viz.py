"""Plots, animations and the 3D twin view of a rover :class:`~.dynamics.Trajectory`.

* :func:`plot_wheel_series` -- per wheel vs distance: sinkage, slip, drawbar pull, normal load, torque, power (PNG).
* :func:`plot_rover_series` -- rover pitch, roll and cumulative drive energy vs distance for several runs (PNG).
* :func:`animate` -- matplotlib 3D animation on the terrain, wheels coloured by sinkage or slip (GIF, and MP4 when
  ffmpeg is available).
* :func:`twin_scene` -- a ``pinneapple_twin3d`` scene (terrain + moving body, rockers, bogies and wheels with the
  wheel sinkage and slip as per-step fields) for the bundled three.js viewer; ``.export(folder)`` writes it.
"""
from __future__ import annotations

import math
import shutil

import numpy as np

from .geometry import WHEEL_NAMES, link_frames, quat_from_matrix, rot_y, rover_meshes

# categorical slots 1-3 of the reference palette: front / middle / rear; left solid, right dashed
POS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES = {"sinkage": ("Sinkage", "mm", 1e3), "slip": ("Slip", "-", 1.0), "drawbar_pull": ("Drawbar pull", "N", 1.0),
          "normal_load": ("Normal load", "N", 1.0), "torque": ("Drive torque", "N m", 1.0),
          "power": ("Drive power", "W", 1.0)}


def _style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9893")
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_wheel_series(traj, path: str, title: str = "", x: str = "distance") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wt = traj.wheel_table()
    xs = traj.distance() if x == "distance" else traj.t
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for ax, (key, (lab, unit, k)) in zip(axes.ravel(), SERIES.items(), strict=True):
        for i, name in enumerate(WHEEL_NAMES):
            ax.plot(xs[1:], k * wt[key][1:, i], color=POS_COLORS[i % 3], lw=1.6, ls="-" if i < 3 else "--",
                    label=name.replace("_", " "))
        ax.set_title(f"{lab} [{unit}]", fontsize=10, color="#0b0b0b", loc="left")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("distance [m]" if x == "distance" else "time [s]", fontsize=9, color="#52514e")
    axes[0, 0].legend(fontsize=7.5, ncol=2, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rover_series(trajs: dict, path: str, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (name, tr) in enumerate(trajs.items()):
        d = tr.distance()
        c = pal[j % len(pal)]
        axes[0].plot(d, np.degrees(tr.pose_array("pitch")), color=c, lw=1.8, label=name)
        axes[1].plot(d, np.degrees(tr.pose_array("roll")), color=c, lw=1.8, label=name)
        axes[2].plot(d, tr.energy(), color=c, lw=1.8, label=name)
    for ax, t in zip(axes, ("Pitch, nose up [deg]", "Roll, left side up [deg]", "Drive energy [J]"), strict=True):
        ax.set_title(t, fontsize=10, loc="left")
        ax.set_xlabel("distance [m]", fontsize=9, color="#52514e")
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _wheel_spin(traj) -> np.ndarray:
    om = traj.array("omega")
    t = traj.t
    return np.r_[0.0, np.cumsum(om[1:] * np.diff(t))]


def _parts_at(traj, k, spin, meshes):
    """World-frame (name, V, F) of every rover part at state k."""
    st = traj.states[k]
    fr = link_frames(traj.rover, st.pose)
    out = [("body", meshes["body"][0] @ fr["R"].T + fr["origin"], meshes["body"][1])]
    for j, s in enumerate(("L", "R")):
        out.append((f"rocker_{s}", meshes[f"rocker_{s}"][0] @ fr["R_rocker"][j].T + fr["rocker_pivot"][j],
                    meshes[f"rocker_{s}"][1]))
        out.append((f"bogie_{s}", meshes[f"bogie_{s}"][0] @ fr["R_bogie"][j].T + fr["bogie_pivot"][j],
                    meshes[f"bogie_{s}"][1]))
    for i, name in enumerate(WHEEL_NAMES):
        Rw = (fr["R_bogie"][i // 3] if i % 3 else fr["R_rocker"][i // 3]) @ rot_y(spin[k])
        out.append((name, meshes["wheel"][0] @ Rw.T + fr["wheel"][i], meshes["wheel"][1]))
    return out


def animate(traj, path_gif: str, *, color_by: str = "sinkage", path_mp4: str | None = None, frames: int = 80,
            window: tuple = (1.7, 1.1), fps: int = 12, title: str = "", view=(26, -58), clim=None) -> dict:
    """3D animation following the rover over the terrain (``window``: half sizes of the terrain patch shown
    around it [m]); wheels coloured by ``color_by`` ("sinkage" [mm] or "slip")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm, colors
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    terrain = traj.terrain
    xy = np.array([[s.pose.x, s.pose.y] for s in traj.states])
    vals = traj.array("sinkage") * 1e3 if color_by == "sinkage" else traj.array("slip")
    lo, hi = clim if clim is not None else (float(np.nanmin(vals)), float(np.nanmax(vals)))
    norm = colors.Normalize(lo, hi if hi > lo else lo + 1e-9)
    cmap = matplotlib.colormaps["plasma"]
    meshes = rover_meshes(traj.rover)
    spin = _wheel_spin(traj)
    ks = np.unique(np.linspace(0, len(traj.states) - 1, min(frames, len(traj.states))).astype(int))
    ls = LightSource(azdeg=300, altdeg=35)
    wx, wy = window

    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    unit = "mm" if color_by == "sinkage" else "-"
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, pad=0.0,
                 label=f"wheel {color_by} [{unit}]")
    fig.subplots_adjust(left=0.0, right=0.95, top=0.92, bottom=0.0)

    def draw(k):
        ax.cla()
        c = traj.states[k]
        x0, x1, y0, y1 = c.pose.x - wx, c.pose.x + wx, c.pose.y - wy, c.pose.y + wy
        X, Y, Z = terrain.grid((x0, x1), (y0, y1), (70, 46))
        rgb = ls.shade(Z, cmap=matplotlib.colormaps["Greys_r"], vert_exag=6, blend_mode="soft",
                       vmin=Z.min() - 0.25, vmax=Z.max() + 0.45)
        ax.plot_surface(X, Y, Z, facecolors=rgb, linewidth=0, antialiased=False, shade=False, rstride=1,
                        cstride=1, zorder=0)
        m = (xy[:, 0] > x0) & (xy[:, 0] < x1)
        ax.plot(xy[m, 0], xy[m, 1], terrain.height(xy[m, 0], xy[m, 1]) + 0.004, color="#2a78d6", lw=1.0, zorder=1)
        for name, V, F in _parts_at(traj, k, spin, meshes):
            if name in WHEEL_NAMES:
                fc = cmap(norm(vals[k, WHEEL_NAMES.index(name)]))
            elif name == "body":
                fc = "#e9e7e2"
            else:
                fc = "#6d6b66"
            ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, linewidth=0, zorder=5,
                                                 shade=True, lightsource=ls))
        zc = float(terrain.height(c.pose.x, c.pose.y))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(zc - 0.6, zc + 0.9)
        ax.set_box_aspect((2 * wx, 2 * wy, 1.5), zoom=1.45)
        ax.view_init(*view)
        ax.set_axis_off()
        ax.set_title(f"{title}\nt = {c.t:5.1f} s   x = {c.pose.x:4.2f} m   pitch {math.degrees(c.pose.pitch):5.1f} deg"
                     f"   roll {math.degrees(c.pose.roll):5.1f} deg", fontsize=9, loc="left")

    anim = animation.FuncAnimation(fig, draw, frames=ks, interval=1000 / fps)
    anim.save(path_gif, writer=animation.PillowWriter(fps=fps))
    out = {"gif": path_gif}
    if path_mp4 and shutil.which("ffmpeg"):
        anim.save(path_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
        out["mp4"] = path_mp4
    plt.close(fig)
    return out


_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # z-up -> y-up (viewer)


def twin_scene(traj, title: str = "Lunar rover", margin: float = 1.5, grid=(160, 100)):
    """A ``pinneapple_twin3d.Scene`` of the run (y up for the viewer): terrain coloured by height, links and wheels
    moving with the trajectory, wheel fields ``sinkage`` [mm] and ``slip`` [-] per step."""
    from pinneapple_twin3d import Scene
    terrain = traj.terrain
    xy = np.array([[s.pose.x, s.pose.y] for s in traj.states])
    xr = (xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    yr = (xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    V, F = terrain.mesh(xr, yr, grid)
    sc = Scene(title, times=traj.t, source="pinneapple_simulation.rover")
    sc.add_part("terrain", V @ _C.T, F, group="terrain", color=(0.55, 0.54, 0.52))
    sc.add_field("terrain", "height", V[:, 2], unit="m")
    meshes = rover_meshes(traj.rover)
    spin = _wheel_spin(traj)
    T = len(traj.states)
    poses = {n: (np.zeros((T, 3)), np.zeros((T, 4))) for n in
             ["body", "rocker_L", "rocker_R", "bogie_L", "bogie_R", *WHEEL_NAMES]}
    for k, st in enumerate(traj.states):
        fr = link_frames(traj.rover, st.pose)
        frames = {"body": (fr["R"], fr["origin"])}
        for j, s in enumerate(("L", "R")):
            frames[f"rocker_{s}"] = (fr["R_rocker"][j], fr["rocker_pivot"][j])
            frames[f"bogie_{s}"] = (fr["R_bogie"][j], fr["bogie_pivot"][j])
        for i, n in enumerate(WHEEL_NAMES):
            Rl = fr["R_bogie"][i // 3] if i % 3 else fr["R_rocker"][i // 3]
            frames[n] = (Rl @ rot_y(spin[k]), fr["wheel"][i])
        for n, (R, p) in frames.items():
            poses[n][0][k] = _C @ p
            poses[n][1][k] = quat_from_matrix(_C @ R @ _C.T)
    colors = {"body": (0.92, 0.91, 0.88)}
    sink = traj.array("sinkage") * 1e3
    slip = traj.array("slip")
    for n in poses:
        mesh = meshes["wheel"] if n in WHEEL_NAMES else meshes[n]
        sc.add_part(n, mesh[0] @ _C.T, mesh[1], group="wheels" if n in WHEEL_NAMES else "rover",
                    color=colors.get(n, (0.42, 0.42, 0.40)))
        sc.add_motion(n, *poses[n])
        if n in WHEEL_NAMES:
            i = WHEEL_NAMES.index(n)
            nv = len(mesh[0])
            sc.add_field(n, "sinkage", np.repeat(sink[:, i:i + 1], nv, 1), unit="mm")
            sc.add_field(n, "slip", np.repeat(slip[:, i:i + 1], nv, 1), unit="-")
    return sc


__all__ = ["plot_wheel_series", "plot_rover_series", "animate", "twin_scene"]
