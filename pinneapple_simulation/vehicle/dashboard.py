"""The animation-centred dashboard of a vehicle run (built on :mod:`pinneapple_tools.visualization.dashboard`).

    from pinneapple_simulation.vehicle import dashboard as vdash
    frames, ks = vdash.render_frames(run, terrain, n=150)                   # rendered animation (planetary renderer)
    db = vdash.vehicle_dashboard(run, frames, ks, model=WongReeceModel(...), surrogate=sur, scene3d=True,
                                 terrain=terrain)
    db.to_html("rover.html"); db.to_png("rover.png"); db.to_video(gif="rover.gif", mp4="rover.mp4")

Charts (units on every axis):

* left -- drawbar pull, driving torque and sinkage versus slip: the single-wheel testbed curve at the static wheel
  load (``model.steady_state``) and each wheel's operating point moving with the cursor;
* right -- speed (and the command), slip per wheel, the PINN against the Wong-Reece reference for one wheel;
* bottom -- per-wheel sinkage and normal load along the traverse, the PINN error for every wheel.

Tiles: distance, mean slip, max sinkage and drive energy so far, PINN error so far. The parameter panel lists the
vehicle, the soil, the command and the surrogate's held-out error.
"""
from __future__ import annotations

import numpy as np

from pinneapple_tools.visualization.dashboard import (
    Chart,
    ColorBar,
    Dashboard,
    Marker,
    Param,
    Series,
    Tile,
)


def render_frames(run, terrain, n: int = 150, *, field="normal_stress", clim=None, size=(1280, 720), camera=None,
                  preset="lunar_low_sun", sun_azimuth_deg=160.0, earth=False, trail: float = 0.0,
                  half_window=(3.2, 3.4), progress=None):
    """``n`` rendered frames spread over the run and their sample indices (``vehicle.viz.render_frame``).
    ``trail`` and ``half_window`` place the detailed terrain window (see ``vehicle.viz.frame_view``)."""
    from .viz import render_frame
    ks = np.unique(np.linspace(0, len(run.t) - 1, n).astype(int))
    if clim is None and field in ("normal_stress", "shear_stress"):
        key = "sigma" if field == "normal_stress" else "tau"
        ter_end = terrain.copy_empty()
        run.replay_terrain(ter_end, 1e9)
        g = 1e-3 * getattr(ter_end, key)
        clim = (0.0, float(np.percentile(g[g > 0], 99))) if (g > 0).any() else (0.0, 1.0)
    cache: dict = {}
    frames = []
    for j, k in enumerate(ks):
        img = render_frame(run, terrain, int(k), None, field=field, clim=clim, camera=camera, size=size, cache=cache,
                           preset=preset, sun_azimuth_deg=sun_azimuth_deg, earth=earth, title="", colorbar=False,
                           trail=trail, half_window=half_window)
        frames.append(np.asarray(img)[..., :3])
        if progress:
            progress(j + 1, len(ks))
    return frames, ks, clim


def _running(fn, a):
    return np.array([fn(a[: i + 1]) for i in range(len(a))])


def vehicle_dashboard(run, frames, ks, *, model=None, surrogate=None, title: str | None = None, subtitle: str = "",
                      field_label="Peak normal stress in the ruts [kPa]", cmap="inferno", clim=(0.0, 1.0),
                      wheel_for_pinn: int = 0, extra_params=(), scene3d: bool = False, terrain=None,
                      footer: str = "") -> Dashboard:
    """Build the dashboard of ``run`` with the rendered ``frames`` at sample indices ``ks``."""
    ks = np.asarray(ks, int)
    spec = run.model.spec
    names = run.corner_names
    W = run.wheel
    t = run.t[ks]
    dist = run.distance()
    charts: list[Chart] = []
    # --- left: single-wheel curves and the live operating points
    if model is not None:
        load = spec.cad.total()["mass"] * run.meta.get("g", 1.62) / len(names)
        s = np.linspace(0.0, 0.8, 41)
        tb = model.steady_state(np.full_like(s, load), s)
        for key, q, lab, sc in (("drawbar_pull", "drawbar_pull", "Drawbar pull [N]", 1.0),
                                ("torque", "drive_torque", "Driving torque [N m]", 1.0),
                                ("sinkage", "sinkage", "Sinkage [mm]", 1e3)):
            charts.append(Chart(f"{lab.split(' [')[0]} vs slip", lab, "slip [-]",
                                [Series(f"testbed {load:.0f} N", sc * tb[key], x=s, color="#7d8794", width=2.2)],
                                [Marker(n_, np.clip(W["slip"][ks, i], -0.2, 1.0), sc * W[q][ks, i])
                                 for i, n_ in enumerate(names)], kind="phase", slot="left"))
    # --- right: speed, slip, PINN vs reference
    cmd = run.meta.get("speed_cmd")
    sp = [Series("speed", run.chassis["speed"][ks])]
    if cmd is not None:
        sp.append(Series("command", np.full(len(ks), float(cmd)), dash=True, color="#7d8794"))
    charts.append(Chart("Speed", "speed [m/s]", "time [s]", sp, slot="right"))
    charts.append(Chart("Slip", "slip [-]", "time [s]", [Series(n_, W["slip"][ks, i]) for i, n_ in enumerate(names)],
                        slot="right", ylim=(-0.3, 1.0)))
    pin_err = None
    if surrogate is not None and model is not None:
        from pinneapple_simulation.terramechanics.surrogate import compare_on_states
        c = compare_on_states(surrogate, model, W["slip"][ks].ravel(), W["sinkage"][ks].ravel())
        inside = c["inside"].reshape(len(ks), -1)
        ref = np.where(inside, c["reference"][:, 0].reshape(len(ks), -1), np.nan)   # only inside the training box
        pr = np.where(inside, c["surrogate"][:, 0].reshape(len(ks), -1), np.nan)
        i = wheel_for_pinn
        charts.append(Chart(f"PINN vs Wong-Reece, {names[i]} drawbar pull", "drawbar pull [N]", "time [s]",
                            [Series("Wong-Reece", ref[:, i], color="#e8e8e8", width=2.6),
                             Series("PINN", pr[:, i], color="#d95926", dash=True)], slot="right",
                            note="same (slip, sinkage) inputs to both; gaps: outside the training box"))
        pin_err = np.abs(pr - ref)
    # --- bottom: along the traverse
    charts.append(Chart("Sinkage per wheel along the traverse", "sinkage [mm]", "distance [m]",
                        [Series(n_, 1e3 * W["sinkage"][ks, i]) for i, n_ in enumerate(names)], x=dist[ks],
                        slot="bottom"))
    charts.append(Chart("Normal load per wheel along the traverse", "normal load [N]", "distance [m]",
                        [Series(n_, W["load"][ks, i]) for i, n_ in enumerate(names)], x=dist[ks], slot="bottom"))
    if pin_err is not None:
        charts.append(Chart("PINN error |PINN - Wong-Reece|, drawbar pull", "error [N]", "time [s]",
                            [Series(n_, pin_err[:, i]) for i, n_ in enumerate(names)], slot="bottom"))
    # --- tiles
    e = run.energy()
    moving = run.chassis["speed"] > 0.02
    slip_run = np.array([W["slip"][: k + 1][moving[: k + 1]].mean() if moving[: k + 1].any() else 0.0 for k in ks])
    tiles = [Tile("Distance", dist[ks], "m", "{:.2f}"), Tile("Mean slip", slip_run, "-", "{:.3f}"),
             Tile("Max sinkage", 1e3 * np.array([W["sinkage"][: k + 1].max() for k in ks]), "mm", "{:.1f}"),
             Tile("Drive energy", e[ks], "J", "{:.0f}")]
    if pin_err is not None:
        tiles.append(Tile("PINN error (rms)", np.sqrt(_running(lambda a: np.nanmean(a ** 2) if np.isfinite(a).any()
                                                               else 0.0, pin_err)), "N", "{:.2f}",
                          note="drawbar pull, all wheels, inside the training box, up to the cursor"))
    # --- parameters
    soil = model.soil if model is not None else None
    tot = spec.cad.total()
    params = [Param("Vehicle", spec.name), Param("Mass", round(tot["mass"], 1), "kg"),
              Param("Wheel radius / width", f"{spec.wheel.radius:.2f} / {spec.wheel.width:.2f}", "m"),
              Param("Grousers", f"{spec.wheel.grousers.count} x {1e3 * spec.wheel.grousers.height:.0f}", "mm"),
              Param("Gravity", run.meta.get("g", 1.62), "m/s²")]
    if soil is not None:
        params += [Param("Soil", soil.name), Param("Cohesion c", soil.c / 1e3, "kPa"),
                   Param("Friction angle", soil.phi_deg, "deg"), Param("k_phi", soil.k_phi / 1e3, "kN/m^(n+2)"),
                   Param("Sinkage exponent n", soil.n0), Param("Shear modulus K", 1e3 * soil.K, "mm")]
    if cmd is not None:
        params.append(Param("Speed command", cmd, "m/s"))
    params.append(Param("Time step", 1e3 * run.meta.get("dt", 0.002), "ms"))
    if surrogate is not None:
        te = surrogate.meta.get("test_errors", {}).get("Fx")
        if te:
            params.append(Param("PINN held-out rmse (Fx)", round(te["rmse_N"], 3), "N"))
    params += list(extra_params)
    db = Dashboard(title or f"{spec.name} on the Moon", t, list(frames), charts=charts, tiles=tiles, params=params,
                   subtitle=subtitle, colorbar=ColorBar(field_label, cmap, *clim), footer=footer)
    if scene3d:
        db.scene3d = vehicle_scene3d(run, ks, terrain)
    return db


def vehicle_scene3d(run, ks, terrain=None, cell: float = 0.03, preset="lunar_low_sun") -> dict:
    """Live-3D payload: CAD bodies (merged per material, body frames), their poses at ``ks`` and the final terrain
    (ruts, marks) shaded by the Sun."""
    from pinneapple_simulation.multibody.tree import quat_from_matrix
    from pinneapple_tools.visualization.dashboard.html import scene3d_payload
    from pinneapple_tools.visualization.studio.lighting import get_preset, threejs_config
    from pinneapple_tools.visualization.studio.scene import MATERIALS
    from pinneapple_tools.visualization.studio.shadows import shade_terrain, terrain_self_shadow
    model = run.model
    bodies = {}
    for b in model.tree.bodies:
        groups: dict[str, list] = {}
        for p in model.spec.cad.parts:
            if p.body == b.name:
                groups.setdefault(p.material, []).append((p.vertices - model.origins[b.name], p.faces))
        parts = []
        for mat, items in groups.items():
            off, V, F = 0, [], []
            for v, f in items:
                V.append(v)
                F.append(f + off)
                off += len(v)
            parts.append((np.concatenate(V), np.concatenate(F), MATERIALS.get(mat, MATERIALS["grey"])["color"]))
        bodies[b.name] = parts
    poses = {b.name: (np.zeros((len(ks), 3)), np.zeros((len(ks), 4))) for b in model.tree.bodies}
    for j, k in enumerate(ks):
        kin = model.tree.kinematics(run.q[k])
        for i, b in enumerate(model.tree.bodies):
            poses[b.name][0][j] = kin.p[i]
            poses[b.name][1][j] = quat_from_matrix(kin.R[i])
    ter3 = None
    if terrain is not None:
        ter = terrain.copy_empty()
        run.replay_terrain(ter, 1e9)
        xy = run.q[:, :2]
        x0, x1 = xy[:, 0].min() - 2.5, xy[:, 0].max() + 2.5
        y0, y1 = xy[:, 1].min() - 2.0, xy[:, 1].max() + 2.0
        X, Y = np.meshgrid(np.arange(max(x0, ter.x0), min(x1, ter.x1), cell),
                           np.arange(max(y0, ter.y0), min(y1, ter.y1), cell))
        Z = ter.surface(X, Y)
        sun = get_preset(preset).sun_direction()
        rgb = shade_terrain(X, Y, Z, sun, (X.mean() - 5, Y.mean() - 6, Z.mean() + 3), (0.42, 0.41, 0.39),
                            terrain_self_shadow(Z, cell, sun), exposure=2.2, fill=0.08)
        ter3 = (X, Y, Z, rgb)
    return scene3d_payload(bodies, poses, ter3, threejs_config(preset))


__all__ = ["render_frames", "vehicle_dashboard", "vehicle_scene3d"]

