"""Generate and read the accretion-flow simulations behind the black-hole weather forecaster (issue #399).

    from pinneapple_physics.blackhole import RIAFConfig
    from pinneapple_physics.blackhole.runs import simulate_run, load_run
    simulate_run(RIAFConfig(nr=256, ntheta=128, alpha=0.1, perturbation_kind="modes", seed=0), "data/bh/run0",
                 t_end=10000.0, every=10.0)
    run = load_run("data/bh/run0")        # {"t", "frames" (T, 5, nr, ntheta), "mdot", "grid", "config"}

A run starts from an equilibrium torus and saves the primitives (rho, v_r, v_theta, v_phi, p) every ``every`` M in
chunks ``frames_XXXX.npy``, with ``times.npy``, ``mdot.npy``, ``grid.npz``, ``config.json`` and a checkpoint
``state.pt``, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .hydro import AccretionFlow, RIAFConfig, torus_state

__all__ = ["simulate_run", "load_run"]


def load_run(folder: str) -> dict:
    """-> dict(t, frames (T, 5, nr, ntheta), mdot, grid, config)."""
    chunks = sorted(glob.glob(os.path.join(folder, "frames_*.npy")))
    frames = np.concatenate([np.load(c) for c in chunks]) if chunks else np.zeros((0,))
    t = np.load(os.path.join(folder, "times.npy"))
    mdot = np.load(os.path.join(folder, "mdot.npy"))
    n = min(len(frames), len(t))
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    return {"t": t[:n], "frames": frames[:n], "mdot": mdot[:n], "grid": dict(np.load(os.path.join(folder, "grid.npz"))),
            "config": cfg}


def simulate_run(cfg: RIAFConfig, folder: str, *, t_end: float = 10000.0, every: float = 10.0, chunk: int = 50,
                 name: str | None = None, log=print) -> str:
    """Run (or resume) one simulation into ``folder``; returns the folder."""
    name = name or os.path.basename(os.path.normpath(folder))
    os.makedirs(folder, exist_ok=True)
    flow = AccretionFlow(cfg)
    state_path = os.path.join(folder, "state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        flow.W.copy_(st["W"])
        flow.t, flow.boundary, flow.mdot_in = st["t"], st["boundary"], st["mdot"]
        times, mdot, k_chunk = list(st["times"]), list(st["mdots"]), st["k_chunk"]
        log(f"resuming {name} at t = {flow.t:.1f} M ({len(times)} frames)")
    else:
        flow.set_primitives(torus_state(flow))
        times, mdot, k_chunk = [0.0], [0.0], 0
        np.save(os.path.join(folder, f"frames_{k_chunk:04d}.npy"), flow.primitives()[None].astype(np.float32))
        k_chunk += 1
        np.savez(os.path.join(folder, "grid.npz"), **flow.grid())
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump({**{k: v for k, v in asdict(cfg).items() if k not in ("dtype",)}, "every": every,
                       "t_end": t_end, "name": name}, f, indent=1)

    buf, t0 = [], time.time()
    t_next = times[-1] + every
    while flow.t < t_end - 1e-9:
        flow.step(min(flow.dt(), t_next - flow.t))
        if not np.isfinite(flow.mdot_in):
            raise FloatingPointError(f"{name}: non-finite state at t = {flow.t:.2f}")
        if flow.t >= t_next - 1e-9:
            buf.append(flow.primitives().astype(np.float32))
            times.append(flow.t)
            mdot.append(flow.mdot_in)
            t_next += every
            if len(buf) == chunk or flow.t >= t_end - 1e-9:
                np.save(os.path.join(folder, f"frames_{k_chunk:04d}.npy"), np.stack(buf))
                k_chunk += 1
                buf = []
                np.save(os.path.join(folder, "times.npy"), np.asarray(times))
                np.save(os.path.join(folder, "mdot.npy"), np.asarray(mdot))
                torch.save({"W": flow.W.clone(), "t": flow.t, "boundary": dict(flow.boundary), "mdot": flow.mdot_in,
                            "times": times, "mdots": mdot, "k_chunk": k_chunk}, state_path)
                el = time.time() - t0
                log(f"{name}: t = {flow.t:8.1f} M  frames {len(times)}  mdot {flow.mdot_in:.4g}  "
                    f"capped {flow.capped}  wall {el/60:.1f} min")
    log(f"{name}: done")
    return folder
