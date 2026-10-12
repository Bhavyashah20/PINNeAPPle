"""Bridge to GeoTaichi (MPM, DEM / level-set DEM, FEM, IGA and their couplings; Taichi; GPL-3.0).

GeoTaichi is run as a separate program: PINNeAPPle never imports or ships its code. Point ``GEOTAICHI_HOME`` at a
GeoTaichi checkout (installed with its own requirements), run one of its scripts, and read the particle frames it
writes (``<path>/particles/MPMParticle<NNNNNN>.npz`` or ``DEMParticle``) into arrays or PhysicalSamples for
training and checks. The same reader reads the frames of PINNeAPPle's own ``geo_mpm`` solver, so a GeoTaichi run
and a PINNeAPPle run of the same problem can be compared point for point.

    from pinneapple_simulation.external_solvers.geotaichi import run_geotaichi, read_geotaichi, frames_to_samples
    run_geotaichi("examples/mpm/column_collapse/DPmaterial.py", workdir="runs/collapse")
    frames = read_geotaichi("runs/collapse/OutputData")          # list of {position, velocity, volume, ...}
    samples = frames_to_samples(frames, source="geotaichi")
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

import numpy as np

from pinneapple_simulation.particle_dynamics.geo_mpm import read_particle_frames

__all__ = ["geotaichi_home", "geotaichi_available", "run_geotaichi", "read_geotaichi", "frames_to_samples"]


def geotaichi_home() -> str | None:
    h = os.environ.get("GEOTAICHI_HOME")
    return h if h and os.path.isdir(os.path.join(h, "geotaichi")) else None


def geotaichi_available() -> bool:
    """True when a GeoTaichi checkout is configured and Taichi is importable by the interpreter that will run it."""
    if geotaichi_home() is None:
        return False
    r = subprocess.run([sys.executable, "-c", "import taichi"], capture_output=True)
    return r.returncode == 0


def run_geotaichi(script: str, workdir: str, args: Sequence[str] = (), python: str | None = None,
                  timeout: float = 24 * 3600, log: str = "log.geotaichi") -> subprocess.CompletedProcess:
    """Run a GeoTaichi script (a path, absolute or relative to GEOTAICHI_HOME) in ``workdir``, in its own process.
    Output and errors go to ``workdir/<log>``; raises RuntimeError on a non-zero exit."""
    home = geotaichi_home()
    if home is None:
        raise RuntimeError("set GEOTAICHI_HOME to a GeoTaichi checkout (https://github.com/Yihao-Shi/GeoTaichi)")
    path = script if os.path.isabs(script) else os.path.join(home, script)
    os.makedirs(workdir, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = home + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(os.path.join(workdir, log), "w") as f:
        r = subprocess.run([python or sys.executable, path, *args], cwd=workdir, env=env, stdout=f,
                           stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"GeoTaichi exited with {r.returncode}; see {os.path.join(workdir, log)}")
    return r


def read_geotaichi(path: str, kind: str = "MPM", allow_pickle: bool = False) -> list[dict]:
    """Particle frames under ``path`` (the run's output directory, which holds ``particles/``). GeoTaichi stores
    some entries as pickled objects; they are skipped unless ``allow_pickle=True`` (only for files you trust)."""
    frames = read_particle_frames(path, kind=kind, allow_pickle=allow_pickle)
    if not frames:
        raise FileNotFoundError(f"no {kind}Particle*.npz under {os.path.join(path, 'particles')}")
    return frames


def frames_to_samples(frames: Sequence[dict], source: str = "geotaichi", fields: Sequence[str] | None = None,
                      times: Sequence[float] | None = None) -> list:
    """One PhysicalSample per frame: particle positions as the point cloud, the other per-particle arrays as fields
    (velocity, volume, stress, ...)."""
    import torch

    from pinneapple_data.physical_sample import PhysicalSample

    out = []
    for i, fr in enumerate(frames):
        pos = np.asarray(fr["position"], float)
        keep = fields or [k for k, v in fr.items() if k != "position" and np.ndim(v) >= 1 and len(v) == len(pos)
                          and np.asarray(v).dtype != object]
        state = {k: torch.as_tensor(np.asarray(fr[k], float), dtype=torch.float32) for k in keep}
        t = float(times[i]) if times is not None else float(np.asarray(fr.get("time", i)))
        out.append(PhysicalSample(state=state, domain={"type": "point_cloud",
                                                        "coords": {"position": torch.as_tensor(pos)}},
                                  provenance={"version": "0.1", "source": source, "frame": i, "time": t},
                                  schema={"units": {"position": "m", "velocity": "m/s", "stress": "Pa"}}))
    return out
