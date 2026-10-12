"""OpenModelica compiler (omc) from Python, without OMPython: build FMUs and simulate models from ``.mo`` files.

omc is used from the PATH when it is installed, otherwise from the official Docker image
(``openmodelica/openmodelica:<version>-minimal``; set ``PINNEAPPLE_OMC_IMAGE`` to change it). Only the
working directory is mounted in the container, so every ``.mo`` file must sit in (or under) it; the helpers
below copy them there.

    from pinneapple_simulation.external_solvers.modelica.omc import build_fmu, simulate_mo
    fmu = build_fmu("Tank.mo", "Tank", "build/")                       # FMI 2.0 Co-Simulation
    res = simulate_mo("Tank.mo", "Tank", stop_time=200, parameters={"Cd": 0.6})
    res["h"], res["time"]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

__all__ = ["OMC_IMAGE", "omc_available", "run_omc_script", "build_fmu", "simulate_mo"]

OMC_IMAGE = os.environ.get("PINNEAPPLE_OMC_IMAGE", "openmodelica/openmodelica:v1.24.4-minimal")

PathLike = str | Path


def _docker_image_present(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def omc_available() -> str | None:
    """``"local"`` when omc is on the PATH, ``"docker"`` when the image is present, else None."""
    if shutil.which("omc"):
        return "local"
    if _docker_image_present(OMC_IMAGE):
        return "docker"
    return None


def run_omc_script(script: str, workdir: PathLike, timeout: float = 1800.0) -> str:
    """Run an omc ``.mos`` script in ``workdir`` and return omc's output. Raises RuntimeError on failure or
    when omc reports an error string."""
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "_pinneapple.mos").write_text(script)
    how = omc_available()
    if how == "local":
        cmd = ["omc", "_pinneapple.mos"]
    elif how == "docker":
        cmd = ["docker", "run", "--rm", "-v", f"{work}:{work}", "-w", str(work)]
        if os.name == "posix":
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
        cmd += [OMC_IMAGE, "omc", "_pinneapple.mos"]
    else:
        raise RuntimeError("OpenModelica not found: install omc or pull the Docker image "
                           f"`docker pull {OMC_IMAGE}`")
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    if r.returncode != 0 or "Error:" in out:
        raise RuntimeError(f"omc failed (code {r.returncode}):\n{out[-3000:]}")
    return out


def _stage(models: PathLike | Sequence[PathLike], work: Path) -> list[str]:
    files = [models] if isinstance(models, (str, Path)) else list(models)
    names = []
    for f in files:
        f = Path(f).resolve()
        dst = work / f.name
        if f != dst:
            shutil.copy(f, dst)
        names.append(f.name)
    return names


def _load(names: Iterable[str], libraries: Sequence[str]) -> str:
    s = "".join(f"loadModel({lib}); getErrorString();\n" for lib in libraries)
    return s + "".join(f'loadFile("{n}"); getErrorString();\n' for n in names)


def build_fmu(models: PathLike | Sequence[PathLike], model_name: str, out_dir: PathLike,
              fmi_version: str = "2.0", fmu_type: str = "cs", libraries: Sequence[str] = ()) -> Path:
    """Compile ``model_name`` (defined in ``models``, one or several ``.mo`` files) into an FMU in ``out_dir``.

    ``fmu_type``: "cs" (Co-Simulation, with the solver inside the FMU), "me" (Model Exchange) or "me_cs".
    ``libraries``: Modelica libraries to load first, e.g. ``("Modelica",)``; the minimal image ships the
    Modelica Standard Library.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = _stage(models, out)
    prefix = model_name.replace(".", "_")
    script = _load(names, libraries) + (
        f'buildModelFMU({model_name}, version="{fmi_version}", fmuType="{fmu_type}", '
        f'fileNamePrefix="{prefix}"); getErrorString();\n')
    run_omc_script(script, out)
    fmu = out / f"{prefix}.fmu"
    if not fmu.exists():
        raise RuntimeError(f"omc reported no error but {fmu} was not written")
    for junk in out.glob(f"{prefix}_*"):          # build leftovers (sources, makefiles, info json)
        if junk.is_dir():
            shutil.rmtree(junk, ignore_errors=True)
    return fmu


def simulate_mo(models: PathLike | Sequence[PathLike], model_name: str, stop_time: float = 1.0,
                start_time: float = 0.0, number_of_intervals: int = 500, tolerance: float = 1e-6,
                parameters: dict[str, float] | None = None, variables: list[str] | None = None,
                libraries: Sequence[str] = (), workdir: PathLike | None = None) -> dict[str, np.ndarray]:
    """Simulate ``model_name`` with omc and return ``{variable: array}`` (with ``"time"``).
    ``parameters`` are applied with ``-override`` (no recompilation per value)."""
    from .om_runner import _read_om_result

    work = Path(workdir or tempfile.mkdtemp(prefix="omc_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = _stage(models, work)
    flags = ""
    if parameters:
        flags = ', simflags="-override=' + ",".join(f"{k}={v!r}" for k, v in parameters.items()) + '"'
    prefix = model_name.replace(".", "_")
    script = _load(names, libraries) + (
        f"simulate({model_name}, startTime={start_time}, stopTime={stop_time}, "
        f'numberOfIntervals={number_of_intervals}, tolerance={tolerance}, fileNamePrefix="{prefix}"{flags}); '
        "getErrorString();\n")
    out = run_omc_script(script, work)
    mat = work / f"{prefix}_res.mat"
    if not mat.exists():
        raise RuntimeError(f"omc wrote no result file:\n{out[-2000:]}")
    return _read_om_result(str(mat), variables)
