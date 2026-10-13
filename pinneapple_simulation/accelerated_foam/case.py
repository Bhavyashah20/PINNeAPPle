"""Small OpenFOAM cases for the pressure-solve benchmarks, a runner and a log parser.

The cases are written from scratch here (no tutorial files are copied):

- :func:`write_cavity` -- the lid-driven cavity, ``icoFoam`` (PISO), N x N cells, 2-D.

The pressure solver is set from a plain dictionary, so the same case can be run with OpenFOAM's
own solver and with the ``pinneWarmStart`` wrapper, at the same tolerance.

    from pinneapple_simulation.accelerated_foam import case
    case.write_cavity("cav_plain", n=64, end_time=0.25, p_solver=case.plain_solver("PCG"))
    run = case.run_case("cav_plain", "icoFoam")
    run.solves["p"].iterations          # total inner iterations of every p solve
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""

#: Places where an OpenFOAM environment script is usually found.
BASHRC_CANDIDATES = (
    "/usr/share/openfoam/etc/bashrc",  # Debian / Ubuntu ``openfoam`` package
    "/usr/lib/openfoam/openfoam/etc/bashrc",  # openfoam.com packages
    "/opt/openfoam/etc/bashrc",
)


def find_bashrc(bashrc: str | None = None) -> str | None:
    """The OpenFOAM environment script: ``bashrc`` if given, else ``$FOAM_BASHRC``, else a usual location."""
    for cand in (bashrc, os.environ.get("FOAM_BASHRC"), *BASHRC_CANDIDATES):
        if cand and Path(cand).is_file():
            return str(cand)
    return None


def openfoam_available(solver: str = "icoFoam") -> bool:
    """True when ``solver`` can be run (on PATH, or through an environment script)."""
    if shutil.which(solver):
        return True
    rc = find_bashrc()
    if not rc:
        return False
    r = subprocess.run(
        ["bash", "-c", f'source "{rc}" >/dev/null 2>&1; command -v {solver}'],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def _w(case: Path, rel: str, cls: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_HEADER.format(cls=cls, obj=p.name) + body)


def _dict_entries(d: dict, indent: str = "        ") -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, str) and (v == "" or "/" in v):
            v = f'"{v}"'
        lines.append(f"{indent}{k:<16}{v};")
    return "\n".join(lines)


def plain_solver(kind: str = "PCG", tolerance: float = 1e-6, rel_tol: float = 0.0, **extra) -> dict:
    """OpenFOAM's own pressure solver: ``PCG`` (DIC preconditioner) or ``GAMG`` (GaussSeidel smoother)."""
    d: dict = {"solver": kind}
    if kind == "PCG":
        d["preconditioner"] = "DIC"
    elif kind == "GAMG":
        d.update(
            smoother="GaussSeidel",
            nCellsInCoarsestLevel=10,
            agglomerator="faceAreaPair",
            mergeLevels=1,
            cacheAgglomeration="true",
        )
    d.update(tolerance=tolerance, relTol=rel_tol, **extra)
    return d


def warm_solver(
    inner: str = "PCG",
    tolerance: float = 1e-6,
    rel_tol: float = 0.0,
    guess: str = "projection",
    n_snapshots: int = 8,
    **extra,
) -> dict:
    """The ``pinneWarmStart`` wrapper around the same inner solver and tolerances as :func:`plain_solver`."""
    d = plain_solver(inner, tolerance, rel_tol)
    d.pop("solver")
    out = {
        "solver": "pinneWarmStart",
        "innerSolver": inner,
        "guess": guess,
        "nSnapshots": n_snapshots,
    }
    out.update(d)
    out.update(extra)
    return out


def write_cavity(
    case_dir,
    n: int = 64,
    end_time: float = 0.25,
    nu: float = 0.01,
    lid: float = 1.0,
    courant: float = 0.5,
    p_solver: dict | None = None,
    p_final_solver: dict | None = None,
    n_correctors: int = 2,
    libs: tuple[str, ...] = (),
    write_interval: float | None = None,
) -> Path:
    """Write a 2-D lid-driven cavity (unit square, ``n`` x ``n`` cells) run by ``icoFoam``.

    ``p_solver`` is the dictionary of the pressure solver for the first PISO correctors and ``p_final_solver``
    the one of the last corrector (default: the same with ``relTol 0``). The fields are written in ASCII
    with 17 significant digits so that two runs can be compared to round-off.
    """
    case = Path(case_dir)
    if case.exists():
        shutil.rmtree(case)
    p_solver = dict(p_solver or plain_solver("PCG"))
    p_final_solver = dict(p_final_solver or {**p_solver, "relTol": 0})
    dt = courant * (1.0 / n) / max(lid, 1e-12)
    n_steps = int(round(end_time / dt))
    dt = end_time / n_steps
    write_interval = end_time if write_interval is None else write_interval
    lib_line = ("libs (" + " ".join(f'"{s}"' for s in libs) + ");\n") if libs else ""

    _w(
        case,
        "system/blockMeshDict",
        "dictionary",
        f"""
convertToMeters 1;
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    movingWall {{ type wall; faces ( (3 7 6 2) ); }}
    fixedWalls {{ type wall; faces ( (0 4 7 3) (2 6 5 1) (1 5 4 0) ); }}
    frontAndBack {{ type empty; faces ( (0 3 2 1) (4 5 6 7) ); }}
);
""",
    )
    _w(
        case,
        "system/controlDict",
        "dictionary",
        f"""
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time!r};
deltaT          {dt!r};
writeControl    runTime;
writeInterval   {write_interval!r};
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable false;
{lib_line}""",
    )
    _w(
        case,
        "system/fvSchemes",
        "dictionary",
        """
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
""",
    )
    _w(
        case,
        "system/fvSolution",
        "dictionary",
        f"""
solvers
{{
    p
    {{
{_dict_entries(p_solver)}
    }}
    pFinal
    {{
{_dict_entries(p_final_solver)}
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }}
}}
PISO
{{
    nCorrectors     {n_correctors};
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
""",
    )
    _w(case, "constant/transportProperties", "dictionary", f"\nnu {nu!r};\n")
    _w(
        case,
        "0/U",
        "volVectorField",
        f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    movingWall {{ type fixedValue; value uniform ({lid!r} 0 0); }}
    fixedWalls {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _w(
        case,
        "0/p",
        "volScalarField",
        """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    movingWall { type zeroGradient; }
    fixedWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
""",
    )
    return case


# --------------------------------------------------------------------------------------------- running


def run_command(
    case_dir,
    command: str,
    bashrc: str | None = None,
    log_name: str | None = None,
    env: dict | None = None,
) -> tuple[str, float]:
    """Run an OpenFOAM command in ``case_dir``; return (log text, wall seconds). Raises on failure."""
    case = Path(case_dir).resolve()
    rc = find_bashrc(bashrc)
    prefix = f'source "{rc}" >/dev/null 2>&1; ' if rc else ""
    full_env = dict(os.environ)
    full_env.update(env or {})
    t0 = time.perf_counter()
    r = subprocess.run(
        ["bash", "-c", f'{prefix}cd "{case}" && {command}'],
        capture_output=True,
        text=True,
        env=full_env,
    )
    wall = time.perf_counter() - t0
    log = r.stdout + r.stderr
    if log_name:
        (case / log_name).write_text(log)
    if r.returncode != 0:
        raise RuntimeError(f"{command} failed in {case} (exit {r.returncode}):\n{log[-3000:]}")
    return log, wall


@dataclass
class FieldSolves:
    """Linear solves of one field, from a solver log."""

    n_solves: int = 0
    iterations: int = 0
    per_solve: list[int] = field(default_factory=list)
    initial_residuals: list[float] = field(default_factory=list)
    final_residuals: list[float] = field(default_factory=list)


@dataclass
class RunResult:
    case_dir: Path
    wall_time: float  # seconds, whole solver process
    execution_time: float  # OpenFOAM's last "ExecutionTime" (CPU seconds)
    solves: dict[str, FieldSolves]
    log: str = ""


_SOLVE_RE = re.compile(
    r"^(\S+):\s+Solving for (\S+), Initial residual = (\S+), Final residual = (\S+), "
    r"No Iterations (\d+)",
    re.M,
)
_EXEC_RE = re.compile(r"^ExecutionTime = (\S+) s", re.M)


def parse_log(log: str) -> tuple[dict[str, FieldSolves], float]:
    """Per-field linear solves and the final ExecutionTime of an OpenFOAM solver log."""
    out: dict[str, FieldSolves] = {}
    for m in _SOLVE_RE.finditer(log):
        fs = out.setdefault(m.group(2), FieldSolves())
        it = int(m.group(5))
        fs.n_solves += 1
        fs.iterations += it
        fs.per_solve.append(it)
        fs.initial_residuals.append(float(m.group(3)))
        fs.final_residuals.append(float(m.group(4)))
    times = _EXEC_RE.findall(log)
    return out, float(times[-1]) if times else float("nan")


def run_case(
    case_dir,
    solver: str = "icoFoam",
    bashrc: str | None = None,
    mesh: bool = True,
    env: dict | None = None,
) -> RunResult:
    """``blockMesh`` (if ``mesh``) then ``solver``; the solver log is parsed into a :class:`RunResult`."""
    if mesh:
        run_command(case_dir, "blockMesh", bashrc, "log.blockMesh")
    log, wall = run_command(case_dir, solver, bashrc, f"log.{solver}", env=env)
    solves, exec_time = parse_log(log)
    return RunResult(Path(case_dir), wall, exec_time, solves, log)


# --------------------------------------------------------------------------------------------- fields


def time_dirs(case_dir) -> list[str]:
    out = []
    for p in Path(case_dir).iterdir():
        try:
            float(p.name)
        except ValueError:
            continue
        if p.is_dir():
            out.append(p.name)
    return sorted(out, key=float)


def read_field(case_dir, name: str, time_name: str | None = None) -> np.ndarray:
    """Internal field ``name`` at ``time_name`` (default: latest), as a NumPy array."""
    from pinneapple_simulation.external_solvers.openfoam import binary_reader

    t = time_name or time_dirs(case_dir)[-1]
    path = Path(case_dir) / t / name
    arr, _uniform, _ncomp = binary_reader.read_internal_field(path.read_bytes(), str(path))
    return np.asarray(arr, dtype=float)


def field_difference(case_a, case_b, name: str, time_name: str | None = None) -> float:
    """Relative L2 difference ``||a - b|| / ||a||`` of a field between two runs at the same time."""
    a = read_field(case_a, name, time_name)
    b = read_field(case_b, name, time_name)
    if a.shape != b.shape:
        raise ValueError(f"{name}: shapes differ {a.shape} vs {b.shape}")
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b)) / (den if den > 0 else 1.0)
