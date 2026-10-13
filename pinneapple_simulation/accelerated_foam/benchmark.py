"""The comparisons behind ``docs/core_concepts/openfoam_acceleration.md``.

- :func:`openfoam_comparison` -- the same case run by plain OpenFOAM and with ``pinneWarmStart``, same inner
  solver and tolerances; iterations and times from the logs, field differences against the plain run and
  against a reference solved to a much tighter tolerance.
- :func:`offline_benchmark` -- the pressure systems of one run, re-solved in Python with each combination of
  initial guess and solver, at OpenFOAM's stopping test.
- :func:`markdown_table` -- format rows for the documentation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import case as fcase
from .ldu import LduSystem, normalised_residual
from .solvers import pcg, torch_pcg
from .warm_start import ProjectionWarmStart


def _run(case_dir, writer_kwargs: dict, solver: str, repeats: int) -> fcase.RunResult:
    best = None
    for _ in range(max(1, repeats)):
        fcase.write_cavity(case_dir, **writer_kwargs)
        r = fcase.run_case(case_dir, solver)
        if best is None or r.wall_time < best.wall_time:
            best = r
    return best


def openfoam_comparison(
    work_dir,
    lib,
    n: int = 128,
    end_time: float = 0.5,
    inners=("PCG", "GAMG"),
    tolerance: float = 1e-6,
    rel_tol: float = 0.05,
    n_snapshots: int = 8,
    repeats: int = 1,
    ref_tolerance: float = 1e-11,
    field_p: str = "p",
) -> list[dict]:
    """Lid-driven cavity (``icoFoam``), plain vs warm-started pressure solves; one row per run."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base = {"n": n, "end_time": end_time, "libs": (str(lib),)}
    ref_dir = work / f"cavity{n}_reference"
    ref_solver = fcase.plain_solver("PCG", ref_tolerance, 0.0)
    _run(ref_dir, {**base, "p_solver": ref_solver, "p_final_solver": ref_solver}, "icoFoam", 1)
    rows = []
    for inner in inners:
        plain_dir = work / f"cavity{n}_{inner}_plain"
        warm_dir = work / f"cavity{n}_{inner}_warm"
        runs = {
            "plain": (
                plain_dir,
                _run(
                    plain_dir,
                    {**base, "p_solver": fcase.plain_solver(inner, tolerance, rel_tol)},
                    "icoFoam",
                    repeats,
                ),
            ),
            "warm": (
                warm_dir,
                _run(
                    warm_dir,
                    {
                        **base,
                        "p_solver": fcase.warm_solver(
                            inner, tolerance, rel_tol, n_snapshots=n_snapshots
                        ),
                    },
                    "icoFoam",
                    repeats,
                ),
            ),
        }
        for kind, (d, r) in runs.items():
            ps = r.solves[field_p]
            rows.append(
                {
                    "case": f"cavity {n}x{n}",
                    "inner": inner,
                    "run": kind,
                    "p_solves": ps.n_solves,
                    "p_iterations": ps.iterations,
                    "wall_s": round(r.wall_time, 3),
                    "exec_s": r.execution_time,
                    "dp_vs_plain": fcase.field_difference(plain_dir, d, "p"),
                    "dU_vs_plain": fcase.field_difference(plain_dir, d, "U"),
                    "dp_vs_ref": fcase.field_difference(ref_dir, d, "p"),
                    "dU_vs_ref": fcase.field_difference(ref_dir, d, "U"),
                }
            )
    return rows


def _foam_dict(case_dir, file: str, entry: str, value: str, bashrc: str | None = None) -> None:
    import shlex

    fcase.run_command(
        case_dir,
        f"foamDictionary {file} -entry {shlex.quote(entry)} -set {shlex.quote(value)}",
        bashrc,
    )


def _find_block(text: str, name: str) -> tuple[int, int]:
    """(start of the body after ``{``, index of the matching ``}``) of the dictionary ``name``."""
    import re

    m = re.search(r"(^|\n)\s*" + re.escape(name) + r"\s*\{", text)
    if m is None:
        raise KeyError(f"no dictionary {name!r}")
    i = depth = m.end()
    depth = 1
    while depth:
        c = text[i]
        depth += (c == "{") - (c == "}")
        i += 1
    return m.end(), i - 1


def solver_entries(fv_solution: str, field_p: str) -> dict:
    """The ``key value;`` entries of ``solvers { <field_p> { ... } }`` in an fvSolution text."""
    lo, hi = _find_block(fv_solution, "solvers")
    blo, bhi = _find_block(fv_solution[lo:hi], field_p)
    out = {}
    for stmt in fv_solution[lo + blo : lo + bhi].split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("$"):
            k, _, v = stmt.partition(" ")
            out[k.strip()] = v.strip()
    return out


def set_solver_entries(fv_solution: str, field_p: str, entries: dict) -> str:
    """Replace the body of ``solvers { <field_p> {...} }``; other entries (``$p`` in ``pFinal``) are untouched."""
    lo, hi = _find_block(fv_solution, "solvers")
    blo, bhi = _find_block(fv_solution[lo:hi], field_p)
    body = "\n" + "".join(f"        {k:<16}{v};\n" for k, v in entries.items()) + "    "
    return fv_solution[: lo + blo] + body + fv_solution[lo + bhi :]


def prepare_tutorial(
    tutorial_dir,
    case_dir,
    p_solver: dict,
    field_p: str,
    libs=(),
    setup=("blockMesh",),
    end_time: float | None = None,
    edit=None,
    bashrc: str | None = None,
    drop_functions: bool = True,
) -> Path:
    """Copy an installed tutorial, set the solver of ``field_p`` (the ``Final`` entry inherits it when the
    tutorial writes it as ``$p``), write fields in ASCII with 17 digits, and run the ``setup`` commands.
    ``drop_functions`` removes the tutorial's function objects (post-processing only; some packaged builds
    fail on them)."""
    import shutil

    case = Path(case_dir)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(tutorial_dir, case, symlinks=True)
    if (case / "0.orig").is_dir() and not (case / "0").exists():
        shutil.copytree(case / "0.orig", case / "0")
    if edit is not None:
        edit(case)
    sets = {
        "writeFormat": "ascii",
        "writePrecision": "17",
        "writeCompression": "off",
        "runTimeModifiable": "false",
    }
    if end_time is not None:
        sets["endTime"] = repr(end_time)
        sets["writeInterval"] = repr(
            end_time
        )  # with the tutorial's writeControl (adjustable hits it exactly)
    if libs:
        sets["libs"] = "(" + " ".join(f'"{x}"' for x in libs) + ")"
    for k, v in sets.items():
        _foam_dict(case, "system/controlDict", k, v, bashrc)
    if drop_functions:
        fcase.run_command(
            case, "foamDictionary system/controlDict -entry functions -remove", bashrc
        )
    fv = case / "system" / "fvSolution"
    fv.write_text(
        set_solver_entries(fv.read_text(), field_p, {k: str(v) for k, v in p_solver.items()})
    )
    for cmd in setup:
        fcase.run_command(case, cmd, bashrc, f"log.{cmd.split()[0]}")
    return case


def read_solver_dict(tutorial_dir, field_p: str) -> dict:
    """The tutorial's own solver controls of ``field_p`` (strings)."""
    return solver_entries((Path(tutorial_dir) / "system" / "fvSolution").read_text(), field_p)


def tutorial_comparison(
    tutorial_dir,
    work_dir,
    lib,
    application: str,
    field_p: str,
    compare_fields=("U",),
    setup=("blockMesh",),
    end_time: float | None = None,
    edit=None,
    n_snapshots: int = 8,
    ref_tolerance: float = 1e-12,
    repeats: int = 1,
    bashrc: str | None = None,
    warm_extra: dict | None = None,
    label: str = "",
) -> list[dict]:
    """An installed tutorial run three ways: its own ``field_p`` solver (plain), the same solver wrapped by
    ``pinneWarmStart`` (warm), and a reference with ``tolerance ref_tolerance; relTol 0``."""
    work = Path(work_dir)
    own = read_solver_dict(tutorial_dir, field_p)
    warm = {
        "solver": "pinneWarmStart",
        "innerSolver": own["solver"],
        "guess": "projection",
        "nSnapshots": str(n_snapshots),
        **{k: v for k, v in own.items() if k != "solver"},
        **{k: str(v) for k, v in (warm_extra or {}).items()},
    }
    ref = {**own, "tolerance": repr(ref_tolerance), "relTol": "0"}
    name = Path(tutorial_dir).name
    dirs, runs = {}, {}
    for kind, ctrl, reps in (
        ("reference", ref, 1),
        ("plain", own, repeats),
        ("warm", warm, repeats),
    ):
        best = None
        for _ in range(max(1, reps)):
            d = prepare_tutorial(
                tutorial_dir,
                work / f"{name}_{kind}",
                ctrl,
                field_p,
                (str(lib),),
                setup,
                end_time,
                edit,
                bashrc,
            )
            log, wall = fcase.run_command(d, application, bashrc, f"log.{application}")
            solves, exec_t = fcase.parse_log(log)
            if best is None or wall < best[0]:
                best = (wall, exec_t, solves)
        dirs[kind], runs[kind] = d, best
    rows = []
    for kind in ("plain", "warm"):
        wall, exec_t, solves = runs[kind]
        ps = solves[field_p]
        row = {
            "case": name,
            "inner": own["solver"]
            + ("-" + own["preconditioner"] if "preconditioner" in own else ""),
            "run": kind + (f" ({label})" if label and kind == "warm" else ""),
            f"{field_p}_solves": ps.n_solves,
            f"{field_p}_iterations": ps.iterations,
            "wall_s": round(wall, 3),
            "exec_s": exec_t,
        }
        for f in (field_p, *compare_fields):
            row[f"d{f}_vs_plain"] = fcase.field_difference(dirs["plain"], dirs[kind], f)
            row[f"d{f}_vs_ref"] = fcase.field_difference(dirs["reference"], dirs[kind], f)
        rows.append(row)
    return rows


def offline_benchmark(
    systems: list[LduSystem],
    guesses: dict,
    solver_specs: dict,
    tolerance: float = 1e-6,
    rel_tol: float = 0.0,
) -> list[dict]:
    """Re-solve a run's systems in order with every (guess, solver) pair.

    ``guesses``: name -> factory returning a fresh warm-start object (``guess``/``update``), or None for the
    incoming field (what OpenFOAM uses). ``solver_specs``: name -> ``callable(system, x0, tolerance, rel_tol)``
    returning a :class:`solvers.SolveResult`. Each guesser is updated with OpenFOAM's solution so all pairs see
    the same sequence. Rows: total iterations, solve and guess times, worst relative difference to OpenFOAM.
    """
    rows = []
    for gname, factory in guesses.items():
        for sname, solve in solver_specs.items():
            ws = factory() if factory else None
            iters = 0
            t_solve = t_guess = 0.0
            worst = 0.0
            r0s = []
            for s in systems:
                t0 = time.perf_counter()
                x0 = s.psi0 if ws is None else ws.guess(s, s.psi0)
                t_guess += time.perf_counter() - t0
                r0s.append(normalised_residual(s, x0))
                res = solve(s, x0, tolerance, rel_tol)
                t_solve += res.wall_time
                iters += res.iterations
                ref = s.psi
                worst = max(
                    worst, float(np.linalg.norm(res.x - ref) / max(np.linalg.norm(ref), 1e-300))
                )
                if ws is not None:
                    ws.update(s.psi)
            rows.append(
                {
                    "guess": gname,
                    "solver": sname,
                    "systems": len(systems),
                    "iterations": iters,
                    "solve_s": round(t_solve, 3),
                    "guess_s": round(t_guess, 3),
                    "mean_initial_residual": float(np.mean(r0s)),
                    "max_rel_diff_vs_openfoam": worst,
                }
            )
    return rows


def default_solver_specs(include_amg: bool = True, include_torch: bool = True) -> dict:
    specs = {
        "PCG-DIC (numpy)": lambda s, x0, tol, rt: pcg(s, x0, "dic", tol, rt),
        "PCG-Jacobi (numpy)": lambda s, x0, tol, rt: pcg(s, x0, "jacobi", tol, rt),
    }
    if include_amg:
        try:
            import pyamg  # noqa: F401

            specs["PCG-AMG (pyamg)"] = lambda s, x0, tol, rt: pcg(s, x0, "amg", tol, rt)
        except ImportError:
            pass
    if include_torch:
        try:
            import torch  # noqa: F401

            specs["PCG-Jacobi (torch f64)"] = lambda s, x0, tol, rt: torch_pcg(
                s, x0, tol, rt, dtype="float64"
            )
            specs["PCG-Jacobi (torch f32+f64 residual)"] = lambda s, x0, tol, rt: torch_pcg(
                s, x0, tol, rt, dtype="float32"
            )
        except ImportError:
            pass
    return specs


def default_guesses(n_snapshots: int = 8) -> dict:
    return {
        "previous field (OpenFOAM default)": None,
        f"projection, last {n_snapshots}": lambda: ProjectionWarmStart(n_snapshots),
    }


def markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    columns = columns or list(rows[0])

    def fmt(v):
        if isinstance(v, float):
            return (
                f"{v:.3g}"
                if (abs(v) < 1e-2 or abs(v) >= 1e4) and v != 0
                else f"{v:.3f}".rstrip("0").rstrip(".")
            )
        return str(v)

    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(out)
