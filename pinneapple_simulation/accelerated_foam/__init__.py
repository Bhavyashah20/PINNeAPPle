"""Faster implicit pressure solves for OpenFOAM, verified against plain OpenFOAM.

OpenFOAM's pressure-based solvers (PISO, PIMPLE, SIMPLE) spend most of a time step in the pressure equation.
This package changes only the *initial guess* of that linear solve and leaves the solver and its stopping test
to OpenFOAM, so the converged fields satisfy the same tolerance as a plain run:

- ``plugin/pinneWarmStart.C`` -- an OpenFOAM linear-solver wrapper (``solver pinneWarmStart;``) that projects
  each new pressure system on the span of the last solutions (Galerkin projection), or asks Python for a guess
  through named pipes, then calls the ordinary inner solver (PCG, GAMG, ...). It can also write every system
  to disk. :func:`build.build_plugin` compiles it against the installed OpenFOAM.
- :mod:`ldu` -- read those systems (lduMatrix arrays) and convert them to SciPy / PyTorch sparse matrices.
- :mod:`solvers` -- CG with Jacobi, DIC and AMG (pyamg) preconditioners and a PyTorch CG (float64 or mixed
  precision), all with OpenFOAM's normalised-residual stopping test.
- :mod:`warm_start` -- projection on past solutions and POD of earlier runs; :mod:`learned` -- a small graph
  network that corrects the guess (PyTorch).
- :mod:`guess_server` -- serve Python guesses to a running OpenFOAM solver.
- :mod:`case` -- write the benchmark cases, run them, parse logs and compare fields; :mod:`benchmark` -- the
  comparisons reported in ``docs/core_concepts/openfoam_acceleration.md``.
"""

from __future__ import annotations

from . import case, ldu, solvers, warm_start
from .ldu import LduSystem, from_csr, normalised_residual, read_dump, read_system
from .warm_start import PODWarmStart, ProjectionWarmStart, galerkin

__all__ = [
    "LduSystem",
    "PODWarmStart",
    "ProjectionWarmStart",
    "case",
    "from_csr",
    "galerkin",
    "ldu",
    "normalised_residual",
    "read_dump",
    "read_system",
    "solvers",
    "warm_start",
]
