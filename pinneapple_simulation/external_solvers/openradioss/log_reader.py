"""Read the energy-balance error that the OpenRadioss engine prints to its log.

During an explicit run the engine prints, at its output cycles, the energy-balance error of the model as
``ERR=<value>%`` (percent). A large value flags a run whose energy balance does not close and whose results should
not be trusted without inspection. ``run_openradioss_case`` writes this output to ``<case>/engine.log``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["read_energy_errors", "read_energy_error"]

_ERR = re.compile(r"ERR=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%")


def _log_text(case_dir_or_log: str | Path) -> str:
    p = Path(case_dir_or_log)
    if p.is_dir():
        p = p / "engine.log"
    return p.read_text(errors="replace")


def read_energy_errors(case_dir_or_log: str | Path) -> np.ndarray:
    """Every ``ERR=<value>%`` printed by the engine, in order, in percent.

    ``case_dir_or_log`` is a case directory (its ``engine.log`` is read) or the path of a log file.
    """
    return np.array([float(v) for v in _ERR.findall(_log_text(case_dir_or_log))], dtype=float)


def read_energy_error(case_dir_or_log: str | Path) -> float:
    """Largest ``|ERR|`` (energy-balance error, percent) printed during the run; ``nan`` if the log has none."""
    errs = read_energy_errors(case_dir_or_log)
    return float(np.abs(errs).max()) if errs.size else float("nan")
