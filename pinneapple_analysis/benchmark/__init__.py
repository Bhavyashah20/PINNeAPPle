"""Equal-budget benchmarks of physics-AI methods: prepare -> validate -> run -> report.

``run`` (or the CLI ``pinneapple benchmark run``) trains every method on every problem under the same
``ComputeBudget``, with an equal hyper-parameter search, several seeds and budget levels, an equal-FLOPs arm, and writes
everything to ``out_dir`` (re-running resumes from its cache). Correctness gates run by default (problem, data,
architecture, training) and are recorded; the report has a Validation section (PINNeAPPle's validation stack on the
prepared artifacts), a trust card and an advisor's "Diagnostics and suggestions".

Extend it with ``@register_problem`` / ``@register_method``, or wrap your arrays with ``ArrayProblem``.
"""
from pinneapple_analysis.cost.equal_budget import ComputeBudget

from .advisor import Diagnosis
from .api import BenchmarkResult, load, prepare, report, run
from .builtin.fields import ArrayProblem, FieldOperatorProblem
from .config import BenchmarkConfig
from .registry import (
           Check,
           Column,
           Method,
           Problem,
           get_method,
           get_problem,
           methods,
           problems,
           register_method,
           register_problem,
)
from .runner import ConfigMismatch, PreparationError

__all__ = ["run", "prepare", "report", "load", "BenchmarkResult", "BenchmarkConfig", "ComputeBudget", "Problem",
           "Method", "ArrayProblem", "FieldOperatorProblem", "Check", "Column", "Diagnosis", "register_problem",
           "register_method", "get_problem", "get_method", "problems", "methods", "ConfigMismatch", "PreparationError"]
