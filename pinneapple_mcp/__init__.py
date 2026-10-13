"""PINNeAPPle as an MCP server: solve, benchmark and validate physics problems from any coding agent.

    pip install "pinneapple[mcp]"
    pinneapple-mcp                       # stdio server; or: python -m pinneapple_mcp

The tools (``pinneapple_mcp.tools``) are plain functions with JSON inputs and outputs, usable without MCP:

* ``list_methods`` / ``describe_method``: solve methods and the method catalog with its validation status;
* ``list_presets``: problem presets;
* ``solve``: a problem (preset or JSON spec) by a method, with physics checks and a trust card (a job);
* ``trust_card``: physics checks of a field given inline or in a file (residual, boundary conditions,
  conservation, reference, applicability);
* ``benchmark``: several methods on one problem with the same training budget (a job);
* ``job_status`` / ``list_jobs`` / ``cancel_job``: long work runs as a job in a separate process;
* ``capabilities``: what this installation offers (optional APIs, limits, working directory).

Safe defaults (``pinneapple_mcp.config``): every result goes to a working directory and its paths are returned;
jobs run in a child process with a wall-time limit, a memory limit and the network switched off; files are read
only from allowed roots. ``import pinneapple_mcp`` does not import the MCP SDK.
"""
from __future__ import annotations

__all__ = ["tools", "config", "build_server", "main"]


def build_server(*args, **kwargs):
    from .server import build_server as _b
    return _b(*args, **kwargs)


def main(argv=None):
    from .server import main as _m
    return _m(argv)
