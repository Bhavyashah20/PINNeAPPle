"""Thin, honest wrapper around the REA command-line tool.

REA ("Reverse Engineer Anything", https://github.com/morluto/rea, MIT) is a
Node.js CLI / MCP server that inspects software *without its source code* --
JavaScript/Electron bundles, websites, native binaries (through Hopper,
Ghidra or IDA), .NET assemblies, WASM, Android packages -- and returns
**Evidence JSON**: every finding carries the artifact's SHA-256, the
operation that produced it, its confidence/authority and its limitations.

This module only runs that CLI and parses what it prints. It never
interprets, completes or "fixes" REA's results: whatever REA did not
observe stays absent. That matches the rule used by
:mod:`pinneapple_llm.research` (results trace back to a real tool output,
never to an LLM's recall).

Resolution of the executable, in order:

1. ``PINNEAPPLE_REA_BIN`` -- an explicit path (or a command line, split on
   whitespace, e.g. ``"npx -y rea-agents@6.4.0"``);
2. ``rea`` on ``PATH`` (``npm install --global rea-agents``);
3. ``npx -y rea-agents@<PINNED_VERSION>`` if ``npx`` is on ``PATH``.

Nothing is installed implicitly beyond what ``npx`` itself caches.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "PINNED_VERSION",
    "ReaUnavailable",
    "ReaError",
    "ReaResult",
    "ReaClient",
    "resolve_rea_command",
]

#: Version this bridge was written and tested against (REA changes quickly;
#: the Evidence schema is versioned per provider, see ``ReaResult.provider``).
PINNED_VERSION = "6.4.0"


class ReaUnavailable(RuntimeError):
    """No REA executable could be found (see module docstring)."""


class ReaError(RuntimeError):
    """REA ran but exited non-zero or printed something that is not JSON."""

    def __init__(self, message: str, *, returncode: int, stderr_tail: str):
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


@dataclass
class ReaResult:
    """One REA invocation: the command, the parsed JSON payload and the
    progress events REA streamed on stderr (kept for the audit trail)."""

    command: List[str]
    payload: Any
    progress: List[Dict[str, Any]] = field(default_factory=list)
    stderr_tail: str = ""

    # Convenience accessors for Evidence-shaped payloads (most commands).
    @property
    def evidence_id(self) -> Optional[str]:
        return self.payload.get("evidence_id") if isinstance(self.payload, dict) else None

    @property
    def subject_sha256(self) -> Optional[str]:
        if not isinstance(self.payload, dict):
            return None
        if "root_sha256" in self.payload:          # HistoricalSourceGraph
            return self.payload["root_sha256"]
        return ((self.payload.get("subject") or {}).get("digest") or {}).get("sha256")

    @property
    def provider(self) -> Optional[Dict[str, Any]]:
        return self.payload.get("provider") if isinstance(self.payload, dict) else None

    @property
    def limitations(self) -> List[str]:
        if not isinstance(self.payload, dict):
            return []
        out = list(self.payload.get("limitations") or [])
        nr = self.payload.get("normalized_result")
        if isinstance(nr, dict):
            out += [x for x in (nr.get("limitations") or []) if x not in out]
        return out


def resolve_rea_command(env: Optional[Dict[str, str]] = None) -> List[str]:
    """Return the argv prefix that runs REA, or raise :class:`ReaUnavailable`."""
    env = os.environ if env is None else env
    explicit = env.get("PINNEAPPLE_REA_BIN")
    if explicit:
        parts = shlex.split(explicit)
        if parts and (shutil.which(parts[0]) or Path(parts[0]).is_file()):
            return parts
        raise ReaUnavailable(f"PINNEAPPLE_REA_BIN={explicit!r} does not point to an executable")
    found = shutil.which("rea")
    if found:
        return [found]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", f"rea-agents@{PINNED_VERSION}"]
    raise ReaUnavailable(
        "REA not found. Install Node.js >= 22.19 and run `npm install --global "
        f"rea-agents@{PINNED_VERSION}`, or set PINNEAPPLE_REA_BIN."
    )


def _split_stderr(stderr: str) -> tuple[List[Dict[str, Any]], str]:
    """REA streams ``{"rea_progress": {...}}`` lines on stderr; keep them
    structured and return the remaining text separately."""
    progress, other = [], []
    for line in stderr.splitlines():
        s = line.strip()
        if s.startswith("{") and '"rea_progress"' in s:
            try:
                progress.append(json.loads(s)["rea_progress"])
                continue
            except (ValueError, KeyError):
                pass
        if s:
            other.append(line)
    return progress, "\n".join(other[-40:])


class ReaClient:
    """Run REA commands and parse their JSON output.

    Parameters
    ----------
    command : argv prefix to run REA; resolved with
        :func:`resolve_rea_command` when omitted.
    timeout_s : wall-clock limit per invocation.
    cwd : working directory for the subprocess (REA itself takes absolute
        target paths, so this rarely matters).
    """

    def __init__(self, command: Optional[Sequence[str]] = None, *,
                 timeout_s: float = 900.0, cwd: Optional[str] = None):
        self.command = list(command) if command else resolve_rea_command()
        self.timeout_s = timeout_s
        self.cwd = cwd

    # -- generic --------------------------------------------------------
    def run(self, subcommand: str, *args: str, json_output: bool = True) -> ReaResult:
        argv = [*self.command, subcommand, *args]
        if json_output:
            argv += ["--format", "json"]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout_s, cwd=self.cwd)
        except FileNotFoundError as exc:  # executable vanished
            raise ReaUnavailable(str(exc)) from exc
        progress, tail = _split_stderr(proc.stderr)
        if proc.returncode != 0:
            raise ReaError(f"rea {subcommand} exited with {proc.returncode}",
                           returncode=proc.returncode, stderr_tail=tail or proc.stdout[-2000:])
        if not json_output:
            return ReaResult(argv, proc.stdout, progress, tail)
        try:
            payload = json.loads(proc.stdout)
        except ValueError as exc:
            raise ReaError(f"rea {subcommand} did not print JSON",
                           returncode=proc.returncode, stderr_tail=proc.stdout[-2000:]) from exc
        return ReaResult(argv, payload, progress, tail)

    # -- common operations ---------------------------------------------
    def version(self) -> str:
        return str(self.run("--version", json_output=False).payload).strip()

    def doctor(self) -> Any:
        return self.run("doctor").payload

    def analyze_javascript_application(self, path: str | os.PathLike, *,
                                       analysis_timeout_ms: Optional[int] = None) -> ReaResult:
        """Static reconstruction of a JavaScript/Electron app directory or ASAR.

        Parses the code as inert text (nothing is executed) and returns the
        application graph plus a semantic graph (functions, call sites,
        bindings) with source ranges.
        """
        args = [str(Path(path).resolve())]
        if analysis_timeout_ms is not None:
            args += ["--analysis-timeout-ms", str(int(analysis_timeout_ms))]
        return self.run("analyze-javascript-application", *args)

    def import_reference_source(self, root: str | os.PathLike) -> ReaResult:
        """Pin a source tree as *historical reference* (inventory, languages,
        root SHA-256, git HEAD). REA has no Python/C++ semantic analyser, so
        for such projects this is the provenance record; the study itself is
        a direct reading of the source (``Finding.method="manual"``)."""
        return self.run("import-reference-source", str(Path(root).resolve()))

    def inspect_binary_layout(self, path: str | os.PathLike) -> ReaResult:
        """ELF sections, segments, symbols and relocations, without executing
        the object. REA delegates to pwntools: set ``REA_PWNTOOLS_PYTHON`` to a
        Python with pwntools 4.15 / pyelftools 0.33 / Unicorn 2.1.2."""
        return self.run("inspect-binary-layout", str(Path(path).resolve()))

    def inspect_artifact(self, path: str | os.PathLike) -> ReaResult:
        """Identify one artifact (binary, archive, package) and list next probes."""
        return self.run("inspect-artifact", str(Path(path).resolve()))
