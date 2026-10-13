"""Limits and locations of the MCP server. Every value has a safe default and an environment variable.

| setting | default | environment variable |
|---|---|---|
| working directory (results, jobs) | ``./pinneapple_mcp_work`` | ``PINNEAPPLE_MCP_WORKDIR`` |
| wall time per job [s] | 600 | ``PINNEAPPLE_MCP_MAX_SECONDS`` |
| memory per job [MB] (address space of the job process) | 8192 | ``PINNEAPPLE_MCP_MAX_MEMORY_MB`` |
| concurrent jobs | 2 | ``PINNEAPPLE_MCP_MAX_JOBS`` |
| training epochs per run | 20000 | ``PINNEAPPLE_MCP_MAX_EPOCHS`` |
| grid points of a checked field | 250000 | ``PINNEAPPLE_MCP_MAX_POINTS`` |
| size of a file read by ``trust_card`` [MB] | 200 | ``PINNEAPPLE_MCP_MAX_FILE_MB`` |
| network inside jobs | off | ``PINNEAPPLE_MCP_ALLOW_NETWORK=1`` |
| roots files may be read from | working directory and current directory | ``PINNEAPPLE_MCP_ROOTS`` (``os.pathsep`` separated) |
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _env(name: str, default: Any, cast=float):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


@dataclass
class Config:
    workdir: Path = field(default_factory=lambda: Path(os.environ.get("PINNEAPPLE_MCP_WORKDIR", "pinneapple_mcp_work")))
    max_seconds: float = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_SECONDS", 600.0))
    max_memory_mb: int = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_MEMORY_MB", 8192, int))
    max_jobs: int = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_JOBS", 2, int))
    max_epochs: int = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_EPOCHS", 20000, int))
    max_points: int = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_POINTS", 250_000, int))
    max_file_mb: float = field(default_factory=lambda: _env("PINNEAPPLE_MCP_MAX_FILE_MB", 200.0))
    allow_network: bool = field(default_factory=lambda: os.environ.get("PINNEAPPLE_MCP_ALLOW_NETWORK", "0") == "1")
    roots: list[Path] = field(default_factory=lambda: [Path(p) for p in os.environ.get(
        "PINNEAPPLE_MCP_ROOTS", "").split(os.pathsep) if p])

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.roots = [Path(r).expanduser().resolve() for r in self.roots] or [self.workdir, Path.cwd().resolve()]
        if self.workdir not in self.roots:
            self.roots.append(self.workdir)

    def ensure(self) -> Config:
        (self.workdir / "jobs").mkdir(parents=True, exist_ok=True)
        return self

    def readable(self, path: str) -> Path:
        """``path`` resolved, if it lies inside an allowed root and is not too large; else ``PermissionError``."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.workdir / p if (self.workdir / p).exists() else Path.cwd() / p
        p = p.resolve()
        if not any(p == r or r in p.parents for r in self.roots):
            raise PermissionError(f"{p} is outside the allowed roots {[str(r) for r in self.roots]} "
                                  "(set PINNEAPPLE_MCP_ROOTS to widen them)")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        if p.stat().st_size > self.max_file_mb * 1e6:
            raise ValueError(f"{p} is larger than {self.max_file_mb:g} MB (PINNEAPPLE_MCP_MAX_FILE_MB)")
        return p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["workdir"] = str(self.workdir)
        d["roots"] = [str(r) for r in self.roots]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Config:
        return cls(**d) if d else cls()


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def set_config(cfg: Config) -> Config:
    global _CONFIG
    _CONFIG = cfg
    return cfg
