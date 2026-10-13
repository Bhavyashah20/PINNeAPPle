"""Long work as jobs: a separate process per job, with a wall-time limit, a memory limit and no network.

Each job has a directory ``<workdir>/jobs/<job_id>/`` holding ``request.json``, ``status.json``, ``log.txt`` and
whatever the job writes (``result.json`` and its files). The status lives on disk, so ``job_status`` still answers
after the server restarts (a job whose process is gone is reported as ``lost``).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Config

_PROCS: dict[str, Any] = {}
_LOCK = threading.Lock()


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(o: Any):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _no_network() -> None:
    """Refuse every non-local socket connection in this process."""
    import socket

    real_connect = socket.socket.connect

    def connect(self, address):  # noqa: ANN001
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host.startswith("127.") or host in ("localhost", "::1")
                                      or self.family == getattr(socket, "AF_UNIX", None)):
            return real_connect(self, address)
        raise PermissionError("network access is disabled in PINNeAPPle MCP jobs (PINNEAPPLE_MCP_ALLOW_NETWORK=1)")

    socket.socket.connect = connect  # type: ignore[method-assign]
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE"):
        os.environ[var] = "offline" if var == "WANDB_MODE" else "1"


def _child(kind: str, args: dict[str, Any], job_dir: str, cfg: dict[str, Any]) -> None:
    jd = Path(job_dir)
    log = open(jd / "log.txt", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        if cfg.get("max_memory_mb"):
            try:
                import resource
                lim = int(cfg["max_memory_mb"]) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except (ImportError, ValueError, OSError):  # not on this platform
                pass
        if not cfg.get("allow_network"):
            _no_network()
        os.environ.setdefault("MPLBACKEND", "Agg")
        from . import tools
        from .config import Config, set_config
        set_config(Config.from_dict(cfg))
        _write_json(jd / "status.json", {"status": "running", "pid": os.getpid(), "started": time.time()})
        result = tools.JOB_KINDS[kind](args, jd)
        _write_json(jd / "result.json", result)
        _write_json(jd / "status.json", {"status": "done", "pid": os.getpid(), "finished": time.time()})
    except BaseException as exc:  # noqa: BLE001 - every failure is reported in the job status
        traceback.print_exc()
        err = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, MemoryError):
            err += f" (memory limit {cfg.get('max_memory_mb')} MB, PINNEAPPLE_MCP_MAX_MEMORY_MB)"
        _write_json(jd / "status.json", {"status": "failed", "error": err, "finished": time.time()})
    finally:
        log.flush()


def _watch(job_id: str, proc, jd: Path, max_seconds: float) -> None:
    proc.join(max_seconds)
    if proc.is_alive():
        proc.kill()                                   # by PID: this job's process only
        proc.join(10)
        _write_json(jd / "status.json", {"status": "timeout", "finished": time.time(),
                                         "error": f"killed after {max_seconds:g} s (PINNEAPPLE_MCP_MAX_SECONDS)"})
    else:
        st = _read(jd / "status.json") or {}
        if st.get("status") in (None, "queued", "running"):
            _write_json(jd / "status.json", {"status": "failed", "finished": time.time(),
                                             "error": f"job process exited with code {proc.exitcode}"})
    with _LOCK:
        _PROCS.pop(job_id, None)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def running() -> list[str]:
    with _LOCK:
        return [j for j, p in _PROCS.items() if p.is_alive()]


def submit(kind: str, args: dict[str, Any], cfg: Config, max_seconds: float | None = None) -> str:
    """Start job ``kind`` (a key of ``tools.JOB_KINDS``) in a new process; returns its id."""
    cfg.ensure()
    if len(running()) >= cfg.max_jobs:
        raise RuntimeError(f"{cfg.max_jobs} jobs are already running (PINNEAPPLE_MCP_MAX_JOBS); wait or cancel one")
    job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    jd = cfg.workdir / "jobs" / job_id
    jd.mkdir(parents=True)
    limit = min(float(max_seconds or cfg.max_seconds), cfg.max_seconds)
    _write_json(jd / "request.json", {"kind": kind, "args": args, "max_seconds": limit, "created": time.time()})
    _write_json(jd / "status.json", {"status": "queued", "created": time.time()})
    ctx = mp.get_context("spawn")                     # a clean interpreter: no state shared with the server
    proc = ctx.Process(target=_child, args=(kind, args, str(jd), cfg.to_dict()), daemon=True)
    proc.start()
    with _LOCK:
        _PROCS[job_id] = proc
    threading.Thread(target=_watch, args=(job_id, proc, jd, limit), daemon=True).start()
    return job_id


def status(job_id: str, cfg: Config, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Status of a job (``queued``, ``running``, ``done``, ``failed``, ``timeout``, ``cancelled``, ``lost``), its
    result when done, and the files in its directory. ``wait_seconds`` waits up to that long for it to finish."""
    if "/" in job_id or "\\" in job_id or job_id.startswith("."):
        raise ValueError("invalid job id")
    jd = cfg.workdir / "jobs" / job_id
    if not jd.is_dir():
        raise KeyError(f"unknown job '{job_id}'")
    deadline = time.time() + max(0.0, float(wait_seconds))
    while True:
        st = _read(jd / "status.json") or {"status": "unknown"}
        if st.get("status") not in ("queued", "running") or time.time() >= deadline:
            break
        time.sleep(0.25)
    if st.get("status") in ("queued", "running"):
        with _LOCK:
            alive = job_id in _PROCS and _PROCS[job_id].is_alive()
        if not alive and st.get("status") == "running":
            st = {**st, "status": "lost", "error": "the job process is gone (server restarted?)"}
    out: dict[str, Any] = {"job_id": job_id, **{k: v for k, v in st.items() if k != "pid"}, "job_dir": str(jd)}
    if st.get("status") == "done":
        out["result"] = _read(jd / "result.json")
    if st.get("status") in ("failed", "timeout", "lost"):
        log = (jd / "log.txt").read_text()[-3000:] if (jd / "log.txt").exists() else ""
        out["log_tail"] = log
    out["files"] = sorted(str(p) for p in jd.rglob("*") if p.is_file())
    return out


def cancel(job_id: str, cfg: Config) -> dict[str, Any]:
    with _LOCK:
        proc = _PROCS.get(job_id)
    if proc is None or not proc.is_alive():
        return status(job_id, cfg)
    proc.kill()
    proc.join(10)
    _write_json(cfg.workdir / "jobs" / job_id / "status.json", {"status": "cancelled", "finished": time.time()})
    return status(job_id, cfg)


def list_jobs(cfg: Config, limit: int = 20) -> list[dict[str, Any]]:
    root = cfg.workdir / "jobs"
    if not root.is_dir():
        return []
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)[:limit]
    out = []
    for d in dirs:
        st = _read(d / "status.json") or {}
        req = _read(d / "request.json") or {}
        out.append({"job_id": d.name, "kind": req.get("kind"), "status": st.get("status"), "error": st.get("error")})
    return out


Runner = Callable[[dict[str, Any], Path], dict[str, Any]]
