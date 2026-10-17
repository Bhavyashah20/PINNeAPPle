"""Shared plumbing for the PINNeAPPle apps (apps/*): login, busy limiter, static vendor files, and the runtime
(job queue, API keys, usage metering, uploads) under /api.

Each app does::

    from appkit import install, install_runtime  # apps/_shared on sys.path
    install(app, prefix="HSS")                    # reads HSS_USER / HSS_PASSWORD
    rt = install_runtime(app, prefix="HSS", name="heatsink", units={"runs": 200})

Environment (all optional, per app prefix; the full list is in apps/_shared/README.md):
  <PREFIX>_USER / <PREFIX>_PASSWORD   HTTP Basic auth on everything except /health (an X-API-Key also passes)
  <PREFIX>_MAX_JOBS                   concurrent jobs of the queue (all server workers together)
  <PREFIX>_FREE_<UNIT>                free use per principal and month, e.g. CAMPAIGN_FREE_CASES_PLANNED
"""
from __future__ import annotations

import base64
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_core.service import (
    APIKeyStore,
    JobQueue,
    UploadRejected,
    UploadStore,
    UploadTooLarge,
    UsageLimitExceeded,
    UsageMeter,
    data_dir,
    limits_from_env,
)


def _vendor_dir() -> str:
    """three.js ships once, inside the library (pinneapple_tools/visualization/studio/web/vendor), so the apps and
    pp.viz's browser viewer serve the same copy. Located without importing the package."""
    import importlib.util
    spec = importlib.util.find_spec("pinneapple_tools")
    root = list(spec.submodule_search_locations)[0]
    return os.path.join(root, "visualization", "studio", "web", "vendor")


VENDOR = _vendor_dir()
STUDIO = os.path.dirname(VENDOR)                      # studio-core.js: colour scales, tubes, slices, edges
SHARED = os.path.join(os.path.dirname(__file__), "static", "shared")


def install(app: FastAPI, prefix: str) -> None:
    """Optional HTTP Basic login + the shared static files: ``/vendor`` (three.js), ``/studio`` (the studio viewer
    core) and ``/shared`` (UI pieces common to every app)."""
    app.state.auth_user = os.environ.get(f"{prefix}_USER")
    app.state.auth_password = os.environ.get(f"{prefix}_PASSWORD")

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        user, password = app.state.auth_user, app.state.auth_password
        rt = getattr(app.state, "runtime", None)
        login = _basic_user(request, user, password) if user and password else None
        key = rt.keys.verify(request.headers.get("x-api-key")) if rt is not None else None
        if rt is not None and request.headers.get("x-api-key") and key is None:
            return JSONResponse({"detail": "invalid or revoked API key"}, status_code=401)
        request.state.login = login
        request.state.api_key = key
        request.state.principal = (f"key:{key['id']}" if key else f"user:{login}" if login
                                   else f"ip:{client_ip(request)}")
        if request.url.path != "/health":
            if user and password and not (login or key):
                return Response(status_code=401,
                                headers={"WWW-Authenticate": f'Basic realm="{app.title}"'})
            if rt is not None and rt.require_key and request.url.path.startswith("/api/") and not (login or key):
                return JSONResponse({"detail": "this server requires an X-API-Key header"}, status_code=401)
        return await call_next(request)

    app.mount("/vendor", StaticFiles(directory=VENDOR), name="vendor")
    app.mount("/studio", StaticFiles(directory=STUDIO), name="studio")
    app.mount("/shared", StaticFiles(directory=SHARED), name="shared")


class BusyLimiter:
    """Caps concurrent heavy requests per worker; the excess gets 429 instead of queueing."""

    def __init__(self, env_var: str, default: int = 2):
        self.slots = threading.BoundedSemaphore(int(os.environ.get(env_var) or default))

    def __enter__(self):
        if not self.slots.acquire(blocking=False):
            raise HTTPException(status_code=429,
                                detail="Server busy with other runs -- please retry in a few seconds.")
        return self

    def __exit__(self, *exc):
        self.slots.release()
        return False


def _basic_user(request: Request, user: str, password: str) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            u, _, p = base64.b64decode(header[6:]).decode().partition(":")
        except (ValueError, UnicodeDecodeError):
            return None
        if secrets.compare_digest(u, user) and secrets.compare_digest(p, password):
            return u
    return None


def client_ip(request: Request) -> str:
    """The client address; behind Caddy the last X-Forwarded-For entry (the one the proxy itself added)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def principal(request: Request) -> str:
    """Who use is counted against: ``key:<id>``, ``user:<login>`` or ``ip:<addr>``."""
    return getattr(request.state, "principal", None) or f"ip:{client_ip(request)}"


class Runtime:
    """The job queue, API keys, usage meter and upload store of one app (``app.state.runtime``)."""

    def __init__(self, name: str, prefix: str, root: Path, jobs: JobQueue, keys: APIKeyStore, meter: UsageMeter,
                 uploads: UploadStore, admin_token: str | None, require_key: bool):
        self.name, self.prefix, self.root = name, prefix, root
        self.jobs, self.keys, self.meter, self.uploads = jobs, keys, meter, uploads
        self.admin_token, self.require_key = admin_token, require_key

    def is_admin(self, request: Request) -> bool:
        if getattr(request.state, "login", None):
            return True
        token = request.headers.get("x-admin-token")
        return bool(self.admin_token and token and secrets.compare_digest(token, self.admin_token))

    def consume(self, request: Request, unit: str, amount: float = 1, ref: str = "") -> None:
        """Count ``amount`` of ``unit`` against the caller; a 429 response if it passes the free limit."""
        self.meter.consume(principal(request), unit, amount, ref)

    def submit(self, request: Request, kind: str, params: dict[str, Any] | None = None, *,
               units: dict[str, float] | None = None, max_retries: int = 0) -> dict[str, Any]:
        """Check the free limits for ``units``, queue the job owned by the caller and count the units."""
        who = principal(request)
        for unit, amount in (units or {}).items():
            self.meter.check(who, unit, amount)
        job_id = self.jobs.submit(kind, params, owner=who, max_retries=max_retries)
        for unit, amount in (units or {}).items():
            self.meter.record(who, unit, amount, ref=job_id)
        return self.jobs.get(job_id)

    def job_for(self, request: Request, job_id: str) -> dict[str, Any]:
        """The job if the caller owns it (or is admin); 404 otherwise."""
        try:
            job = self.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown job {job_id}") from None
        if job["owner"] != principal(request) and not self.is_admin(request):
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    def upload_path(self, request: Request, upload_id: str) -> Path:
        try:
            meta = self.uploads.get(upload_id)
        except KeyError:
            raise HTTPException(404, f"unknown upload {upload_id}") from None
        if meta.get("owner") != principal(request) and not self.is_admin(request):
            raise HTTPException(404, f"unknown upload {upload_id}")
        return self.uploads.path(upload_id)

    def describe(self) -> dict[str, Any]:
        return {"app": self.name, "max_concurrent_jobs": self.jobs.max_concurrent, "executor": self.jobs.executor,
                "job_kinds": sorted(self.jobs.stats()["kinds"]), "free_limits_per_month": self.meter.limits,
                "upload_max_mb": self.uploads.max_bytes / 2**20,
                "upload_suffixes": sorted(self.uploads.allowed) if self.uploads.allowed else None,
                "api_key_required": self.require_key}


def install_runtime(app: FastAPI, prefix: str, name: str, *, units: dict[str, float | None] | None = None,
                    max_jobs: int = 2, executor: str = "thread", job_max_seconds: float | None = None,
                    upload_max_mb: float = 50, upload_suffixes: set[str] | None = None) -> Runtime:
    """Job queue, API keys, usage metering and uploads for an app, with their routes under ``/api``.

    ``units`` are the metered units and their default free limit per principal and month (None = unlimited).
    Call after :func:`install`. Register job functions with ``rt.jobs.register(kind, fn)``.
    """
    env = os.environ
    root = data_dir(name, prefix)
    db = root / "runtime.sqlite"
    jobs = JobQueue(db, app=name, max_concurrent=int(env.get(f"{prefix}_MAX_JOBS") or max_jobs),
                    executor=env.get(f"{prefix}_JOB_EXECUTOR") or executor, workdir=root / "jobs",
                    max_seconds=float(env.get(f"{prefix}_JOB_MAX_SECONDS") or 0) or job_max_seconds)
    meter = UsageMeter(db, app=name, limits=limits_from_env(prefix, dict(units or {})))
    uploads = UploadStore(root / "uploads", max_bytes=int(float(env.get(f"{prefix}_UPLOAD_MAX_MB") or upload_max_mb)
                                                          * 2**20), allowed_suffixes=upload_suffixes)
    rt = Runtime(name, prefix, root, jobs, APIKeyStore(db, app=name), meter, uploads,
                 admin_token=env.get(f"{prefix}_ADMIN_TOKEN") or None,
                 require_key=(env.get(f"{prefix}_REQUIRE_API_KEY") or "").lower() in ("1", "true", "yes"))
    app.state.runtime = rt

    @app.exception_handler(UsageLimitExceeded)
    async def _limit(request: Request, exc: UsageLimitExceeded):
        return JSONResponse(exc.to_dict(), status_code=429)

    api = APIRouter(prefix="/api", tags=["runtime"])

    def admin(request: Request) -> None:
        if not rt.is_admin(request):
            raise HTTPException(403, f"needs the app login or X-Admin-Token ({prefix}_ADMIN_TOKEN)")

    @api.get("/runtime")
    def runtime_info():
        return rt.describe()

    @api.get("/jobs")
    def list_jobs(request: Request, status: str | None = None, kind: str | None = None, limit: int = 50):
        owner = None if rt.is_admin(request) else principal(request)
        return rt.jobs.list(owner=owner, status=status, kind=kind, limit=min(limit, 500))

    @api.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str):
        return rt.job_for(request, job_id)

    @api.get("/jobs/{job_id}/logs")
    def job_logs(request: Request, job_id: str, since: int = 0):
        rt.job_for(request, job_id)
        return rt.jobs.logs(job_id, since=since)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(request: Request, job_id: str):
        rt.job_for(request, job_id)
        return rt.jobs.cancel(job_id)

    @api.post("/jobs/{job_id}/retry")
    def retry_job(request: Request, job_id: str):
        rt.job_for(request, job_id)
        try:
            return rt.jobs.retry(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @api.get("/usage")
    def usage(request: Request):
        out = rt.meter.usage(principal(request))
        if rt.is_admin(request):
            out["totals"] = rt.meter.totals()
        return out

    @api.get("/keys")
    def list_keys(request: Request):
        admin(request)
        return rt.keys.list()

    @api.post("/keys")
    def create_key(request: Request, body: dict[str, Any]):
        admin(request)
        return rt.keys.create(str(body.get("name") or "unnamed"), scopes=body.get("scopes"))

    @api.delete("/keys/{key_id}")
    def revoke_key(request: Request, key_id: str):
        admin(request)
        try:
            return rt.keys.revoke(key_id)
        except KeyError:
            raise HTTPException(404, f"unknown key {key_id}") from None

    @api.post("/uploads")
    async def upload(request: Request, file: UploadFile):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > rt.uploads.max_bytes + 64 * 1024:
            raise HTTPException(413, f"upload larger than {rt.uploads.max_bytes / 2**20:g} MB")
        try:
            return rt.uploads.save(file.filename or "upload", file.file, owner=principal(request))
        except UploadTooLarge as exc:
            raise HTTPException(413, str(exc)) from None
        except UploadRejected as exc:
            raise HTTPException(415, str(exc)) from None

    @api.get("/uploads")
    def list_uploads(request: Request):
        return rt.uploads.list(owner=None if rt.is_admin(request) else principal(request))

    @api.delete("/uploads/{upload_id}")
    def delete_upload(request: Request, upload_id: str):
        rt.upload_path(request, upload_id)
        rt.uploads.delete(upload_id)
        return {"deleted": upload_id}

    app.include_router(api)
    return rt
