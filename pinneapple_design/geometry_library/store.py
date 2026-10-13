"""Pre-generated files: the manifest (SHA-256 of every file), the local copies, the Hugging Face Hub, the cache.

Light files (STL + glTF) are committed under ``data/geometries/<name>/`` and the thumbnails under
``docs/assets/geometry_library/``; every format of both levels
of detail is published to the Hub dataset ``PINNeAPPle-Labs/geometry-library`` by the ``geometry-library``
workflow. :func:`load` looks, in order, in the repository copy, the cache (``~/.cache/pinneapple/geometries`` or
``$PINNEAPPLE_GEOMETRY_CACHE``) and the Hub, checks the SHA-256 against the manifest, and, if the file cannot be
had (offline, not uploaded yet, or a hash mismatch), builds it from the code with the default parameters, which
reproduce the stored files.

Environment: ``PINNEAPPLE_GEOMETRY_HUB`` (another Hub repo id), ``PINNEAPPLE_GEOMETRY_CACHE``,
``PINNEAPPLE_OFFLINE=1`` or ``HF_HUB_OFFLINE=1`` (never touch the network).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

from .exporters import normalise_format

HUB_REPO = "PINNeAPPle-Labs/geometry-library"
HUB_REPO_TYPE = "dataset"
MANIFEST = Path(__file__).with_name("manifest.json")
REPO_DATA = Path(__file__).resolve().parents[2] / "data" / "geometries"
COMMITTED = {("light", "stl"), ("light", "glb")}       # the files that live in git (the rest is on the Hub)


def file_name(name: str, lod: str, fmt: str) -> str:
    return f"{name}.{lod}.{normalise_format(fmt)}"


THUMBS = Path(__file__).resolve().parents[2] / "docs" / "assets" / "geometry_library"


def thumbnail_path(name: str) -> Path:
    return THUMBS / f"{name}.png"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"version": 1, "hub_repo": HUB_REPO, "repo_type": HUB_REPO_TYPE, "geometries": {}}


def write_manifest(m: dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")


def cache_dir() -> Path:
    d = os.environ.get("PINNEAPPLE_GEOMETRY_CACHE")
    return Path(d) if d else Path.home() / ".cache" / "pinneapple" / "geometries"


def offline() -> bool:
    return os.environ.get("PINNEAPPLE_OFFLINE", "") not in ("", "0") or os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")


def hub_url(name: str, fname: str) -> str:
    repo = os.environ.get("PINNEAPPLE_GEOMETRY_HUB", HUB_REPO)
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{name}/{fname}"


def _entry(name: str, fname: str) -> dict[str, Any] | None:
    return read_manifest().get("geometries", {}).get(name, {}).get("files", {}).get(fname)


def _verified(path: Path, entry: dict[str, Any] | None) -> bool:
    return path.exists() and (entry is None or sha256(path) == entry["sha256"])


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-geometry-library"})
        with urllib.request.urlopen(req, timeout=timeout) as r, tempfile.NamedTemporaryFile(
                dir=dest.parent, delete=False) as tmp:
            shutil.copyfileobj(r, tmp)
        os.replace(tmp.name, dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def load(name: str, format: str = "stl", lod: str = "light", *, allow_build: bool = True,
         allow_download: bool = True) -> Path:
    """Path to the pre-generated file of ``name`` (``format``: stl, glb, obj, usd / usda, step; ``lod``: light or
    fine). See the module docstring for the lookup order."""
    from .core import LODS, get_spec
    get_spec(name)                                   # unknown names fail fast
    if lod not in LODS:
        raise ValueError(f"lod must be one of {sorted(LODS)}")
    fmt = normalise_format(format)
    fname = file_name(name, lod, fmt)
    entry = _entry(name, fname)
    local = REPO_DATA / name / fname
    if _verified(local, entry):
        return local
    cached = cache_dir() / name / fname
    if _verified(cached, entry):
        return cached
    if allow_download and entry is not None and not offline():
        if _download(hub_url(name, fname), cached):
            if _verified(cached, entry):
                return cached
            warnings.warn(f"{fname}: the downloaded file does not match the manifest SHA-256; rebuilding it", stacklevel=2)
            cached.unlink(missing_ok=True)
    if not allow_build:
        raise FileNotFoundError(f"{fname} is not available locally or on the Hub ({hub_url(name, fname)})")
    if entry is not None:
        warnings.warn(f"{fname} could not be fetched; building it from the code (default parameters)", stacklevel=2)
    from .core import build
    g = build(name, lod=lod)
    g.export(str(cached), fmt)
    if entry is not None and sha256(cached) != entry["sha256"]:
        warnings.warn(f"{fname}: the rebuilt file differs bitwise from the manifest (platform floating point); "
                      "the geometry agrees within the tested tolerance", stacklevel=2)
    return cached


__all__ = ["load", "read_manifest", "write_manifest", "sha256", "file_name", "hub_url", "cache_dir", "HUB_REPO",
           "REPO_DATA", "MANIFEST", "COMMITTED", "thumbnail_path"]
